// BatchedPuct — C++ PUCT trees for the Q_ref producer.
//
// Division of labour: C++ owns trees (select / expand / backup), move
// generation (chess.hpp), prior masking over the 1924-way policy, and the
// 68-token board encoding for the oracle. Python owns only the batched GPU
// eval. Semantics mirror chessdecoder/agent/rl/qref.py exactly (parity
// tested): C_PUCT, FPU reduction, terminal values, castling dual-spelling
// prior collapse, unkeyed-move exclusion, unvisited-q floor.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "chess.hpp"

namespace py = pybind11;
using chess::Board;
using chess::Move;
using chess::Movelist;

static constexpr float C_PUCT = 1.5f;
static constexpr float FPU_RED = 0.2f;
static constexpr int POLICY_DIM = 1924;

// uci key -> policy index (both castling spellings are distinct keys)
static std::unordered_map<std::string, int> g_move_map;
// 68-token encoding constants (from chessdecoder.dataloader.loader)
static int g_start, g_end, g_empty, g_wtm, g_btm, g_nocastle;
static int g_piece[12];                      // PNBRQK pnbrqk
static std::unordered_map<std::string, int> g_castle_map;

static int piece_token(const chess::Piece &p) {
    static const std::string order = "PNBRQKpnbrqk";
    char c = static_cast<std::string>(p)[0];
    auto i = order.find(c);
    return i == std::string::npos ? g_empty : g_piece[i];
}

// python move_keys(): [uci] + king-from + rook-file + rank for castling.
// chess.hpp castling moves have to() = rook square, so the lc0 spelling is
// just from+to; uci::moveToUci gives the e1g1 form.
static void move_key_logits(const Board &b, const Move &mv,
                            const float *pol, float &best_logit, bool &found) {
    best_logit = -1e30f;
    found = false;
    std::string k1 = chess::uci::moveToUci(mv);
    auto it = g_move_map.find(k1);
    if (it != g_move_map.end()) { best_logit = pol[it->second]; found = true; }
    if (mv.typeOf() == Move::CASTLING) {
        std::string k2;
        k2 += static_cast<std::string>(mv.from().file());
        k2 += static_cast<std::string>(mv.from().rank());
        k2 += static_cast<std::string>(mv.to().file());
        k2 += static_cast<std::string>(mv.to().rank());
        auto it2 = g_move_map.find(k2);
        if (it2 != g_move_map.end() && (!found || pol[it2->second] > best_logit)) {
            best_logit = pol[it2->second];
            found = true;
        }
    }
}

struct Node {
    std::vector<Move> moves;
    std::vector<float> P, W;
    std::vector<int32_t> N, child;           // child id, -1 = unexpanded
    float v = 0.f;
    bool expanded = false;
    float terminal_v = 1e9f;                  // 1e9 = not terminal
};

struct Tree {
    Board root;
    std::vector<Node> nodes;
    // per-wave state
    std::vector<std::pair<int, int>> path;    // (node, child idx)
    int leaf = -1;
    Board leaf_board;
    bool want_eval = false;
    bool done_wave = false;

    explicit Tree(const std::string &fen) : root(fen) { nodes.emplace_back(); }

    void select() {
        path.clear();
        want_eval = false;
        Board b = root;
        int nid = 0;
        while (true) {
            Node &n = nodes[nid];
            if (n.terminal_v < 1e8f) {        // known terminal: backup, done
                backup(n.terminal_v);
                return;
            }
            if (!n.expanded) {
                // fresh leaf: terminal check, else needs oracle eval
                auto [reason, result] = b.isGameOver();
                if (reason != chess::GameResultReason::NONE) {
                    float tv = (reason == chess::GameResultReason::CHECKMATE)
                                   ? -1.f : 0.f;
                    n.terminal_v = tv;
                    backup(tv);
                    return;
                }
                leaf = nid;
                leaf_board = b;
                want_eval = true;
                return;
            }
            float sqrt_total = std::sqrt(std::max<float>(
                1.f, std::accumulate(n.N.begin(), n.N.end(), 0)));
            int best = 0;
            float best_s = -1e30f;
            for (size_t i = 0; i < n.moves.size(); ++i) {
                float q = n.N[i] > 0 ? n.W[i] / n.N[i] : n.v - FPU_RED;
                float u = C_PUCT * n.P[i] * sqrt_total / (1.f + n.N[i]);
                if (q + u > best_s) { best_s = q + u; best = static_cast<int>(i); }
            }
            path.emplace_back(nid, best);
            b.makeMove(n.moves[best]);
            if (n.child[best] == -1) {
                n.child[best] = static_cast<int32_t>(nodes.size());
                nodes.emplace_back();
            }
            nid = n.child[best];
        }
    }

    void backup(float v_leaf) {
        float v = v_leaf;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            v = -v;
            Node &n = nodes[it->first];
            n.N[it->second] += 1;
            n.W[it->second] += v;
        }
    }

    void expand_backup(float value, const float *pol) {
        Node &n = nodes[leaf];
        Movelist ml;
        chess::movegen::legalmoves(ml, leaf_board);
        std::vector<float> logits;
        for (const auto &mv : ml) {
            float lg; bool found;
            move_key_logits(leaf_board, mv, pol, lg, found);
            if (!found) continue;             // unkeyed (underpromo): exclude
            n.moves.push_back(mv);
            logits.push_back(lg);
        }
        float mx = *std::max_element(logits.begin(), logits.end());
        float sum = 0.f;
        n.P.resize(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            n.P[i] = std::exp(logits[i] - mx);
            sum += n.P[i];
        }
        for (auto &p : n.P) p /= sum;
        n.N.assign(n.moves.size(), 0);
        n.W.assign(n.moves.size(), 0.f);
        n.child.assign(n.moves.size(), -1);
        n.v = value;
        n.expanded = true;
        backup(value);
    }
};

static void encode68(const Board &b, int32_t *out) {
    out[0] = g_start;
    for (int sq = 0; sq < 64; ++sq)
        out[1 + sq] = piece_token(b.at(static_cast<chess::Square>(sq)));
    out[65] = g_end;
    std::string cr = b.getCastleString();
    out[66] = cr.empty() ? g_nocastle : g_castle_map.at(cr);
    out[67] = b.sideToMove() == chess::Color::WHITE ? g_wtm : g_btm;
}

class BatchedPuct {
  public:
    explicit BatchedPuct(const std::vector<std::string> &fens) {
        for (const auto &f : fens) trees_.emplace_back(f);
    }

    // one wave: run select in every tree; terminal sims back up internally.
    // returns (tree indices needing eval, ids [n,68] int32 oracle input)
    std::pair<std::vector<int>, py::array_t<int32_t>> select() {
        std::vector<int> idx;
        for (size_t t = 0; t < trees_.size(); ++t) {
            trees_[t].select();
            if (trees_[t].want_eval) idx.push_back(static_cast<int>(t));
        }
        py::array_t<int32_t> ids({static_cast<py::ssize_t>(idx.size()),
                                  static_cast<py::ssize_t>(68)});
        auto buf = ids.mutable_unchecked<2>();
        for (size_t j = 0; j < idx.size(); ++j)
            encode68(trees_[idx[j]].leaf_board, &buf(j, 0));
        return {idx, ids};
    }

    // values [n] stm-POV, policy [n,1924] raw logits
    void expand_backup(const std::vector<int> &idx,
                       py::array_t<float> values, py::array_t<float> policy) {
        auto v = values.unchecked<1>();
        auto p = policy.unchecked<2>();
        for (py::ssize_t j = 0; j < v.shape(0); ++j)
            trees_[idx[j]].expand_backup(v(j), &p(j, 0));
    }

    // per tree: (moves uci, q, visits, oracle_greedy, search_best)
    py::list results() {
        py::list out;
        for (auto &t : trees_) {
            Node &r = t.nodes[0];
            py::list moves;
            std::vector<float> q(r.moves.size());
            std::vector<int> vis(r.moves.size());
            float min_vq = 1e9f;
            for (size_t i = 0; i < r.moves.size(); ++i)
                if (r.N[i] > 0)
                    min_vq = std::min(min_vq, r.W[i] / r.N[i]);
            float floor = min_vq < 1e8f ? std::max(-1.f, min_vq - 0.1f) : -1.f;
            int best = 0, greedy = 0;
            double best_key = -1e18;
            for (size_t i = 0; i < r.moves.size(); ++i) {
                moves.append(chess::uci::moveToUci(r.moves[i]));
                q[i] = r.N[i] > 0 ? r.W[i] / r.N[i] : floor;
                vis[i] = r.N[i];
                if (r.P[i] > r.P[greedy]) greedy = static_cast<int>(i);
                double key = static_cast<double>(r.N[i]) + q[i];
                if (key > best_key) { best_key = key; best = static_cast<int>(i); }
            }
            out.append(py::make_tuple(moves, q, vis, greedy, best));
        }
        return out;
    }

  private:
    std::vector<Tree> trees_;
};

PYBIND11_MODULE(_puct_cpp, m) {
    m.def("init_maps", [](const std::unordered_map<std::string, int> &move_map,
                          const std::unordered_map<std::string, int> &consts,
                          const std::unordered_map<std::string, int> &castle_map) {
        g_move_map = move_map;
        g_start = consts.at("start"); g_end = consts.at("end");
        g_empty = consts.at("empty"); g_wtm = consts.at("wtm");
        g_btm = consts.at("btm"); g_nocastle = consts.at("nocastle");
        static const std::string order = "PNBRQKpnbrqk";
        for (int i = 0; i < 12; ++i)
            g_piece[i] = consts.at(std::string(1, order[i]));
        g_castle_map = castle_map;
    });
    py::class_<BatchedPuct>(m, "BatchedPuct")
        .def(py::init<const std::vector<std::string> &>())
        .def("select", &BatchedPuct::select)
        .def("expand_backup", &BatchedPuct::expand_backup)
        .def("results", &BatchedPuct::results);
}

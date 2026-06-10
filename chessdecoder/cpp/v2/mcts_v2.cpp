// V2 PUCT MCTS implementation.
//
// Each MCTS node = one chess::Board state. Leaf evaluation calls BoardForward
// on the position's 68-token encoding only — no history, no thinking trace
// (the "first-board policy" mode the V2 architecture supports natively).
//
// Batched leaf expansion + virtual loss: each PUCT-search round collects up
// to cfg.max_batch_leaves leaves before calling the network, so the GPU
// stays fed. Virtual loss is the standard AlphaZero trick — temporarily
// subtract from W/add to N along the just-traversed path so the next
// selection avoids the same line until the real backup arrives.
#include "mcts_v2.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace v2 {

namespace {

// Map a chess::Move to the V2 move sub-vocab id (or -1 if not in vocab).
// V2 was trained with king-takes-rook castling notation (e1h1/e1a1/...) but
// chess-library emits king-jumps-two (e1g1/e1c1/...) by default — translate.
int move_to_sub_id(const chess::Move& m, const chess::Board& board,
                   const Vocab& vocab) {
  std::string uci = chess::uci::moveToUci(m, /*chess960=*/false);
  if (m.typeOf() == chess::Move::CASTLING) {
    if (uci == "e1g1") uci = "e1h1";
    else if (uci == "e1c1") uci = "e1a1";
    else if (uci == "e8g8") uci = "e8h8";
    else if (uci == "e8c8") uci = "e8a8";
  }
  (void)board;
  return vocab.uci_to_move_sub_id(uci);
}

// UCI string used in user-facing output (visits policy + chosen action).
// The training-time string is the king-takes-rook form for castles; the
// played move side wants the standard form (e1g1) because that's what
// downstream PGN / engine play expects. Mirror the V2 predict_move
// post-translation table.
std::string display_uci(const std::string& sub_uci) {
  if (sub_uci == "e1h1") return "e1g1";
  if (sub_uci == "e1a1") return "e1c1";
  if (sub_uci == "e8h8") return "e8g8";
  if (sub_uci == "e8a8") return "e8c8";
  return sub_uci;
}

struct Edge {
  chess::Move move;
  int sub_id = -1;     // policy sub-vocab id (for prior lookup)
  float prior = 0.f;   // P(s,a) — softmax over legal moves only
  int N = 0;           // real visits
  float W = 0.f;       // sum of backed-up values (from this node's stm POV)
  int VL = 0;          // virtual loss counter
  int child = -1;      // node id of child, or -1 if not expanded
};

struct Node {
  chess::Board board;
  bool expanded = false;
  bool terminal = false;
  // value at terminal positions, from the side-to-move at this node's POV.
  // (LOSE for stm = checkmated = -1; DRAW = 0.)
  float terminal_value = 0.f;
  std::vector<Edge> edges;
  int parent = -1;
  int parent_edge = -1;
};

struct PendingLeaf {
  int node;                        // leaf node id
  std::vector<int> path_nodes;     // path from root to leaf (excl. leaf)
  std::vector<int> path_edges;     // edge indices taken at each path node
};

// PUCT score for one edge under virtual loss.
inline float puct_score(const Edge& e, int parent_visits, float cpuct) {
  const float Neff = static_cast<float>(e.N + e.VL);
  const float Q = (e.N + e.VL > 0) ? (e.W - e.VL) / Neff : 0.f;
  const float U = cpuct * e.prior *
                  std::sqrt(static_cast<float>(std::max(1, parent_visits))) /
                  (1.f + Neff);
  return Q + U;
}

inline int select_child(const Node& n, float cpuct) {
  int best = 0;
  float best_score = -1e30f;
  // parent_visits = sum_a N(s,a) — equivalently 1 + sum_a N (the +1 from
  // the leaf eval itself). Using sum N here keeps the formula standard.
  int parent_visits = 0;
  for (const auto& e : n.edges) parent_visits += e.N + e.VL;
  for (size_t i = 0; i < n.edges.size(); ++i) {
    float s = puct_score(n.edges[i], parent_visits, cpuct);
    if (s > best_score) {
      best_score = s;
      best = static_cast<int>(i);
    }
  }
  return best;
}

// Apply virtual loss along a path (root -> leaf parent).
void apply_virtual_loss(std::vector<Node>& nodes, const PendingLeaf& p) {
  for (size_t i = 0; i < p.path_nodes.size(); ++i) {
    auto& e = nodes[p.path_nodes[i]].edges[p.path_edges[i]];
    e.VL += 1;
  }
}

void undo_virtual_loss(std::vector<Node>& nodes, const PendingLeaf& p) {
  for (size_t i = 0; i < p.path_nodes.size(); ++i) {
    auto& e = nodes[p.path_nodes[i]].edges[p.path_edges[i]];
    e.VL -= 1;
  }
}

// Back up a leaf value along the path. `leaf_value` is from the leaf node's
// own side-to-move POV. Each step up, the perspective flips (parent's stm
// is opposite of child's stm), so we negate at each level.
//
// Implementation: we record `node_value` for each level walking back from
// leaf. The "value" each edge sees is the value FROM the parent's POV (since
// the edge belongs to the parent). The child was reached from the parent;
// the leaf's W/L is from the leaf-stm's POV; the parent saw the move and
// expects the OPPOSITE sign.
void backup(std::vector<Node>& nodes, const PendingLeaf& p,
            float leaf_value) {
  // value seen by each *ancestor edge*: alternates starting from -leaf_value
  // at the deepest edge (parent of the leaf), then +leaf_value at grandparent,
  // etc.
  float v = -leaf_value;
  for (int i = static_cast<int>(p.path_nodes.size()) - 1; i >= 0; --i) {
    auto& e = nodes[p.path_nodes[i]].edges[p.path_edges[i]];
    e.N += 1;
    e.W += v;
    v = -v;
  }
}

}  // namespace

V2Mcts::V2Mcts(std::shared_ptr<BoardForward> net,
               std::shared_ptr<Vocab> vocab,
               MctsConfig cfg)
    : net_(std::move(net)), vocab_(std::move(vocab)), cfg_(cfg) {}

MctsResult V2Mcts::search(const std::string& fen) {
  std::vector<Node> nodes;
  nodes.reserve(cfg_.simulations * 2 + 16);

  // ---- root setup
  Node root;
  root.board = chess::Board::fromFen(fen);
  nodes.push_back(std::move(root));

  // Helper: legal move generation + edge construction from a leaf's NN eval.
  auto expand_with_priors = [&](int node_id,
                                const std::vector<float>& policy_probs) {
    Node& n = nodes[node_id];
    chess::Movelist legal;
    chess::movegen::legalmoves(legal, n.board);

    if (legal.empty()) {
      // Position is terminal but caller already marked it; nothing to expand.
      return;
    }

    std::vector<float> prior_raw;
    prior_raw.reserve(legal.size());
    float sum = 0.f;
    for (size_t i = 0; i < legal.size(); ++i) {
      const auto& mv = legal[i];
      int sub = move_to_sub_id(mv, n.board, *vocab_);
      float p = (sub >= 0 && sub < static_cast<int>(policy_probs.size()))
                    ? policy_probs[sub]
                    : 0.f;
      prior_raw.push_back(p);
      sum += p;
    }
    // Renormalize over legal moves. If model assigned ~0 mass to legal moves
    // (very rare in practice but possible at OOD positions), fall back to
    // uniform.
    if (sum <= 1e-8f) {
      float u = 1.f / static_cast<float>(legal.size());
      for (auto& p : prior_raw) p = u;
    } else {
      for (auto& p : prior_raw) p /= sum;
    }

    n.edges.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      Edge e;
      e.move = legal[i];
      e.sub_id = move_to_sub_id(legal[i], n.board, *vocab_);
      e.prior = prior_raw[i];
      n.edges.push_back(std::move(e));
    }
    n.expanded = true;
  };

  // Selection loop: from root, descend by PUCT until reaching an unexpanded
  // node (a leaf to evaluate). Returns the leaf node id + the path taken.
  // If the leaf turns out to be terminal, it's expanded inline and we return
  // -1 to signal "no NN eval needed for this slot."
  auto select_leaf = [&](PendingLeaf& out) -> int {
    int cur = 0;  // root
    while (nodes[cur].expanded && !nodes[cur].terminal) {
      int ei = select_child(nodes[cur], cfg_.cpuct);
      out.path_nodes.push_back(cur);
      out.path_edges.push_back(ei);
      Edge& e = nodes[cur].edges[ei];

      if (e.child < 0) {
        // Need to create child node.
        Node child;
        child.board = nodes[cur].board;
        child.board.makeMove(e.move);
        child.parent = cur;
        child.parent_edge = ei;
        int new_id = static_cast<int>(nodes.size());
        nodes.push_back(std::move(child));
        e.child = new_id;
      }
      cur = e.child;
    }
    out.node = cur;
    return cur;
  };

  int sims_done = 0;

  // Expand root first (synchronously) so subsequent selections have edges.
  {
    auto root_ids_arr = vocab_->fen_to_board_ids(fen);
    std::vector<int64_t> root_ids(root_ids_arr.begin(), root_ids_arr.end());
    auto eval = net_->forward_one(root_ids);
    auto over = nodes[0].board.isGameOver();
    if (over.second != chess::GameResult::NONE) {
      nodes[0].terminal = true;
      nodes[0].expanded = true;
      nodes[0].terminal_value =
          (over.second == chess::GameResult::LOSE)   ? -1.f
          : (over.second == chess::GameResult::WIN)  ?  1.f
                                                     :  0.f;
    } else {
      expand_with_priors(0, eval.policy);
    }
  }

  while (sims_done < cfg_.simulations) {
    // ---- collect a batch of leaves
    int budget = std::min(cfg_.max_batch_leaves <= 0 ? 1 : cfg_.max_batch_leaves,
                          cfg_.simulations - sims_done);
    std::vector<PendingLeaf> pending;
    pending.reserve(budget);
    std::vector<std::vector<int64_t>> batch_inputs;
    batch_inputs.reserve(budget);

    for (int b = 0; b < budget; ++b) {
      PendingLeaf p;
      int leaf = select_leaf(p);
      Node& ln = nodes[leaf];

      // If this leaf was already expanded as terminal (handled inline), or
      // happens to be terminal: skip NN eval and back up terminal value.
      if (!ln.expanded) {
        auto over = ln.board.isGameOver();
        if (over.second != chess::GameResult::NONE) {
          ln.terminal = true;
          ln.expanded = true;
          ln.terminal_value =
              (over.second == chess::GameResult::LOSE)   ? -1.f
              : (over.second == chess::GameResult::WIN)  ?  1.f
                                                         :  0.f;
        }
      }

      if (ln.terminal) {
        backup(nodes, p, ln.terminal_value);
        sims_done += 1;
        // No virtual loss applied to this path (we didn't enter the queue),
        // so nothing to undo.
        continue;
      }

      // Real leaf — queue for NN eval, apply virtual loss along its path.
      apply_virtual_loss(nodes, p);
      auto ids = vocab_->fen_to_board_ids(ln.board.getFen(/*move_counters=*/true));
      batch_inputs.emplace_back(ids.begin(), ids.end());
      pending.push_back(std::move(p));
    }

    if (pending.empty()) continue;  // all picks were terminal -> next round

    // ---- batched network call
    auto evals = net_->forward_batch(batch_inputs);

    // ---- expand + back up each leaf
    for (size_t i = 0; i < pending.size(); ++i) {
      PendingLeaf& p = pending[i];
      const LeafEval& e = evals[i];

      Node& ln = nodes[p.node];
      if (!ln.expanded) {
        expand_with_priors(p.node, e.policy);
      }

      // Leaf value: Q = W - L, from the leaf's side-to-move POV.
      float leaf_v = e.w - e.l;

      undo_virtual_loss(nodes, p);
      backup(nodes, p, leaf_v);
      sims_done += 1;
    }
  }

  // ---- assemble result
  MctsResult r;
  r.sims_done = sims_done;
  r.root_w = 0.f;
  r.root_d = 0.f;
  r.root_l = 0.f;

  // root WDL: from the root's own leaf eval (the synchronous one we did
  // above isn't stashed — re-evaluate once at the end is one call, cheap).
  // Skip if root is terminal.
  if (!nodes[0].terminal) {
    auto root_arr = vocab_->fen_to_board_ids(fen);
    std::vector<int64_t> root_ids(root_arr.begin(), root_arr.end());
    auto root_eval = net_->forward_one(root_ids);
    r.root_w = root_eval.w;
    r.root_d = root_eval.d;
    r.root_l = root_eval.l;
  }

  // visit-count policy + per-move Q
  int total_visits = 0;
  for (const auto& e : nodes[0].edges) total_visits += e.N;

  std::vector<std::pair<std::string, int>> visit_pairs;
  visit_pairs.reserve(nodes[0].edges.size());
  for (const auto& e : nodes[0].edges) {
    const std::string& sub_uci = vocab_->move_sub_id_to_uci(e.sub_id);
    std::string disp = display_uci(sub_uci);
    r.policy.emplace_back(
        disp, total_visits > 0
                  ? static_cast<float>(e.N) / static_cast<float>(total_visits)
                  : 0.f);
    r.q_values.emplace_back(disp, e.N > 0 ? e.W / static_cast<float>(e.N) : 0.f);
    visit_pairs.emplace_back(disp, e.N);
  }

  // Action selection.
  if (visit_pairs.empty()) {
    r.action = "";
    return r;
  }

  if (cfg_.temperature <= 0.f) {
    // Argmax visits.
    int best = 0;
    for (size_t i = 1; i < visit_pairs.size(); ++i) {
      if (visit_pairs[i].second > visit_pairs[best].second) best = i;
    }
    r.action = visit_pairs[best].first;
  } else {
    // Softmax over (visits / temperature). Sample.
    double sum = 0.0;
    std::vector<double> w(visit_pairs.size());
    for (size_t i = 0; i < visit_pairs.size(); ++i) {
      w[i] = std::pow(static_cast<double>(visit_pairs[i].second),
                      1.0 / cfg_.temperature);
      sum += w[i];
    }
    if (sum <= 0.0) {
      r.action = visit_pairs[0].first;
    } else {
      static std::mt19937 rng{std::random_device{}()};
      std::uniform_real_distribution<double> U(0.0, sum);
      double u = U(rng);
      double cum = 0.0;
      int chosen = 0;
      for (size_t i = 0; i < w.size(); ++i) {
        cum += w[i];
        if (u <= cum) {
          chosen = static_cast<int>(i);
          break;
        }
      }
      r.action = visit_pairs[chosen].first;
    }
  }
  return r;
}

}  // namespace v2

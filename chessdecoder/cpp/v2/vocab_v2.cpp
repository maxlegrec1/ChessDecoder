// V2 vocab implementation.
//
// FEN tokenization mirrors chessdecoder/dataloader/data.py:fen_to_position_tokens
// exactly (and we test this via the bindings — see scripts/test_vocab_v2_parity.py).
//
// Vocab is loaded from a line-oriented vocab.txt produced by the Python
// exporter (avoids vendoring a JSON parser for one schema we control). Format:
//   BOARD <n>
//   <token> <id>           (n lines)
//   MOVES <n>
//   <uci>                  (n lines, in sub-vocab id order)
#include "vocab_v2.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace v2 {

namespace {

// Piece char (FEN convention) -> board token name.
const char* piece_char_to_token(char c) {
  switch (c) {
    case 'P': return "white_pawn";
    case 'N': return "white_knight";
    case 'B': return "white_bishop";
    case 'R': return "white_rook";
    case 'Q': return "white_queen";
    case 'K': return "white_king";
    case 'p': return "black_pawn";
    case 'n': return "black_knight";
    case 'b': return "black_bishop";
    case 'r': return "black_rook";
    case 'q': return "black_queen";
    case 'k': return "black_king";
    default:  return nullptr;
  }
}

// Split FEN into its 6 fields. Returns "" for missing trailing fields (legal
// short FENs only carry the first 4-5).
std::vector<std::string> split_fen(const std::string& fen) {
  std::vector<std::string> out;
  std::istringstream is(fen);
  std::string field;
  while (is >> field) out.push_back(field);
  while (out.size() < 6) out.emplace_back("");
  return out;
}

}  // namespace

Vocab Vocab::from_json(const std::string& path) {
  // `path` may be a vocab.json OR a vocab.txt; we look for the line-oriented
  // sibling. The Python exporter always writes both side-by-side.
  std::string txt_path = path;
  auto dot = txt_path.find_last_of('.');
  if (dot != std::string::npos &&
      (txt_path.substr(dot) == ".json" || txt_path.substr(dot) == ".txt")) {
    txt_path = txt_path.substr(0, dot) + ".txt";
  } else {
    txt_path += ".txt";
  }

  std::ifstream f(txt_path);
  if (!f) {
    throw std::runtime_error("vocab_v2: cannot open " + txt_path);
  }

  Vocab v;
  std::string header;
  int n_board = 0;

  if (!(f >> header >> n_board) || header != "BOARD") {
    throw std::runtime_error("vocab_v2: expected 'BOARD <n>' header in " +
                             txt_path);
  }
  for (int i = 0; i < n_board; ++i) {
    std::string tok;
    int64_t idx;
    if (!(f >> tok >> idx)) {
      throw std::runtime_error("vocab_v2: malformed BOARD row at i=" +
                               std::to_string(i));
    }
    v.token_to_idx_[tok] = idx;
  }

  int n_move = 0;
  if (!(f >> header >> n_move) || header != "MOVES") {
    throw std::runtime_error("vocab_v2: expected 'MOVES <n>' header");
  }
  v.move_sub_to_uci_.reserve(n_move);
  for (int i = 0; i < n_move; ++i) {
    std::string uci;
    if (!(f >> uci)) {
      throw std::runtime_error("vocab_v2: malformed MOVES row at i=" +
                               std::to_string(i));
    }
    v.uci_to_move_sub_[uci] = static_cast<int>(v.move_sub_to_uci_.size());
    v.move_sub_to_uci_.push_back(std::move(uci));
  }
  return v;
}

std::array<int64_t, kBoardSeqLen> Vocab::fen_to_board_ids(
    const std::string& fen) const {
  auto fields = split_fen(fen);
  const std::string& piece_field = fields[0];
  const std::string& stm_field = fields[1];
  const std::string& cas_field = fields[2];

  // ---- 1) parse piece placement into board[64], indexed in python-chess
  // SQUARES order (a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63 — file-major).
  // FEN walks ranks 8 down to 1; within each rank, files a to h.
  std::array<char, 64> board{};
  board.fill(0);  // 0 = empty (sentinel)
  int rank = 7;   // FEN starts at rank 8 (index 7)
  int file = 0;
  for (char c : piece_field) {
    if (c == '/') {
      --rank;
      file = 0;
    } else if (c >= '1' && c <= '8') {
      file += (c - '0');
    } else {
      int sq = rank * 8 + file;
      if (sq < 0 || sq > 63) {
        throw std::runtime_error("vocab_v2: FEN piece placement out of range");
      }
      board[sq] = c;
      ++file;
    }
  }

  // ---- 2) emit 68 tokens
  auto idx_of = [&](const char* name) -> int64_t {
    auto it = token_to_idx_.find(name);
    if (it == token_to_idx_.end()) {
      throw std::runtime_error(
          std::string("vocab_v2: token not in vocab: ") + name);
    }
    return it->second;
  };

  std::array<int64_t, kBoardSeqLen> out{};
  int o = 0;
  out[o++] = idx_of("start_pos");
  for (int sq = 0; sq < 64; ++sq) {
    char c = board[sq];
    if (c == 0) {
      out[o++] = idx_of("empty");
    } else {
      const char* tok = piece_char_to_token(c);
      if (tok == nullptr) {
        throw std::runtime_error("vocab_v2: unknown piece char in FEN");
      }
      out[o++] = idx_of(tok);
    }
  }
  out[o++] = idx_of("end_pos");

  // ---- 3) castling field: vocab tokens are sorted-K Q k q strings, or
  // "no_castling_rights". The Python tokenizer takes the rights present
  // *in K, Q, k, q order* (lines 35-38 of dataloader/data.py).
  std::string rights;
  for (char c : {'K', 'Q', 'k', 'q'}) {
    if (cas_field.find(c) != std::string::npos) rights += c;
  }
  out[o++] = idx_of(rights.empty() ? "no_castling_rights" : rights.c_str());

  // ---- 4) side to move
  out[o++] = idx_of(stm_field == "b" ? "black_to_move" : "white_to_move");

  return out;
}

const std::string& Vocab::move_sub_id_to_uci(int sub_id) const {
  static const std::string empty;
  if (sub_id < 0 || sub_id >= static_cast<int>(move_sub_to_uci_.size()))
    return empty;
  return move_sub_to_uci_[sub_id];
}

int Vocab::uci_to_move_sub_id(const std::string& uci) const {
  auto it = uci_to_move_sub_.find(uci);
  return it == uci_to_move_sub_.end() ? -1 : it->second;
}

}  // namespace v2

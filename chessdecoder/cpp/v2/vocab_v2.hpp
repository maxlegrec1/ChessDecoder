// V2 board + move sub-vocabulary (C++ port).
//
// The model expects a single `BoardForward` TorchScript module whose contract
// is fixed at export time. We mirror three constant artifacts from Python:
//
//   1) The 68-token board layout produced by `fen_to_position_tokens`
//      (start_pos + 64 squares + end_pos + castling + side_to_move).
//   2) The 41-token board sub-vocab (we don't *predict* board tokens in MCTS,
//      but we need the ids to tokenize FENs into the encoder's input).
//   3) The 1924-token move sub-vocab (policy_head output space).
//
// All three are loaded from a single `vocab.json` produced by the Python
// exporter alongside the .ts file — same convention as cpp/decoder/ used.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace v2 {

constexpr int kBoardSeqLen = 68;     // start_pos + 64 + end_pos + castling + stm
constexpr int kMoveVocab   = 1924;   // policy_head output

class Vocab {
 public:
  // Load board + move sub-vocab from a JSON dumped by the Python exporter.
  // The JSON layout matches Python: { "board_token_to_idx": {...},
  // "move_token_to_idx": {...}, "move_idx_to_full_idx": [...] }.
  static Vocab from_json(const std::string& path);

  // Tokenize a FEN to 68 full-vocab token ids.
  // (Encoder embedding table is over the full 1968-token vocab.)
  std::array<int64_t, kBoardSeqLen> fen_to_board_ids(const std::string& fen) const;

  // Decode a move sub-vocab id (0..1923) to UCI ("e2e4" / "e7e8q" / "e1g1").
  const std::string& move_sub_id_to_uci(int sub_id) const;

  // Encode a UCI string to its move sub-vocab id, or -1 if unknown.
  int uci_to_move_sub_id(const std::string& uci) const;

 private:
  // full vocab id of each board token in the order written by
  // fen_to_position_tokens
  std::unordered_map<std::string, int64_t> token_to_idx_;

  // sub-vocab move id -> UCI string
  std::vector<std::string> move_sub_to_uci_;

  // UCI -> sub-vocab id
  std::unordered_map<std::string, int> uci_to_move_sub_;
};

}  // namespace v2

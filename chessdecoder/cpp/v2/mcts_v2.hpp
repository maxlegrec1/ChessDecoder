// V2 PUCT MCTS — single-board evaluation per node.
//
// Each MCTS node corresponds to a chess::Board state. Leaf evaluation calls
// BoardForward on the position's 68-token encoding (no history, no thinking
// sequence — the "first board policy" mode per the design discussion). PUCT
// with virtual loss + batched leaf expansion so the GPU stays fed.
//
// Tree algorithm cribbed from chessdecoder/cpp/mcts/mcts_leela.cpp
// (LeelaMctsRunner), but trimmed: no TRT, no Leela-policy-index mapping,
// no parallel-tree harness. Single tree per call.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "board_forward.hpp"
#include "chess.hpp"
#include "vocab_v2.hpp"

namespace v2 {

struct MctsResult {
  std::string action;                                // chosen UCI move
  std::vector<std::pair<std::string, float>> policy; // visit-count distribution
  std::vector<std::pair<std::string, float>> q_values; // per-move avg Q
  // (W, D, L) at root (root's own WDL estimate from leaf eval)
  float root_w, root_d, root_l;
  int sims_done;
};

struct MctsConfig {
  int simulations = 800;
  float cpuct = 1.5f;
  float temperature = 1.0f;  // visit-count temperature for action selection
  // Virtual-loss + batched leaf expansion. 0 disables (one leaf / forward).
  int max_batch_leaves = 32;
};

class V2Mcts {
 public:
  V2Mcts(std::shared_ptr<BoardForward> net,
         std::shared_ptr<Vocab> vocab,
         MctsConfig cfg = {});

  // Run a search from `fen` (starting fresh tree). Returns root action +
  // visit-count policy + per-move Q values.
  MctsResult search(const std::string& fen);

 private:
  std::shared_ptr<BoardForward> net_;
  std::shared_ptr<Vocab> vocab_;
  MctsConfig cfg_;
};

}  // namespace v2

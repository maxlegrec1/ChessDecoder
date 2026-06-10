// V2 single-board policy/value forward.
//
// Wraps the `BoardForward` TorchScript module produced by
// chessdecoder/export/export_v2.py. The contract:
//
//   inputs : board_ids   [B, 68]  int64    (output of vocab_v2.fen_to_board_ids)
//   outputs: policy_logits [B, 1924] float
//            wdl           [B, 3]    float   ((W, D, L) probabilities)
//
// One call ≡ one node-leaf evaluation in MCTS. The batched form is the hot
// path: leaf collection batches up to `max_batch` positions per network call
// so the GPU stays fed under virtual-loss MCTS.
#pragma once

#include <string>
#include <vector>

#include <torch/script.h>

namespace v2 {

struct LeafEval {
  // policy[1924] over the move sub-vocab (full softmax — caller masks to legal)
  std::vector<float> policy;
  // (W, D, L) — direct readout of WDLHead's mean cell
  float w, d, l;
};

class BoardForward {
 public:
  // Construct + load the TS module onto `device` (default: cuda:0).
  // `module_path` is the .ts produced by export_v2.py.
  BoardForward(const std::string& module_path,
               const std::string& device = "cuda:0");

  // Single forward. `board_ids` is exactly kBoardSeqLen ints.
  LeafEval forward_one(const std::vector<int64_t>& board_ids);

  // Batched forward. Each row of `board_ids_batch` is one position (68 ints).
  // Returns one LeafEval per row.
  std::vector<LeafEval> forward_batch(
      const std::vector<std::vector<int64_t>>& board_ids_batch);

 private:
  torch::jit::Module module_;
  torch::Device device_;
};

}  // namespace v2

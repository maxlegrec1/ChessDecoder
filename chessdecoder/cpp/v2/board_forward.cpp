// V2 BoardForward wrapper implementation.
//
// Single libtorch jit::Module forward. Inputs are int64 ids, outputs are
// fp32 logits + WDL. Both legs are simple, no KV-cache state to manage
// (this is what makes V2 cpp ~10x smaller than V1).
#include "board_forward.hpp"

#include <stdexcept>

namespace v2 {

namespace {
constexpr int64_t kBoardLen = 68;
constexpr int64_t kMoveVocab = 1924;
constexpr int64_t kWdlSize = 3;
}  // namespace

BoardForward::BoardForward(const std::string& module_path,
                           const std::string& device)
    : device_(device) {
  try {
    module_ = torch::jit::load(module_path, device_);
    module_.eval();
  } catch (const c10::Error& e) {
    throw std::runtime_error("BoardForward: failed to load " + module_path +
                             ": " + e.what());
  }
}

LeafEval BoardForward::forward_one(const std::vector<int64_t>& board_ids) {
  return forward_batch({board_ids})[0];
}

std::vector<LeafEval> BoardForward::forward_batch(
    const std::vector<std::vector<int64_t>>& board_ids_batch) {
  const int64_t B = static_cast<int64_t>(board_ids_batch.size());
  if (B == 0) return {};

  // Pack into one [B, 68] int64 tensor on CPU, then move to device.
  // (TorchScript trace baked input on CPU at export; we move to device here.)
  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  auto ids_cpu = torch::empty({B, kBoardLen}, opts);
  auto* ids_ptr = ids_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < B; ++i) {
    if (static_cast<int64_t>(board_ids_batch[i].size()) != kBoardLen) {
      throw std::runtime_error(
          "BoardForward::forward_batch: each row must have 68 ids");
    }
    std::memcpy(ids_ptr + i * kBoardLen, board_ids_batch[i].data(),
                kBoardLen * sizeof(int64_t));
  }
  auto ids = ids_cpu.to(device_, /*non_blocking=*/true);

  // Forward: returns a tuple (policy_logits [B,1924], wdl [B,3]).
  torch::NoGradGuard nograd;
  std::vector<torch::jit::IValue> inputs{ids};
  auto out = module_.forward(inputs);
  if (!out.isTuple()) {
    throw std::runtime_error(
        "BoardForward: expected (policy, wdl) tuple from TorchScript");
  }
  auto tup = out.toTuple();
  if (tup->elements().size() != 2) {
    throw std::runtime_error("BoardForward: tuple has wrong arity");
  }
  auto policy_logits = tup->elements()[0].toTensor();   // [B,1924]
  auto wdl = tup->elements()[1].toTensor();             // [B,3]

  // Softmax on policy_logits (raw logits → probabilities).
  // We do softmax here so the MCTS side gets priors directly. Legal-move
  // masking happens at the MCTS leaf-expansion step (after we know the board).
  auto policy_probs = torch::softmax(policy_logits.to(torch::kFloat32),
                                     /*dim=*/-1);

  auto policy_cpu = policy_probs.to(torch::kCPU).contiguous();
  auto wdl_cpu = wdl.to(torch::kCPU).contiguous().to(torch::kFloat32);

  const float* p = policy_cpu.data_ptr<float>();
  const float* w = wdl_cpu.data_ptr<float>();

  std::vector<LeafEval> out_vec(B);
  for (int64_t i = 0; i < B; ++i) {
    out_vec[i].policy.assign(p + i * kMoveVocab,
                             p + (i + 1) * kMoveVocab);
    out_vec[i].w = w[i * kWdlSize + 0];
    out_vec[i].d = w[i * kWdlSize + 1];
    out_vec[i].l = w[i * kWdlSize + 2];
  }
  return out_vec;
}

}  // namespace v2

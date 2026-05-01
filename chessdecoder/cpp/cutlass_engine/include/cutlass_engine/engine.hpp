#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cutlass_engine/allocator.hpp"
#include "cutlass_engine/config.hpp"
#include "cutlass_engine/kv_cache.hpp"
#include "cutlass_engine/model.hpp"
#include "cutlass_engine/scheduler.hpp"
#include "cutlass_engine/stream.hpp"
#include "cutlass_engine/weights.hpp"

namespace cutlass_engine {

class ThinkingEngine {
public:
    // Construct: parse config.json, alloc arena, load weights, allocate model
    // workspace + KV cache.
    ThinkingEngine(const std::string& backbone_pt_unused,    // ignored for now (kept for API parity)
                   const std::string& weights_dir,
                   const std::string& vocab_json_unused,     // ignored
                   const std::string& config_json,
                   int batch_size);

    ~ThinkingEngine();

    // Sampling controls (mirror existing ThinkingBatchedInferenceEngine).
    void set_board_temperature(float t)   { board_t_ = t; }
    void set_think_temperature(float t)   { think_t_ = t; }
    void set_policy_temperature(float t)  { policy_t_ = t; }
    void set_wl_temperature(float t)      { wl_t_ = t; }
    void set_d_temperature(float t)       { d_t_ = t; }

    // Run rollouts on `fens`. Returns one result per FEN, in submission order.
    // N can exceed batch_size — continuous batching kicks in.
    std::vector<RolloutResult> predict_moves(const std::vector<std::string>& fens,
                                             float fallback_temperature);

    // Hot-swap weights without reallocation (RL update path).
    void update_weights(const std::string& weights_dir);

    int batch_size() const { return cfg_.batch_size; }

private:
    ModelConfig cfg_;
    Arena arena_;
    PinnedArena pinned_arena_;
    Stream stream_;

    ModelWeights w_{};
    ChessDecoderModel model_;
    KvCache kv_;
    Scheduler sched_;

    float board_t_{0.0f};
    float think_t_{0.0f};
    float policy_t_{0.0f};
    float wl_t_{0.5f};
    float d_t_{0.5f};
};

}  // namespace cutlass_engine

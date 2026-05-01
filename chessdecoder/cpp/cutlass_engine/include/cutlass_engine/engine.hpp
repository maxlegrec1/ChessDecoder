#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cstdint>

#include "cutlass_engine/allocator.hpp"
#include "cutlass_engine/config.hpp"
#include "cutlass_engine/kv_cache.hpp"
#include "cutlass_engine/model.hpp"
#include "cutlass_engine/scheduler.hpp"
#include "cutlass_engine/stream.hpp"
#include "cutlass_engine/weights.hpp"

namespace decoder { class DecoderVocab; }

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
    // N can exceed batch_size — chunked continuous batching kicks in.
    std::vector<RolloutResult> predict_moves(const std::vector<std::string>& fens,
                                             float fallback_temperature);

    // Full thinking-trace inference (state machine: MOVE → WL_D → BOARD → ...
    // → AFTER_END_VAR → FINAL).  Each FEN produces a full ThinkingResult-like
    // payload (token_ids, block_ids, wl/d entries, final_move + final wl/d).
    //
    // temp=0 → argmax (deterministic, matches Python run_thinking).
    // max_iters caps the number of variation iterations to avoid runaway.
    std::vector<RolloutResult> predict_moves_thinking(
        const std::vector<std::string>& fens, float temperature,
        int max_seq_len_cap, int max_iters);

    // Hot-swap weights without reallocation (RL update path).
    void update_weights(const std::string& weights_dir);

    int batch_size() const { return cfg_.batch_size; }

    // ---- Test/debug surface ----------------------------------------------
    // Run a single decode step over [B,1] input. Caller fills:
    //   ids        device int32 [B]
    //   pos        device int32 [B]
    //   active     device int32 [B]   (1 = run; 0 = skip + zero output)
    //   past_len   device int32 [B]   (will be incremented by 1 on active slots)
    // Returns hidden state in `out_h` (device __half [B, E]).
    // No fourier override.
    void forward_decode_test(std::uintptr_t ids, std::uintptr_t pos,
                             std::uintptr_t active, std::uintptr_t past_len,
                             std::uintptr_t out_h);

    // Debug: stop after `stop_after_layer` layers and return h_in + residual.
    // stop_after_layer == -1 means "before any layer" (just embedding).
    // stop_after_layer == cfg.num_layers means "all layers, before final norm".
    void forward_decode_partial(std::uintptr_t ids, std::uintptr_t pos,
                                std::uintptr_t active, std::uintptr_t past_len,
                                int stop_after_layer,
                                std::uintptr_t out_h_in,
                                std::uintptr_t out_residual);

private:
    ModelConfig cfg_;
    Arena arena_;
    PinnedArena pinned_arena_;
    Stream stream_;

    ModelWeights w_{};
    ChessDecoderModel model_;
    KvCache kv_;
    Scheduler sched_;
    std::unique_ptr<decoder::DecoderVocab> vocab_;

    // Per-call scratch.  Sized at construction.
    int32_t* d_ids_buf_{nullptr};       // [B, max_init_S] int32
    int32_t* d_pos_buf_{nullptr};       // [B, max_init_S] int32
    int32_t* d_block_buf_{nullptr};     // [B, max_init_S] int32
    int32_t* d_active_buf_{nullptr};    // [B] int32
    __half*  d_hidden_buf_{nullptr};    // [B, max_init_S, E]
    __half*  d_last_h_buf_{nullptr};    // [B, E] last-position hidden
    __half*  d_logits_buf_{nullptr};    // [B, move_vocab] (head outputs)
    bool*    d_legal_mask_{nullptr};    // [B, move_vocab]
    int32_t* d_idx_out_{nullptr};       // [B] int32

    int max_init_S_{71};

    // Thinking-path scratch: max-S sized [B, max_S] tensors for the variable-
    // length forward calls.  Sized at construction; reused across calls.
    int32_t* d_th_ids_{nullptr};
    int32_t* d_th_pos_{nullptr};
    int32_t* d_th_block_{nullptr};
    bool*    d_th_wl_pos_{nullptr};
    bool*    d_th_d_pos_{nullptr};
    __half*  d_th_wl_val_{nullptr};
    __half*  d_th_d_val_{nullptr};
    __half*  d_th_hidden_{nullptr};   // [B, max_S, E]
    __half*  d_th_last_h_{nullptr};   // [B, E] gather of hidden at chosen position

    float board_t_{0.0f};
    float think_t_{0.0f};
    float policy_t_{0.0f};
    float wl_t_{0.5f};
    float d_t_{0.5f};
};

}  // namespace cutlass_engine

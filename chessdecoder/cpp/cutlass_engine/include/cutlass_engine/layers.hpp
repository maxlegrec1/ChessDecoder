#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "cutlass_engine/config.hpp"
#include "cutlass_engine/kv_cache.hpp"
#include "cutlass_engine/weights.hpp"

namespace cutlass_engine {

// Per-call workspace — pre-allocated in the engine arena, sized for the
// largest call mode. All FP16.
struct LayerWorkspace {
    __half* h_in{nullptr};       // [M, E]
    __half* h_out{nullptr};      // [M, E]
    __half* residual{nullptr};   // [M, E]
    __half* qkv{nullptr};        // [M, 3*E]
    __half* attn_out{nullptr};   // [M, E]
    __half* gate_up{nullptr};    // [M, 2*d_ff]
    __half* mlp_inner{nullptr};  // [M, d_ff]
    __half* q_buf{nullptr};      // [M, E]   (reused across layers)
    __half* k_buf{nullptr};      // [M, E]
    __half* v_buf{nullptr};      // [M, E]
    int32_t* pos{nullptr};       // [M] int32
};

void allocate_layer_workspace(LayerWorkspace& ws, const ModelConfig& cfg,
                              int max_M, Arena& arena);

enum class ForwardMode : uint8_t {
    Decode = 0,
    PrefillBlock = 1,
};

// Static context every layer needs (shapes + RoPE tables). Passed into
// attention_block_forward instead of being threaded through every arg.
struct LayerContext {
    const ModelConfig* cfg{nullptr};
    const ModelWeights* w{nullptr};
};

// Run one attention block.  After this:
//   - kv cache (K/V) is updated for `layer_idx` in Decode mode (always),
//     or in PrefillBlock mode if `write_kv_in_prefill` is true (used for
//     init prefill / refill, populating positions [past_len, past_len+S)).
//   - ws.h_in    contains attn_out @ W_out  (delta, to be residual-added by MLP)
//   - ws.residual contains the residual stream up to (but not including) this delta
void attention_block_forward(const LayerContext& ctx,
                             const LayerWeights& Lw,
                             LayerWorkspace& ws,
                             KvCache& kv,
                             int layer_idx,
                             ForwardMode mode,
                             int B, int S,
                             const int32_t* block_id,
                             cudaStream_t stream,
                             bool write_kv_in_prefill = false);

void mlp_block_forward(const LayerContext& ctx,
                       const LayerWeights& Lw,
                       LayerWorkspace& ws,
                       int M,
                       cudaStream_t stream);

void transformer_layer_forward(const LayerContext& ctx,
                               const LayerWeights& Lw,
                               LayerWorkspace& ws,
                               KvCache& kv,
                               int layer_idx,
                               ForwardMode mode,
                               int B, int S,
                               const int32_t* block_id,
                               cudaStream_t stream,
                               bool write_kv_in_prefill = false);

}  // namespace cutlass_engine

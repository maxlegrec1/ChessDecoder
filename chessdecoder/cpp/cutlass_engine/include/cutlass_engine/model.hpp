#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "cutlass_engine/config.hpp"
#include "cutlass_engine/kv_cache.hpp"
#include "cutlass_engine/layers.hpp"
#include "cutlass_engine/weights.hpp"

namespace cutlass_engine {

class ChessDecoderModel {
public:
    void initialize(const ModelConfig& cfg, const ModelWeights& w,
                    Arena& arena, int max_M);

    // Decode forward: S=1, returns hidden state [B, E] in `out_h`.
    // Caller must have populated:
    //   ids       [B] int32        — input token id per slot
    //   pos       [B] int32        — RoPE position per slot (== past_len[b])
    //   wl_pos/d_pos: per-slot bool flags for fourier override (or null)
    //   wl_val/d_val: per-slot FP16 values for fourier override (or null)
    // After the call, kv_cache.past_len[b] is incremented by 1 for active slots.
    void forward_decode(const int32_t* ids,
                        const int32_t* pos,
                        const bool* wl_pos, const bool* d_pos,
                        const __half* wl_val, const __half* d_val,
                        KvCache& kv,
                        __half* out_h,                     // [B, E] FP16
                        cudaStream_t stream);

    // Prefill forward over a [B, S] block (init or refill). Self-attention
    // only — does NOT update the cache. Caller post-processes by copying the
    // last-position K/V (or all positions) into cache via kv_scatter.
    //
    // block_id [B, S] int32 — for prefix mask
    void forward_prefill_block(const int32_t* ids,        // [B, S]
                               const int32_t* pos,        // [B, S]
                               const int32_t* block_id,   // [B, S]
                               const bool* wl_pos, const bool* d_pos,
                               const __half* wl_val, const __half* d_val,
                               int B, int S,
                               KvCache& kv,
                               __half* out_h,             // [B, S, E]
                               cudaStream_t stream);

    int batch_size() const { return cfg_->batch_size; }
    int embed_dim() const { return cfg_->embed_dim; }
    const ModelConfig& cfg() const { return *cfg_; }
    const ModelWeights& weights() const { return *w_; }

private:
    const ModelConfig* cfg_{nullptr};
    const ModelWeights* w_{nullptr};
    LayerWorkspace ws_{};
    LayerContext ctx_{};
};

}  // namespace cutlass_engine

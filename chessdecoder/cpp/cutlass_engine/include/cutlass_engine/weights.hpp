#pragma once

#include <cuda_fp16.h>
#include <string>
#include <vector>

#include "cutlass_engine/allocator.hpp"
#include "cutlass_engine/config.hpp"

namespace cutlass_engine {

// Per-layer weight slabs, all FP16. Allocated inside the engine arena and
// populated via host-side memcpy from the .bin files.
struct LayerWeights {
    // Pre-attn RMSNorm weight [E].
    __half* sa_norm{nullptr};
    // Pre-mlp RMSNorm weight [E].
    __half* mlp_norm{nullptr};

    // Fused QKV proj: weight [3*E, E], no bias.  We keep weights stored as
    // [out, in] (PyTorch convention) — GEMM consumes them transposed.
    __half* qkv_w{nullptr};

    // Output proj: weight [E, E], no bias.
    __half* out_w{nullptr};

    // Fused gate+up: weight [2*d_ff, E], no bias.
    __half* gate_up_w{nullptr};

    // Down proj: weight [E, d_ff], no bias.
    __half* down_w{nullptr};
};

struct ModelWeights {
    // Token embedding [V, E].
    __half* tok_embedding{nullptr};

    // Final norm weight [E].
    __half* final_norm{nullptr};

    // Per-layer.
    std::vector<LayerWeights> layers;

    // Heads.
    __half* board_head_w{nullptr};       // [board_vocab, E]
    __half* board_head_b{nullptr};       // [board_vocab]
    __half* policy_head_w{nullptr};
    __half* policy_head_b{nullptr};
    __half* thinking_policy_head_w{nullptr};
    __half* thinking_policy_head_b{nullptr};

    // Value heads (W1: [H, E], W2: [n_buckets, H], biases included).
    __half* wl_w1_w{nullptr};
    __half* wl_w1_b{nullptr};
    __half* wl_w2_w{nullptr};
    __half* wl_w2_b{nullptr};
    __half* d_w1_w{nullptr};
    __half* d_w1_b{nullptr};
    __half* d_w2_w{nullptr};
    __half* d_w2_b{nullptr};

    // Bucket centers FP32 [n_buckets].
    float* wl_bucket_centers{nullptr};
    float* d_bucket_centers{nullptr};

    // Fourier encoder.
    // frequencies [1, F]  (learned), proj_w [E, 2*F], proj_b [E].
    __half* fourier_freq{nullptr};
    __half* fourier_proj_w{nullptr};
    __half* fourier_proj_b{nullptr};

    // RoPE precomputed cos/sin table [max_seq_len, head_dim/2] FP32 each.
    float* rope_cos{nullptr};
    float* rope_sin{nullptr};
};

// Allocate slabs in `arena` and populate from the export dir at
// `weights_dir`. Throws on any missing file. Backbone weights are expected at
// weights_dir/backbone/{layer_<i>_<name>.bin, tok_embedding.bin, final_norm.bin}.
// Heads at weights_dir/{board,policy,thinking_policy,wl,d}_*.bin (existing layout).
// Bucket centers at weights_dir/{wl,d}_bucket_centers.bin.
ModelWeights load_weights(const std::string& weights_dir,
                          const ModelConfig& cfg, Arena& arena);

// Re-upload weights into the same slabs (no realloc). Used by update_weights().
void reupload_weights(const std::string& weights_dir,
                      const ModelConfig& cfg, ModelWeights& w);

}  // namespace cutlass_engine

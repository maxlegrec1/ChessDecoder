#pragma once

#include <torch/script.h>
#include <cstdint>
#include <string>

namespace decoder
{

/// Batched TorchScript backbone wrapper with B>1 KV cache management.
///
/// Simpler than TorchCausalBackbone: no CUDA graphs (batching already
/// provides good GPU utilization), no tiered board generation.
///
/// Manages two independent batched KV caches:
/// 1. Causal  — for autoregressive board generation (causal attention)
/// 2. Prefix  — for block-bidirectional attention (move/value prediction)
class BatchedBackbone
{
public:
    /// Load TorchScript model and initialize empty KV caches.
    BatchedBackbone(const std::string& pt_path, int num_layers, int num_heads,
                    int head_dim, int embed_dim, int max_seq_len, int batch_size);

    // ==================== Causal mode ====================

    /// Batched causal forward with auto-generated causal mask.
    /// Updates causal KV cache.
    /// Returns hidden states [B, S, E] FP16 on CUDA.
    torch::Tensor causalForward(torch::Tensor input_ids,         // [B, S] int64
                                torch::Tensor input_pos,         // [B, S] int64
                                torch::Tensor override_values,   // [B, S] FP16
                                torch::Tensor override_mask);    // [B, S] bool

    /// Batched 1-token causal incremental.
    /// Returns hidden states [B, 1, E] FP16 on CUDA.
    torch::Tensor causalIncremental(torch::Tensor input_ids,     // [B, 1] int64
                                    torch::Tensor input_pos,     // [B, 1] int64
                                    torch::Tensor override_values,
                                    torch::Tensor override_mask);

    void resetCausal();
    int causalLen() const { return causal_len_; }

    // ==================== Prefix mode ====================

    /// Batched prefix forward with externally provided mask.
    /// Updates prefix KV cache.
    /// Returns hidden states [B, S, E] FP16 on CUDA.
    torch::Tensor prefixForward(torch::Tensor input_ids,         // [B, S] int64
                                torch::Tensor input_pos,         // [B, S] int64
                                torch::Tensor attention_mask,    // [B, 1, S, S+past]
                                torch::Tensor override_values,   // [B, S] FP16
                                torch::Tensor override_mask);    // [B, S] bool

    /// Batched 1-token prefix incremental (all-zeros mask: attend to all past).
    /// Returns hidden states [B, 1, E] FP16 on CUDA.
    torch::Tensor prefixIncremental(torch::Tensor input_ids,     // [B, 1] int64
                                    torch::Tensor input_pos,     // [B, 1] int64
                                    torch::Tensor override_values,
                                    torch::Tensor override_mask);

    /// Batched block prefix forward (all-zeros mask: bidirectional within block).
    /// Returns hidden states [B, S, E] FP16 on CUDA.
    torch::Tensor prefixBlockForward(torch::Tensor input_ids,    // [B, S] int64
                                     torch::Tensor input_pos,    // [B, S] int64
                                     torch::Tensor override_values,
                                     torch::Tensor override_mask);

    void resetPrefix();
    int prefixLen() const { return prefix_len_; }

    int embedDim() const { return embed_dim_; }
    int batchSize() const { return B_; }

private:
    /// Core forward: call model, update specified cache.
    /// If update_causal=true, appends to causal cache.
    /// If update_prefix=true, appends to prefix cache.
    torch::Tensor forwardImpl(torch::Tensor ids, torch::Tensor pos,
                              torch::Tensor mask,
                              torch::Tensor past_k, torch::Tensor past_v,
                              torch::Tensor ov, torch::Tensor om,
                              bool update_causal, bool update_prefix);

    torch::jit::Module model_;
    int B_, num_layers_, num_heads_, head_dim_, embed_dim_, max_seq_len_;

    // Causal KV cache [NL, B, NH, causal_len, HD] FP16 CUDA
    torch::Tensor causal_k_, causal_v_;
    int causal_len_;

    // Prefix KV cache [NL, B, NH, prefix_len, HD] FP16 CUDA
    torch::Tensor prefix_k_, prefix_v_;
    int prefix_len_;
};

} // namespace decoder

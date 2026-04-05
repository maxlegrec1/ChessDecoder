#pragma once

#include <torch/script.h>
#include <cstdint>
#include <string>
#include <vector>

namespace decoder
{

/// Batched TorchScript backbone with B>1 KV cache management.
///
/// Uses pre-allocated mask buffers (one per cache) that are updated
/// incrementally as tokens are added.  Invalid positions stay at -inf,
/// valid positions are set to 0.0.  The incremental mask for any forward
/// call is just a slice of the buffer — zero allocations per step.
///
/// No CUDA graphs — keeps memory footprint minimal so batch size can be
/// maximized. At batch=32-64, the GPU is well-utilized from batching alone.
class BatchedBackbone
{
public:
    BatchedBackbone(const std::string& pt_path, int num_layers, int num_heads,
                    int head_dim, int embed_dim, int max_seq_len, int batch_size);

    // ==================== Causal mode ====================

    torch::Tensor causalForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                torch::Tensor override_values, torch::Tensor override_mask,
                                torch::Tensor active, torch::Tensor num_real);

    torch::Tensor causalIncremental(torch::Tensor input_ids, torch::Tensor input_pos,
                                    torch::Tensor override_values, torch::Tensor override_mask,
                                    torch::Tensor active);

    void resetCausal();
    int causalLen() const { return causal_len_; }

    // ==================== Prefix mode ====================

    torch::Tensor prefixForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                torch::Tensor attention_mask,
                                torch::Tensor override_values, torch::Tensor override_mask,
                                torch::Tensor active);

    torch::Tensor prefixIncremental(torch::Tensor input_ids, torch::Tensor input_pos,
                                    torch::Tensor override_values, torch::Tensor override_mask,
                                    torch::Tensor active);

    torch::Tensor prefixBlockForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                     torch::Tensor override_values, torch::Tensor override_mask,
                                     torch::Tensor active);

    void resetPrefix();
    int prefixLen() const { return prefix_len_; }

    int embedDim() const { return embed_dim_; }
    int batchSize() const { return B_; }

    torch::Tensor causalProbe(torch::Tensor ids, torch::Tensor pos,
                              torch::Tensor ov, torch::Tensor om);
    torch::Tensor prefixProbe(torch::Tensor ids, torch::Tensor pos,
                              torch::Tensor ov, torch::Tensor om);

    /// Pre-mark a range of causal positions as valid for active elements.
    /// Call before a series of causalIncremental steps when you know all
    /// positions will be valid (e.g. 68-step board generation).
    void markCausalValidRange(int start, int count, torch::Tensor active);
    void markPrefixValidRange(int start, int count, torch::Tensor active);

    // No-ops for API compatibility
    void syncCausalToGraph() {}
    void syncGraphToCausal() {}
    void syncPrefixToGraph() {}
    void syncGraphToPrefix() {}

private:
    torch::Tensor forwardImpl(torch::Tensor ids, torch::Tensor pos,
                              torch::Tensor mask,
                              torch::Tensor past_k, torch::Tensor past_v,
                              torch::Tensor ov, torch::Tensor om,
                              bool update_causal, bool update_prefix);

    /// Mark position `pos` as valid (0.0) in the mask buffer for active elements.
    /// Positions default to -inf; only active elements are set to 0.0.
    void markCausalValid(int pos, torch::Tensor active);
    void markPrefixValid(int pos, torch::Tensor active);

    torch::jit::Module model_;
    int B_, num_layers_, num_heads_, head_dim_, embed_dim_, max_seq_len_;

    torch::Tensor causal_k_, causal_v_;
    int causal_len_;
    torch::Tensor prefix_k_, prefix_v_;
    int prefix_len_;

    /// Pre-allocated mask buffers: [B, 1, 1, max_seq_len] FP32 on CUDA.
    /// Initialized to -inf.  Valid positions are set to 0.0 incrementally.
    /// The mask for any incremental step is just a slice — zero allocations.
    torch::Tensor causal_mask_buf_;
    torch::Tensor prefix_mask_buf_;
};

} // namespace decoder

#pragma once

#include <torch/script.h>
#include <cstdint>
#include <string>

namespace decoder
{

/// Batched TorchScript backbone with B>1 KV cache management.
///
/// No CUDA graphs — keeps memory footprint minimal so batch size can be
/// maximized. At batch=32-64, the GPU is well-utilized from batching alone.
/// The memory saved by not allocating graph buffers allows 2-4x larger batches.
class BatchedBackbone
{
public:
    BatchedBackbone(const std::string& pt_path, int num_layers, int num_heads,
                    int head_dim, int embed_dim, int max_seq_len, int batch_size);

    // ==================== Causal mode ====================

    torch::Tensor causalForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                torch::Tensor override_values, torch::Tensor override_mask);

    torch::Tensor causalIncremental(torch::Tensor input_ids, torch::Tensor input_pos,
                                    torch::Tensor override_values, torch::Tensor override_mask);

    void resetCausal();
    int causalLen() const { return causal_len_; }

    // ==================== Prefix mode ====================

    torch::Tensor prefixForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                torch::Tensor attention_mask,
                                torch::Tensor override_values, torch::Tensor override_mask);

    torch::Tensor prefixIncremental(torch::Tensor input_ids, torch::Tensor input_pos,
                                    torch::Tensor override_values, torch::Tensor override_mask);

    torch::Tensor prefixBlockForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                     torch::Tensor override_values, torch::Tensor override_mask);

    void resetPrefix();
    int prefixLen() const { return prefix_len_; }

    int embedDim() const { return embed_dim_; }
    int batchSize() const { return B_; }

    // No-ops for API compatibility with engine's sync calls
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

    torch::jit::Module model_;
    int B_, num_layers_, num_heads_, head_dim_, embed_dim_, max_seq_len_;

    torch::Tensor causal_k_, causal_v_;
    int causal_len_;
    torch::Tensor prefix_k_, prefix_v_;
    int prefix_len_;
};

} // namespace decoder

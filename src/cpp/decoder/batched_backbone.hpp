#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <torch/script.h>
#include <cstdint>
#include <string>

namespace decoder
{

/// Batched TorchScript backbone wrapper with B>1 KV cache management
/// and CUDA graph acceleration for 1-token incremental forwards.
///
/// Manages two independent batched KV caches:
/// 1. Causal  — for autoregressive board generation (causal attention)
/// 2. Prefix  — for block-bidirectional attention (move/value prediction)
///
/// CUDA graphs are captured for causalIncremental and prefixIncremental
/// (the hot paths called 67× and 3× per variation respectively).
/// Multi-token forwards (prefill, catch-up, block) use dynamic dispatch.
class BatchedBackbone
{
public:
    BatchedBackbone(const std::string& pt_path, int num_layers, int num_heads,
                    int head_dim, int embed_dim, int max_seq_len, int batch_size);

    // ==================== Causal mode ====================

    /// Batched causal forward (prefill/catch-up). Dynamic dispatch.
    /// Returns hidden states [B, S, E] FP16 on CUDA.
    torch::Tensor causalForward(torch::Tensor input_ids,
                                torch::Tensor input_pos,
                                torch::Tensor override_values,
                                torch::Tensor override_mask);

    /// Batched 1-token causal incremental via CUDA graph.
    /// Returns hidden states [B, 1, E] FP16 on CUDA.
    torch::Tensor causalIncremental(torch::Tensor input_ids,
                                    torch::Tensor input_pos,
                                    torch::Tensor override_values,
                                    torch::Tensor override_mask);

    /// Sync dynamic causal cache → graph buffer.
    /// Call after causalForward() to switch to graph-accelerated mode.
    void syncCausalToGraph();

    /// Sync graph buffer → dynamic causal cache.
    /// Call before causalForward() when switching from graph to dynamic mode.
    void syncGraphToCausal();

    void resetCausal();
    int causalLen() const { return causal_len_; }

    // ==================== Prefix mode ====================

    /// Batched prefix forward with custom mask. Dynamic dispatch.
    /// Returns hidden states [B, S, E] FP16 on CUDA.
    torch::Tensor prefixForward(torch::Tensor input_ids,
                                torch::Tensor input_pos,
                                torch::Tensor attention_mask,
                                torch::Tensor override_values,
                                torch::Tensor override_mask);

    /// Batched 1-token prefix incremental via CUDA graph.
    /// Returns hidden states [B, 1, E] FP16 on CUDA.
    torch::Tensor prefixIncremental(torch::Tensor input_ids,
                                    torch::Tensor input_pos,
                                    torch::Tensor override_values,
                                    torch::Tensor override_mask);

    /// Batched block prefix forward (bidirectional). Dynamic dispatch.
    /// Returns hidden states [B, S, E] FP16 on CUDA.
    torch::Tensor prefixBlockForward(torch::Tensor input_ids,
                                     torch::Tensor input_pos,
                                     torch::Tensor override_values,
                                     torch::Tensor override_mask);

    /// Sync dynamic prefix cache → graph buffer.
    /// Call after prefixForward()/prefixBlockForward() to switch to graph mode.
    void syncPrefixToGraph();

    /// Sync graph buffer → dynamic prefix cache.
    /// Call before prefixForward()/prefixBlockForward() when switching from graph to dynamic.
    void syncGraphToPrefix();

    void resetPrefix();
    int prefixLen() const { return prefix_len_; }

    int embedDim() const { return embed_dim_; }
    int batchSize() const { return B_; }

private:
    /// Dynamic forward (no graph). Updates specified cache.
    torch::Tensor forwardImpl(torch::Tensor ids, torch::Tensor pos,
                              torch::Tensor mask,
                              torch::Tensor past_k, torch::Tensor past_v,
                              torch::Tensor ov, torch::Tensor om,
                              bool update_causal, bool update_prefix);

    void captureGraphs();

    torch::jit::Module model_;
    int B_, num_layers_, num_heads_, head_dim_, embed_dim_, max_seq_len_;

    // ---- Dynamic caches (for multi-token forwards) ----
    torch::Tensor causal_k_, causal_v_;   // [NL, B, NH, causal_len, HD]
    int causal_len_;
    torch::Tensor prefix_k_, prefix_v_;   // [NL, B, NH, prefix_len, HD]
    int prefix_len_;

    // ---- CUDA Graph: causal incremental ----
    torch::Tensor cg_ids_, cg_pos_, cg_mask_, cg_ov_, cg_om_;  // Inputs [B, 1]
    torch::Tensor cg_buf_k_, cg_buf_v_;                         // [NL, B, NH, MAX_LEN, HD]
    torch::Tensor cg_out_h_, cg_out_pk_, cg_out_pv_;           // Outputs
    at::cuda::CUDAGraph causal_graph_;
    int cg_len_;

    // ---- CUDA Graph: prefix incremental ----
    torch::Tensor pg_ids_, pg_pos_, pg_mask_, pg_ov_, pg_om_;
    torch::Tensor pg_buf_k_, pg_buf_v_;
    torch::Tensor pg_out_h_, pg_out_pk_, pg_out_pv_;
    at::cuda::CUDAGraph prefix_graph_;
    int pg_len_;

    bool graphs_captured_;
};

} // namespace decoder

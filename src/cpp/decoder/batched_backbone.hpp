#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <torch/script.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace decoder
{

/// Per-tier CUDA graph for batched causal board generation.
struct BatchedBoardTier
{
    int max_len;
    torch::Tensor mask;        // [B, 1, 1, max_len + 1]
    torch::Tensor buf_k;       // [NL, B, NH, max_len, HD]
    torch::Tensor buf_v;
    torch::Tensor out_h;       // [B, 1, E]
    torch::Tensor out_pk;      // [NL, B, NH, max_len+1, HD]
    torch::Tensor out_pv;
    at::cuda::CUDAGraph graph;
    int len{0};
};

/// Batched backbone with tiered CUDA graphs for causal incremental
/// and a single CUDA graph for prefix incremental.
///
/// Tiered causal graphs (128, 256, 512, 1024) reduce memory bandwidth
/// by using the smallest buffer that fits the current sequence position.
/// At batch=32 with 4 tiers, total graph memory is ~5GB (vs ~12GB for
/// a single 4096-buffer graph).
class BatchedBackbone
{
public:
    BatchedBackbone(const std::string& pt_path, int num_layers, int num_heads,
                    int head_dim, int embed_dim, int max_seq_len, int batch_size);

    // ==================== Causal mode ====================

    /// Multi-token causal forward (prefill/catch-up). Dynamic dispatch.
    torch::Tensor causalForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                torch::Tensor override_values, torch::Tensor override_mask);

    /// 1-token causal incremental via tiered CUDA graph.
    /// Automatically selects the smallest tier that fits.
    /// Falls back to dynamic dispatch if no tier fits.
    torch::Tensor causalIncremental(torch::Tensor input_ids, torch::Tensor input_pos,
                                    torch::Tensor override_values, torch::Tensor override_mask);

    /// Sync dynamic causal cache → graph buffer for the appropriate tier.
    /// Call after causalForward() before starting graph-accelerated board gen.
    void syncCausalToGraph();

    /// Sync graph buffer → dynamic causal cache.
    void syncGraphToCausal();

    void resetCausal();
    int causalLen() const { return causal_len_; }

    // ==================== Prefix mode ====================

    torch::Tensor prefixForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                torch::Tensor attention_mask,
                                torch::Tensor override_values, torch::Tensor override_mask);

    /// 1-token prefix incremental via CUDA graph (buffer size = prefix_graph_len_).
    torch::Tensor prefixIncremental(torch::Tensor input_ids, torch::Tensor input_pos,
                                    torch::Tensor override_values, torch::Tensor override_mask);

    torch::Tensor prefixBlockForward(torch::Tensor input_ids, torch::Tensor input_pos,
                                     torch::Tensor override_values, torch::Tensor override_mask);

    void syncPrefixToGraph();
    void syncGraphToPrefix();

    void resetPrefix();
    int prefixLen() const { return prefix_len_; }

    int embedDim() const { return embed_dim_; }
    int batchSize() const { return B_; }

private:
    torch::Tensor forwardImpl(torch::Tensor ids, torch::Tensor pos,
                              torch::Tensor mask,
                              torch::Tensor past_k, torch::Tensor past_v,
                              torch::Tensor ov, torch::Tensor om,
                              bool update_causal, bool update_prefix);

    /// Dynamic 1-token causal incremental (fallback when no tier fits).
    torch::Tensor causalIncrementalDynamic(torch::Tensor ids, torch::Tensor pos,
                                           torch::Tensor ov, torch::Tensor om);

    void captureGraphs();

    /// Select smallest causal tier whose buffer fits max_needed_pos.
    /// Returns tier index, or -1 if none fits.
    int selectCausalTier(int max_needed_pos) const;

    torch::jit::Module model_;
    int B_, num_layers_, num_heads_, head_dim_, embed_dim_, max_seq_len_;

    // Dynamic caches
    torch::Tensor causal_k_, causal_v_;
    int causal_len_;
    torch::Tensor prefix_k_, prefix_v_;
    int prefix_len_;

    // Shared input tensors for causal tier graphs
    torch::Tensor cg_ids_, cg_pos_, cg_ov_, cg_om_;

    // Tiered causal graphs
    static constexpr int kCausalTierSizes[] = {128, 256, 512, 1024};
    static constexpr int kNumCausalTiers = 4;
    std::vector<std::unique_ptr<BatchedBoardTier>> causal_tiers_;
    int active_tier_;  // currently synced tier (-1 = none)

    // Prefix graph (single, moderate buffer)
    static constexpr int kPrefixGraphLen = 512;
    torch::Tensor pg_ids_, pg_pos_, pg_mask_, pg_ov_, pg_om_;
    torch::Tensor pg_buf_k_, pg_buf_v_;
    torch::Tensor pg_out_h_, pg_out_pk_, pg_out_pv_;
    at::cuda::CUDAGraph prefix_graph_;
    int pg_len_;

    bool graphs_captured_;
};

} // namespace decoder

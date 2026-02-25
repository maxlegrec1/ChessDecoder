#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <torch/script.h>

#include <cstdint>
#include <string>

namespace decoder
{

/// Wraps a TorchScript causal backbone with KV cache management.
/// Replaces TRT-based CausalBackbone + PrefixBackbone with a single libtorch model
/// that uses the exact same CUDA kernels as PyTorch, guaranteeing 100% match.
///
/// Manages TWO independent KV caches:
/// 1. Causal KV cache — for autoregressive board generation (causal attention)
/// 2. Prefix KV cache — for block-bidirectional attention (move/value prediction)
///
/// CUDA Graph acceleration:
/// - 1-token causal incremental uses a captured CUDA graph for ~5x speedup
/// - Uses padded fixed-size KV buffers so graph shapes are constant
class TorchCausalBackbone
{
public:
    /// Load TorchScript model and initialize KV caches.
    TorchCausalBackbone(const std::string& pt_path, int num_layers, int num_heads,
                        int head_dim, int embed_dim, int max_cache_len);

    // ==================== Causal mode (board generation) ====================

    /// Run causal forward with auto-generated causal attention mask.
    /// Updates internal causal KV cache.
    void forward(const int64_t* input_ids, const int64_t* input_pos,
                 int seq_len, int past_len,
                 const float* override_values, const uint8_t* override_flags,
                 float* hidden_out);

    /// Run forward with an externally provided attention mask.
    /// Updates internal causal KV cache.
    void forwardWithMask(const int64_t* input_ids, const int64_t* input_pos,
                         int seq_len, int past_len,
                         const float* attention_mask,
                         const float* override_values, const uint8_t* override_flags,
                         float* hidden_out);

    /// Reset causal KV cache (both dynamic and graph buffer).
    void resetCache();

    /// Current causal cache length.
    int cacheLen() const { return cache_len_; }

    // ==================== Prefix mode (uncached, full recomputation) ========

    /// Run prefix forward without modifying any KV cache.
    /// Uses empty past KV internally.
    void forwardPrefix(const int64_t* input_ids, const int64_t* input_pos,
                       int seq_len,
                       const float* attention_mask,
                       const float* override_values, const uint8_t* override_flags,
                       float* hidden_out);

    // ==================== Prefix KV cache mode (incremental) ================

    /// Initialize prefix KV cache with a full prefix forward.
    /// Stores the resulting KV in the prefix cache.
    /// Outputs hidden state at `extract_pos` (single position, embed_dim floats).
    void prefixInit(const int64_t* input_ids, const int64_t* input_pos,
                    int seq_len,
                    const float* attention_mask,
                    const float* override_values, const uint8_t* override_flags,
                    int extract_pos, float* hidden_out);

    /// Incremental prefix forward: 1 token against prefix KV cache.
    /// Mask is all zeros (attend to everything — correct for orphan tokens).
    /// Updates prefix KV cache. Outputs hidden state (embed_dim floats).
    void prefixIncremental(int64_t token_id, int64_t position,
                           float override_value, uint8_t override_flag,
                           float* hidden_out);

    /// Block prefix forward: N tokens with full attention against prefix KV cache.
    /// Mask is all zeros (attend to everything — correct for board blocks where
    /// all new tokens share a block_id and have bidirectional intra-block attention).
    /// Updates prefix KV cache. Outputs hidden state at `extract_pos`.
    void prefixBlockForward(const int64_t* input_ids, const int64_t* input_pos,
                            int seq_len,
                            const float* override_values, const uint8_t* override_flags,
                            int extract_pos, float* hidden_out);

    /// Reset prefix KV cache.
    void resetPrefixCache();

    /// Current prefix cache length.
    int prefixCacheLen() const { return prefix_cache_len_; }

    int embedDim() const { return embed_dim_; }

    // ==================== CUDA Graph accelerated paths ====================

    /// 1-token causal incremental via CUDA graph replay.
    /// Returns hidden state as GPU FP16 tensor [1, 1, E].
    /// Also updates the graph KV buffer internally.
    torch::Tensor causalIncrementalGraph(int64_t token_id, int64_t position,
                                         float override_value, uint8_t override_flag);

    /// Transfer dynamic causal KV cache into the graph buffer.
    /// Call this after a multi-token causal forward (prefill) to
    /// switch to graph-accelerated incremental mode.
    void syncCausalCacheToGraph();

    /// Transfer graph causal KV buffer back to the dynamic cache.
    /// Call this after graph-accelerated board generation when
    /// a subsequent non-graph forward is needed.
    void syncGraphToCausalCache();

    /// 1-token prefix incremental via CUDA graph replay.
    /// Returns hidden state as GPU FP16 tensor [1, 1, E].
    /// Also updates the graph prefix KV buffer internally.
    torch::Tensor prefixIncrementalGraph(int64_t token_id, int64_t position,
                                          float override_value, uint8_t override_flag);

    /// Transfer dynamic prefix KV cache into the graph buffer.
    void syncPrefixCacheToGraph();

    /// Transfer graph prefix KV buffer back to the dynamic cache.
    void syncGraphToPrefixCache();

    /// Transfer block forward results from dynamic prefix cache to graph buffer.
    /// Call after prefixBlockForward() to keep graph buffer in sync.
    void syncPrefixCacheToGraphAfterBlock();

    /// Board generation step: graph replay + GPU head eval + LUT, all on GPU.
    /// Assumes cg_ids_ already contains the previous token (set via setGraphInput
    /// or a previous causalBoardStep). Returns the full token index as a GPU scalar.
    /// No CPU sync — use this in a loop and sync once at the end.
    torch::Tensor causalBoardStep(
        const torch::Tensor& head_w_t,   // [E, V] FP16 pre-transposed weight
        const torch::Tensor& head_b,     // [V] FP16 bias
        const torch::Tensor& lut,        // [V] int64 board_sub_idx → full_idx
        int64_t position);               // absolute position for this step

    /// Set the causal graph input token ID (GPU-side, for starting a board gen loop).
    void setGraphInput(int64_t token_id);

private:
    /// Internal forward: runs the model, optionally updates causal cache.
    void forwardInternal(const int64_t* input_ids_ptr, const int64_t* input_pos_ptr,
                         int seq_len, torch::Tensor mask,
                         torch::Tensor past_k, torch::Tensor past_v,
                         const float* override_values_ptr, const uint8_t* override_flags_ptr,
                         float* hidden_out, bool update_cache);

    /// Internal prefix forward: runs the model against prefix KV cache.
    /// Updates prefix cache. Extracts hidden at extract_pos.
    void prefixForwardInternal(const int64_t* input_ids_ptr, const int64_t* input_pos_ptr,
                               int seq_len, torch::Tensor mask,
                               const float* override_values_ptr, const uint8_t* override_flags_ptr,
                               int extract_pos, float* hidden_out);

    /// Capture CUDA graphs for 1-token incremental (causal + prefix).
    void captureGraphs();

    torch::jit::Module model_;

    // Causal KV cache (dynamic, for multi-token forwards)
    torch::Tensor past_keys_;   // [NL, 1, NH, cache_len, HD] FP16 on CUDA
    torch::Tensor past_values_;

    // Prefix KV cache (dynamic, for multi-token forwards)
    torch::Tensor prefix_past_keys_;   // [NL, 1, NH, prefix_cache_len, HD] FP16 on CUDA
    torch::Tensor prefix_past_values_;

    int num_layers_, num_heads_, head_dim_, embed_dim_;
    int max_cache_len_;
    int cache_len_;
    int prefix_cache_len_;

    // ---- CUDA Graph state (causal) ----
    // Static input tensors (fixed addresses, values updated between replays)
    torch::Tensor cg_ids_, cg_pos_, cg_mask_, cg_ov_, cg_om_;
    torch::Tensor cg_buf_k_, cg_buf_v_;     // [NL, 1, NH, MAX_LEN, HD] padded buffer
    // Graph output tensors (written by replay)
    torch::Tensor cg_out_h_, cg_out_pk_, cg_out_pv_;
    at::cuda::CUDAGraph causal_graph_;
    int cg_len_;  // current fill level of graph causal buffer
    bool graphs_captured_;

    // ---- CUDA Graph state (prefix) ----
    torch::Tensor pg_ids_, pg_pos_, pg_mask_, pg_ov_, pg_om_;
    torch::Tensor pg_buf_k_, pg_buf_v_;
    torch::Tensor pg_out_h_, pg_out_pk_, pg_out_pv_;
    at::cuda::CUDAGraph prefix_graph_;
    int pg_len_;
};

} // namespace decoder

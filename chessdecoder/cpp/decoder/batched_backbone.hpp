#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <torch/script.h>
#include <cstdint>
#include <memory>
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

    // ==================== Phase 5: CUDA graph capture =====================

    /// Capture a CUDA graph for a single board-generation step:
    ///   forward_new (S=1, causal) + board_head matmul + argmax + LUT lookup
    ///   + ids feedback (in-place write into the graph's static input).
    /// Replay via boardStepGraph(...) is launch-overhead-free — eliminates
    /// ~80 kernel launches per step (the dominant cost on B200 at small B
    /// per-token compute).
    ///
    /// Caller-owned head tensors (board_w_t [E, board_vocab] FP16 transposed,
    /// board_b [board_vocab] FP16, board_lut [board_vocab] int64) are
    /// referenced by the captured graph. They must outlive the engine.
    void captureBoardStepGraph(torch::Tensor board_w_t,
                               torch::Tensor board_b,
                               torch::Tensor board_lut);

    /// Replay the captured board-step graph for one new board token.
    /// `active` [B] bool — slots where the new K/V's mask position should
    /// be marked valid for the NEXT replay (inactive slots stay -inf so
    /// they don't attend to the freshly-written K of an unused rollout).
    torch::Tensor boardStepGraph(torch::Tensor active);

    /// Initialize board_step_ids_ to a per-slot start token (e.g. start_pos)
    /// before the first boardStepGraph() call of a 67-step run.
    void setBoardStepInitToken(torch::Tensor full_idx_b1);

    /// Initialize board_step_pos_ to the per-slot starting RoPE position
    /// (e.g. seq_pos[b]) before the first boardStepGraph() call.
    void setBoardStepInitPos(torch::Tensor pos_b1);

    /// Pre-mark mask valid at [causal_len_, causal_len_+count) for active
    /// slots before a sequence of boardStepGraph() replays. (Mirrors
    /// markCausalValidRange but the graph itself doesn't update mask.)
    /// Called once per board-gen run with count=68.
    void prepareBoardStepMask(int count, torch::Tensor active);

    /// Whether captureBoardStepGraph has run successfully.
    bool boardStepGraphReady() const { return board_step_ready_; }

    /// Hidden state from the most recent boardStepGraph() replay.
    /// Shape [B, 1, E] FP16. Caller .squeeze(1) for [B, E].
    torch::Tensor boardStepLastH() const { return board_step_out_h_; }

    /// Phase 4: refill finished slots in place without disturbing other slots.
    ///
    /// For each `slot_active[b]==true` slot:
    ///   1. Wipe its causal_mask_buf_ and prefix_mask_buf_ rows entirely to -inf.
    ///   2. Run a fresh prefill (forward_new on past_len=0) and write the new
    ///      K/V into causal_k_/v_/prefix_k_/v_ at slot b, physical positions
    ///      [0, init_len). Other slots' rows are untouched.
    ///   3. Mark mask valid at [0, init_len) for slot b.
    ///
    /// `causal_len_` and `prefix_len_` are NOT modified — slot b's logical
    /// positions decouple from physical (RoPE uses input_pos arg). After
    /// refill, slot b's mask says valid at [0, init_len) only. Subsequent
    /// causalIncremental writes at causal_len_ (global) get masked-valid by
    /// the existing markCausalValid path when slot is active.
    ///
    /// Returns: saved_h `[B, E]` — last-position hidden state from the
    /// refill prefill. Caller updates saved_h only at refilled slot indices.
    torch::Tensor resetSlotsForRefill(
        torch::Tensor slot_active,    // [B] bool
        torch::Tensor init_ids,       // [B, init_len] int64
        torch::Tensor init_pos,       // [B, init_len] int64
        torch::Tensor prefix_mask,    // [B, 1, init_len, init_len] FP32 — block-aware
        torch::Tensor init_ov,        // [B, init_len] FP16
        torch::Tensor init_om);       // [B, init_len] bool

    // No-ops for API compatibility
    void syncCausalToGraph() {}
    void syncGraphToCausal() {}
    void syncPrefixToGraph() {}
    void syncGraphToPrefix() {}

private:
    /// Run the model with `past_k_buf[:, :, :, :past_len, :]` as the past
    /// (a view, no alloc), then copy the model's present_keys/values into
    /// the appropriate fixed buffer in-place. With pre-allocated cache
    /// buffers, the only alloc per call is the model's internal cat() —
    /// allocator caching covers most of that as shapes stabilize.
    torch::Tensor forwardImpl(torch::Tensor ids, torch::Tensor pos,
                              torch::Tensor mask,
                              torch::Tensor past_k_buf, torch::Tensor past_v_buf,
                              int past_len,
                              torch::Tensor ov, torch::Tensor om,
                              bool update_causal, bool update_prefix);

    /// Mark position `pos` as valid (0.0) in the mask buffer for active elements.
    /// Positions default to -inf; only active elements are set to 0.0.
    void markCausalValid(int pos, torch::Tensor active);
    void markPrefixValid(int pos, torch::Tensor active);

    torch::jit::Module model_;
    /// Cached handle to the TorchScript `forward_new` method (returns new-only
    /// K/V instead of present K/V). Avoids `model_.get_method()` lookup on the
    /// hot path.
    std::unique_ptr<torch::jit::Method> forward_new_method_;
    int num_layers_, num_heads_, head_dim_, embed_dim_, max_seq_len_, B_;

    torch::Tensor causal_k_, causal_v_;
    int causal_len_;
    torch::Tensor prefix_k_, prefix_v_;
    int prefix_len_;

    /// Pre-allocated mask buffers: [B, 1, 1, max_seq_len] FP32 on CUDA.
    /// Initialized to -inf.  Valid positions are set to 0.0 incrementally.
    /// The mask for any incremental step is just a slice — zero allocations.
    torch::Tensor causal_mask_buf_;
    torch::Tensor prefix_mask_buf_;

    // ==================== CUDA graph state (Phase 5) ====================
    /// Past-K size used by the captured board-step graph. Capped well below
    /// max_seq_len_ to keep attn_weights' M dim small (per-step compute and
    /// memory traffic scale linearly with M; on B200 the difference between
    /// M=4096 and M=1500 is ~3x in attention cost). Rollouts that exceed
    /// this past size fall back to the non-graph causalIncremental path.
    static constexpr int kBoardStepGraphM = 1536;
    bool board_step_ready_{false};
    at::cuda::CUDAGraph board_step_graph_;
    torch::Tensor board_step_ids_;          // [B, 1] int64 — graph input + feedback
    torch::Tensor board_step_pos_;          // [B, 1] int64 — graph input, in-place add
    torch::Tensor board_step_inc_;          // [B, 1] int64 — increment (1)
    torch::Tensor board_step_ov_;           // [B, 1] FP16
    torch::Tensor board_step_om_;           // [B, 1] bool
    torch::Tensor board_step_mask_;         // [B, 1, 1, kBoardStepGraphM+1] FP32
    torch::Tensor board_step_past_k_;       // [NL, B, NH, kBoardStepGraphM, HD] FP16
    torch::Tensor board_step_past_v_;       // (graph-owned past KV cache)
    torch::Tensor board_step_out_h_;        // [B, 1, E] FP16
    torch::Tensor board_step_out_kn_;       // [NL, B, NH, 1, HD] FP16
    torch::Tensor board_step_out_vn_;       // [NL, B, NH, 1, HD] FP16
    torch::Tensor board_step_out_full_idx_; // [B] int64 — sampled full-vocab token
    /// Held alive for the lifetime of the captured graph.
    torch::Tensor board_step_w_t_;
    torch::Tensor board_step_b_;
    torch::Tensor board_step_lut_;
};

} // namespace decoder

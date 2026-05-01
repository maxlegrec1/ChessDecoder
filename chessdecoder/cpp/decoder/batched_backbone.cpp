#include "batched_backbone.hpp"

#include <c10/cuda/CUDAStream.h>
#include <iostream>

namespace decoder
{

BatchedBackbone::BatchedBackbone(
    const std::string& pt_path, int num_layers, int num_heads,
    int head_dim, int embed_dim, int max_seq_len, int batch_size)
    : num_layers_(num_layers), num_heads_(num_heads)
    , head_dim_(head_dim), embed_dim_(embed_dim)
    , max_seq_len_(max_seq_len), B_(batch_size)
    , causal_len_(0), prefix_len_(0)
{
    torch::NoGradGuard no_grad;

    model_ = torch::jit::load(pt_path, torch::kCUDA);
    model_.eval();

    // Warmup
    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    // Cache the forward_new method handle once. Phase 0.1 full uses forward_new
    // (returns [NL, B, NH, S, HD] new-only K/V) instead of forward (returns
    // [NL, B, NH, past_len+S, HD] present K/V) — kills the per-call allocation
    // of a tensor that grows with past_len.
    forward_new_method_ = std::make_unique<torch::jit::Method>(
        model_.get_method("forward_new"));

    std::vector<torch::IValue> warmup_inputs{
        torch::zeros({B_, 1}, opts_int), torch::zeros({B_, 1}, opts_int),
        torch::zeros({B_, 1, 1, 1}, opts_fp32),
        torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts_fp16),
        torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts_fp16),
        torch::zeros({B_, 1}, opts_fp16), torch::zeros({B_, 1}, opts_bool)
    };
    (*forward_new_method_)(warmup_inputs);
    c10::cuda::getCurrentCUDAStream().synchronize();

    // Pre-allocate mask buffers filled with -inf
    causal_mask_buf_ = torch::full({B_, 1, 1, max_seq_len_}, -1e9f, opts_fp32);
    prefix_mask_buf_ = torch::full({B_, 1, 1, max_seq_len_}, -1e9f, opts_fp32);

    // Pre-allocate KV cache at max size — never reassigned, just sliced + copied
    // into. Eliminates the per-forward reallocation that was driving 150+ GB
    // of nvidia-smi VRAM jitter under the old `causal_k_ = pk` pattern.
    causal_k_ = torch::zeros({num_layers_, B_, num_heads_, max_seq_len_, head_dim_}, opts_fp16);
    causal_v_ = torch::zeros({num_layers_, B_, num_heads_, max_seq_len_, head_dim_}, opts_fp16);
    prefix_k_ = torch::zeros({num_layers_, B_, num_heads_, max_seq_len_, head_dim_}, opts_fp16);
    prefix_v_ = torch::zeros({num_layers_, B_, num_heads_, max_seq_len_, head_dim_}, opts_fp16);

    resetCausal();
    resetPrefix();

    // Report fixed VRAM footprint for the cache (informational; helps diagnose
    // OOMs early).
    int64_t kv_bytes = 4LL * num_layers_ * B_ * num_heads_ * max_seq_len_ * head_dim_ * 2; // 4 buffers, fp16
    int64_t mask_bytes = 2LL * B_ * max_seq_len_ * 4; // 2 mask buffers, fp32
    std::cout << "[BatchedBackbone] " << num_layers << " layers, "
              << embed_dim << "d, batch=" << B_
              << " | fixed KV="
              << (kv_bytes / (1024 * 1024)) << " MiB"
              << " mask=" << (mask_bytes / (1024 * 1024)) << " MiB"
              << std::endl;
}

// ============================================================================
// Core forward
// ============================================================================

torch::Tensor BatchedBackbone::forwardImpl(
    torch::Tensor ids, torch::Tensor pos, torch::Tensor mask,
    torch::Tensor past_k_buf, torch::Tensor past_v_buf, int past_len,
    torch::Tensor ov, torch::Tensor om,
    bool update_causal, bool update_prefix)
{
    torch::NoGradGuard no_grad;

    // Slice the persistent buffer to the current valid range. Views, no alloc.
    auto past_k = past_k_buf.slice(3, 0, past_len);
    auto past_v = past_v_buf.slice(3, 0, past_len);

    // forward_new returns NEW-only K/V [NL, B, NH, S, HD] — proportional to
    // S, not past_len+S. Per-call allocation drops by 1000x at past_len=900,
    // S=1.
    std::vector<torch::IValue> inputs{ids, pos, mask, past_k, past_v, ov, om};
    auto result = (*forward_new_method_)(inputs);
    auto outputs = result.toTuple()->elements();
    auto hidden = outputs[0].toTensor();
    auto kn = outputs[1].toTensor();   // [NL, B, NH, S, HD] new only
    auto vn = outputs[2].toTensor();
    int S = kn.size(3);
    int new_total = past_len + S;

    if (update_causal) {
        // Write only the new tail [past_len, past_len+S) into the persistent
        // buffer; positions [0, past_len) already hold valid past K/V.
        causal_k_.slice(3, past_len, new_total).copy_(kn);
        causal_v_.slice(3, past_len, new_total).copy_(vn);
        causal_len_ = new_total;
    }
    if (update_prefix) {
        prefix_k_.slice(3, past_len, new_total).copy_(kn);
        prefix_v_.slice(3, past_len, new_total).copy_(vn);
        prefix_len_ = new_total;
    }
    return hidden;
}

// ============================================================================
// Mask buffer helpers
// ============================================================================

void BatchedBackbone::markCausalValid(int pos, torch::Tensor active)
{
    // Positions start at -inf. Set to 0.0 where active[b] = true (in-place).
    causal_mask_buf_.select(3, pos).select(2, 0).select(1, 0).masked_fill_(active, 0.0f);
}

void BatchedBackbone::markPrefixValid(int pos, torch::Tensor active)
{
    prefix_mask_buf_.select(3, pos).select(2, 0).select(1, 0).masked_fill_(active, 0.0f);
}

void BatchedBackbone::markCausalValidRange(int start, int count, torch::Tensor active)
{
    // Mark [start, start+count) as valid for active elements — single kernel.
    // mask_buf shape: [B, 1, 1, max_seq_len]
    // slice: [B, 1, 1, count] → squeeze to [B, count] for masked_fill_
    auto slice = causal_mask_buf_.slice(3, start, start + count)
                     .squeeze(2).squeeze(1);  // [B, count]
    slice.masked_fill_(active.unsqueeze(1), 0.0f);
}

void BatchedBackbone::markPrefixValidRange(int start, int count, torch::Tensor active)
{
    auto slice = prefix_mask_buf_.slice(3, start, start + count)
                     .squeeze(2).squeeze(1);
    slice.masked_fill_(active.unsqueeze(1), 0.0f);
}

// ============================================================================
// Causal mode
// ============================================================================

void BatchedBackbone::resetCausal()
{
    // Buffers are persistent — just reset the valid length and clear the mask.
    // KV cache content past causal_len_ is never read (mask is -inf there) so
    // we don't need to zero it.
    causal_len_ = 0;
    causal_mask_buf_.fill_(-1e9f);
}

torch::Tensor BatchedBackbone::causalForward(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om,
    torch::Tensor active, torch::Tensor num_real)
{
    // Multi-token causal forward with per-element real token counts.
    int S = ids.size(1);
    int old_len = causal_len_;
    int total = old_len + S;

    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Build mask [B, 1, S, total]:
    //   cached part [0, old_len): use causal_mask_buf_ (already maintained)
    //   new part [old_len, total): causal within real tokens per element
    auto mask = torch::full({B_, 1, S, total}, -1e9f, opts_fp32);

    // Cached: broadcast the [B, 1, 1, old_len] mask across S query positions
    if (old_len > 0) {
        auto cached = causal_mask_buf_.slice(3, 0, old_len);  // [B, 1, 1, old_len]
        mask.slice(3, 0, old_len).copy_(cached.expand({B_, 1, S, old_len}));
    }

    // New tokens: causal within real tokens
    auto j_range = torch::arange(S, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto new_valid = j_range.unsqueeze(0) < num_real.unsqueeze(1);  // [B, S]
    auto rows = torch::arange(S, torch::kCUDA).unsqueeze(1);
    auto cols = torch::arange(S, torch::kCUDA).unsqueeze(0);
    auto causal_within = (cols <= rows);  // [S, S]
    auto new_attend = new_valid.unsqueeze(1) & causal_within.unsqueeze(0);  // [B, S, S]
    auto new_mask = torch::where(new_attend,
                                 torch::tensor(0.0f, opts_fp32),
                                 torch::tensor(-1e9f, opts_fp32));
    mask.slice(3, old_len, total).copy_(new_mask.unsqueeze(1));

    auto h = forwardImpl(ids, pos, mask, causal_k_, causal_v_, old_len, ov, om, true, false);

    // Update mask buffer: mark positions [old_len, old_len+S) based on num_real.
    // For each position j, valid if j < num_real[b].
    // Use vectorized approach: build [B, S] bool mask and apply in one go.
    auto j_indices = torch::arange(S, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto valid_mask = j_indices.unsqueeze(0) < num_real.unsqueeze(1);  // [B, S] bool
    auto buf_slice = causal_mask_buf_.slice(3, old_len, old_len + S)
                         .squeeze(2).squeeze(1);  // [B, S]
    buf_slice.masked_fill_(valid_mask, 0.0f);

    return h;
}

torch::Tensor BatchedBackbone::causalIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om,
    torch::Tensor active)
{
    int new_pos = causal_len_;
    int total = new_pos + 1;

    // Mark new position valid.  If caller pre-marked via markCausalValidRange,
    // this is a redundant masked_fill_ on already-0.0 values (cheap no-op).
    markCausalValid(new_pos, active);

    // Slice + contiguous: the slice of the pre-allocated buffer is non-contiguous
    // (stride[0]=max_seq_len vs size[3]=total). Making it contiguous avoids
    // slow paths in the attention kernel.
    auto mask = causal_mask_buf_.slice(3, 0, total).contiguous();

    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, new_pos, ov, om, true, false);
}

// ============================================================================
// Prefix mode
// ============================================================================

void BatchedBackbone::resetPrefix()
{
    prefix_len_ = 0;
    prefix_mask_buf_.fill_(-1e9f);
}

torch::Tensor BatchedBackbone::prefixForward(
    torch::Tensor ids, torch::Tensor pos, torch::Tensor mask,
    torch::Tensor ov, torch::Tensor om,
    torch::Tensor active)
{
    // Full forward with explicit mask (used for init).
    auto h = forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, prefix_len_, ov, om, false, true);

    // Mark all positions as valid — single kernel
    int S = ids.size(1);
    markPrefixValidRange(0, S, active);

    return h;
}

torch::Tensor BatchedBackbone::prefixIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om,
    torch::Tensor active)
{
    int total = prefix_len_ + 1;

    // Mark new position in buffer
    markPrefixValid(prefix_len_, active);

    auto mask = prefix_mask_buf_.slice(3, 0, total).contiguous();

    auto h = forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, prefix_len_, ov, om, false, true);

    return h;
}

torch::Tensor BatchedBackbone::prefixBlockForward(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om,
    torch::Tensor active)
{
    // Block forward: S new tokens attend to all valid cached + all within block.
    int S = ids.size(1);
    int old_len = prefix_len_;
    int total = old_len + S;
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Build mask [B, 1, S, total]
    auto mask = torch::full({B_, 1, S, total}, -1e9f, opts_fp32);

    // Cached part: use prefix mask buffer (already maintained)
    if (old_len > 0) {
        auto cached = prefix_mask_buf_.slice(3, 0, old_len);
        mask.slice(3, 0, old_len).copy_(cached.expand({B_, 1, S, old_len}));
    }

    // Block part: all attend to all within block if active
    auto block_val = torch::where(active.view({B_, 1, 1, 1}),
                                  torch::tensor(0.0f, opts_fp32),
                                  torch::tensor(-1e9f, opts_fp32));
    mask.slice(3, old_len, total).copy_(block_val.expand({B_, 1, S, S}));

    auto h = forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, old_len, ov, om, false, true);

    // Mark block positions as valid — single kernel
    markPrefixValidRange(old_len, S, active);

    return h;
}

// ============================================================================
// Probe: forward WITHOUT updating any cache or mask buffers
// ============================================================================

torch::Tensor BatchedBackbone::causalProbe(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    int S = ids.size(1);
    int total = causal_len_ + S;
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Build mask: cached part from buffer + new tokens attend to everything
    auto mask = torch::full({B_, 1, S, total}, -1e9f, opts_fp32);
    if (causal_len_ > 0) {
        auto cached = causal_mask_buf_.slice(3, 0, causal_len_);
        mask.slice(3, 0, causal_len_).copy_(cached.expand({B_, 1, S, causal_len_}));
    }
    // Probe tokens see each other
    mask.slice(3, causal_len_, total).fill_(0.0f);

    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, causal_len_, ov, om,
                       false, false);
}

torch::Tensor BatchedBackbone::prefixProbe(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    int S = ids.size(1);
    int total = prefix_len_ + S;
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto mask = torch::full({B_, 1, S, total}, -1e9f, opts_fp32);
    if (prefix_len_ > 0) {
        auto cached = prefix_mask_buf_.slice(3, 0, prefix_len_);
        mask.slice(3, 0, prefix_len_).copy_(cached.expand({B_, 1, S, prefix_len_}));
    }
    mask.slice(3, prefix_len_, total).fill_(0.0f);

    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, prefix_len_, ov, om,
                       false, false);
}

// ============================================================================
// Phase 4: per-slot refill (continuous batching)
// ============================================================================

torch::Tensor BatchedBackbone::resetSlotsForRefill(
    torch::Tensor slot_active,
    torch::Tensor init_ids, torch::Tensor init_pos,
    torch::Tensor prefix_mask,
    torch::Tensor init_ov, torch::Tensor init_om)
{
    torch::NoGradGuard no_grad;

    int init_len = init_ids.size(1);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // 1. Wipe slot rows of mask buffers entirely to -inf. masked_fill_
    //    broadcasts slot_active [B] across the [1, 1, max_seq_len] dims.
    auto reset_view = slot_active.view({B_, 1, 1, 1});
    causal_mask_buf_.masked_fill_(reset_view, -1e9f);
    prefix_mask_buf_.masked_fill_(reset_view, -1e9f);

    // 2. Build empty past for forward_new (past_len=0).
    auto past_k_empty = torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts_fp16);
    auto past_v_empty = torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts_fp16);

    // 3. Causal prefill: causal-only attention mask within [init_len, init_len].
    auto causal_pattern = torch::tril(torch::ones({init_len, init_len}, opts_fp32));
    auto causal_mask = torch::where(
        causal_pattern.to(torch::kBool),
        torch::tensor(0.0f, opts_fp32),
        torch::tensor(-1e9f, opts_fp32))
        .view({1, 1, init_len, init_len})
        .expand({B_, 1, init_len, init_len})
        .contiguous();

    std::vector<torch::IValue> causal_inputs{
        init_ids, init_pos, causal_mask, past_k_empty, past_v_empty, init_ov, init_om};
    auto c_result = (*forward_new_method_)(causal_inputs);
    auto c_outputs = c_result.toTuple()->elements();
    auto causal_kn = c_outputs[1].toTensor();   // [NL, B, NH, init_len, HD]
    auto causal_vn = c_outputs[2].toTensor();

    // 4. Prefix prefill: block-aware mask supplied by caller.
    std::vector<torch::IValue> prefix_inputs{
        init_ids, init_pos, prefix_mask, past_k_empty, past_v_empty, init_ov, init_om};
    auto p_result = (*forward_new_method_)(prefix_inputs);
    auto p_outputs = p_result.toTuple()->elements();
    auto prefix_h = p_outputs[0].toTensor();    // [B, init_len, E]
    auto prefix_kn = p_outputs[1].toTensor();
    auto prefix_vn = p_outputs[2].toTensor();

    // 5. Scatter K/V into selected slot rows at physical [0, init_len). Loop
    //    over K is short (typically 1-2 refills per cycle). One D2H sync.
    auto slot_idx = torch::nonzero(slot_active).squeeze(1);  // [K] int64 on CUDA
    auto slot_idx_cpu = slot_idx.cpu();
    auto si_a = slot_idx_cpu.accessor<int64_t, 1>();
    int K = (int)slot_idx.size(0);

    for (int k = 0; k < K; k++)
    {
        int64_t b = si_a[k];
        // causal_k_[:, b, :, 0:init_len, :].copy_(causal_kn[:, b, :, :, :])
        causal_k_.select(1, b).slice(2, 0, init_len).copy_(causal_kn.select(1, b));
        causal_v_.select(1, b).slice(2, 0, init_len).copy_(causal_vn.select(1, b));
        prefix_k_.select(1, b).slice(2, 0, init_len).copy_(prefix_kn.select(1, b));
        prefix_v_.select(1, b).slice(2, 0, init_len).copy_(prefix_vn.select(1, b));
    }

    // 6. Mark mask valid at [0, init_len) for the refilled slots.
    auto causal_slice = causal_mask_buf_.slice(3, 0, init_len)
                            .squeeze(2).squeeze(1);   // [B, init_len]
    causal_slice.masked_fill_(slot_active.unsqueeze(1), 0.0f);
    auto prefix_slice = prefix_mask_buf_.slice(3, 0, init_len)
                            .squeeze(2).squeeze(1);
    prefix_slice.masked_fill_(slot_active.unsqueeze(1), 0.0f);

    // NOTE: causal_len_ / prefix_len_ are intentionally NOT modified.
    // resetSlotsForRefill is called mid-rollout when other slots are at higher
    // global positions. Slot b's logical seq_pos resets to init_len (caller's
    // responsibility); RoPE uses input_pos arg, decoupled from physical cache
    // index. Slot b's mask -inf at [init_len, current_global_len) keeps stale
    // physical positions from being attended.

    // Return last-position hidden state. Caller picks rows for refilled slots.
    return prefix_h.index({torch::indexing::Slice(), init_len - 1}).contiguous();
}

} // namespace decoder

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

    model_.forward({
        torch::zeros({B_, 1}, opts_int), torch::zeros({B_, 1}, opts_int),
        torch::zeros({B_, 1, 1, 1}, opts_fp32),
        torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts_fp16),
        torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts_fp16),
        torch::zeros({B_, 1}, opts_fp16), torch::zeros({B_, 1}, opts_bool)
    });
    c10::cuda::getCurrentCUDAStream().synchronize();

    // Pre-allocate mask buffers filled with -inf
    causal_mask_buf_ = torch::full({B_, 1, 1, max_seq_len_}, -1e9f, opts_fp32);
    prefix_mask_buf_ = torch::full({B_, 1, 1, max_seq_len_}, -1e9f, opts_fp32);

    resetCausal();
    resetPrefix();

    std::cout << "[BatchedBackbone] " << num_layers << " layers, "
              << embed_dim << "d, batch=" << B_
              << " (pre-allocated mask buffers, zero per-step alloc)"
              << std::endl;
}

// ============================================================================
// Core forward
// ============================================================================

torch::Tensor BatchedBackbone::forwardImpl(
    torch::Tensor ids, torch::Tensor pos, torch::Tensor mask,
    torch::Tensor past_k, torch::Tensor past_v,
    torch::Tensor ov, torch::Tensor om,
    bool update_causal, bool update_prefix)
{
    torch::NoGradGuard no_grad;
    auto result = model_.forward({ids, pos, mask, past_k, past_v, ov, om});
    auto outputs = result.toTuple()->elements();
    auto hidden = outputs[0].toTensor();
    auto pk = outputs[1].toTensor();
    auto pv = outputs[2].toTensor();

    if (update_causal) { causal_k_ = pk; causal_v_ = pv; causal_len_ = pk.size(3); }
    if (update_prefix) { prefix_k_ = pk; prefix_v_ = pv; prefix_len_ = pk.size(3); }
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
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    causal_k_ = torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts);
    causal_v_ = torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts);
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

    auto h = forwardImpl(ids, pos, mask, causal_k_, causal_v_, ov, om, true, false);

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

    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, ov, om, true, false);
}

// ============================================================================
// Prefix mode
// ============================================================================

void BatchedBackbone::resetPrefix()
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    prefix_k_ = torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts);
    prefix_v_ = torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts);
    prefix_len_ = 0;
    prefix_mask_buf_.fill_(-1e9f);
}

torch::Tensor BatchedBackbone::prefixForward(
    torch::Tensor ids, torch::Tensor pos, torch::Tensor mask,
    torch::Tensor ov, torch::Tensor om,
    torch::Tensor active)
{
    // Full forward with explicit mask (used for init).
    auto h = forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om, false, true);

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

    auto h = forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om, false, true);

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

    auto h = forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om, false, true);

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

    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, ov, om,
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

    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om,
                       false, false);
}

} // namespace decoder

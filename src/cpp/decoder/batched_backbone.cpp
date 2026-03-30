#include "batched_backbone.hpp"

#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <stdexcept>

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

    // Warmup forward with batch_size to trigger CUDA initialization and
    // verify the TorchScript model supports B>1
    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    auto dummy_ids = torch::zeros({B_, 1}, opts_int);
    auto dummy_pos = torch::zeros({B_, 1}, opts_int);
    auto dummy_mask = torch::zeros({B_, 1, 1, 1}, opts_fp32);
    auto dummy_pk = torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts_fp16);
    auto dummy_pv = torch::zeros({num_layers_, B_, num_heads_, 0, head_dim_}, opts_fp16);
    auto dummy_ov = torch::zeros({B_, 1}, opts_fp16);
    auto dummy_om = torch::zeros({B_, 1}, opts_bool);

    model_.forward({dummy_ids, dummy_pos, dummy_mask, dummy_pk, dummy_pv, dummy_ov, dummy_om});
    c10::cuda::getCurrentCUDAStream().synchronize();

    resetCausal();
    resetPrefix();

    std::cout << "[BatchedBackbone] Loaded: " << num_layers << " layers, "
              << embed_dim << "d, batch=" << B_ << ", max_seq=" << max_seq_len
              << std::endl;
}

// ============================================================================
// Core forward
// ============================================================================

torch::Tensor BatchedBackbone::forwardImpl(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor mask,
    torch::Tensor past_k, torch::Tensor past_v,
    torch::Tensor ov, torch::Tensor om,
    bool update_causal, bool update_prefix)
{
    torch::NoGradGuard no_grad;

    auto result = model_.forward({ids, pos, mask, past_k, past_v, ov, om});
    auto outputs = result.toTuple()->elements();

    auto hidden = outputs[0].toTensor();      // [B, S, E]
    auto present_k = outputs[1].toTensor();   // [NL, B, NH, past+S, HD]
    auto present_v = outputs[2].toTensor();   // [NL, B, NH, past+S, HD]

    if (update_causal)
    {
        causal_k_ = present_k;
        causal_v_ = present_v;
        causal_len_ = present_k.size(3);
    }
    if (update_prefix)
    {
        prefix_k_ = present_k;
        prefix_v_ = present_v;
        prefix_len_ = present_k.size(3);
    }

    return hidden;
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
}

torch::Tensor BatchedBackbone::causalForward(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    int S = ids.size(1);
    int total = causal_len_ + S;

    // Build causal mask [B, 1, S, total]
    auto mask = torch::full({B_, 1, S, total}, -1e9f,
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    // Unmask causal positions: for query i, attend to all j <= past_len + i
    auto rows = torch::arange(S, torch::kCUDA).unsqueeze(1);    // [S, 1]
    auto cols = torch::arange(total, torch::kCUDA).unsqueeze(0); // [1, total]
    auto causal_pattern = (cols <= (rows + causal_len_));         // [S, total]
    // Expand to [B, 1, S, total] and set 0 where causal
    mask.masked_fill_(causal_pattern.unsqueeze(0).unsqueeze(0), 0.0f);

    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, ov, om,
                       /*update_causal=*/true, /*update_prefix=*/false);
}

torch::Tensor BatchedBackbone::causalIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    // 1 new token attending to all past + itself → all-zeros mask
    int total = causal_len_ + 1;
    auto mask = torch::zeros({B_, 1, 1, total},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, ov, om,
                       /*update_causal=*/true, /*update_prefix=*/false);
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
}

torch::Tensor BatchedBackbone::prefixForward(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor mask,
    torch::Tensor ov, torch::Tensor om)
{
    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om,
                       /*update_causal=*/false, /*update_prefix=*/true);
}

torch::Tensor BatchedBackbone::prefixIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    // 1 new token attending to all past + itself → all-zeros mask
    int total = prefix_len_ + 1;
    auto mask = torch::zeros({B_, 1, 1, total},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om,
                       /*update_causal=*/false, /*update_prefix=*/true);
}

torch::Tensor BatchedBackbone::prefixBlockForward(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    int S = ids.size(1);
    int total = prefix_len_ + S;

    // All-zeros mask: new board tokens attend to everything
    // (bidirectional within the block, plus causal to all past)
    auto mask = torch::zeros({B_, 1, S, total},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om,
                       /*update_causal=*/false, /*update_prefix=*/true);
}

} // namespace decoder

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

    resetCausal();
    resetPrefix();

    std::cout << "[BatchedBackbone] " << num_layers << " layers, "
              << embed_dim << "d, batch=" << B_ << " (no graphs, max memory for batching)"
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
    auto mask = torch::full({B_, 1, S, total}, -1e9f,
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto rows = torch::arange(S, torch::kCUDA).unsqueeze(1);
    auto cols = torch::arange(total, torch::kCUDA).unsqueeze(0);
    mask.masked_fill_((cols <= (rows + causal_len_)).unsqueeze(0).unsqueeze(0), 0.0f);
    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, ov, om, true, false);
}

torch::Tensor BatchedBackbone::causalIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    int total = causal_len_ + 1;
    auto mask = torch::zeros({B_, 1, 1, total},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
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
}

torch::Tensor BatchedBackbone::prefixForward(
    torch::Tensor ids, torch::Tensor pos, torch::Tensor mask,
    torch::Tensor ov, torch::Tensor om)
{
    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om, false, true);
}

torch::Tensor BatchedBackbone::prefixIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    int total = prefix_len_ + 1;
    auto mask = torch::zeros({B_, 1, 1, total},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om, false, true);
}

torch::Tensor BatchedBackbone::prefixBlockForward(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    int S = ids.size(1);
    int total = prefix_len_ + S;
    auto mask = torch::zeros({B_, 1, S, total},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om, false, true);
}

} // namespace decoder

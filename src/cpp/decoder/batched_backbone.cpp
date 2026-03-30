#include "batched_backbone.hpp"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
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
    , active_tier_(-1), pg_len_(0), graphs_captured_(false)
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
    captureGraphs();

    std::cout << "[BatchedBackbone] " << num_layers << " layers, "
              << embed_dim << "d, batch=" << B_ << std::endl;
}

// ============================================================================
// CUDA Graph capture: tiered causal + single prefix
// ============================================================================

void BatchedBackbone::captureGraphs()
{
    torch::NoGradGuard no_grad;
    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    auto pool = at::cuda::graph_pool_handle();
    auto stream = at::cuda::getStreamFromPool(false);

    {
        c10::cuda::CUDAStreamGuard guard(stream);

        // Shared input tensors for all causal tier graphs
        cg_ids_ = torch::zeros({B_, 1}, opts_int);
        cg_pos_ = torch::zeros({B_, 1}, opts_int);
        cg_ov_ = torch::zeros({B_, 1}, opts_fp16);
        cg_om_ = torch::zeros({B_, 1}, opts_bool);

        // ---- Causal tier graphs ----
        causal_tiers_.resize(kNumCausalTiers);
        size_t tier_mem = 0;

        for (int t = 0; t < kNumCausalTiers; t++)
        {
            causal_tiers_[t] = std::make_unique<BatchedBoardTier>();
            auto& tier = *causal_tiers_[t];
            tier.max_len = kCausalTierSizes[t];
            tier.mask = torch::full({B_, 1, 1, tier.max_len + 1}, -1e9f, opts_fp32);
            tier.buf_k = torch::zeros({num_layers_, B_, num_heads_, tier.max_len, head_dim_}, opts_fp16);
            tier.buf_v = torch::zeros({num_layers_, B_, num_heads_, tier.max_len, head_dim_}, opts_fp16);
            tier.len = 0;

            // Memory accounting
            tier_mem += 2ULL * num_layers_ * B_ * num_heads_ * tier.max_len * head_dim_ * 2;

            // Warmup
            for (int i = 0; i < 3; i++)
                model_.forward({cg_ids_, cg_pos_, tier.mask, tier.buf_k, tier.buf_v, cg_ov_, cg_om_});
            stream.synchronize();

            // Capture
            tier.graph.capture_begin(pool);
            {
                auto output = model_.forward({cg_ids_, cg_pos_, tier.mask, tier.buf_k, tier.buf_v, cg_ov_, cg_om_});
                auto elements = output.toTuple()->elements();
                tier.out_h = elements[0].toTensor();
                tier.out_pk = elements[1].toTensor();
                tier.out_pv = elements[2].toTensor();
            }
            tier.graph.capture_end();
        }

        // ---- Prefix graph (single, moderate buffer) ----
        pg_ids_ = torch::zeros({B_, 1}, opts_int);
        pg_pos_ = torch::zeros({B_, 1}, opts_int);
        pg_mask_ = torch::full({B_, 1, 1, kPrefixGraphLen + 1}, -1e9f, opts_fp32);
        pg_ov_ = torch::zeros({B_, 1}, opts_fp16);
        pg_om_ = torch::zeros({B_, 1}, opts_bool);
        pg_buf_k_ = torch::zeros({num_layers_, B_, num_heads_, kPrefixGraphLen, head_dim_}, opts_fp16);
        pg_buf_v_ = torch::zeros({num_layers_, B_, num_heads_, kPrefixGraphLen, head_dim_}, opts_fp16);
        pg_len_ = 0;

        size_t prefix_mem = 2ULL * num_layers_ * B_ * num_heads_ * kPrefixGraphLen * head_dim_ * 2;

        for (int i = 0; i < 3; i++)
            model_.forward({pg_ids_, pg_pos_, pg_mask_, pg_buf_k_, pg_buf_v_, pg_ov_, pg_om_});
        stream.synchronize();

        prefix_graph_.capture_begin(pool);
        {
            auto output = model_.forward({pg_ids_, pg_pos_, pg_mask_, pg_buf_k_, pg_buf_v_, pg_ov_, pg_om_});
            auto elements = output.toTuple()->elements();
            pg_out_h_ = elements[0].toTensor();
            pg_out_pk_ = elements[1].toTensor();
            pg_out_pv_ = elements[2].toTensor();
        }
        prefix_graph_.capture_end();

        std::cout << "[BatchedBackbone] Causal tiers: ";
        for (int t = 0; t < kNumCausalTiers; t++)
            std::cout << kCausalTierSizes[t] << (t < kNumCausalTiers - 1 ? ", " : "");
        std::cout << " (" << tier_mem / (1024*1024) << " MB)" << std::endl;
        std::cout << "[BatchedBackbone] Prefix graph: " << kPrefixGraphLen
                  << " (" << prefix_mem / (1024*1024) << " MB)" << std::endl;
    }

    graphs_captured_ = true;
    c10::cuda::getCurrentCUDAStream().synchronize();
}

// ============================================================================
// Dynamic forward
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
    active_tier_ = -1;

    if (graphs_captured_)
    {
        for (auto& tier : causal_tiers_)
        {
            tier->buf_k.zero_();
            tier->buf_v.zero_();
            tier->mask.fill_(-1e9f);
            using namespace torch::indexing;
            tier->mask.index_put_({Slice(), Slice(), Slice(), tier->max_len}, 0.0f);
            tier->len = 0;
        }
    }
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

int BatchedBackbone::selectCausalTier(int max_needed_pos) const
{
    for (int t = 0; t < kNumCausalTiers; t++)
        if (causal_tiers_[t]->max_len >= max_needed_pos)
            return t;
    return -1;
}

torch::Tensor BatchedBackbone::causalIncrementalDynamic(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    int total = causal_len_ + 1;
    auto mask = torch::zeros({B_, 1, 1, total},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, ov, om, true, false);
}

torch::Tensor BatchedBackbone::causalIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    // Select tier based on current position
    int tier_idx = selectCausalTier(causal_len_ + 1);

    if (tier_idx < 0 || !graphs_captured_)
        return causalIncrementalDynamic(ids, pos, ov, om);

    // If we need to switch tiers, sync old tier → dynamic → new tier
    if (tier_idx != active_tier_)
    {
        // First: sync old tier back to dynamic cache (if any)
        if (active_tier_ >= 0)
        {
            auto& old_tier = *causal_tiers_[active_tier_];
            int old_len = old_tier.len;
            using namespace torch::indexing;
            causal_k_ = old_tier.buf_k.index({Slice(), Slice(), Slice(), Slice(0, old_len)}).clone();
            causal_v_ = old_tier.buf_v.index({Slice(), Slice(), Slice(), Slice(0, old_len)}).clone();
            causal_len_ = old_len;
        }

        // Then: copy dynamic cache into the new tier's buffer
        auto& tier = *causal_tiers_[tier_idx];
        int len = causal_len_;
        if (len > 0)
        {
            using namespace torch::indexing;
            tier.buf_k.index({Slice(), Slice(), Slice(), Slice(0, len)})
                .copy_(causal_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
            tier.buf_v.index({Slice(), Slice(), Slice(), Slice(0, len)})
                .copy_(causal_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
        }
        tier.mask.fill_(-1e9f);
        if (len > 0)
        {
            using namespace torch::indexing;
            tier.mask.index_put_({Slice(), Slice(), Slice(), Slice(0, len)}, 0.0f);
        }
        tier.mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                              torch::indexing::Slice(), tier.max_len}, 0.0f);
        tier.len = len;
        active_tier_ = tier_idx;
    }

    torch::NoGradGuard no_grad;
    auto& tier = *causal_tiers_[active_tier_];

    cg_ids_.copy_(ids);
    cg_pos_.copy_(pos);
    cg_ov_.copy_(ov);
    cg_om_.copy_(om);

    tier.graph.replay();

    using namespace torch::indexing;
    tier.buf_k.index({Slice(), Slice(), Slice(), tier.len})
        .copy_(tier.out_pk.index({Slice(), Slice(), Slice(), tier.max_len}));
    tier.buf_v.index({Slice(), Slice(), Slice(), tier.len})
        .copy_(tier.out_pv.index({Slice(), Slice(), Slice(), tier.max_len}));
    tier.mask.index_put_({Slice(), Slice(), Slice(), tier.len}, 0.0f);
    tier.len++;

    causal_len_ = tier.len;

    return tier.out_h;  // [B, 1, E]
}

void BatchedBackbone::syncCausalToGraph()
{
    int tier_idx = selectCausalTier(causal_len_);
    if (tier_idx < 0) { active_tier_ = -1; return; }

    auto& tier = *causal_tiers_[tier_idx];
    int len = causal_len_;
    if (len > 0)
    {
        using namespace torch::indexing;
        tier.buf_k.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(causal_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
        tier.buf_v.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(causal_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
    }
    tier.mask.fill_(-1e9f);
    if (len > 0)
    {
        using namespace torch::indexing;
        tier.mask.index_put_({Slice(), Slice(), Slice(), Slice(0, len)}, 0.0f);
    }
    tier.mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                          torch::indexing::Slice(), tier.max_len}, 0.0f);
    tier.len = len;
    active_tier_ = tier_idx;
}

void BatchedBackbone::syncGraphToCausal()
{
    if (active_tier_ < 0) return;
    auto& tier = *causal_tiers_[active_tier_];
    int len = tier.len;
    using namespace torch::indexing;
    causal_k_ = tier.buf_k.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    causal_v_ = tier.buf_v.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    causal_len_ = len;
    active_tier_ = -1;
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

    if (graphs_captured_)
    {
        pg_buf_k_.zero_();
        pg_buf_v_.zero_();
        pg_mask_.fill_(-1e9f);
        using namespace torch::indexing;
        pg_mask_.index_put_({Slice(), Slice(), Slice(), kPrefixGraphLen}, 0.0f);
        pg_len_ = 0;
    }
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
    // If prefix exceeds graph buffer, fall back to dynamic
    if (prefix_len_ >= kPrefixGraphLen || !graphs_captured_)
    {
        int total = prefix_len_ + 1;
        auto mask = torch::zeros({B_, 1, 1, total},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om, false, true);
    }

    torch::NoGradGuard no_grad;

    pg_ids_.copy_(ids);
    pg_pos_.copy_(pos);
    pg_ov_.copy_(ov);
    pg_om_.copy_(om);

    prefix_graph_.replay();

    using namespace torch::indexing;
    pg_buf_k_.index({Slice(), Slice(), Slice(), pg_len_})
        .copy_(pg_out_pk_.index({Slice(), Slice(), Slice(), kPrefixGraphLen}));
    pg_buf_v_.index({Slice(), Slice(), Slice(), pg_len_})
        .copy_(pg_out_pv_.index({Slice(), Slice(), Slice(), kPrefixGraphLen}));
    pg_mask_.index_put_({Slice(), Slice(), Slice(), pg_len_}, 0.0f);

    pg_len_++;
    prefix_len_ = pg_len_;

    return pg_out_h_;  // [B, 1, E]
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

void BatchedBackbone::syncPrefixToGraph()
{
    if (!graphs_captured_ || prefix_len_ > kPrefixGraphLen) return;
    int len = prefix_len_;
    if (len > 0)
    {
        using namespace torch::indexing;
        pg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(prefix_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
        pg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(prefix_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
    }
    pg_mask_.fill_(-1e9f);
    if (len > 0)
    {
        using namespace torch::indexing;
        pg_mask_.index_put_({Slice(), Slice(), Slice(), Slice(0, len)}, 0.0f);
    }
    pg_mask_.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(), kPrefixGraphLen}, 0.0f);
    pg_len_ = len;
}

void BatchedBackbone::syncGraphToPrefix()
{
    if (!graphs_captured_ || pg_len_ == 0) return;
    int len = pg_len_;
    using namespace torch::indexing;
    prefix_k_ = pg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    prefix_v_ = pg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    prefix_len_ = len;
}

} // namespace decoder

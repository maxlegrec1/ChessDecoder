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
    , cg_len_(0), pg_len_(0), graphs_captured_(false)
{
    torch::NoGradGuard no_grad;

    model_ = torch::jit::load(pt_path, torch::kCUDA);
    model_.eval();

    // Warmup with batch_size to verify B>1 support
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
    captureGraphs();

    std::cout << "[BatchedBackbone] Loaded: " << num_layers << " layers, "
              << embed_dim << "d, batch=" << B_ << ", max_seq=" << max_seq_len
              << " [CUDA graphs captured]" << std::endl;
}

// ============================================================================
// CUDA Graph capture
// ============================================================================

void BatchedBackbone::captureGraphs()
{
    torch::NoGradGuard no_grad;
    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    auto pool = at::cuda::graph_pool_handle();
    auto capture_stream = at::cuda::getStreamFromPool(false);

    {
        c10::cuda::CUDAStreamGuard guard(capture_stream);

        // ---- Causal graph: [B, 1] incremental ----
        cg_ids_ = torch::zeros({B_, 1}, opts_int);
        cg_pos_ = torch::zeros({B_, 1}, opts_int);
        cg_mask_ = torch::full({B_, 1, 1, max_seq_len_ + 1}, -1e9f, opts_fp32);
        cg_ov_ = torch::zeros({B_, 1}, opts_fp16);
        cg_om_ = torch::zeros({B_, 1}, opts_bool);
        cg_buf_k_ = torch::zeros({num_layers_, B_, num_heads_, max_seq_len_, head_dim_}, opts_fp16);
        cg_buf_v_ = torch::zeros({num_layers_, B_, num_heads_, max_seq_len_, head_dim_}, opts_fp16);
        cg_len_ = 0;

        // Warmup
        for (int i = 0; i < 3; i++)
            model_.forward({cg_ids_, cg_pos_, cg_mask_, cg_buf_k_, cg_buf_v_, cg_ov_, cg_om_});
        capture_stream.synchronize();

        // Capture
        causal_graph_.capture_begin(pool);
        {
            auto output = model_.forward({cg_ids_, cg_pos_, cg_mask_, cg_buf_k_, cg_buf_v_, cg_ov_, cg_om_});
            auto elements = output.toTuple()->elements();
            cg_out_h_ = elements[0].toTensor();
            cg_out_pk_ = elements[1].toTensor();
            cg_out_pv_ = elements[2].toTensor();
        }
        causal_graph_.capture_end();

        // ---- Prefix graph: [B, 1] incremental ----
        pg_ids_ = torch::zeros({B_, 1}, opts_int);
        pg_pos_ = torch::zeros({B_, 1}, opts_int);
        pg_mask_ = torch::full({B_, 1, 1, max_seq_len_ + 1}, -1e9f, opts_fp32);
        pg_ov_ = torch::zeros({B_, 1}, opts_fp16);
        pg_om_ = torch::zeros({B_, 1}, opts_bool);
        pg_buf_k_ = torch::zeros({num_layers_, B_, num_heads_, max_seq_len_, head_dim_}, opts_fp16);
        pg_buf_v_ = torch::zeros({num_layers_, B_, num_heads_, max_seq_len_, head_dim_}, opts_fp16);
        pg_len_ = 0;

        for (int i = 0; i < 3; i++)
            model_.forward({pg_ids_, pg_pos_, pg_mask_, pg_buf_k_, pg_buf_v_, pg_ov_, pg_om_});
        capture_stream.synchronize();

        prefix_graph_.capture_begin(pool);
        {
            auto output = model_.forward({pg_ids_, pg_pos_, pg_mask_, pg_buf_k_, pg_buf_v_, pg_ov_, pg_om_});
            auto elements = output.toTuple()->elements();
            pg_out_h_ = elements[0].toTensor();
            pg_out_pk_ = elements[1].toTensor();
            pg_out_pv_ = elements[2].toTensor();
        }
        prefix_graph_.capture_end();
    }

    graphs_captured_ = true;
    c10::cuda::getCurrentCUDAStream().synchronize();

    // Log memory
    size_t graph_mem = 2ULL * num_layers_ * B_ * num_heads_ * max_seq_len_ * head_dim_ * 2 * 2;  // 2 graphs × k+v
    std::cout << "[BatchedBackbone] Graph buffers: " << graph_mem / (1024 * 1024) << " MB" << std::endl;
}

// ============================================================================
// Dynamic forward (shared)
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

    auto hidden = outputs[0].toTensor();
    auto present_k = outputs[1].toTensor();
    auto present_v = outputs[2].toTensor();

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

    if (graphs_captured_)
    {
        cg_buf_k_.zero_();
        cg_buf_v_.zero_();
        cg_mask_.fill_(-1e9f);
        // Unmask position max_seq_len_ (where cat places new KV)
        using namespace torch::indexing;
        cg_mask_.index_put_({Slice(), Slice(), Slice(), max_seq_len_}, 0.0f);
        cg_len_ = 0;
    }
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
    auto rows = torch::arange(S, torch::kCUDA).unsqueeze(1);
    auto cols = torch::arange(total, torch::kCUDA).unsqueeze(0);
    auto causal_pattern = (cols <= (rows + causal_len_));
    mask.masked_fill_(causal_pattern.unsqueeze(0).unsqueeze(0), 0.0f);

    return forwardImpl(ids, pos, mask, causal_k_, causal_v_, ov, om,
                       true, false);
}

torch::Tensor BatchedBackbone::causalIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    torch::NoGradGuard no_grad;

    // Update graph input tensors (same addresses, different values)
    cg_ids_.copy_(ids);
    cg_pos_.copy_(pos);
    cg_ov_.copy_(ov);
    cg_om_.copy_(om);

    // Replay
    causal_graph_.replay();

    // New KV is at position max_seq_len_ (after cat). Copy to cg_len_.
    using namespace torch::indexing;
    cg_buf_k_.index({Slice(), Slice(), Slice(), cg_len_})
        .copy_(cg_out_pk_.index({Slice(), Slice(), Slice(), max_seq_len_}));
    cg_buf_v_.index({Slice(), Slice(), Slice(), cg_len_})
        .copy_(cg_out_pv_.index({Slice(), Slice(), Slice(), max_seq_len_}));

    // Unmask position cg_len_ for next call
    cg_mask_.index_put_({Slice(), Slice(), Slice(), cg_len_}, 0.0f);

    cg_len_++;
    causal_len_ = cg_len_;

    return cg_out_h_;  // [B, 1, E]
}

void BatchedBackbone::syncCausalToGraph()
{
    using namespace torch::indexing;
    int len = causal_len_;
    if (len > 0 && len <= max_seq_len_)
    {
        cg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(causal_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
        cg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(causal_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}));

        cg_mask_.fill_(-1e9f);
        cg_mask_.index_put_({Slice(), Slice(), Slice(), Slice(0, len)}, 0.0f);
        cg_mask_.index_put_({Slice(), Slice(), Slice(), max_seq_len_}, 0.0f);
    }
    cg_len_ = len;
}

void BatchedBackbone::syncGraphToCausal()
{
    using namespace torch::indexing;
    int len = cg_len_;
    causal_k_ = cg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    causal_v_ = cg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    causal_len_ = len;
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
        pg_mask_.index_put_({Slice(), Slice(), Slice(), max_seq_len_}, 0.0f);
        pg_len_ = 0;
    }
}

torch::Tensor BatchedBackbone::prefixForward(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor mask,
    torch::Tensor ov, torch::Tensor om)
{
    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om,
                       false, true);
}

torch::Tensor BatchedBackbone::prefixIncremental(
    torch::Tensor ids, torch::Tensor pos,
    torch::Tensor ov, torch::Tensor om)
{
    torch::NoGradGuard no_grad;

    pg_ids_.copy_(ids);
    pg_pos_.copy_(pos);
    pg_ov_.copy_(ov);
    pg_om_.copy_(om);

    prefix_graph_.replay();

    using namespace torch::indexing;
    pg_buf_k_.index({Slice(), Slice(), Slice(), pg_len_})
        .copy_(pg_out_pk_.index({Slice(), Slice(), Slice(), max_seq_len_}));
    pg_buf_v_.index({Slice(), Slice(), Slice(), pg_len_})
        .copy_(pg_out_pv_.index({Slice(), Slice(), Slice(), max_seq_len_}));

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

    return forwardImpl(ids, pos, mask, prefix_k_, prefix_v_, ov, om,
                       false, true);
}

void BatchedBackbone::syncPrefixToGraph()
{
    using namespace torch::indexing;
    int len = prefix_len_;
    if (len > 0 && len <= max_seq_len_)
    {
        pg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(prefix_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
        pg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(prefix_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}));

        pg_mask_.fill_(-1e9f);
        pg_mask_.index_put_({Slice(), Slice(), Slice(), Slice(0, len)}, 0.0f);
        pg_mask_.index_put_({Slice(), Slice(), Slice(), max_seq_len_}, 0.0f);
    }
    pg_len_ = len;
}

void BatchedBackbone::syncGraphToPrefix()
{
    using namespace torch::indexing;
    int len = pg_len_;
    prefix_k_ = pg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    prefix_v_ = pg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    prefix_len_ = len;
}

} // namespace decoder

#include "torch_backbone.hpp"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace decoder
{

TorchCausalBackbone::TorchCausalBackbone(
    const std::string& pt_path, int num_layers, int num_heads,
    int head_dim, int embed_dim, int max_cache_len)
    : num_layers_(num_layers), num_heads_(num_heads)
    , head_dim_(head_dim), embed_dim_(embed_dim)
    , max_cache_len_(max_cache_len)
    , cache_len_(0), prefix_cache_len_(0)
    , cg_len_(0), graphs_captured_(false), pg_len_(0)
{
    torch::NoGradGuard no_grad;

    model_ = torch::jit::load(pt_path, torch::kCUDA);
    model_.eval();

    // Warm up: run one forward pass to trigger lazy CUDA initialization
    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    auto dummy_ids = torch::zeros({1, 1}, opts_int);
    auto dummy_pos = torch::zeros({1, 1}, opts_int);
    auto dummy_mask = torch::zeros({1, 1, 1, 1}, opts_fp32);
    auto dummy_pk = torch::zeros({num_layers, 1, num_heads, 0, head_dim}, opts_fp16);
    auto dummy_pv = torch::zeros({num_layers, 1, num_heads, 0, head_dim}, opts_fp16);
    auto dummy_ov = torch::zeros({1, 1}, opts_fp16);
    auto dummy_om = torch::zeros({1, 1}, opts_bool);

    std::vector<torch::jit::IValue> warmup_inputs = {
        dummy_ids, dummy_pos, dummy_mask, dummy_pk, dummy_pv, dummy_ov, dummy_om
    };
    model_.forward(warmup_inputs);
    c10::cuda::getCurrentCUDAStream().synchronize();

    resetCache();
    resetPrefixCache();

    // Capture CUDA graphs for 1-token incremental forward
    captureGraphs();

    std::cout << "[TorchBackbone] Loaded: " << pt_path
              << " (" << num_layers << " layers, " << embed_dim << "d)"
              << " [CUDA graphs captured]" << std::endl;
}

// ============================================================================
// CUDA Graph capture
// ============================================================================

void TorchCausalBackbone::captureGraphs()
{
    torch::NoGradGuard no_grad;
    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    // Shared memory pool so both graphs can coexist
    auto pool = at::cuda::graph_pool_handle();

    // ---- Causal graph ----
    // Pre-allocate static input tensors with fixed shapes
    cg_ids_ = torch::zeros({1, 1}, opts_int);
    cg_pos_ = torch::zeros({1, 1}, opts_int);
    cg_mask_ = torch::full({1, 1, 1, max_cache_len_ + 1}, -1e9f, opts_fp32);
    cg_ov_ = torch::zeros({1, 1}, opts_fp16);
    cg_om_ = torch::zeros({1, 1}, opts_bool);
    cg_buf_k_ = torch::zeros({num_layers_, 1, num_heads_, max_cache_len_, head_dim_}, opts_fp16);
    cg_buf_v_ = torch::zeros({num_layers_, 1, num_heads_, max_cache_len_, head_dim_}, opts_fp16);
    cg_len_ = 0;

    // Create a non-default stream for graph capture (required by CUDA)
    auto capture_stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    {
        c10::cuda::CUDAStreamGuard guard(capture_stream);

        // Warmup on capture stream
        for (int i = 0; i < 3; i++)
        {
            std::vector<torch::jit::IValue> inputs = {
                cg_ids_, cg_pos_, cg_mask_, cg_buf_k_, cg_buf_v_, cg_ov_, cg_om_
            };
            model_.forward(inputs);
        }
        capture_stream.synchronize();

        // Capture causal graph
        causal_graph_.capture_begin(pool);
        {
            std::vector<torch::jit::IValue> inputs = {
                cg_ids_, cg_pos_, cg_mask_, cg_buf_k_, cg_buf_v_, cg_ov_, cg_om_
            };
            auto output = model_.forward(inputs);
            auto elements = output.toTuple()->elements();
            cg_out_h_ = elements[0].toTensor();
            cg_out_pk_ = elements[1].toTensor();
            cg_out_pv_ = elements[2].toTensor();
        }
        causal_graph_.capture_end();

        // ---- Prefix graph ----
        pg_ids_ = torch::zeros({1, 1}, opts_int);
        pg_pos_ = torch::zeros({1, 1}, opts_int);
        pg_mask_ = torch::full({1, 1, 1, max_cache_len_ + 1}, -1e9f, opts_fp32);
        pg_ov_ = torch::zeros({1, 1}, opts_fp16);
        pg_om_ = torch::zeros({1, 1}, opts_bool);
        pg_buf_k_ = torch::zeros({num_layers_, 1, num_heads_, max_cache_len_, head_dim_}, opts_fp16);
        pg_buf_v_ = torch::zeros({num_layers_, 1, num_heads_, max_cache_len_, head_dim_}, opts_fp16);
        pg_len_ = 0;

        for (int i = 0; i < 3; i++)
        {
            std::vector<torch::jit::IValue> inputs = {
                pg_ids_, pg_pos_, pg_mask_, pg_buf_k_, pg_buf_v_, pg_ov_, pg_om_
            };
            model_.forward(inputs);
        }
        capture_stream.synchronize();

        prefix_graph_.capture_begin(pool);
        {
            std::vector<torch::jit::IValue> inputs = {
                pg_ids_, pg_pos_, pg_mask_, pg_buf_k_, pg_buf_v_, pg_ov_, pg_om_
            };
            auto output = model_.forward(inputs);
            auto elements = output.toTuple()->elements();
            pg_out_h_ = elements[0].toTensor();
            pg_out_pk_ = elements[1].toTensor();
            pg_out_pv_ = elements[2].toTensor();
        }
        prefix_graph_.capture_end();
    }

    graphs_captured_ = true;
    // Sync on default stream to ensure capture is fully done
    c10::cuda::getCurrentCUDAStream().synchronize();
}

// ============================================================================
// CUDA Graph: causal incremental
// ============================================================================

torch::Tensor TorchCausalBackbone::causalIncrementalGraph(
    int64_t token_id, int64_t position,
    float override_value, uint8_t override_flag)
{
    torch::NoGradGuard no_grad;

    // Update static input values
    cg_ids_[0][0] = token_id;
    cg_pos_[0][0] = position;

    if (override_flag)
    {
        cg_ov_.index_put_({0, 0}, static_cast<at::Half>(override_value));
        cg_om_.index_put_({0, 0}, true);
    }
    else
    {
        cg_ov_.index_put_({0, 0}, static_cast<at::Half>(0.0f));
        cg_om_.index_put_({0, 0}, false);
    }

    // Replay the captured graph
    // Mask state: positions [0..cg_len_-1] and [max_cache_len_] are unmasked
    causal_graph_.replay();

    // After cat(past_k[MAX_LEN], new_k[1]), the new token's KV is at position
    // max_cache_len_ (the last position in the output), NOT at cg_len_.
    // Copy it to position cg_len_ in the buffer for the next replay.
    using namespace torch::indexing;
    cg_buf_k_.index_put_({Slice(), Slice(), Slice(), cg_len_},
                          cg_out_pk_.index({Slice(), Slice(), Slice(), max_cache_len_}));
    cg_buf_v_.index_put_({Slice(), Slice(), Slice(), cg_len_},
                          cg_out_pv_.index({Slice(), Slice(), Slice(), max_cache_len_}));

    // Unmask position cg_len_ for the next call (now it has real KV data)
    cg_mask_.index_put_({0, 0, 0, cg_len_}, 0.0f);

    cg_len_++;
    cache_len_ = cg_len_;

    return cg_out_h_;  // [1, 1, E] FP16 on CUDA
}

void TorchCausalBackbone::setGraphInput(int64_t token_id)
{
    cg_ids_[0][0] = token_id;
    // Ensure override values are zero (for board tokens)
    cg_ov_.index_put_({0, 0}, static_cast<at::Half>(0.0f));
    cg_om_.index_put_({0, 0}, false);
}

torch::Tensor TorchCausalBackbone::causalBoardStep(
    const torch::Tensor& head_w_t,
    const torch::Tensor& head_b,
    const torch::Tensor& lut,
    int64_t position)
{
    torch::NoGradGuard no_grad;

    // Set position (async CPU→GPU write)
    cg_pos_[0][0] = position;

    // Replay captured graph (cg_ids_ already set by previous step or setGraphInput)
    causal_graph_.replay();

    // Copy new KV from max_cache_len_ position to cg_len_ (all GPU ops)
    using namespace torch::indexing;
    cg_buf_k_.index_put_({Slice(), Slice(), Slice(), cg_len_},
                          cg_out_pk_.index({Slice(), Slice(), Slice(), max_cache_len_}));
    cg_buf_v_.index_put_({Slice(), Slice(), Slice(), cg_len_},
                          cg_out_pv_.index({Slice(), Slice(), Slice(), max_cache_len_}));
    cg_mask_.index_put_({0, 0, 0, cg_len_}, 0.0f);
    cg_len_++;
    cache_len_ = cg_len_;

    // GPU board head eval: FP16 matmul + argmax + LUT lookup
    auto logits = torch::mm(cg_out_h_.view({1, embed_dim_}), head_w_t) + head_b;
    auto sub_idx = torch::argmax(logits, 1).squeeze();
    auto full_idx = lut.index({sub_idx});

    // Feed back to graph input for next step (GPU→GPU, no sync)
    cg_ids_.index_put_({0, 0}, full_idx);

    return full_idx;  // GPU scalar tensor (int64)
}

void TorchCausalBackbone::syncCausalCacheToGraph()
{
    using namespace torch::indexing;
    int len = cache_len_;
    if (len > 0 && len <= max_cache_len_)
    {
        cg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(past_keys_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
        cg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(past_values_.index({Slice(), Slice(), Slice(), Slice(0, len)}));

        // Reset mask: -1e9 everywhere, then unmask [0..len-1] and [MAX_LEN]
        // Position MAX_LEN is where cat() puts the new token's KV
        cg_mask_.fill_(-1e9f);
        cg_mask_.index_put_({0, 0, 0, Slice(0, len)}, 0.0f);
        cg_mask_.index_put_({0, 0, 0, max_cache_len_}, 0.0f);
    }
    cg_len_ = len;
}

void TorchCausalBackbone::syncGraphToCausalCache()
{
    using namespace torch::indexing;
    int len = cg_len_;
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    past_keys_ = cg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    past_values_ = cg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    cache_len_ = len;
}

// ============================================================================
// CUDA Graph: prefix incremental
// ============================================================================

torch::Tensor TorchCausalBackbone::prefixIncrementalGraph(
    int64_t token_id, int64_t position,
    float override_value, uint8_t override_flag)
{
    torch::NoGradGuard no_grad;

    pg_ids_[0][0] = token_id;
    pg_pos_[0][0] = position;

    if (override_flag)
    {
        pg_ov_.index_put_({0, 0}, static_cast<at::Half>(override_value));
        pg_om_.index_put_({0, 0}, true);
    }
    else
    {
        pg_ov_.index_put_({0, 0}, static_cast<at::Half>(0.0f));
        pg_om_.index_put_({0, 0}, false);
    }

    // Mask state: positions [0..pg_len_-1] and [max_cache_len_] are unmasked
    prefix_graph_.replay();

    // New token's KV is at position max_cache_len_ after cat()
    using namespace torch::indexing;
    pg_buf_k_.index_put_({Slice(), Slice(), Slice(), pg_len_},
                          pg_out_pk_.index({Slice(), Slice(), Slice(), max_cache_len_}));
    pg_buf_v_.index_put_({Slice(), Slice(), Slice(), pg_len_},
                          pg_out_pv_.index({Slice(), Slice(), Slice(), max_cache_len_}));

    // Unmask position pg_len_ for the next call
    pg_mask_.index_put_({0, 0, 0, pg_len_}, 0.0f);

    pg_len_++;
    prefix_cache_len_ = pg_len_;

    return pg_out_h_;  // [1, 1, E] FP16 on CUDA
}

void TorchCausalBackbone::syncPrefixCacheToGraph()
{
    using namespace torch::indexing;
    int len = prefix_cache_len_;
    if (len > 0 && len <= max_cache_len_)
    {
        pg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(prefix_past_keys_.index({Slice(), Slice(), Slice(), Slice(0, len)}));
        pg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)})
            .copy_(prefix_past_values_.index({Slice(), Slice(), Slice(), Slice(0, len)}));

        // Unmask [0..len-1] and [MAX_LEN] (new token always at MAX_LEN after cat)
        pg_mask_.fill_(-1e9f);
        pg_mask_.index_put_({0, 0, 0, Slice(0, len)}, 0.0f);
        pg_mask_.index_put_({0, 0, 0, max_cache_len_}, 0.0f);
    }
    pg_len_ = len;
}

void TorchCausalBackbone::syncGraphToPrefixCache()
{
    using namespace torch::indexing;
    int len = pg_len_;
    // Clone (not view!) — graph replay writes to pg_buf_k_, which would corrupt a view.
    prefix_past_keys_ = pg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    prefix_past_values_ = pg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
    prefix_cache_len_ = len;
}

void TorchCausalBackbone::syncPrefixCacheToGraphAfterBlock()
{
    // After a dynamic prefixBlockForward, the dynamic prefix cache grew.
    // Sync the new entries to the graph buffer.
    syncPrefixCacheToGraph();
}

// ============================================================================
// Causal mode (non-graph)
// ============================================================================

void TorchCausalBackbone::resetCache()
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    past_keys_ = torch::zeros({num_layers_, 1, num_heads_, 0, head_dim_}, opts);
    past_values_ = torch::zeros({num_layers_, 1, num_heads_, 0, head_dim_}, opts);
    cache_len_ = 0;

    // Also reset graph buffer
    if (graphs_captured_)
    {
        cg_buf_k_.zero_();
        cg_buf_v_.zero_();
        cg_mask_.fill_(-1e9f);
        // Always unmask position max_cache_len_ (where cat places new token)
        cg_mask_.index_put_({0, 0, 0, max_cache_len_}, 0.0f);
        cg_len_ = 0;
    }
}

void TorchCausalBackbone::forward(
    const int64_t* input_ids, const int64_t* input_pos,
    int seq_len, int past_len,
    const float* override_values, const uint8_t* override_flags,
    float* hidden_out)
{
    int total_len = past_len + seq_len;

    // Build causal attention mask on CPU
    auto mask = torch::full({1, 1, seq_len, total_len}, -1e9f, torch::kFloat32);
    auto mask_a = mask.accessor<float, 4>();
    for (int i = 0; i < seq_len; i++)
    {
        int end = past_len + i + 1;
        for (int j = 0; j < end; j++)
            mask_a[0][0][i][j] = 0.0f;
    }
    mask = mask.to(torch::kCUDA);

    forwardInternal(input_ids, input_pos, seq_len, mask,
                    past_keys_, past_values_,
                    override_values, override_flags,
                    hidden_out, /*update_cache=*/true);
}

void TorchCausalBackbone::forwardWithMask(
    const int64_t* input_ids, const int64_t* input_pos,
    int seq_len, int past_len,
    const float* attention_mask,
    const float* override_values, const uint8_t* override_flags,
    float* hidden_out)
{
    int total_len = past_len + seq_len;
    auto mask = torch::from_blob(
        const_cast<float*>(attention_mask),
        {1, 1, seq_len, total_len}, torch::kFloat32
    ).to(torch::kCUDA);

    forwardInternal(input_ids, input_pos, seq_len, mask,
                    past_keys_, past_values_,
                    override_values, override_flags,
                    hidden_out, /*update_cache=*/true);
}

void TorchCausalBackbone::forwardInternal(
    const int64_t* input_ids_ptr, const int64_t* input_pos_ptr,
    int seq_len, torch::Tensor mask,
    torch::Tensor past_k, torch::Tensor past_v,
    const float* override_values_ptr, const uint8_t* override_flags_ptr,
    float* hidden_out, bool update_cache)
{
    torch::NoGradGuard no_grad;

    auto ids = torch::from_blob(
        const_cast<int64_t*>(input_ids_ptr), {1, seq_len}, torch::kInt64
    ).to(torch::kCUDA);

    auto pos = torch::from_blob(
        const_cast<int64_t*>(input_pos_ptr), {1, seq_len}, torch::kInt64
    ).to(torch::kCUDA);

    auto ov = torch::from_blob(
        const_cast<float*>(override_values_ptr), {1, seq_len}, torch::kFloat32
    ).to(torch::kFloat16).to(torch::kCUDA);

    auto om = torch::from_blob(
        const_cast<uint8_t*>(override_flags_ptr), {1, seq_len}, torch::kByte
    ).to(torch::kBool).to(torch::kCUDA);

    std::vector<torch::jit::IValue> inputs = {
        ids, pos, mask, past_k, past_v, ov, om
    };

    auto output = model_.forward(inputs);
    auto elements = output.toTuple()->elements();

    auto hidden = elements[0].toTensor();
    auto present_k = elements[1].toTensor();
    auto present_v = elements[2].toTensor();

    if (update_cache)
    {
        past_keys_ = present_k;
        past_values_ = present_v;
        cache_len_ = static_cast<int>(present_k.size(3));
    }

    // Copy all hidden states to host: FP16 GPU -> FP32 CPU
    auto hidden_fp32 = hidden.to(torch::kFloat32).cpu().contiguous();
    std::memcpy(hidden_out, hidden_fp32.data_ptr<float>(),
                static_cast<size_t>(seq_len) * embed_dim_ * sizeof(float));
}

// ============================================================================
// Prefix mode (uncached)
// ============================================================================

void TorchCausalBackbone::forwardPrefix(
    const int64_t* input_ids, const int64_t* input_pos,
    int seq_len,
    const float* attention_mask,
    const float* override_values, const uint8_t* override_flags,
    float* hidden_out)
{
    auto mask = torch::from_blob(
        const_cast<float*>(attention_mask),
        {1, 1, seq_len, seq_len}, torch::kFloat32
    ).to(torch::kCUDA);

    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto empty_k = torch::zeros({num_layers_, 1, num_heads_, 0, head_dim_}, opts);
    auto empty_v = torch::zeros({num_layers_, 1, num_heads_, 0, head_dim_}, opts);

    forwardInternal(input_ids, input_pos, seq_len, mask,
                    empty_k, empty_v,
                    override_values, override_flags,
                    hidden_out, /*update_cache=*/false);
}

// ============================================================================
// Prefix KV cache mode (incremental, non-graph)
// ============================================================================

void TorchCausalBackbone::resetPrefixCache()
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    prefix_past_keys_ = torch::zeros({num_layers_, 1, num_heads_, 0, head_dim_}, opts);
    prefix_past_values_ = torch::zeros({num_layers_, 1, num_heads_, 0, head_dim_}, opts);
    prefix_cache_len_ = 0;

    if (graphs_captured_)
    {
        pg_buf_k_.zero_();
        pg_buf_v_.zero_();
        pg_mask_.fill_(-1e9f);
        // Always unmask position max_cache_len_ (where cat places new token)
        pg_mask_.index_put_({0, 0, 0, max_cache_len_}, 0.0f);
        pg_len_ = 0;
    }
}

void TorchCausalBackbone::prefixInit(
    const int64_t* input_ids, const int64_t* input_pos,
    int seq_len,
    const float* attention_mask,
    const float* override_values, const uint8_t* override_flags,
    int extract_pos, float* hidden_out)
{
    auto mask = torch::from_blob(
        const_cast<float*>(attention_mask),
        {1, 1, seq_len, seq_len}, torch::kFloat32
    ).to(torch::kCUDA);

    prefixForwardInternal(input_ids, input_pos, seq_len, mask,
                          override_values, override_flags,
                          extract_pos, hidden_out);
}

void TorchCausalBackbone::prefixIncremental(
    int64_t token_id, int64_t position,
    float override_value, uint8_t override_flag,
    float* hidden_out)
{
    int total_len = prefix_cache_len_ + 1;

    // All-zeros mask: 1 token attending to everything (causal for orphan tokens)
    auto mask = torch::zeros({1, 1, 1, total_len},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    prefixForwardInternal(&token_id, &position, 1, mask,
                          &override_value, &override_flag,
                          0, hidden_out);
}

void TorchCausalBackbone::prefixBlockForward(
    const int64_t* input_ids, const int64_t* input_pos,
    int seq_len,
    const float* override_values, const uint8_t* override_flags,
    int extract_pos, float* hidden_out)
{
    int total_len = prefix_cache_len_ + seq_len;

    // All-zeros mask: all tokens attend to everything
    // (bidirectional within block + causal to past)
    auto mask = torch::zeros({1, 1, seq_len, total_len},
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    prefixForwardInternal(input_ids, input_pos, seq_len, mask,
                          override_values, override_flags,
                          extract_pos, hidden_out);
}

void TorchCausalBackbone::prefixForwardInternal(
    const int64_t* input_ids_ptr, const int64_t* input_pos_ptr,
    int seq_len, torch::Tensor mask,
    const float* override_values_ptr, const uint8_t* override_flags_ptr,
    int extract_pos, float* hidden_out)
{
    torch::NoGradGuard no_grad;

    auto ids = torch::from_blob(
        const_cast<int64_t*>(input_ids_ptr), {1, seq_len}, torch::kInt64
    ).to(torch::kCUDA);

    auto pos = torch::from_blob(
        const_cast<int64_t*>(input_pos_ptr), {1, seq_len}, torch::kInt64
    ).to(torch::kCUDA);

    auto ov = torch::from_blob(
        const_cast<float*>(override_values_ptr), {1, seq_len}, torch::kFloat32
    ).to(torch::kFloat16).to(torch::kCUDA);

    auto om = torch::from_blob(
        const_cast<uint8_t*>(override_flags_ptr), {1, seq_len}, torch::kByte
    ).to(torch::kBool).to(torch::kCUDA);

    std::vector<torch::jit::IValue> inputs = {
        ids, pos, mask, prefix_past_keys_, prefix_past_values_, ov, om
    };

    auto output = model_.forward(inputs);
    auto elements = output.toTuple()->elements();

    auto hidden = elements[0].toTensor();    // [1, seq_len, E] FP16
    auto present_k = elements[1].toTensor(); // [NL, 1, NH, total, HD] FP16
    auto present_v = elements[2].toTensor();

    // Update prefix KV cache
    prefix_past_keys_ = present_k;
    prefix_past_values_ = present_v;
    prefix_cache_len_ = static_cast<int>(present_k.size(3));

    // Extract hidden state at the requested position only (saves CPU<->GPU bandwidth)
    auto h_pos = hidden.index({0, extract_pos}).to(torch::kFloat32).cpu().contiguous();
    std::memcpy(hidden_out, h_pos.data_ptr<float>(), embed_dim_ * sizeof(float));
}

} // namespace decoder

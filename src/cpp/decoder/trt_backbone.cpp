#include "trt_backbone.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace decoder
{

namespace
{

nvinfer1::Dims makeDims5(int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 5;
    dims.d[0] = d0;
    dims.d[1] = d1;
    dims.d[2] = d2;
    dims.d[3] = d3;
    dims.d[4] = d4;
    return dims;
}

} // anonymous namespace

// ============= CudaStream =============

CudaStream::CudaStream()
{
    if (cudaStreamCreate(&stream_) != cudaSuccess)
        throw std::runtime_error("Failed to create CUDA stream");
}

CudaStream::~CudaStream()
{
    if (stream_) cudaStreamDestroy(stream_);
}

void CudaStream::sync() const
{
    cudaStreamSynchronize(stream_);
}

// ============= GpuBuffer =============

GpuBuffer::GpuBuffer(size_t bytes) : bytes_(bytes)
{
    if (bytes > 0)
    {
        if (cudaMalloc(&ptr_, bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed for " + std::to_string(bytes) + " bytes");
    }
}

GpuBuffer::~GpuBuffer()
{
    if (ptr_) cudaFree(ptr_);
}

GpuBuffer::GpuBuffer(GpuBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_)
{
    o.ptr_ = nullptr;
    o.bytes_ = 0;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& o) noexcept
{
    if (this != &o)
    {
        if (ptr_) cudaFree(ptr_);
        ptr_ = o.ptr_;
        bytes_ = o.bytes_;
        o.ptr_ = nullptr;
        o.bytes_ = 0;
    }
    return *this;
}

void GpuBuffer::copyFromHost(const void* src, size_t bytes, cudaStream_t stream)
{
    if (stream)
        cudaMemcpyAsync(ptr_, src, bytes, cudaMemcpyHostToDevice, stream);
    else
        cudaMemcpy(ptr_, src, bytes, cudaMemcpyHostToDevice);
}

void GpuBuffer::copyToHost(void* dst, size_t bytes, cudaStream_t stream) const
{
    if (stream)
        cudaMemcpyAsync(dst, ptr_, bytes, cudaMemcpyDeviceToHost, stream);
    else
        cudaMemcpy(dst, ptr_, bytes, cudaMemcpyDeviceToHost);
}

void GpuBuffer::zero(cudaStream_t stream)
{
    if (stream)
        cudaMemsetAsync(ptr_, 0, bytes_, stream);
    else
        cudaMemset(ptr_, 0, bytes_);
}

// ============= TrtEngine =============

void TrtEngine::Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cerr << "[TensorRT] " << msg << std::endl;
    }
}

TrtEngine::TrtEngine(const std::string& trt_path)
{
    std::ifstream file(trt_path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open TRT engine: " + trt_path);

    file.seekg(0, std::ios::end);
    size_t fsize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(fsize);
    file.read(data.data(), fsize);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");

    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) throw std::runtime_error("Failed to deserialize TRT engine: " + trt_path);
}

nvinfer1::IExecutionContext* TrtEngine::createContext()
{
    auto* ctx = engine_->createExecutionContext();
    if (!ctx) throw std::runtime_error("Failed to create TRT execution context");
    return ctx;
}

// ============= CausalBackbone =============

CausalBackbone::CausalBackbone(const std::string& trt_path, int num_layers, int num_heads,
                               int head_dim, int embed_dim, int max_cache_len)
    : engine_(trt_path)
    , num_layers_(num_layers), num_heads_(num_heads)
    , head_dim_(head_dim), embed_dim_(embed_dim)
    , max_cache_len_(max_cache_len)
{
    context_.reset(engine_.createContext());

    // Pre-allocate GPU buffers for max sizes
    size_t max_seq = max_cache_len;
    d_input_ids_ = GpuBuffer(max_seq * sizeof(int64_t));
    d_input_pos_ = GpuBuffer(max_seq * sizeof(int64_t));
    d_attention_mask_ = GpuBuffer(max_seq * max_seq * sizeof(float));
    d_override_values_ = GpuBuffer(max_seq * sizeof(float));
    d_override_mask_ = GpuBuffer(max_seq * sizeof(bool));
    d_hidden_states_ = GpuBuffer(max_seq * embed_dim * sizeof(float));

    // KV cache: [NL, 1, NH, max_cache_len, HD]
    size_t kv_size = (size_t)num_layers * num_heads * max_cache_len * head_dim * sizeof(float);
    d_past_keys_ = GpuBuffer(kv_size);
    d_past_values_ = GpuBuffer(kv_size);
    d_present_keys_ = GpuBuffer(kv_size);
    d_present_values_ = GpuBuffer(kv_size);

    resetCache();
}

void CausalBackbone::resetCache()
{
    cache_len_ = 0;
    d_past_keys_.zero();
    d_past_values_.zero();
}

void CausalBackbone::loadExternalKV(const void* src_keys_gpu, const void* src_values_gpu,
                                    int seq_len, cudaStream_t src_stream)
{
    // Wait for source stream to finish writing
    if (src_stream)
        cudaStreamSynchronize(src_stream);

    // Copy K/V from prefix engine's GPU buffers to our past_keys/values
    // Shape: [NL, 1, NH, seq_len, HD]
    size_t kv_bytes = (size_t)num_layers_ * num_heads_ * seq_len * head_dim_ * sizeof(float);
    cudaMemcpyAsync(d_past_keys_.data(), src_keys_gpu, kv_bytes,
                    cudaMemcpyDeviceToDevice, stream_.get());
    cudaMemcpyAsync(d_past_values_.data(), src_values_gpu, kv_bytes,
                    cudaMemcpyDeviceToDevice, stream_.get());
    stream_.sync();
    cache_len_ = seq_len;
}

void CausalBackbone::buildAttentionMask(int seq_len, int past_len)
{
    // Build causal mask: [1, 1, seq_len, past_len + seq_len]
    int total_len = past_len + seq_len;
    std::vector<float> mask(seq_len * total_len);

    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < total_len; j++)
        {
            // Position i can attend to all past (j < past_len)
            // and to positions up to i in the new tokens (j < past_len + i + 1)
            if (j < past_len + i + 1)
                mask[i * total_len + j] = 0.0f;
            else
                mask[i * total_len + j] = -1e9f;
        }
    }

    d_attention_mask_.copyFromHost(mask.data(), mask.size() * sizeof(float), stream_.get());
}

void CausalBackbone::forward(const int64_t* input_ids, const int64_t* input_pos,
                             int seq_len, int past_len,
                             const float* override_values, const uint8_t* override_flags,
                             float* hidden_out)
{
    buildAttentionMask(seq_len, past_len);
    forwardInternal(input_ids, input_pos, seq_len, past_len,
                    override_values, override_flags, hidden_out);
}

void CausalBackbone::forwardWithMask(const int64_t* input_ids, const int64_t* input_pos,
                                     int seq_len, int past_len,
                                     const float* attention_mask,
                                     const float* override_values, const uint8_t* override_flags,
                                     float* hidden_out)
{
    int total_len = past_len + seq_len;
    d_attention_mask_.copyFromHost(attention_mask,
                                  (size_t)seq_len * total_len * sizeof(float),
                                  stream_.get());
    forwardInternal(input_ids, input_pos, seq_len, past_len,
                    override_values, override_flags, hidden_out);
}

void CausalBackbone::forwardInternal(const int64_t* input_ids, const int64_t* input_pos,
                                     int seq_len, int past_len,
                                     const float* override_values, const uint8_t* override_flags,
                                     float* hidden_out)
{
    auto* ctx = context_.get();
    int total_len = past_len + seq_len;

    // Set input shapes
    ctx->setInputShape("input_ids", nvinfer1::Dims2{1, seq_len});
    ctx->setInputShape("input_pos", nvinfer1::Dims2{1, seq_len});
    ctx->setInputShape("attention_mask", nvinfer1::Dims4{1, 1, seq_len, total_len});
    ctx->setInputShape("past_keys", makeDims5(num_layers_, 1, num_heads_, past_len, head_dim_));
    ctx->setInputShape("past_values", makeDims5(num_layers_, 1, num_heads_, past_len, head_dim_));
    ctx->setInputShape("override_values", nvinfer1::Dims2{1, seq_len});
    ctx->setInputShape("override_mask", nvinfer1::Dims2{1, seq_len});

    // Copy inputs to GPU
    d_input_ids_.copyFromHost(input_ids, seq_len * sizeof(int64_t), stream_.get());
    d_input_pos_.copyFromHost(input_pos, seq_len * sizeof(int64_t), stream_.get());
    d_override_values_.copyFromHost(override_values, seq_len * sizeof(float), stream_.get());
    d_override_mask_.copyFromHost(override_flags, seq_len * sizeof(uint8_t), stream_.get());

    // Bind tensors
    ctx->setTensorAddress("input_ids", d_input_ids_.data());
    ctx->setTensorAddress("input_pos", d_input_pos_.data());
    ctx->setTensorAddress("attention_mask", d_attention_mask_.data());
    ctx->setTensorAddress("past_keys", d_past_keys_.data());
    ctx->setTensorAddress("past_values", d_past_values_.data());
    ctx->setTensorAddress("override_values", d_override_values_.data());
    ctx->setTensorAddress("override_mask", d_override_mask_.data());
    ctx->setTensorAddress("hidden_states", d_hidden_states_.data());
    ctx->setTensorAddress("present_keys", d_present_keys_.data());
    ctx->setTensorAddress("present_values", d_present_values_.data());

    // Execute
    if (!ctx->enqueueV3(stream_.get()))
        throw std::runtime_error("CausalBackbone TRT execution failed");

    // Copy hidden states back
    d_hidden_states_.copyToHost(hidden_out, (size_t)seq_len * embed_dim_ * sizeof(float), stream_.get());

    stream_.sync();

    // Swap present/past KV buffers (pointer swap, no data copy)
    std::swap(d_past_keys_, d_present_keys_);
    std::swap(d_past_values_, d_present_values_);

    cache_len_ = total_len;
}

// ============= PrefixBackbone =============

PrefixBackbone::PrefixBackbone(const std::string& trt_path, int num_heads,
                               int head_dim, int embed_dim, int max_seq_len)
    : engine_(trt_path)
    , num_heads_(num_heads), head_dim_(head_dim)
    , embed_dim_(embed_dim), max_seq_len_(max_seq_len)
{
    context_.reset(engine_.createContext());

    d_input_ids_ = GpuBuffer(max_seq_len * sizeof(int64_t));
    d_input_pos_ = GpuBuffer(max_seq_len * sizeof(int64_t));
    d_attention_mask_ = GpuBuffer((size_t)max_seq_len * max_seq_len * sizeof(float));
    d_override_values_ = GpuBuffer(max_seq_len * sizeof(float));
    d_override_mask_ = GpuBuffer(max_seq_len * sizeof(bool));
    d_hidden_states_ = GpuBuffer((size_t)max_seq_len * embed_dim * sizeof(float));
}

void PrefixBackbone::forward(const int64_t* input_ids, const int64_t* input_pos,
                             int seq_len,
                             const float* attention_mask,
                             const float* override_values, const uint8_t* override_flags,
                             float* hidden_out)
{
    auto* ctx = context_.get();

    ctx->setInputShape("input_ids", nvinfer1::Dims2{1, seq_len});
    ctx->setInputShape("input_pos", nvinfer1::Dims2{1, seq_len});
    ctx->setInputShape("attention_mask", nvinfer1::Dims4{1, 1, seq_len, seq_len});
    ctx->setInputShape("override_values", nvinfer1::Dims2{1, seq_len});
    ctx->setInputShape("override_mask", nvinfer1::Dims2{1, seq_len});

    d_input_ids_.copyFromHost(input_ids, seq_len * sizeof(int64_t), stream_.get());
    d_input_pos_.copyFromHost(input_pos, seq_len * sizeof(int64_t), stream_.get());
    d_attention_mask_.copyFromHost(attention_mask, (size_t)seq_len * seq_len * sizeof(float), stream_.get());
    d_override_values_.copyFromHost(override_values, seq_len * sizeof(float), stream_.get());
    d_override_mask_.copyFromHost(override_flags, seq_len * sizeof(uint8_t), stream_.get());

    ctx->setTensorAddress("input_ids", d_input_ids_.data());
    ctx->setTensorAddress("input_pos", d_input_pos_.data());
    ctx->setTensorAddress("attention_mask", d_attention_mask_.data());
    ctx->setTensorAddress("override_values", d_override_values_.data());
    ctx->setTensorAddress("override_mask", d_override_mask_.data());
    ctx->setTensorAddress("hidden_states", d_hidden_states_.data());

    if (!ctx->enqueueV3(stream_.get()))
        throw std::runtime_error("PrefixBackbone TRT execution failed");

    d_hidden_states_.copyToHost(hidden_out, (size_t)seq_len * embed_dim_ * sizeof(float), stream_.get());

    stream_.sync();
}

} // namespace decoder

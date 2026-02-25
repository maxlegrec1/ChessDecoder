#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <memory>
#include <string>
#include <vector>

namespace decoder
{

/// RAII CUDA stream
class CudaStream
{
public:
    CudaStream();
    ~CudaStream();
    cudaStream_t get() const noexcept { return stream_; }
    void sync() const;
private:
    cudaStream_t stream_{};
};

/// RAII GPU buffer
class GpuBuffer
{
public:
    GpuBuffer() = default;
    explicit GpuBuffer(size_t bytes);
    ~GpuBuffer();
    GpuBuffer(GpuBuffer&& o) noexcept;
    GpuBuffer& operator=(GpuBuffer&& o) noexcept;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    void* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return bytes_; }

    void copyFromHost(const void* src, size_t bytes, cudaStream_t stream = nullptr);
    void copyToHost(void* dst, size_t bytes, cudaStream_t stream = nullptr) const;
    void zero(cudaStream_t stream = nullptr);
private:
    void* ptr_{nullptr};
    size_t bytes_{0};
};

/// TRT engine wrapper for a single .trt file.
class TrtEngine
{
public:
    explicit TrtEngine(const std::string& trt_path);

    nvinfer1::ICudaEngine* engine() { return engine_.get(); }
    nvinfer1::IExecutionContext* createContext();

private:
    class Logger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
};

/// Wraps causal backbone TRT engine with KV cache management.
class CausalBackbone
{
public:
    CausalBackbone(const std::string& trt_path, int num_layers, int num_heads,
                   int head_dim, int embed_dim, int max_cache_len);

    /// Run causal forward pass with auto-generated causal attention mask.
    /// override_values: scalar float per position (Fourier-encoded inside TRT)
    void forward(const int64_t* input_ids, const int64_t* input_pos,
                 int seq_len, int past_len,
                 const float* override_values, const uint8_t* override_flags,
                 float* hidden_out);

    /// Run forward pass with an externally provided attention mask.
    /// attention_mask: host float array [seq_len * (past_len + seq_len)], 0 or -1e9.
    /// override_values: scalar float per position (Fourier-encoded inside TRT)
    void forwardWithMask(const int64_t* input_ids, const int64_t* input_pos,
                         int seq_len, int past_len,
                         const float* attention_mask,
                         const float* override_values, const uint8_t* override_flags,
                         float* hidden_out);

    /// Reset KV cache (call before new sequence).
    void resetCache();

    /// Get current cache length (number of valid positions).
    int cacheLen() const { return cache_len_; }

    /// Load external K/V from GPU into the past KV buffers.
    /// Used to inject prefix K/V for incremental orphan token inference.
    /// src_keys/src_values: GPU pointers, shape [NL, 1, NH, seq_len, HD]
    void loadExternalKV(const void* src_keys_gpu, const void* src_values_gpu,
                        int seq_len, cudaStream_t src_stream);

    CudaStream& stream() { return stream_; }

private:
    void buildAttentionMask(int seq_len, int past_len);
    void forwardInternal(const int64_t* input_ids, const int64_t* input_pos,
                         int seq_len, int past_len,
                         const float* override_values, const uint8_t* override_flags,
                         float* hidden_out);

    TrtEngine engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    CudaStream stream_;

    int num_layers_, num_heads_, head_dim_, embed_dim_;
    int max_cache_len_;
    int cache_len_{0};

    // GPU buffers
    GpuBuffer d_input_ids_;
    GpuBuffer d_input_pos_;
    GpuBuffer d_attention_mask_;
    GpuBuffer d_past_keys_;
    GpuBuffer d_past_values_;
    GpuBuffer d_present_keys_;
    GpuBuffer d_present_values_;
    GpuBuffer d_override_values_;
    GpuBuffer d_override_mask_;
    GpuBuffer d_hidden_states_;
};

/// Wraps prefix backbone TRT engine (hidden states output only).
class PrefixBackbone
{
public:
    PrefixBackbone(const std::string& trt_path, int num_heads,
                   int head_dim, int embed_dim, int max_seq_len);

    /// Run prefix forward pass.
    /// attention_mask: host float array of length seq_len * seq_len (0 or -1e9)
    /// hidden_out: host float array of length seq_len * embed_dim (output)
    /// override_values: scalar float per position (Fourier-encoded inside TRT)
    void forward(const int64_t* input_ids, const int64_t* input_pos,
                 int seq_len,
                 const float* attention_mask,
                 const float* override_values, const uint8_t* override_flags,
                 float* hidden_out);

    CudaStream& stream() { return stream_; }

private:
    TrtEngine engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    CudaStream stream_;

    int num_heads_, head_dim_;
    int embed_dim_, max_seq_len_;

    GpuBuffer d_input_ids_;
    GpuBuffer d_input_pos_;
    GpuBuffer d_attention_mask_;
    GpuBuffer d_override_values_;
    GpuBuffer d_override_mask_;
    GpuBuffer d_hidden_states_;
};

} // namespace decoder

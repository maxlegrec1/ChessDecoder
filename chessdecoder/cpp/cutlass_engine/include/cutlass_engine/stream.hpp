#pragma once

#include <cuda_runtime.h>

#include "cutlass_engine/check.hpp"

namespace cutlass_engine {

// RAII wrapper around a non-blocking CUDA stream. The engine owns one of these
// for the lifetime of inference; all kernels are launched on it.
class Stream {
public:
    Stream() {
        CE_CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
    }

    ~Stream() {
        if (s_) {
            cudaStreamDestroy(s_);
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t get() const { return s_; }
    operator cudaStream_t() const { return s_; }  // NOLINT — implicit on purpose

    void sync() const { CE_CUDA_CHECK(cudaStreamSynchronize(s_)); }

private:
    cudaStream_t s_{nullptr};
};

}  // namespace cutlass_engine

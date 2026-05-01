#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cutlass_engine {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "CUDA error at %s:%d — %s (%d)",
                      file, line, cudaGetErrorString(err), int(err));
        throw std::runtime_error(buf);
    }
}

}  // namespace cutlass_engine

#define CE_CUDA_CHECK(expr) ::cutlass_engine::cuda_check((expr), __FILE__, __LINE__)
#define CE_CUDA_LAST() ::cutlass_engine::cuda_check(cudaGetLastError(), __FILE__, __LINE__)

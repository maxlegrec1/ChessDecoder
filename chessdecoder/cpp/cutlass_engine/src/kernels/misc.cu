#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>

namespace cutlass_engine {

namespace {

__global__ void gather_bucket_kernel(const float* __restrict__ centers,
                                     const int32_t* __restrict__ idx,
                                     float* __restrict__ out, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    out[b] = centers[idx[b]];
}

__global__ void cast_fp32_fp16_kernel(const float* __restrict__ in,
                                      __half* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2half_rn(in[i]);
}

}  // namespace

void gather_bucket_center(const float* centers, const int32_t* idx,
                          float* out, int B, cudaStream_t stream) {
    int blocks = (B + 63) / 64;
    gather_bucket_kernel<<<blocks, 64, 0, stream>>>(centers, idx, out, B);
    CE_CUDA_LAST();
}

void cast_fp32_to_fp16(const float* in, __half* out, int N, cudaStream_t stream) {
    int blocks = (N + 255) / 256;
    cast_fp32_fp16_kernel<<<blocks, 256, 0, stream>>>(in, out, N);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

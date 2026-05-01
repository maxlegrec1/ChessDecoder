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

// For each slot b: if active[b]==0, restore dst[b, :] from backup[b, :].
// Used after a batched forward to preserve inactive slots' hidden state.
__global__ void restore_inactive_kernel(__half* __restrict__ dst,
                                        const __half* __restrict__ backup,
                                        const int32_t* __restrict__ active,
                                        int B, int E) {
    int b = blockIdx.y;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || e >= E) return;
    if (active[b] == 0) {
        dst[b * E + e] = backup[b * E + e];
    }
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

void restore_inactive_last_h(__half* dst, const __half* backup,
                             const int32_t* active, int B, int E,
                             cudaStream_t stream) {
    constexpr int TX = 128;
    dim3 block(TX);
    dim3 grid((E + TX - 1) / TX, B);
    restore_inactive_kernel<<<grid, block, 0, stream>>>(dst, backup, active, B, E);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

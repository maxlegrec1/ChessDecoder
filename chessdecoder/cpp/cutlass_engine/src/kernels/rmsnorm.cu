#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

namespace cutlass_engine {

namespace {

// Block-wide reduction sum using shared memory + warp shuffles.
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[BLOCK / 32];

    // warp reduce
    unsigned mask = 0xFFFFFFFFu;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(mask, v, off);
    }
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    // first warp reduces across warps
    if (wid == 0) {
        float w = (threadIdx.x < (BLOCK / 32)) ? warp_sums[threadIdx.x] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            w += __shfl_xor_sync(mask, w, off);
        }
        if (lane == 0) warp_sums[0] = w;
    }
    __syncthreads();
    return warp_sums[0];
}

template <int BLOCK>
__global__ void rmsnorm_kernel(const __half* __restrict__ x,
                               const __half* __restrict__ w,
                               __half* __restrict__ y,
                               int E, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* x_row = x + row * E;
    __half* y_row = y + row * E;

    // 1. squared sum
    float ss = 0.0f;
    for (int e = tid; e < E; e += BLOCK) {
        float v = __half2float(x_row[e]);
        ss += v * v;
    }
    ss = block_reduce_sum<BLOCK>(ss);

    float scale = rsqrtf(ss / float(E) + eps);

    // 2. write y = x * scale * w
    for (int e = tid; e < E; e += BLOCK) {
        float v = __half2float(x_row[e]) * scale;
        v *= __half2float(w[e]);
        y_row[e] = __float2half_rn(v);
    }
}

template <int BLOCK>
__global__ void rmsnorm_residual_kernel(const __half* __restrict__ x,
                                        const __half* __restrict__ residual,
                                        const __half* __restrict__ w,
                                        __half* __restrict__ y,
                                        __half* __restrict__ out_residual,
                                        int E, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* x_row = x + row * E;
    const __half* r_row = residual + row * E;
    __half* y_row = y + row * E;
    __half* out_r_row = out_residual + row * E;

    // 1. compute sum = x + residual, store in out_residual, accumulate ss
    float ss = 0.0f;
    for (int e = tid; e < E; e += BLOCK) {
        float v = __half2float(x_row[e]) + __half2float(r_row[e]);
        out_r_row[e] = __float2half_rn(v);
        ss += v * v;
    }
    ss = block_reduce_sum<BLOCK>(ss);

    float scale = rsqrtf(ss / float(E) + eps);

    for (int e = tid; e < E; e += BLOCK) {
        float v = __half2float(out_r_row[e]) * scale * __half2float(w[e]);
        y_row[e] = __float2half_rn(v);
    }
}

}  // namespace

void rmsnorm_fp16(const __half* x, const __half* w, __half* y,
                  int M, int E, float eps, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    rmsnorm_kernel<BLOCK><<<M, BLOCK, 0, stream>>>(x, w, y, E, eps);
    CE_CUDA_LAST();
}

void rmsnorm_residual_fp16(const __half* x, const __half* residual,
                           const __half* w, __half* y, __half* out_residual,
                           int M, int E, float eps, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    rmsnorm_residual_kernel<BLOCK><<<M, BLOCK, 0, stream>>>(
        x, residual, w, y, out_residual, E, eps);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

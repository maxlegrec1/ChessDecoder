#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>

namespace cutlass_engine {

namespace {

__device__ __forceinline__ float silu(float x) {
    // x * sigmoid(x)
    return x / (1.0f + __expf(-x));
}

__global__ void swiglu_kernel(const __half* __restrict__ gate_up,
                              __half* __restrict__ y,
                              int M, int d_ff) {
    // gate_up is [M, 2*d_ff] = [M, gate(:d_ff) | up(d_ff:2*d_ff)]
    // PyTorch's torchtune FeedForward order is: gate_proj output then up_proj
    // output, fused side-by-side. But our fused gate_up_w stacks with gate
    // weights first, so output[:, :d_ff] = gate, output[:, d_ff:] = up.
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d_ff) return;

    const __half* row_ptr = gate_up + row * (2 * d_ff);
    float g = __half2float(row_ptr[col]);
    float u = __half2float(row_ptr[d_ff + col]);
    float v = silu(g) * u;
    y[row * d_ff + col] = __float2half_rn(v);
}

__global__ void mish_kernel(__half* __restrict__ x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = __half2float(x[idx]);
    float sp = log1pf(__expf(v));    // softplus
    float t = tanhf(sp);
    x[idx] = __float2half_rn(v * t);
}

}  // namespace

void swiglu_fp16(const __half* gate_up, __half* y, int M, int d_ff,
                 cudaStream_t stream) {
    constexpr int TX = 256;
    dim3 block(TX);
    dim3 grid(M, (d_ff + TX - 1) / TX);
    swiglu_kernel<<<grid, block, 0, stream>>>(gate_up, y, M, d_ff);
    CE_CUDA_LAST();
}

void mish_inplace_fp16(__half* x, int N, cudaStream_t stream) {
    constexpr int TX = 256;
    int blocks = (N + TX - 1) / TX;
    mish_kernel<<<blocks, TX, 0, stream>>>(x, N);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

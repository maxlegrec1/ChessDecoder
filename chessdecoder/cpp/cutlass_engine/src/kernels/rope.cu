#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>

namespace cutlass_engine {

namespace {

// Apply RoPE in-place to one (token, head) pair. HD/2 threads cooperate.
//
// Convention: matches torchtune RotaryPositionalEmbeddings, which uses
// "interleaved" pairing — element 2j and 2j+1 are the cos/sin pair.
// (This is the same convention HuggingFace's "interleaved" RoPE uses; some
// LLaMA impls split the dim in halves instead. We chose interleaved since
// torchtune uses it.)
__global__ void rope_kernel(__half* __restrict__ Q, __half* __restrict__ K,
                            const int32_t* __restrict__ pos,
                            const float* __restrict__ cos_table,
                            const float* __restrict__ sin_table,
                            int M, int num_heads, int head_dim,
                            int rope_max_seq) {
    const int m = blockIdx.x;       // token index, m ∈ [0, M)
    const int h = blockIdx.y;       // head index
    const int j = threadIdx.x;      // pair index in [0, HD/2)
    const int half = head_dim / 2;
    if (j >= half) return;

    const int p = pos[m];
    if (p < 0 || p >= rope_max_seq) return;

    float c = cos_table[p * half + j];
    float s = sin_table[p * half + j];

    const int base = m * (num_heads * head_dim) + h * head_dim + 2 * j;

    // Q
    {
        float q0 = __half2float(Q[base + 0]);
        float q1 = __half2float(Q[base + 1]);
        Q[base + 0] = __float2half_rn(q0 * c - q1 * s);
        Q[base + 1] = __float2half_rn(q0 * s + q1 * c);
    }
    // K
    {
        float k0 = __half2float(K[base + 0]);
        float k1 = __half2float(K[base + 1]);
        K[base + 0] = __float2half_rn(k0 * c - k1 * s);
        K[base + 1] = __float2half_rn(k0 * s + k1 * c);
    }
}

}  // namespace

void rope_apply_qk_fp16(__half* Q, __half* K, const int32_t* pos,
                        const float* cos_t, const float* sin_t,
                        int M, int num_heads, int head_dim, int rope_max_seq,
                        cudaStream_t stream) {
    const int half = head_dim / 2;
    dim3 block(half);
    dim3 grid(M, num_heads);
    rope_kernel<<<grid, block, 0, stream>>>(Q, K, pos, cos_t, sin_t, M,
                                             num_heads, head_dim, rope_max_seq);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

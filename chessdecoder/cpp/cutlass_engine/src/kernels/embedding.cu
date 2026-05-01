#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>
#include <math_constants.h>

namespace cutlass_engine {

namespace {

constexpr float TWO_PI = 6.28318530717958647692f;

// Per-token gather of tok_embedding[input_ids[m], :] into out[m, :].
__global__ void gather_token_embedding(const int32_t* __restrict__ ids,
                                       const __half* __restrict__ tok_emb,
                                       __half* __restrict__ out,
                                       int M, int E) {
    int m = blockIdx.x;
    int e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int id = ids[m];
    out[m * E + e] = tok_emb[id * E + e];
}

// Override embedding rows with the Fourier MLP output where wl_pos OR d_pos.
// One block per "marked" row (we run a tiny prefix-sum-of-flags step before to
// build a compact list of marked rows; here we do it the simple way: branchy
// per row).
//
// Each block handles one row; threadIdx.x covers E and 2*F separately.
__global__ void fourier_override(const bool* __restrict__ wl_pos,
                                 const bool* __restrict__ d_pos,
                                 const __half* __restrict__ wl_val,
                                 const __half* __restrict__ d_val,
                                 const __half* __restrict__ freq,    // [F]
                                 const __half* __restrict__ proj_w,  // [E, 2*F]
                                 const __half* __restrict__ proj_b,  // [E]
                                 __half* __restrict__ out,
                                 int M, int E, int F) {
    int m = blockIdx.x;
    bool is_wl = (wl_pos != nullptr) && wl_pos[m];
    bool is_d  = (d_pos  != nullptr) && d_pos[m];
    if (!is_wl && !is_d) return;

    const int tid = threadIdx.x;

    // Determine x value.
    float x = is_wl ? __half2float(wl_val[m]) : __half2float(d_val[m]);

    // Compute features f = [cos(2π x ω_j), sin(2π x ω_j)] in shared mem.
    extern __shared__ float feats[];  // size 2*F
    for (int j = tid; j < F; j += blockDim.x) {
        float omega = __half2float(freq[j]);
        float a = TWO_PI * x * omega;
        feats[j] = __cosf(a);
        feats[F + j] = __sinf(a);
    }
    __syncthreads();

    // Compute out[m, e] = sum_k proj_w[e, k] * feats[k] + proj_b[e].
    // proj_w stored row-major [E, 2*F], so weight for output e is at proj_w + e*2F.
    for (int e = tid; e < E; e += blockDim.x) {
        const __half* wrow = proj_w + e * (2 * F);
        float acc = __half2float(proj_b[e]);
#pragma unroll 8
        for (int k = 0; k < 2 * F; ++k) {
            acc += __half2float(wrow[k]) * feats[k];
        }
        out[m * E + e] = __float2half_rn(acc);
    }
}

}  // namespace

void embed_with_fourier_fp16(const int32_t* ids, const __half* tok_emb,
                             const bool* wl_pos, const bool* d_pos,
                             const __half* wl_val, const __half* d_val,
                             const __half* fr, const __half* pw, const __half* pb,
                             __half* out, int M, int E, int /*V*/, int F,
                             cudaStream_t stream) {
    // 1. base embedding gather
    constexpr int TX = 256;
    dim3 block1(TX);
    dim3 grid1(M, (E + TX - 1) / TX);
    gather_token_embedding<<<grid1, block1, 0, stream>>>(ids, tok_emb, out, M, E);
    CE_CUDA_LAST();

    // 2. fourier override at marked rows
    if (wl_pos != nullptr || d_pos != nullptr) {
        const int TX2 = 128;
        size_t shmem = sizeof(float) * 2 * F;
        fourier_override<<<M, TX2, shmem, stream>>>(
            wl_pos, d_pos, wl_val, d_val, fr, pw, pb, out, M, E, F);
        CE_CUDA_LAST();
    }
}

}  // namespace cutlass_engine

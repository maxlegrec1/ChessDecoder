#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>

namespace cutlass_engine {

// --------------------------------------------------------------------------
// Block-aware prefill self-attention.
// One block per (b, h). HD threads. S is small (≤ ~256 in practice — 71 for
// the FEN init block plus a 68-token board block prefill). We just do an
// O(S^2) algorithm per (b, h), which at B=64 NH=16 S=71 is ~5M ops per
// layer per call — totally fine for a once-per-FEN code path.
//
// The mask:
//   mask[i, j] = (block_id[b, i] == block_id[b, j]) || (j <= i)
// (causal between blocks; full attention within block — same as PyTorch
//  prefix mode in the reference model.)
// --------------------------------------------------------------------------

namespace {

template <int HD>
__global__ void fmha_prefill_kernel(const __half* __restrict__ Q,
                                    const __half* __restrict__ K,
                                    const __half* __restrict__ V,
                                    const int32_t* __restrict__ block_id,
                                    const int32_t* __restrict__ active,
                                    __half* __restrict__ O,
                                    int B, int S, int NH, float scale) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (active[b] == 0) {
        // Zero O for inactive slots.
        for (int idx = threadIdx.x; idx < S * HD; idx += blockDim.x) {
            int s = idx / HD;
            int d = idx - s * HD;
            O[((b * S + s) * NH + h) * HD + d] = __float2half_rn(0.0f);
        }
        return;
    }
    const int tid = threadIdx.x;

    extern __shared__ float scratch[];   // size: 2 * S * HD floats + S floats
    float* K_sh = scratch;                                  // [S, HD]
    float* V_sh = scratch + S * HD;                         // [S, HD]
    int32_t* B_sh = (int32_t*)(scratch + 2 * S * HD);       // [S]

    // Load K, V, block_id into shared.
    for (int idx = tid; idx < S * HD; idx += blockDim.x) {
        int s = idx / HD;
        int d = idx - s * HD;
        K_sh[s * HD + d] = __half2float(K[((b * S + s) * NH + h) * HD + d]);
        V_sh[s * HD + d] = __half2float(V[((b * S + s) * NH + h) * HD + d]);
    }
    if (tid < S) {
        B_sh[tid] = block_id[b * S + tid];
    }
    __syncthreads();

    // For each query token i ∈ [0, S):
    for (int i = 0; i < S; ++i) {
        // Load Q[i] into register array (HD threads, each holds q[tid]).
        float q_lane = 0.0f;
        if (tid < HD) {
            q_lane = __half2float(Q[((b * S + i) * NH + h) * HD + tid]);
        }
        int32_t bi = B_sh[i];

        // Compute s[j] = q . K[j] * scale, for j ∈ [0, S), masked.
        // We loop j in shared mem.
        // Stream stats for online softmax in lane 0.
        float m_run = -3.4e38f;
        float l_run = 0.0f;
        float o_lane = 0.0f;

        for (int j = 0; j < S; ++j) {
            int32_t bj = B_sh[j];
            bool valid = (bi == bj) || (j <= i);
            // Compute partial dot.
            float partial = (tid < HD) ? q_lane * K_sh[j * HD + tid] : 0.0f;
            unsigned mask = 0xFFFFFFFFu;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(mask, partial, off);
            }
            __shared__ float warp_sums[8];
            int lane = tid & 31;
            int wid  = tid >> 5;
            if (lane == 0) warp_sums[wid] = partial;
            __syncthreads();

            float total = 0.0f;
            int n_warps = (HD + 31) / 32;
            for (int w = 0; w < n_warps; ++w) total += warp_sums[w];
            float s = total * scale;
            if (!valid) s = -3.4e38f;

            float new_m = fmaxf(m_run, s);
            float coef  = (m_run > -3.0e38f) ? __expf(m_run - new_m) : 0.0f;
            float ek    = (s > -3.0e38f) ? __expf(s - new_m) : 0.0f;
            float v_lane = (tid < HD) ? V_sh[j * HD + tid] : 0.0f;
            o_lane = o_lane * coef + ek * v_lane;
            l_run  = l_run  * coef + ek;
            m_run  = new_m;
            __syncthreads();
        }

        // Write O[i].
        if (tid < HD) {
            float out = o_lane / fmaxf(l_run, 1e-30f);
            O[((b * S + i) * NH + h) * HD + tid] = __float2half_rn(out);
        }
        __syncthreads();
    }
}

}  // namespace

void fmha_prefill_dispatch(const __half* Q, const __half* K, const __half* V,
                           const int32_t* block_id, const int32_t* active,
                           __half* O, int B, int S, int NH, int HD,
                           float scale, cudaStream_t stream) {
    if (HD == 64) {
        dim3 grid(B, NH);
        dim3 block(64);
        std::size_t shmem = sizeof(float) * (2 * S * HD) + sizeof(int32_t) * S;
        fmha_prefill_kernel<64><<<grid, block, shmem, stream>>>(
            Q, K, V, block_id, active, O, B, S, NH, scale);
        CE_CUDA_LAST();
    } else {
        // unsupported
    }
}

}  // namespace cutlass_engine

#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>

namespace cutlass_engine {

// --------------------------------------------------------------------------
// Tiled block-aware prefill self-attention.
//
// Each block handles one (b, h, q_tile) where q_tile partitions S into
// chunks of T_Q queries.  Threads = T_Q (one query per thread).  Each
// thread holds its Q vector and its O accumulator in registers.
//
// K and V are streamed from global memory in tiles of T_K keys at a time
// — shared memory holds only T_K*HD halfs per buffer (~8 KB at HD=64),
// independent of total S.  Supports S up to max_seq_len (4096+).
//
// Mask:
//   valid[q, k] = (block_id[b, q] == block_id[b, k]) || (k <= q)
// (causal + same-block bidirectional)
// --------------------------------------------------------------------------

namespace {

template <int HD, int T_Q, int T_K>
__global__ void fmha_prefill_tiled_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const int32_t* __restrict__ block_id,
    const int32_t* __restrict__ active,
    __half* __restrict__ O,
    int B, int S, int NH, float scale)
{
    const int b      = blockIdx.x;
    const int h      = blockIdx.y;
    const int q_tile = blockIdx.z;
    const int q_local = threadIdx.x;
    const int q_global = q_tile * T_Q + q_local;

    const bool valid_q = (q_global < S);
    const bool slot_active = (active[b] != 0);

    // Inactive slot: zero outputs and exit.
    if (!slot_active) {
        if (valid_q) {
            #pragma unroll
            for (int d = 0; d < HD; ++d) {
                O[((b * S + q_global) * NH + h) * HD + d] = __float2half_rn(0.0f);
            }
        }
        return;
    }

    // Load Q[q] into per-thread registers.
    float q_reg[HD];
    int32_t block_q = -1;
    if (valid_q) {
        #pragma unroll
        for (int d = 0; d < HD; ++d) {
            q_reg[d] = __half2float(Q[((b * S + q_global) * NH + h) * HD + d]);
        }
        block_q = block_id[b * S + q_global];
    } else {
        #pragma unroll
        for (int d = 0; d < HD; ++d) q_reg[d] = 0.0f;
    }

    // Shared mem: K_tile, V_tile (FP16), block_k_tile (int32).
    __shared__ __half K_tile[T_K * HD];
    __shared__ __half V_tile[T_K * HD];
    __shared__ int32_t block_k_tile[T_K];

    // Per-thread online softmax accumulators (in registers).
    float m_run = -3.4e38f;
    float l_run = 0.0f;
    float o_reg[HD];
    #pragma unroll
    for (int d = 0; d < HD; ++d) o_reg[d] = 0.0f;

    // Iterate over K tiles.
    for (int k_base = 0; k_base < S; k_base += T_K) {
        int k_count = min(T_K, S - k_base);

        // Cooperatively load K_tile, V_tile, block_k_tile into shared mem.
        __syncthreads();
        for (int idx = q_local; idx < k_count * HD; idx += T_Q) {
            int k = idx / HD;
            int d = idx - k * HD;
            K_tile[k * HD + d] = K[((b * S + k_base + k) * NH + h) * HD + d];
            V_tile[k * HD + d] = V[((b * S + k_base + k) * NH + h) * HD + d];
        }
        if (q_local < k_count) {
            block_k_tile[q_local] = block_id[b * S + k_base + q_local];
        }
        __syncthreads();

        if (!valid_q) continue;

        // Per-key attention update.
        for (int k = 0; k < k_count; ++k) {
            int k_global = k_base + k;
            int32_t block_k = block_k_tile[k];

            // Mask: same-block || causal.
            bool valid = (block_q == block_k) || (k_global <= q_global);

            // Q . K dot product (HD elements, accumulated in float).
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < HD; ++d) {
                s += q_reg[d] * __half2float(K_tile[k * HD + d]);
            }
            s *= scale;
            if (!valid) s = -3.4e38f;

            // Online softmax update.
            float new_m = fmaxf(m_run, s);
            float coef  = (m_run > -3.0e38f) ? __expf(m_run - new_m) : 0.0f;
            float ek    = (s > -3.0e38f) ? __expf(s - new_m) : 0.0f;
            #pragma unroll
            for (int d = 0; d < HD; ++d) {
                float v = __half2float(V_tile[k * HD + d]);
                o_reg[d] = o_reg[d] * coef + ek * v;
            }
            l_run = l_run * coef + ek;
            m_run = new_m;
        }
    }

    // Write O[q] = o_reg / l_run.
    if (valid_q) {
        float inv_l = 1.0f / fmaxf(l_run, 1e-30f);
        #pragma unroll
        for (int d = 0; d < HD; ++d) {
            O[((b * S + q_global) * NH + h) * HD + d] =
                __float2half_rn(o_reg[d] * inv_l);
        }
    }
}

}  // namespace

void fmha_prefill_dispatch(const __half* Q, const __half* K, const __half* V,
                           const int32_t* block_id, const int32_t* active,
                           __half* O, int B, int S, int NH, int HD,
                           float scale, cudaStream_t stream) {
    constexpr int T_Q = 64;
    constexpr int T_K = 64;
    int n_q_tiles = (S + T_Q - 1) / T_Q;
    dim3 grid(B, NH, n_q_tiles);
    dim3 block(T_Q);

    if (HD == 32) {
        fmha_prefill_tiled_kernel<32, T_Q, T_K><<<grid, block, 0, stream>>>(
            Q, K, V, block_id, active, O, B, S, NH, scale);
    } else if (HD == 64) {
        fmha_prefill_tiled_kernel<64, T_Q, T_K><<<grid, block, 0, stream>>>(
            Q, K, V, block_id, active, O, B, S, NH, scale);
    } else if (HD == 128) {
        fmha_prefill_tiled_kernel<128, T_Q, T_K><<<grid, block, 0, stream>>>(
            Q, K, V, block_id, active, O, B, S, NH, scale);
    } else {
        // unsupported
    }
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

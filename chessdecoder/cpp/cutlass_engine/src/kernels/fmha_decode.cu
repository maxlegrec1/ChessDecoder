#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cstdio>
#include <cuda_fp16.h>
#include <stdexcept>

namespace cutlass_engine {

// --------------------------------------------------------------------------
// FlashAttention-style decode kernel (S_q = 1).
// Online softmax over the key cache. Per-slot `past_len` controls how many
// cache positions to attend over.
//
// Inputs:
//   Q       [B, NH, HD]                    FP16
//   K_cache [num_layers, B, NH, max_seq, HD] FP16 — slice [layer, b, h, :p, :]
//   V_cache same shape — slice [layer, b, h, :p, :]
//   past_len[B]                             int32
// Output:
//   O       [B, NH, HD]                    FP16
//
// One block per (b, h). HD threads per block (HD ≤ 128). Each thread holds
// q[tid] in a register. Iterates over key positions in tiles of T_K.
// --------------------------------------------------------------------------

namespace {

template <int HD, int T_K>
__global__ void fmha_decode_kernel(const __half* __restrict__ Q,
                                   const __half* __restrict__ K_cache,
                                   const __half* __restrict__ V_cache,
                                   const int32_t* __restrict__ past_len,
                                   const int32_t* __restrict__ active,
                                   __half* __restrict__ O,
                                   int B, int NH, int max_seq_len,
                                   int layer_idx, float scale) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (active[b] == 0) {
        // Inactive slot — write zeros and return.
        if (threadIdx.x < HD) {
            O[(b * NH + h) * HD + threadIdx.x] = __float2half_rn(0.0f);
        }
        return;
    }
    const int tid = threadIdx.x;
    // Decode-mode contract: kv_scatter has already written the new K/V at
    // physical position past_len[b]. We attend over [0, past_len[b] + 1)
    // so the new position is included (self-attention).
    const int p_max = past_len[b] + 1;
    if (p_max <= 0) {
        // No keys to attend. Output 0; the engine will only read O at S_q=1
        // of an empty cache during pure-prefill init, which doesn't happen.
        if (tid < HD) {
            O[(b * NH + h) * HD + tid] = __float2half_rn(0.0f);
        }
        return;
    }

    // ---- Load Q[b,h,:] into register ----
    float q_reg = 0.0f;
    if (tid < HD) {
        q_reg = __half2float(Q[(b * NH + h) * HD + tid]);
    }

    // ---- Cache base for this (layer, b, h) ----
    const std::size_t per_h = (std::size_t)max_seq_len * HD;
    const std::size_t per_b = (std::size_t)NH * per_h;
    const std::size_t base  = (std::size_t)layer_idx * B * per_b
                            + (std::size_t)b * per_b
                            + (std::size_t)h * per_h;
    const __half* K_bh = K_cache + base;
    const __half* V_bh = V_cache + base;

    // Online-softmax accumulators. Each thread holds o[tid] across its tile loop.
    float m_run = -3.4e38f;  // running max
    float l_run = 0.0f;       // running denom
    float o_reg = 0.0f;       // running numerator partial for this lane

    // Shared memory: K_tile [T_K, HD], V_tile [T_K, HD], s_tile[T_K].
    __shared__ __half K_tile[T_K * HD];
    __shared__ __half V_tile[T_K * HD];
    __shared__ float  s_tile[T_K];

    for (int kp_base = 0; kp_base < p_max; kp_base += T_K) {
        int kp_count = min(T_K, p_max - kp_base);

        // Load K_tile, V_tile cooperatively.
        for (int idx = tid; idx < kp_count * HD; idx += blockDim.x) {
            int k = idx / HD;
            int d = idx - k * HD;
            K_tile[k * HD + d] = K_bh[(kp_base + k) * HD + d];
            V_tile[k * HD + d] = V_bh[(kp_base + k) * HD + d];
        }
        __syncthreads();

        // Compute s_tile[k] = q . K_tile[k] * scale.
        // Each thread contributes q_reg * K_tile[k][tid]; then warp reduce.
        for (int k = 0; k < kp_count; ++k) {
            float partial = 0.0f;
            if (tid < HD) {
                partial = q_reg * __half2float(K_tile[k * HD + tid]);
            }
            // Warp reduction (single warp covers HD=64; if HD>32 we need
            // two-step). HD ≤ 128 → up to 4 warps.
            unsigned mask = 0xFFFFFFFFu;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(mask, partial, off);
            }
            // Now lane 0 of each warp has its warp's sum.
            __shared__ float warp_sums[8];
            int lane = tid & 31;
            int wid  = tid >> 5;
            if (lane == 0) warp_sums[wid] = partial;
            __syncthreads();
            float total = 0.0f;
            int n_warps = (HD + 31) / 32;
            if (tid == 0) {
                for (int w = 0; w < n_warps; ++w) total += warp_sums[w];
                s_tile[k] = total * scale;
            }
            __syncthreads();
        }

        // Online softmax update — same for every lane; we work on s_tile[k]
        // and update m_run, l_run, o_reg.
        for (int k = 0; k < kp_count; ++k) {
            float s = s_tile[k];
            float new_m = fmaxf(m_run, s);
            float coef  = (m_run > -3.0e38f) ? __expf(m_run - new_m) : 0.0f;
            float ek    = __expf(s - new_m);
            float v_lane = (tid < HD) ? __half2float(V_tile[k * HD + tid]) : 0.0f;
            o_reg = o_reg * coef + ek * v_lane;
            l_run = l_run * coef + ek;
            m_run = new_m;
        }
        __syncthreads();
    }

    // Final write: O[b,h,:] = o_reg / l_run
    if (tid < HD) {
        float out = o_reg / fmaxf(l_run, 1e-30f);
        O[(b * NH + h) * HD + tid] = __float2half_rn(out);
    }
}

}  // namespace

// We specialize on HD; only the model's head_dim is used in practice. For
// ChessDecoder HD=64. We also support HD=128 for forward-compat. We dispatch
// at the call site.

}  // namespace cutlass_engine

// Dispatch lives outside the anonymous namespace.
namespace cutlass_engine {

void fmha_decode_dispatch(const __half* Q, const __half* K_cache, const __half* V_cache,
                          const int32_t* past_len, const int32_t* active,
                          __half* O,
                          int B, int NH, int HD,
                          int max_seq_len, int layer_idx, float scale,
                          cudaStream_t stream) {
    dim3 grid(B, NH);
    if (HD == 32) {
        constexpr int T_K = 64;
        dim3 block(32);
        fmha_decode_kernel<32, T_K><<<grid, block, 0, stream>>>(
            Q, K_cache, V_cache, past_len, active, O, B, NH,
            max_seq_len, layer_idx, scale);
    } else if (HD == 64) {
        constexpr int T_K = 64;
        dim3 block(64);
        fmha_decode_kernel<64, T_K><<<grid, block, 0, stream>>>(
            Q, K_cache, V_cache, past_len, active, O, B, NH,
            max_seq_len, layer_idx, scale);
    } else if (HD == 128) {
        constexpr int T_K = 32;
        dim3 block(128);
        fmha_decode_kernel<128, T_K><<<grid, block, 0, stream>>>(
            Q, K_cache, V_cache, past_len, active, O, B, NH,
            max_seq_len, layer_idx, scale);
    } else {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "fmha_decode_dispatch: unsupported HD=%d (supported: 32, 64, 128)", HD);
        throw std::runtime_error(buf);
    }
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

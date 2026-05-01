#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <cstdint>

namespace cutlass_engine {

namespace {

// One block per row. We do a parallel reduction with (val, idx) pairs.
// Threads cover [0, V) in strides; warp + block reduce keeps the max.
template <int BLOCK>
__global__ void argmax_kernel(const __half* __restrict__ logits,
                              int32_t* __restrict__ idx_out,
                              int V) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* row = logits + b * V;

    float best_v = -3.4e38f;
    int best_i = 0;
    for (int i = tid; i < V; i += BLOCK) {
        float v = __half2float(row[i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }

    __shared__ float sval[BLOCK];
    __shared__ int   sidx[BLOCK];
    sval[tid] = best_v;
    sidx[tid] = best_i;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float v_other = sval[tid + off];
            int   i_other = sidx[tid + off];
            if (v_other > sval[tid]) {
                sval[tid] = v_other;
                sidx[tid] = i_other;
            }
        }
        __syncthreads();
    }

    if (tid == 0) idx_out[b] = sidx[0];
}

template <int BLOCK>
__global__ void gumbel_argmax_kernel(const __half* __restrict__ logits,
                                     uint64_t* __restrict__ ph_seed,
                                     uint64_t* __restrict__ ph_offset,
                                     int32_t* __restrict__ idx_out,
                                     float inv_temp,
                                     int V) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* row = logits + b * V;

    // Single-thread-per-slot RNG state. Thread 0 will draw V uniforms;
    // we instead let each thread draw its own using its (b, tid, i) coords.
    curandStatePhilox4_32_10_t st;
    curand_init(ph_seed[b], (uint64_t)b * V + tid, ph_offset[b], &st);

    float best_v = -3.4e38f;
    int best_i = 0;
    for (int i = tid; i < V; i += BLOCK) {
        float u = curand_uniform(&st);
        // u in (0, 1].  -log(-log(u)) is +Gumbel; we add to logits/T.
        float g = -__logf(-__logf(u + 1e-20f) + 1e-20f);
        float v = __half2float(row[i]) * inv_temp + g;
        if (v > best_v) { best_v = v; best_i = i; }
    }

    __shared__ float sval[BLOCK];
    __shared__ int   sidx[BLOCK];
    sval[tid] = best_v;
    sidx[tid] = best_i;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float vo = sval[tid + off];
            int   io = sidx[tid + off];
            if (vo > sval[tid]) { sval[tid] = vo; sidx[tid] = io; }
        }
        __syncthreads();
    }
    if (tid == 0) {
        idx_out[b] = sidx[0];
        // Bump offset so subsequent calls draw fresh entropy.
        atomicAdd(reinterpret_cast<unsigned long long*>(&ph_offset[b]),
                  (unsigned long long)V);
    }
}

__global__ void apply_legal_mask_kernel(__half* __restrict__ logits,
                                        const bool* __restrict__ legal,
                                        int V) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    __half* row = logits + b * V;
    for (int i = tid; i < V; i += stride) {
        if (!legal[b * V + i]) {
            row[i] = __float2half_rn(-65504.0f);
        }
    }
}

template <int BLOCK>
__global__ void log_prob_at_idx_kernel(const __half* __restrict__ logits,
                                       const int32_t* __restrict__ idx,
                                       float inv_temp,
                                       float* __restrict__ out_logp,
                                       int V) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* row = logits + b * V;

    // 1. row max (numerical-stability)
    float my_max = -3.4e38f;
    for (int i = tid; i < V; i += BLOCK) {
        float v = __half2float(row[i]) * inv_temp;
        if (v > my_max) my_max = v;
    }
    __shared__ float smax[BLOCK];
    smax[tid] = my_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off && smax[tid + off] > smax[tid]) smax[tid] = smax[tid + off];
        __syncthreads();
    }
    float row_max = smax[0];

    // 2. log-sum-exp
    float my_sum = 0.0f;
    for (int i = tid; i < V; i += BLOCK) {
        float v = __half2float(row[i]) * inv_temp - row_max;
        my_sum += __expf(v);
    }
    __shared__ float ssum[BLOCK];
    ssum[tid] = my_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) ssum[tid] += ssum[tid + off];
        __syncthreads();
    }
    float lse = row_max + __logf(ssum[0]);

    // 3. logp = (logits[idx]/T) - lse
    if (tid == 0) {
        int chosen = idx[b];
        float logit = __half2float(row[chosen]) * inv_temp;
        out_logp[b] = logit - lse;
    }
}

}  // namespace

void argmax_fp16(const __half* logits, int32_t* idx_out,
                 int B, int V, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    argmax_kernel<BLOCK><<<B, BLOCK, 0, stream>>>(logits, idx_out, V);
    CE_CUDA_LAST();
}

void gumbel_argmax_fp16(const __half* logits,
                        uint64_t* ph_seed, uint64_t* ph_offset,
                        int32_t* idx_out,
                        float temperature,
                        int B, int V, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    float inv_t = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    gumbel_argmax_kernel<BLOCK><<<B, BLOCK, 0, stream>>>(
        logits, ph_seed, ph_offset, idx_out, inv_t, V);
    CE_CUDA_LAST();
}

void apply_legal_mask_fp16(__half* logits, const bool* legal,
                           int B, int V, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    apply_legal_mask_kernel<<<B, BLOCK, 0, stream>>>(logits, legal, V);
    CE_CUDA_LAST();
}

void log_prob_at_idx_fp16(const __half* logits, const int32_t* idx,
                          float temperature,
                          float* out_logp,
                          int B, int V, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    float inv_t = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    log_prob_at_idx_kernel<BLOCK><<<B, BLOCK, 0, stream>>>(
        logits, idx, inv_t, out_logp, V);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

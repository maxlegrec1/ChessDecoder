#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>

namespace cutlass_engine {

namespace {

// Layout: K_cache, V_cache are
//    [num_layers, B, NH, max_seq_len, HD] FP16, contiguous.
// new_k, new_v are
//    [B, S, NH, HD] FP16, contiguous.
//
// For each active slot b and step s in [0, S):
//   dst_pos = past_len[b] + s
//   K_cache[layer, b, h, dst_pos, :] = new_k[b, s, h, :]
//
// One block per (b, h); threads cover S * HD.
__global__ void kv_scatter_kernel(const __half* __restrict__ new_k,
                                  const __half* __restrict__ new_v,
                                  __half* __restrict__ K_cache,
                                  __half* __restrict__ V_cache,
                                  const int32_t* __restrict__ past_len,
                                  const int32_t* __restrict__ active,
                                  int B, int S, int NH, int HD,
                                  int max_seq_len, int layer_idx,
                                  int /*num_layers*/) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (active[b] == 0) return;

    const int p0 = past_len[b];

    const std::size_t per_b   = (std::size_t)NH * max_seq_len * HD;
    const std::size_t cache_off_layer = (std::size_t)layer_idx * B * per_b;
    __half* k_dst = K_cache + cache_off_layer + (std::size_t)b * per_b;
    __half* v_dst = V_cache + cache_off_layer + (std::size_t)b * per_b;

    const std::size_t per_b_src = (std::size_t)S * NH * HD;
    const __half* k_src = new_k + (std::size_t)b * per_b_src;
    const __half* v_src = new_v + (std::size_t)b * per_b_src;

    // Each thread writes one (s, hd) element of the head h.
    // s ∈ [0, S),  d ∈ [0, HD).
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total = S * HD;
    int stride = blockDim.x * blockDim.y;
    for (int idx = tid; idx < total; idx += stride) {
        int s = idx / HD;
        int d = idx - s * HD;
        std::size_t src = ((std::size_t)s * NH + h) * HD + d;
        std::size_t dst = ((std::size_t)h * max_seq_len + (p0 + s)) * HD + d;
        k_dst[dst] = k_src[src];
        v_dst[dst] = v_src[src];
    }
}

__global__ void past_len_inc_kernel(int32_t* past_len,
                                    const int32_t* active,
                                    int S, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    if (active[b]) past_len[b] += S;
}

}  // namespace

void kv_scatter_fp16(const __half* new_k, const __half* new_v,
                     __half* K_cache, __half* V_cache,
                     const int32_t* past_len, const int32_t* active,
                     int B, int S, int NH, int HD,
                     int max_seq_len, int layer_idx, int num_layers,
                     cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid(B, NH);
    kv_scatter_kernel<<<grid, block, 0, stream>>>(
        new_k, new_v, K_cache, V_cache, past_len, active,
        B, S, NH, HD, max_seq_len, layer_idx, num_layers);
    CE_CUDA_LAST();
}

void past_len_increment(int32_t* past_len, const int32_t* active,
                        int S, int B, cudaStream_t stream) {
    int blocks = (B + 63) / 64;
    past_len_inc_kernel<<<blocks, 64, 0, stream>>>(past_len, active, S, B);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

// Helpers for CUTLASS FMHA block-aware mask integration:
//   1. compute_effective_limit_kernel: from block_id[B, S] compute
//      effective_limit[b][q] = max(q, end_of_block(b, q)) where end_of_block
//      is the largest k with block_id[b, k] == block_id[b, q]. The
//      block-aware causal mask
//          valid[b, q, k] = (block_id[b, q] == block_id[b, k]) || (k <= q)
//      reduces to the simpler "k <= effective_limit[b, q]" because for any
//      k inside the same block (block_id[b, k] == block_id[b, q]):
//          k <= end_of_block(b, q) <= effective_limit[b, q]
//      and for k > q outside the block:
//          k > q AND block_id[b, k] != block_id[b, q] ⇒ invalid
//      and for k <= q (causal):
//          k <= q <= effective_limit[b, q] ⇒ valid.
//
//   2. write_g_block_aware_arr_kernel: copy a [B, S] eff_limit buffer into
//      the global __device__ array `g_block_aware_eff_limit_arr` and store
//      max_S into `g_block_aware_max_S`. This kernel runs on stream and
//      sequences before the FMHA kernel, so the FMHA reads the correct
//      values without needing cudaMemcpyToSymbol (which is host-side).

#include "cutlass_engine/check.hpp"
#include "collective/fmha_fusion.hpp"  // declares the __device__ globals

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace cutlass_engine {

namespace {

// One block per (b, q). Each thread scans the row to find end_of_block.
// Naive O(S) per thread; total work O(B * S^2) which at our sizes (B<=256,
// S<=4096) is ~4M ops — comparable to one moderate gemm and fits easily.
__global__ void compute_effective_limit_kernel(
    const int32_t* __restrict__ block_id,
    int32_t* __restrict__ eff_limit,
    int B, int S) {
    int b = blockIdx.x;
    int q = blockIdx.y * blockDim.x + threadIdx.x;
    if (q >= S) return;

    const int32_t* row = block_id + b * S;
    int32_t bq = row[q];
    int end = q;
    // Scan from S-1 down to find the max k with same block_id.
    // Since blocks are typically contiguous, breaking at first mismatch
    // from S-1 backwards may be sub-optimal; we instead scan from q forward
    // to find the last same-block index.
    for (int k = q + 1; k < S; ++k) {
        if (row[k] == bq) end = k;
    }
    eff_limit[b * S + q] = end > q ? end : q;
}

// Copies the precomputed eff_limit buffer into the global __device__ array
// and writes max_S. We can't use cudaMemcpyToSymbol on a stream, so we
// launch a tiny kernel to do the copy + scalar store, ensuring stream
// ordering against the upcoming FMHA launch.
__global__ void write_g_block_aware_arr_kernel(
    const int32_t* __restrict__ src, int total_elems, int max_S_value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        g_block_aware_eff_limit_arr[i] = src[i];
    }
    if (i == 0) {
        g_block_aware_max_S = max_S_value;
    }
}

}  // namespace

// Public host wrappers.
void fmha_compute_effective_limit(const int32_t* block_id,
                                  int32_t* eff_limit_out,
                                  int B, int S, cudaStream_t stream) {
    constexpr int TX = 128;
    dim3 block(TX);
    dim3 grid(B, (S + TX - 1) / TX);
    compute_effective_limit_kernel<<<grid, block, 0, stream>>>(
        block_id, eff_limit_out, B, S);
    CE_CUDA_LAST();
}

void fmha_publish_block_aware_globals(const int32_t* eff_limit_buf,
                                      int B, int S, cudaStream_t stream) {
    int total = B * S;
    constexpr int TX = 256;
    dim3 block(TX);
    dim3 grid((total + TX - 1) / TX);
    write_g_block_aware_arr_kernel<<<grid, block, 0, stream>>>(
        eff_limit_buf, total, S);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

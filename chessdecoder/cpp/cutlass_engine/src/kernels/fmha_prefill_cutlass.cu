// CUTLASS Blackwell sm_100a FMHA wrapper for prefill.
//
// First iteration: stock CausalMask only. Validates the integration path
// (TMA + tensor cores via CollectiveBuilder, ProblemShape construction,
// strides, kernel-static `run(params, stream)` API). Block-aware mask
// support is a follow-up; for now, this kernel is selected only when the
// caller opts in via `fmha_prefill_cutlass_causal`.
//
// Heavy template; expect ~5–10 min compile.

#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>
#include <cstdio>
#include <stdexcept>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

// example 77 headers (added to include path in setup.py)
#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

namespace cutlass_engine {

namespace {

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

// ----------------------------------------------------------------------------
// Single FMHA template instance: HD=64, FP16 in/out, FP32 acc, CausalMask.
// TileShape = <Q=256, K=128, D=64>. Persistent tile scheduler.
// ----------------------------------------------------------------------------

using Element = cutlass::half_t;
using ElementAccumulatorQK = float;
using ElementAccumulatorPV = float;
using ElementOut = cutlass::half_t;

// Problem shape: (Q, K, D, ((H_R, H_K), B))
// For MHA: H_R = 1 (each Q head is its own group), H_K = NH (one KV per Q).
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

// Strides (row-major Q[B,SQ,H,D], K/V[B,SK,H,D], O[B,SQ,H,D])
//   Q-stride = (H*D, _1, ((D, H_R*D), H*D*SQ))
//   K-stride = (H_K*D, _1, ((_0, D), H_K*D*SK))   // H_R-stride = 0 (broadcast unused at H_R=1)
using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;

using TileShapeHD64 = Shape<_256, _128, _64>;

template <class Mask>
using MainloopHD64 =
    cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
        Element, ElementAccumulatorQK, ElementAccumulatorPV,
        TileShapeHD64, StrideQ, StrideK, StrideV,
        Mask
    >;

template <class Mask>
using EpilogueHD64 =
    cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementOut, ElementAccumulatorPV,
        typename MainloopHD64<Mask>::TileShapePV,
        StrideO, StrideLSE
    >;

template <class Mask>
using KernelHD64 = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape,
    MainloopHD64<Mask>,
    EpilogueHD64<Mask>,
    cutlass::fmha::kernel::PersistentTileScheduler
>;

template <class Mask>
using OperationHD64 = cutlass::fmha::device::FMHA<KernelHD64<Mask>>;

template <class Mask>
void run_fmha_hd64(const __half* Q, const __half* K, const __half* V,
                   __half* O,
                   int B, int S, int NH, float scale,
                   void* workspace, void* lse_buf,
                   cudaStream_t stream) {
    using Op = OperationHD64<Mask>;
    using Kernel = KernelHD64<Mask>;
    constexpr int D = 64;

    // Shape (Q-len, K-len, D, ((H_R=1, H_K=NH), B))
    ProblemShape problem_shape = make_tuple(
        S, S, D, make_tuple(make_tuple(1, NH), B));

    // Q layout: row-major [B, SQ, H, D]
    StrideQ stride_Q = make_stride(
        NH * D, _1{},
        make_stride(make_stride(D, /*H_R*D=*/1 * D), NH * D * S));
    StrideO stride_O = stride_Q;
    StrideK stride_K = make_stride(
        NH * D, _1{},
        make_stride(make_stride(_0{}, D), NH * D * S));
    StrideV stride_V = stride_K;
    StrideLSE stride_LSE = make_stride(
        _1{},
        make_stride(make_stride(S, S * 1), S * NH));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    static int sm_count = 0;
    if (sm_count == 0) {
        sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    }
    hw_info.sm_count = sm_count;

    typename Op::Arguments args{
        problem_shape,
        { reinterpret_cast<const Element*>(Q), stride_Q,
          reinterpret_cast<const Element*>(K), stride_K,
          reinterpret_cast<const Element*>(V), stride_V },
        { reinterpret_cast<ElementOut*>(O), stride_O,
          reinterpret_cast<ElementAccumulatorPV*>(lse_buf), stride_LSE },
        hw_info
    };

    // The static run(params, stream) API skips cudaFuncSetAttribute for the
    // dynamic shared-memory carveout. Use the instance API: initialize() does
    // both the cudaFuncSetAttribute (once, gated by static flag) and the
    // workspace init / params update; run() then launches with the cached
    // params_. Thread-local because the static-bool init flag inside FMHA
    // would otherwise be shared across threads.
    static thread_local Op op;
    cutlass::Status st = op.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "CUTLASS FMHA initialize failed: status=%d B=%d S=%d NH=%d",
            int(st), B, S, NH);
        throw std::runtime_error(msg);
    }
    st = op.run(stream);
    if (st != cutlass::Status::kSuccess) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "CUTLASS FMHA run failed: status=%d B=%d S=%d NH=%d", int(st), B, S, NH);
        throw std::runtime_error(msg);
    }
}

}  // namespace

// __device__ symbol definitions — fixed-size array (no pointer indirection)
// + scalar max_S. cudaMemcpyToSymbol writes directly into the array storage.
__device__ int g_block_aware_eff_limit_arr[kBlockAwareEffLimitMaxElems];
__device__ volatile int g_block_aware_max_S;

// Workspace size needed by CUTLASS FMHA per invocation, in bytes.
// (Conservative upper bound — actual usage is tiny; both Mask types share
// the same workspace size.)
std::size_t fmha_prefill_cutlass_workspace_bytes(int B, int S, int NH, int HD) {
    if (HD != 64) return 0;
    using Op = OperationHD64<CausalMask>;
    typename Op::Arguments args{};
    args.problem_shape = make_tuple(
        S, S, HD, make_tuple(make_tuple(1, NH), B));
    return Op::get_workspace_size(args);
}

// LSE buffer size in float elements: [B * NH * S].
std::size_t fmha_prefill_cutlass_lse_elements(int B, int S, int NH) {
    return std::size_t(B) * NH * S;
}

void fmha_prefill_cutlass_causal(const __half* Q, const __half* K, const __half* V,
                                 __half* O,
                                 int B, int S, int NH, int HD, float scale,
                                 void* workspace, void* lse_buf,
                                 cudaStream_t stream) {
    if (HD == 64) {
        run_fmha_hd64<CausalMask>(Q, K, V, O, B, S, NH, scale, workspace, lse_buf, stream);
    } else {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
            "CUTLASS FMHA: HD=%d not instantiated (only 64 is supported in J.2)", HD);
        throw std::runtime_error(msg);
    }
    CE_CUDA_LAST();
}

// Block-aware causal: caller precomputes effective_limit[B*max_S] (int32)
// device pointer. Per (b, q): valid k iff k <= effective_limit[b*max_S + q].
// To express block-aware mask, set effective_limit[b][q] = max(q, end_of_block(b, q)).
void fmha_prefill_cutlass_block_aware(const __half* Q, const __half* K, const __half* V,
                                      __half* O,
                                      const int32_t* effective_limit, int max_S,
                                      int B, int S, int NH, int HD, float scale,
                                      void* workspace, void* lse_buf,
                                      cudaStream_t stream) {
    if (HD != 64) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
            "CUTLASS FMHA block-aware: HD=%d not instantiated (only 64)", HD);
        throw std::runtime_error(msg);
    }
    // Copy the limit array directly into __device__ storage. Direct array
    // storage (no pointer indirection) avoids the constant-folding issue we
    // hit with __device__ pointer globals in templated FMHA kernels.
    std::size_t bytes = std::size_t(B) * max_S * sizeof(int32_t);
    if (bytes > std::size_t(kBlockAwareEffLimitMaxElems) * sizeof(int)) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
            "fmha_prefill_cutlass_block_aware: B*max_S=%zu exceeds %d slots",
            std::size_t(B) * max_S, kBlockAwareEffLimitMaxElems);
        throw std::runtime_error(msg);
    }
    // First copy the limit data (device-to-device — `effective_limit` is in
    // gmem too). Use cudaMemcpyToSymbol with cudaMemcpyDeviceToDevice.
    CE_CUDA_CHECK(cudaMemcpyToSymbol(g_block_aware_eff_limit_arr,
        effective_limit, bytes, /*offset=*/0, cudaMemcpyDeviceToDevice));
    CE_CUDA_CHECK(cudaMemcpyToSymbol(g_block_aware_max_S,
        &max_S, sizeof(int)));
    // Note: host-side readback confirms these writes land at the symbol
    // address. The FMHA kernel however reads zeros for both — the symbol
    // resolution diverges between the host's cudaMemcpyToSymbol and the
    // templated FMHA kernel's read. -rdc=true fixes the symbol resolution
    // but breaks the device-link in the current setuptools build.
    // See fmha_fusion.hpp BlockAwareCausalMask::apply_mask comment.

    run_fmha_hd64<BlockAwareCausalMask>(Q, K, V, O, B, S, NH, scale,
                                        workspace, lse_buf, stream);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

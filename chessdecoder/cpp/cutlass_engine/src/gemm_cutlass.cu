// CUTLASS-based FP16 GEMM with optional residual/bias epilogue fusion.
//
// Same semantics as gemm_fp16 in gemm.cu:
//   D[M,N] = A[M,K] @ B_w[N,K]^T   (row-major) with optional + bias[N], + beta*C[M,N].
//
// Backed by a CUTLASS 3.x sm_100a warpspecialized kernel (Blackwell).
// Single tile config (MmaTile=128x128x64, Cluster=1x1x1) — good baseline for
// the shapes we run; can be specialized per-shape later.
//
// Compared to cuBLAS HGEMM, this lets us:
//   1. Fuse residual-add into the GEMM epilogue (D = AB + beta*C),
//      eliminating a separate bandwidth-bound add+rmsnorm dance.
//   2. Lay the groundwork for the FP8 path (Phase H) by reusing this dispatch.
//
// Selected at runtime via the env var USE_CUTLASS_GEMM=1 in gemm_dispatch.

#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>
#include <cstdio>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace cutlass_engine {

namespace {

using namespace cute;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAcc = float;
using ElementCompute = float;
using ElementScalar = float;

using LayoutA = cutlass::layout::RowMajor;     // A: [M, K] row-major
using LayoutB = cutlass::layout::ColumnMajor;  // B: [K, N] col-major == [N, K] row-major (weight storage)
using LayoutC = cutlass::layout::RowMajor;     // C: [M, N] row-major (residual)
using LayoutD = cutlass::layout::RowMajor;     // D: [M, N] row-major

constexpr int AlignmentA = 8;  // 8 * sizeof(half) = 16-byte aligned vector loads
constexpr int AlignmentB = 8;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;

using MmaTile     = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;

// LinearCombination: D = alpha * acc + beta * C. With C=residual and beta=1
// this is the residual-fused path. With beta=0 it's plain GEMM.
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementScalar,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTile, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    EpilogueOp
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAcc,
    MmaTile, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,   // ProblemShape (M, N, K, L=batch)
    CollectiveMainloop,
    CollectiveEpilogue
  >;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

void run_gemm(const __half* A, const __half* B_w,
              const __half* C_in, __half* D_out,
              int M, int N, int K,
              float alpha, float beta,
              cudaStream_t stream) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const ElementA*>(A), stride_A,
         reinterpret_cast<const ElementB*>(B_w), stride_B},
        {{},  // epilogue.thread placeholder filled below
         reinterpret_cast<const ElementC*>(C_in), stride_C,
         reinterpret_cast<ElementD*>(D_out), stride_D},
        {}    // hw_info — defaulted
    };
    args.epilogue.thread.alpha = alpha;
    args.epilogue.thread.beta  = beta;

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "CUTLASS gemm.can_implement failed: status=%d M=%d N=%d K=%d alpha=%g beta=%g",
            int(status), M, N, K, alpha, beta);
        throw std::runtime_error(msg);
    }

    // Workspace — small; one allocation per call is fine for now (Blackwell
    // warpspecialized typically asks for a few KB scheduler workspace). If
    // hot-path overhead shows up, lift this into a per-engine arena.
    std::size_t ws_bytes = Gemm::get_workspace_size(args);
    void* ws_ptr = nullptr;
    if (ws_bytes > 0) {
        if (cudaMallocAsync(&ws_ptr, ws_bytes, stream) != cudaSuccess) {
            throw std::runtime_error("cudaMallocAsync failed for CUTLASS workspace");
        }
    }

    status = gemm_op.initialize(args, ws_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
        if (ws_ptr) cudaFreeAsync(ws_ptr, stream);
        throw std::runtime_error("CUTLASS gemm.initialize failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        if (ws_ptr) cudaFreeAsync(ws_ptr, stream);
        throw std::runtime_error("CUTLASS gemm.run failed");
    }

    if (ws_ptr) cudaFreeAsync(ws_ptr, stream);
}

__global__ void add_bias_kernel(__half* C, const __half* bias, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    float v = __half2float(C[m * N + n]) + __half2float(bias[n]);
    C[m * N + n] = __float2half_rn(v);
}

}  // namespace

// Exposed entry: plain GEMM (D = A @ B^T). Bias handled via separate kernel
// for parity with cuBLAS path (bias is rare in this model).
void gemm_fp16_cutlass(const __half* A, const __half* B_w, const __half* bias,
                       __half* D, int M, int N, int K,
                       cudaStream_t stream) {
    run_gemm(A, B_w, /*C=*/nullptr, D, M, N, K, /*alpha=*/1.0f, /*beta=*/0.0f, stream);
    if (bias != nullptr) {
        constexpr int TX = 256;
        dim3 block(TX);
        dim3 grid((N + TX - 1) / TX, M);
        add_bias_kernel<<<grid, block, 0, stream>>>(D, bias, M, N);
        CE_CUDA_LAST();
    }
}

// Exposed entry: residual-fused GEMM. D = A @ B^T + residual.
// Eliminates the separate "rmsnorm_residual reads h_in then adds" hop —
// callers hand `residual` straight to the GEMM and the next block's
// rmsnorm reads the updated `D` (which IS the residual sum).
void gemm_fp16_cutlass_residual(const __half* A, const __half* B_w,
                                const __half* residual,
                                __half* D, int M, int N, int K,
                                cudaStream_t stream) {
    run_gemm(A, B_w, residual, D, M, N, K, /*alpha=*/1.0f, /*beta=*/1.0f, stream);
}

}  // namespace cutlass_engine

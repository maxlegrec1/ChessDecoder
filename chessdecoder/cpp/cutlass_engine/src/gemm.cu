#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// FP16 GEMM wrapper.
//
// For Phase B / FP16 we use cuBLAS as the backend. cuBLAS at FP16 (HGEMM /
// CUBLAS_COMPUTE_16F with FP32 accumulator) hits ~95% of optimal on Blackwell
// for the shapes we run (M,N,K typically 64×4096×1024, 64×3072×1024, 64×1024×1024).
//
// Phase H / FP8 swaps this for a handcrafted CUTLASS kernel — cuBLAS exposes
// FP8 GEMM but not all the per-channel scaling we want.
//
// Bias is fused via a separate `addmm_bias` kernel — this is a single launch
// per matmul (still well within bandwidth budget) and avoids the cuBLAS-LT
// epilogue setup cost. We can flip to LT in Phase G if profiling shows bias
// addition dominating.

namespace cutlass_engine {

namespace {

// One cuBLAS handle per process. Lazily initialized.
cublasHandle_t& get_handle() {
    static thread_local cublasHandle_t h = nullptr;
    if (h == nullptr) {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
    }
    return h;
}

__global__ void add_bias_kernel(__half* C, const __half* bias, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    float v = __half2float(C[m * N + n]) + __half2float(bias[n]);
    C[m * N + n] = __float2half_rn(v);
}

__global__ void add_bias_fp32_kernel(float* C, const __half* bias, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    C[m * N + n] = C[m * N + n] + __half2float(bias[n]);
}

}  // namespace

std::size_t gemm_fp16_workspace_bytes(int /*M*/, int /*N*/, int /*K*/) {
    // cuBLAS GEMM doesn't need user workspace at this sizing.
    return 0;
}

namespace {
// Check USE_CUTLASS_GEMM=1 once. Cached because env reads are not free.
bool use_cutlass_backend() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("USE_CUTLASS_GEMM");
        cached = (v && v[0] == '1') ? 1 : 0;
        std::fprintf(stderr, "[cutlass_engine] gemm backend: %s\n",
                     cached ? "CUTLASS" : "cuBLAS");
    }
    return cached == 1;
}
}  // namespace

void gemm_fp16(const __half* A, const __half* B_w, const __half* bias,
               __half* C, int M, int N, int K,
               void* /*workspace*/, std::size_t /*workspace_bytes*/,
               cudaStream_t stream) {
    // CUTLASS backend requires alignment 8 on N and K (8-half = 16B vector
    // loads). Head GEMMs (N=1924 policy, N=41 board, etc.) don't fit; fall
    // back to cuBLAS for those. Transformer block GEMMs all qualify
    // (N ∈ {3072, 1024}, K ∈ {1024, 1536}).
    if (use_cutlass_backend() && (N % 8 == 0) && (K % 8 == 0)) {
        gemm_fp16_cutlass(A, B_w, bias, C, M, N, K, stream);
        return;
    }
    // We compute C[M,N] = A[M,K] @ B_w[N,K]^T, row-major.
    // cuBLAS is column-major, so we issue the equivalent
    //   C^T[N,M] = (B_w[N,K])  @  (A[M,K])^T
    // i.e. cublasGemmEx with op_A = N (no transpose on B_w because it's
    // already [N,K] column-major-equivalent), op_B = T (transpose A).
    cublasHandle_t h = get_handle();
    cublasSetStream(h, stream);

    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);

    // From cuBLAS column-major perspective:
    //   A_cb = B_w (interpreted column-major as [K, N])
    //   B_cb = A   (interpreted column-major as [K, M])
    //   C_cb = C   (interpreted column-major as [N, M])
    // Both A_cb and B_cb need transA=N (no extra transpose needed because
    // row-major[N,K] == column-major[K,N], and we want C_cb = A_cb @ B_cb).
    //
    // Wait — that gives C_cb[N,M] = (B_w col-major)[K,N]^T @ (A col-major)[K,M]
    // which is cublas op_A=T, op_B=N.
    cublasStatus_t st = cublasGemmEx(
        h,
        CUBLAS_OP_T,                // transA=T: B_w col-maj is [K,N], we want [N,K]
        CUBLAS_OP_N,                // transB=N: A col-maj is [K,M]
        N, M, K,
        &alpha,
        B_w, CUDA_R_16F, K,         // ldA in column-major == K
        A,   CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx FP16 failed");
    }

    if (bias != nullptr) {
        constexpr int TX = 256;
        dim3 block(TX);
        dim3 grid((N + TX - 1) / TX, M);
        add_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
        CE_CUDA_LAST();
    }
}

void gemm_fp16_out_fp32(const __half* A, const __half* B_w, const __half* bias,
                        float* C, int M, int N, int K,
                        void* /*ws*/, std::size_t /*wsb*/,
                        cudaStream_t stream) {
    cublasHandle_t h = get_handle();
    cublasSetStream(h, stream);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B_w, CUDA_R_16F, K,
        A,   CUDA_R_16F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx FP16->FP32 failed");
    }
    if (bias != nullptr) {
        constexpr int TX = 256;
        dim3 block(TX);
        dim3 grid((N + TX - 1) / TX, M);
        add_bias_fp32_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
        CE_CUDA_LAST();
    }
}

}  // namespace cutlass_engine

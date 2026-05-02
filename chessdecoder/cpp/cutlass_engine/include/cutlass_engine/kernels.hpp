#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace cutlass_engine {

// ============================================================================
// Custom CUDA kernels exposed to layer-level code. All take a CUDA stream
// and never sync — they enqueue and return.
// ============================================================================

// ---------- RMSNorm ----------
//
// y[b,t,e] = x[b,t,e] * rsqrt(mean(x[b,t,:]^2) + eps) * w[e]
//
// Shapes: x,y both [M, E] flat (M = B*S). w [E].
// FP16 in/out, FP32 reduce. eps is the model's default 1e-6.
void rmsnorm_fp16(const __half* x, const __half* w, __half* y,
                  int M, int E, float eps, cudaStream_t stream);

// Fused-residual variant. Used between layers:
//   sum = x + residual
//   y = rmsnorm(sum) * w
//   out_residual = sum
// Shapes match rmsnorm_fp16; out_residual may alias x.
void rmsnorm_residual_fp16(const __half* x, const __half* residual,
                           const __half* w, __half* y, __half* out_residual,
                           int M, int E, float eps, cudaStream_t stream);

// ---------- RoPE ----------
//
// Applied in-place on Q and K after qkv_proj. Shapes: Q,K both
// [M, NH, HD] flat (M = B*S). pos [M] int32. cos,sin [max_seq_len, HD/2] FP32.
void rope_apply_qk_fp16(__half* Q, __half* K, const int32_t* pos,
                        const float* cos, const float* sin,
                        int M, int num_heads, int head_dim, int rope_max_seq,
                        cudaStream_t stream);

// ---------- SwiGLU ----------
//
// Given gate_up [M, 2*d_ff] (concatenation of gate, up along last dim),
// computes y[M, d_ff] = silu(gate) * up.
void swiglu_fp16(const __half* gate_up, __half* y,
                 int M, int d_ff, cudaStream_t stream);

// ---------- Mish ----------
//
// In-place mish(x) = x * tanh(softplus(x)). Used by value-head MLP hidden.
void mish_inplace_fp16(__half* x, int N, cudaStream_t stream);

// ---------- Embedding + Fourier override ----------
//
// h[b,t,:] = wl_pos[b,t] ? fourier(wl_val[b,t])
//          : d_pos[b,t]  ? fourier(d_val[b,t])
//          :                tok_emb[input_ids[b,t], :]
//
// For the common path (no overrides), this is a single gather. The wl/d
// override path runs the Fourier MLP on the marked positions only.
//
// Pass nullptr for {wl_pos, d_pos, wl_val, d_val} to skip the override step.
void embed_with_fourier_fp16(
    const int32_t* input_ids,           // [M] int32
    const __half* tok_embedding,        // [V, E]
    const bool* wl_pos, const bool* d_pos,           // [M] (or nullptr)
    const __half* wl_val, const __half* d_val,       // [M] FP16 (or nullptr)
    const __half* fourier_freq,         // [F]
    const __half* fourier_proj_w,       // [E, 2*F]
    const __half* fourier_proj_b,       // [E]
    __half* out,                         // [M, E]
    int M, int E, int V, int F,
    cudaStream_t stream);

// ---------- KV write ----------
//
// After qkv_proj+rope produces K,V of shape [M, NH, HD] (M=B*S), scatter into
//   K_cache[layer, b, :, past_len[b]+s, :] and same for V.
// new_k, new_v are [B, S, NH, HD].
//
// Strides assume the layout in kv_cache.hpp: contiguous
// [num_layers, B, NH, max_seq_len, HD].
void kv_scatter_fp16(const __half* new_k, const __half* new_v,
                     __half* K_cache, __half* V_cache,
                     const int32_t* past_len,         // [B] int32
                     const int32_t* slot_active,      // [B] int32 (0/1) — gates write
                     int B, int S, int num_heads, int head_dim,
                     int max_seq_len, int layer_idx, int num_layers,
                     cudaStream_t stream);

// Increment past_len[b] by S for active slots only. One launch per layer block.
void past_len_increment(int32_t* past_len, const int32_t* slot_active,
                        int S, int B, cudaStream_t stream);

// ---------- Sampler ----------
//
// Argmax over [B, V] -> idx_out[B] int32. Assumes V fits in shared memory of
// 1024 threads × 4 bytes; we use a multi-block reduction otherwise.
void argmax_fp16(const __half* logits, int32_t* idx_out,
                 int B, int V, cudaStream_t stream);

// Fused argmax + LUT lookup. sub_idx_out gets the argmax sub-vocab idx;
// full_idx_out gets lut[sub_idx_out]. Used to chain sampling on-device
// without a host roundtrip (the BOARD loop's 68-step generation).
void argmax_lut_scatter_fp16(const __half* logits, const int32_t* lut,
                             int32_t* sub_idx_out, int32_t* full_idx_out,
                             int B, int V, cudaStream_t stream);

// Multinomial via Gumbel-max:
//   g[i] = logits[i]/T + (-log(-log(u[i])))     (u uniform in (0,1))
//   idx = argmax_i g[i]
// Equivalent in distribution to softmax-multinomial. Philox state is per-slot.
void gumbel_argmax_fp16(const __half* logits,
                        uint64_t* philox_seed,    // [B] u64 — base seed
                        uint64_t* philox_offset,  // [B] u64 — incremented each call
                        int32_t* idx_out,
                        float temperature,
                        int B, int V, cudaStream_t stream);

// Apply legal-mask in-place: logits[b, v] <- -65504 if !legal[b, v].
void apply_legal_mask_fp16(__half* logits, const bool* legal,
                           int B, int V, cudaStream_t stream);

// Compute log-prob of a chosen index after softmax over (logits/T).
// out_logp[b] = log_softmax(logits[b] / T)[idx[b]].
void log_prob_at_idx_fp16(const __half* logits, const int32_t* idx,
                          float temperature,
                          float* out_logp,    // [B] FP32
                          int B, int V, cudaStream_t stream);

// ---------- GEMM wrapper (FP16) ----------
//
// Out [M,N] = A [M,K] @ B^T [N,K] (+ bias [N] if not nullptr).
// Layout: row-major. PyTorch convention — weight stored as [out_dim, in_dim].
// FP16 in, FP32 acc, FP16 out.
//
// `workspace` must be ≥ gemm_fp16_workspace_bytes(M,N,K) bytes.
void gemm_fp16(const __half* A, const __half* B_w, const __half* bias,
               __half* C, int M, int N, int K,
               void* workspace, std::size_t workspace_bytes,
               cudaStream_t stream);

std::size_t gemm_fp16_workspace_bytes(int M, int N, int K);

// FP32-output variant for the head paths where we want exact softmax later.
void gemm_fp16_out_fp32(const __half* A, const __half* B_w, const __half* bias,
                        float* C, int M, int N, int K,
                        void* workspace, std::size_t workspace_bytes,
                        cudaStream_t stream);

// CUTLASS-backed FP16 GEMM (alternative implementation, env-var selected via
// USE_CUTLASS_GEMM=1 inside gemm_fp16). Exposed directly for callers that
// want fused-residual without going through the env switch.
void gemm_fp16_cutlass(const __half* A, const __half* B_w, const __half* bias,
                       __half* D, int M, int N, int K,
                       cudaStream_t stream);

// D[M,N] = A @ B_w^T + residual[M,N]. Single-pass fused epilogue.
// Equivalent to gemm_fp16(...) followed by elementwise D += residual, but
// the residual read happens inside the GEMM epilogue (one pass).
void gemm_fp16_cutlass_residual(const __half* A, const __half* B_w,
                                const __half* residual,
                                __half* D, int M, int N, int K,
                                cudaStream_t stream);

// ---------- FlashAttention ----------
//
// Decode (S_q = 1). Q [B, NH, HD]. K/V cache [num_layers, B, NH, max_seq, HD].
// past_len[B] = current valid length per slot. active[B] in {0,1} gates work.
// scale = 1/sqrt(head_dim).
void fmha_decode_dispatch(const __half* Q, const __half* K_cache, const __half* V_cache,
                          const int32_t* past_len, const int32_t* active,
                          __half* O,
                          int B, int NH, int HD,
                          int max_seq_len, int layer_idx, float scale,
                          cudaStream_t stream);

// Prefill (S_q variable). Q [B, S, NH, HD], K/V [B, S, NH, HD] (just the
// new K/V — past is empty for init prefill). Mask is block-aware:
//   mask[b, i, j] = (block_id[b,i] == block_id[b,j]) || (j <= i)
// where j is the key index (same block as i, so always within S).
//
// `S` is the number of new tokens; this is a pure self-attention over the
// new block only. Used at FEN init / refill.
//
// O [B, S, NH, HD].
void fmha_prefill_dispatch(const __half* Q, const __half* K, const __half* V,
                           const int32_t* block_id,    // [B, S] int32
                           const int32_t* active,      // [B] int32
                           __half* O,
                           int B, int S, int NH, int HD,
                           float scale, cudaStream_t stream);

// CUTLASS Blackwell FMHA (sm_100a, TMA + tensor cores). Causal mask only in
// J.2; block-aware mask is a follow-up. Caller must allocate workspace and
// LSE buffer of the sizes returned below.
void fmha_prefill_cutlass_causal(const __half* Q, const __half* K, const __half* V,
                                 __half* O,
                                 int B, int S, int NH, int HD, float scale,
                                 void* workspace, void* lse_buf,
                                 cudaStream_t stream);

std::size_t fmha_prefill_cutlass_workspace_bytes(int B, int S, int NH, int HD);
std::size_t fmha_prefill_cutlass_lse_elements(int B, int S, int NH);

// ---------- Misc ----------

// out[b] = centers[idx[b]]. centers FP32, idx int32, out FP32.
void gather_bucket_center(const float* centers, const int32_t* idx,
                          float* out, int B, cudaStream_t stream);

// FP32 -> FP16 cast (used by misc paths).
void cast_fp32_to_fp16(const float* in, __half* out, int N, cudaStream_t stream);

// For each slot b where active[b]==0, restore dst[b, :] from backup[b, :].
// Used after a batched forward to preserve inactive slots' last_h.
void restore_inactive_last_h(__half* dst, const __half* backup,
                             const int32_t* active, int B, int E,
                             cudaStream_t stream);

}  // namespace cutlass_engine

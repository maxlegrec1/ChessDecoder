#include "cutlass_engine/layers.hpp"
#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cmath>

namespace cutlass_engine {

namespace {

__global__ void split_qkv_kernel(const __half* qkv,
                                 __half* Q, __half* K, __half* V,
                                 int M, int E) {
    int m = blockIdx.x;
    int e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const __half* row = qkv + m * (3 * E);
    Q[m * E + e] = row[e];
    K[m * E + e] = row[E + e];
    V[m * E + e] = row[2 * E + e];
}

void split_qkv(const __half* qkv, __half* Q, __half* K, __half* V,
               int M, int E, cudaStream_t stream) {
    constexpr int TX = 256;
    dim3 block(TX);
    dim3 grid(M, (E + TX - 1) / TX);
    split_qkv_kernel<<<grid, block, 0, stream>>>(qkv, Q, K, V, M, E);
    CE_CUDA_LAST();
}

}  // namespace

void allocate_layer_workspace(LayerWorkspace& ws, const ModelConfig& cfg,
                              int max_M, Arena& arena) {
    const int E = cfg.embed_dim;
    const int d_ff = cfg.d_ff;
    ws.h_in     = arena.allocT<__half>(max_M * E);
    ws.h_out    = arena.allocT<__half>(max_M * E);
    ws.residual = arena.allocT<__half>(max_M * E);
    ws.qkv      = arena.allocT<__half>(max_M * 3 * E);
    ws.attn_out = arena.allocT<__half>(max_M * E);
    ws.gate_up  = arena.allocT<__half>(max_M * 2 * d_ff);
    ws.mlp_inner= arena.allocT<__half>(max_M * d_ff);
    ws.q_buf    = arena.allocT<__half>(max_M * E);
    ws.k_buf    = arena.allocT<__half>(max_M * E);
    ws.v_buf    = arena.allocT<__half>(max_M * E);
    ws.pos      = arena.allocT<int32_t>(max_M);
}

void attention_block_forward(const LayerContext& ctx, const LayerWeights& Lw,
                             LayerWorkspace& ws, KvCache& kv, int layer_idx,
                             ForwardMode mode, int B, int S,
                             const int32_t* block_id, cudaStream_t stream,
                             bool write_kv_in_prefill) {
    const auto& cfg = *ctx.cfg;
    const auto& w   = *ctx.w;
    const int M = B * S;
    const int E = cfg.embed_dim;
    const int NH = cfg.num_heads;
    const int HD = cfg.head_dim;

    rmsnorm_residual_fp16(ws.h_in, ws.residual, Lw.sa_norm,
                          ws.h_out, ws.residual,
                          M, E, 1e-6f, stream);

    gemm_fp16(ws.h_out, Lw.qkv_w, nullptr,
              ws.qkv, M, 3 * E, E, nullptr, 0, stream);

    split_qkv(ws.qkv, ws.q_buf, ws.k_buf, ws.v_buf, M, E, stream);

    rope_apply_qk_fp16(ws.q_buf, ws.k_buf, ws.pos,
                       w.rope_cos, w.rope_sin,
                       M, NH, HD, cfg.max_seq_len, stream);

    const float scale = 1.0f / std::sqrt(float(HD));
    if (mode == ForwardMode::Decode) {
        kv_scatter_fp16(ws.k_buf, ws.v_buf, kv.K(), kv.V(),
                        kv.past_len(), kv.slot_active(),
                        B, S, NH, HD, cfg.max_seq_len, layer_idx,
                        cfg.num_layers, stream);
        fmha_decode_dispatch(ws.q_buf, kv.K(), kv.V(),
                             kv.past_len(), kv.slot_active(),
                             ws.attn_out, B, NH, HD, cfg.max_seq_len,
                             layer_idx, scale, stream);
    } else {
        if (write_kv_in_prefill) {
            // Init / refill prefill: also populate the KV cache so subsequent
            // forward_decode calls can read these positions.  Writes at
            // past_len[b]..past_len[b]+S for active slots.
            kv_scatter_fp16(ws.k_buf, ws.v_buf, kv.K(), kv.V(),
                            kv.past_len(), kv.slot_active(),
                            B, S, NH, HD, cfg.max_seq_len, layer_idx,
                            cfg.num_layers, stream);
        }
        fmha_prefill_dispatch(ws.q_buf, ws.k_buf, ws.v_buf, block_id,
                              kv.slot_active(), ws.attn_out,
                              B, S, NH, HD, scale, stream);
    }

    gemm_fp16(ws.attn_out, Lw.out_w, nullptr,
              ws.h_in, M, E, E, nullptr, 0, stream);
}

void mlp_block_forward(const LayerContext& ctx, const LayerWeights& Lw,
                       LayerWorkspace& ws, int M, cudaStream_t stream) {
    const auto& cfg = *ctx.cfg;
    const int E = cfg.embed_dim;
    const int d_ff = cfg.d_ff;

    rmsnorm_residual_fp16(ws.h_in, ws.residual, Lw.mlp_norm,
                          ws.h_out, ws.residual,
                          M, E, 1e-6f, stream);

    gemm_fp16(ws.h_out, Lw.gate_up_w, nullptr,
              ws.gate_up, M, 2 * d_ff, E, nullptr, 0, stream);

    swiglu_fp16(ws.gate_up, ws.mlp_inner, M, d_ff, stream);

    gemm_fp16(ws.mlp_inner, Lw.down_w, nullptr,
              ws.h_in, M, E, d_ff, nullptr, 0, stream);
}

void transformer_layer_forward(const LayerContext& ctx, const LayerWeights& Lw,
                               LayerWorkspace& ws, KvCache& kv, int layer_idx,
                               ForwardMode mode, int B, int S,
                               const int32_t* block_id, cudaStream_t stream,
                               bool write_kv_in_prefill) {
    attention_block_forward(ctx, Lw, ws, kv, layer_idx, mode, B, S, block_id,
                            stream, write_kv_in_prefill);
    mlp_block_forward(ctx, Lw, ws, B * S, stream);
}

}  // namespace cutlass_engine

#include "cutlass_engine/model.hpp"
#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

namespace cutlass_engine {

void ChessDecoderModel::initialize(const ModelConfig& cfg, const ModelWeights& w,
                                   Arena& arena, int max_M) {
    cfg_ = &cfg;
    w_ = &w;
    ctx_.cfg = &cfg;
    ctx_.w   = &w;
    allocate_layer_workspace(ws_, cfg, max_M, arena);
}

void ChessDecoderModel::forward_decode(const int32_t* ids,
                                       const int32_t* pos,
                                       const bool* wl_pos, const bool* d_pos,
                                       const __half* wl_val, const __half* d_val,
                                       KvCache& kv,
                                       __half* out_h,
                                       cudaStream_t stream) {
    const int B = cfg_->batch_size;
    const int S = 1;
    const int M = B * S;
    const int E = cfg_->embed_dim;

    // 1. Embedding (with optional Fourier override).
    embed_with_fourier_fp16(ids, w_->tok_embedding,
                            wl_pos, d_pos, wl_val, d_val,
                            w_->fourier_freq, w_->fourier_proj_w, w_->fourier_proj_b,
                            ws_.h_in, M, E, cfg_->vocab_size, cfg_->num_fourier_freq,
                            stream);

    // 2. Initialize residual stream to zero — will be filled by rmsnorm_residual.
    CE_CUDA_CHECK(cudaMemsetAsync(ws_.residual, 0, M * E * sizeof(__half), stream));

    // 3. Copy pos → ws_.pos (one D2D memcpy of B int32s).
    CE_CUDA_CHECK(cudaMemcpyAsync(ws_.pos, pos, M * sizeof(int32_t),
                                  cudaMemcpyDeviceToDevice, stream));

    // 4. Run all 12 layers.
    for (int li = 0; li < cfg_->num_layers; ++li) {
        transformer_layer_forward(ctx_, w_->layers[li], ws_, kv, li,
                                  ForwardMode::Decode, B, S, /*block_id=*/nullptr,
                                  stream);
    }

    // 5. Final norm: out = rmsnorm(h_in + residual) * w_final
    //    We use the fused-residual variant; the residual is the running skip.
    rmsnorm_residual_fp16(ws_.h_in, ws_.residual, w_->final_norm,
                          out_h, ws_.residual, M, E, 1e-6f, stream);

    // 6. Increment past_len for active slots.
    past_len_increment(kv.past_len(), kv.slot_active(), S, B, stream);
}

void ChessDecoderModel::forward_prefill_block(const int32_t* ids,
                                              const int32_t* pos,
                                              const int32_t* block_id,
                                              const bool* wl_pos, const bool* d_pos,
                                              const __half* wl_val, const __half* d_val,
                                              int B, int S,
                                              KvCache& /*kv*/,
                                              __half* out_h,
                                              cudaStream_t stream) {
    const int M = B * S;
    const int E = cfg_->embed_dim;

    embed_with_fourier_fp16(ids, w_->tok_embedding,
                            wl_pos, d_pos, wl_val, d_val,
                            w_->fourier_freq, w_->fourier_proj_w, w_->fourier_proj_b,
                            ws_.h_in, M, E, cfg_->vocab_size, cfg_->num_fourier_freq,
                            stream);
    CE_CUDA_CHECK(cudaMemsetAsync(ws_.residual, 0, M * E * sizeof(__half), stream));
    CE_CUDA_CHECK(cudaMemcpyAsync(ws_.pos, pos, M * sizeof(int32_t),
                                  cudaMemcpyDeviceToDevice, stream));

    // Prefill mode: KV cache is NOT used by attention; the caller will scatter
    // the resulting K/V into cache themselves once the block is finalized. We
    // pass a temporary kv-like wrapper for the API contract.
    KvCache dummy_kv;  // not allocated; only past_len/slot_active needed for active mask
    // Reuse the caller's kv slot_active by faking it — but since prefill mode
    // doesn't read from cache, we just need slot_active to be valid. The clean
    // version threads it through; for now we let attention_block_forward dispatch
    // PrefillBlock and ignore dummy_kv's K/V buffers entirely.
    (void)dummy_kv;

    // FIXME: prefill currently reuses the live kv's slot_active mask via the
    // ForwardMode::PrefillBlock branch. The cleaner fix is a PrefillContext
    // that carries the active mask explicitly. For now the caller must pass
    // an externally-owned active vector; not wired in this scaffold.

    for (int li = 0; li < cfg_->num_layers; ++li) {
        transformer_layer_forward(ctx_, w_->layers[li], ws_, dummy_kv, li,
                                  ForwardMode::PrefillBlock, B, S, block_id,
                                  stream);
    }
    rmsnorm_residual_fp16(ws_.h_in, ws_.residual, w_->final_norm,
                          out_h, ws_.residual, M, E, 1e-6f, stream);
}

}  // namespace cutlass_engine

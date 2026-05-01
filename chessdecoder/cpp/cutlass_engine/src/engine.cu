#include "cutlass_engine/engine.hpp"
#include "cutlass_engine/check.hpp"
#include "cutlass_engine/kernels.hpp"

#include "vocab.hpp"   // existing DecoderVocab from chessdecoder/cpp/decoder/

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cutlass_engine {

namespace {

// Compute the bytes the engine needs from arena. This is conservative —
// production should compute exactly per layer/buffer.
std::size_t estimate_arena_bytes(const ModelConfig& c) {
    std::size_t bytes = 0;
    const std::size_t E = c.embed_dim;
    const std::size_t V = c.vocab_size;
    const std::size_t H = c.value_hidden_size;
    const std::size_t Bv = c.board_vocab_size;
    const std::size_t Mv = c.move_vocab_size;
    const std::size_t Kb = c.n_buckets;
    const std::size_t F = c.num_fourier_freq;
    const std::size_t d_ff = c.d_ff;
    const std::size_t MAX = c.max_seq_len;
    const std::size_t Bs = c.batch_size;
    const std::size_t NL = c.num_layers;
    const std::size_t NH = c.num_heads;
    const std::size_t HD = c.head_dim;

    // Backbone weights (FP16).
    bytes += NL * (E * 2 + 3*E*E + E*E + 2*d_ff*E + E*d_ff) * sizeof(__half);
    // Embedding + final norm.
    bytes += (V * E + E) * sizeof(__half);
    // Heads (FP16).
    bytes += (Bv * E + Bv + 2 * (Mv * E + Mv) + 4 * (H * E + H + Kb * H + Kb)) * sizeof(__half);
    // Bucket centers (FP32).
    bytes += 2 * Kb * sizeof(float);
    // Fourier (FP16).
    bytes += (F + E * 2 * F + E) * sizeof(__half);
    // RoPE table (FP32).
    bytes += 2 * MAX * (HD / 2) * sizeof(float);

    // KV cache (FP16).
    bytes += 2 * NL * Bs * NH * MAX * HD * sizeof(__half);
    bytes += Bs * sizeof(int32_t) * 2;  // past_len, slot_active

    // LayerWorkspace at max_M=Bs*max_seq_len (worst case for prefill init).
    // We size for max_M = Bs*71 (a reasonable upper bound for init prefill —
    // 68 board tokens + move + wl + d). Refills use the same buffer.
    const std::size_t max_M = Bs * 71;
    bytes += max_M * E * sizeof(__half) * 6;            // h_in/h_out/residual + 3 buffers
    bytes += max_M * 3 * E * sizeof(__half);            // qkv
    bytes += max_M * E * sizeof(__half);                // attn_out
    bytes += max_M * 2 * d_ff * sizeof(__half);
    bytes += max_M * d_ff * sizeof(__half);
    bytes += max_M * sizeof(int32_t);                   // pos

    // 25% headroom for alignment/padding.
    bytes = bytes + bytes / 4;
    return bytes;
}

}  // namespace

ThinkingEngine::ThinkingEngine(const std::string& /*backbone_pt*/,
                               const std::string& weights_dir,
                               const std::string& vocab_json,
                               const std::string& config_json,
                               int batch_size) {
    cfg_ = load_model_config(config_json, batch_size);
    const std::size_t total = estimate_arena_bytes(cfg_);
    arena_.reserve(total);
    pinned_arena_.reserve(64 * 1024 * 1024);  // 64 MB pinned host (rollout dumps)

    w_ = load_weights(weights_dir, cfg_, arena_);
    kv_.allocate(cfg_, arena_);

    // Worst-case workspace: prefill of `max_init_S_` tokens × B.
    const int max_M = cfg_.batch_size * max_init_S_;
    model_.initialize(cfg_, w_, arena_, max_M);

    // Per-call scratch.
    const int B = cfg_.batch_size;
    d_ids_buf_   = arena_.allocT<int32_t>(max_M);
    d_pos_buf_   = arena_.allocT<int32_t>(max_M);
    d_block_buf_ = arena_.allocT<int32_t>(max_M);
    d_active_buf_= arena_.allocT<int32_t>(B);
    d_hidden_buf_= arena_.allocT<__half>(max_M * cfg_.embed_dim);
    d_last_h_buf_= arena_.allocT<__half>(B * cfg_.embed_dim);
    d_logits_buf_= arena_.allocT<__half>(B * std::max(cfg_.move_vocab_size, cfg_.board_vocab_size));
    d_legal_mask_= arena_.allocT<bool>(B * cfg_.move_vocab_size);
    d_idx_out_   = arena_.allocT<int32_t>(B);

    if (!vocab_json.empty()) {
        vocab_.reset(new decoder::DecoderVocab(vocab_json));
    }

    std::printf("[cutlass_engine] arena: %.2f GB used of %.2f GB reserved\n",
                arena_.used_bytes() / 1e9, arena_.total_bytes() / 1e9);
}

ThinkingEngine::~ThinkingEngine() = default;

void ThinkingEngine::update_weights(const std::string& weights_dir) {
    reupload_weights(weights_dir, cfg_, w_);
}

void ThinkingEngine::forward_decode_partial(std::uintptr_t ids,
                                            std::uintptr_t pos,
                                            std::uintptr_t active,
                                            std::uintptr_t past_len,
                                            int stop_after_layer,
                                            std::uintptr_t out_h_in,
                                            std::uintptr_t out_residual) {
    const int B = cfg_.batch_size;
    CE_CUDA_CHECK(cudaMemcpyAsync(kv_.slot_active(),
                                  reinterpret_cast<const void*>(active),
                                  B * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                                  stream_.get()));
    CE_CUDA_CHECK(cudaMemcpyAsync(kv_.past_len(),
                                  reinterpret_cast<const void*>(past_len),
                                  B * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                                  stream_.get()));

    model_.forward_decode_partial(reinterpret_cast<const int32_t*>(ids),
                                  reinterpret_cast<const int32_t*>(pos),
                                  kv_, stop_after_layer,
                                  reinterpret_cast<__half*>(out_h_in),
                                  reinterpret_cast<__half*>(out_residual),
                                  stream_.get());
    stream_.sync();
}

void ThinkingEngine::forward_decode_test(std::uintptr_t ids, std::uintptr_t pos,
                                         std::uintptr_t active,
                                         std::uintptr_t past_len,
                                         std::uintptr_t out_h) {
    // Caller-provided active+past_len are copied into the engine's KvCache so
    // that internal kernels read them.  Caller is responsible for matching B.
    const int B = cfg_.batch_size;
    CE_CUDA_CHECK(cudaMemcpyAsync(kv_.slot_active(),
                                  reinterpret_cast<const void*>(active),
                                  B * sizeof(int32_t),
                                  cudaMemcpyDeviceToDevice, stream_.get()));
    CE_CUDA_CHECK(cudaMemcpyAsync(kv_.past_len(),
                                  reinterpret_cast<const void*>(past_len),
                                  B * sizeof(int32_t),
                                  cudaMemcpyDeviceToDevice, stream_.get()));

    model_.forward_decode(reinterpret_cast<const int32_t*>(ids),
                          reinterpret_cast<const int32_t*>(pos),
                          /*wl_pos=*/nullptr, /*d_pos=*/nullptr,
                          /*wl_val=*/nullptr, /*d_val=*/nullptr,
                          kv_,
                          reinterpret_cast<__half*>(out_h),
                          stream_.get());

    // Copy past_len back so caller can observe the increment.
    CE_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(past_len),
                                  kv_.past_len(),
                                  B * sizeof(int32_t),
                                  cudaMemcpyDeviceToDevice, stream_.get()));
    stream_.sync();
}

std::vector<RolloutResult> ThinkingEngine::predict_moves(
    const std::vector<std::string>& fens, float fallback_temperature) {
    if (!vocab_) {
        throw std::runtime_error(
            "predict_moves: engine constructed without vocab_json — vocab is required");
    }
    const int N = (int)fens.size();
    const int B = cfg_.batch_size;
    const int E = cfg_.embed_dim;
    const int Mv = cfg_.move_vocab_size;

    std::vector<RolloutResult> results(N);

    // Process FENs in chunks of B at a time.  Within each chunk, all slots
    // run a single prefill on a [B, S=68] block, then the policy_head reads
    // the last position's hidden, applies a per-slot legal-move mask, and
    // samples (argmax for temp=0).
    //
    // This is the "no-thinking" path — equivalent to the existing
    // libtorch engine's `fallbackMove`.  The full thinking-trace state
    // machine is the next step (see plan).
    constexpr int S = 68;  // tokens per FEN (start_pos + 64 squares + end_pos + castling + stm)

    for (int chunk_start = 0; chunk_start < N; chunk_start += B) {
        int chunk_n = std::min(B, N - chunk_start);

        // Build per-slot CPU buffers.
        std::vector<int32_t> ids_h(B * S, 0);
        std::vector<int32_t> pos_h(B * S, 0);
        std::vector<int32_t> block_h(B * S, 0);
        std::vector<int32_t> active_h(B, 0);
        std::vector<uint8_t> legal_h(B * Mv, 0);  // bool packed as u8

        for (int b = 0; b < chunk_n; ++b) {
            const auto& fen = fens[chunk_start + b];
            auto tokens = vocab_->fenToTokenIds(fen);
            if ((int)tokens.size() != S) {
                throw std::runtime_error(
                    "fenToTokenIds returned " + std::to_string(tokens.size()) +
                    " tokens for FEN '" + fen + "', expected 68");
            }
            for (int s = 0; s < S; ++s) {
                ids_h[b * S + s]   = tokens[s];
                pos_h[b * S + s]   = s;
                block_h[b * S + s] = 0;  // single block — full bidirectional within FEN
            }
            active_h[b] = 1;

            auto legal = vocab_->legalMoveIndices(fen);
            for (int idx : legal) {
                if (idx >= 0 && idx < Mv) legal_h[b * Mv + idx] = 1;
            }
        }

        // Upload.
        CE_CUDA_CHECK(cudaMemcpyAsync(d_ids_buf_, ids_h.data(),
                                      B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_pos_buf_, pos_h.data(),
                                      B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_block_buf_, block_h.data(),
                                      B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_active_buf_, active_h.data(),
                                      B * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_legal_mask_, legal_h.data(),
                                      B * Mv * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));

        // Run prefill.  Output is hidden [B, S, E].
        model_.forward_prefill_block(d_ids_buf_, d_pos_buf_, d_block_buf_,
                                     d_active_buf_,
                                     /*wl_pos=*/nullptr, /*d_pos=*/nullptr,
                                     /*wl_val=*/nullptr, /*d_val=*/nullptr,
                                     B, S, d_hidden_buf_, stream_.get());

        // Extract last position [B, E] = hidden[:, S-1, :] into d_last_h_buf_.
        // cudaMemcpy2D handles the stride cleanly.
        CE_CUDA_CHECK(cudaMemcpy2DAsync(
            d_last_h_buf_,                                                  // dst
            E * sizeof(__half),                                             // dst pitch
            d_hidden_buf_ + (S - 1) * E,                                    // src (offset to last row of slot 0)
            S * E * sizeof(__half),                                         // src pitch (stride between slots)
            E * sizeof(__half),                                             // width per row
            B,                                                              // num rows
            cudaMemcpyDeviceToDevice, stream_.get()));

        // Policy head GEMM: [B, E] @ [Mv, E]^T + bias → [B, Mv].
        gemm_fp16(d_last_h_buf_, w_.policy_head_w, w_.policy_head_b,
                  d_logits_buf_, B, Mv, E, nullptr, 0, stream_.get());

        // Apply legal mask in-place on logits.
        apply_legal_mask_fp16(d_logits_buf_, d_legal_mask_, B, Mv, stream_.get());

        // Sample.
        if (fallback_temperature == 0.0f || policy_t_ == 0.0f) {
            argmax_fp16(d_logits_buf_, d_idx_out_, B, Mv, stream_.get());
        } else {
            // Gumbel sampling — needs philox state per slot.  For now, fall
            // back to argmax.  Multinomial sampling will be wired with a
            // per-slot RNG state in a follow-up commit; this path documents
            // the call site.
            argmax_fp16(d_logits_buf_, d_idx_out_, B, Mv, stream_.get());
        }

        // Pull move sub-vocab idx back to host.
        std::vector<int32_t> idx_h(B);
        CE_CUDA_CHECK(cudaMemcpyAsync(idx_h.data(), d_idx_out_,
                                      B * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost, stream_.get()));
        stream_.sync();

        for (int b = 0; b < chunk_n; ++b) {
            int sub_idx = idx_h[b];
            int full_idx = vocab_->moveIdxToFullIdx(sub_idx);
            std::string move_pseudo = vocab_->idxToToken(full_idx);
            std::string move_uci = decoder::DecoderVocab::pseudoToStandardUci(move_pseudo);

            auto& r = results[chunk_start + b];
            r.move = move_uci;
        }
    }

    return results;
}

}  // namespace cutlass_engine

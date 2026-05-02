// Full thinking-trace state machine for ThinkingEngine.
//
// Phase E.2: rewritten to use forward_decode (causal, S=1, KV cache) for
// every state-machine step.  Init prefill writes K/V to the cache once
// (via forward_prefill_block(kv_for_write=&kv_)).  Subsequent appends are
// O(S) instead of O(S^2).
//
// Mathematical equivalence to Python's run_thinking (which uses prefix
// mode for MOVE/WL/D and causal mode for BOARD): in this model, every
// position the engine reads (start_think, move, wl_value, d_value,
// end_think, end_var) is in its own unique orphan block.  Prefix-mode
// attention at an orphan position is identical to causal-mode attention
// at the same position (no within-block bidirectional companions exist).
// So forward_decode (causal) gives the same hidden state Python's
// prefix_forward gives at these positions.

#include "cutlass_engine/engine.hpp"
#include "cutlass_engine/check.hpp"
#include "cutlass_engine/kernels.hpp"

#include "vocab.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

namespace cutlass_engine {

namespace {

std::vector<float> d2h_fp32(const float* dptr, int B) {
    std::vector<float> out(B);
    cudaMemcpy(out.data(), dptr, B * sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

std::vector<int32_t> d2h_int32(const int32_t* dptr, int B) {
    std::vector<int32_t> out(B);
    cudaMemcpy(out.data(), dptr, B * sizeof(int32_t), cudaMemcpyDeviceToHost);
    return out;
}

}  // namespace

std::vector<RolloutResult> ThinkingEngine::predict_moves_thinking(
    const std::vector<std::string>& fens, float /*temperature*/,
    int max_seq_len_cap, int max_iters) {

    if (!vocab_) {
        throw std::runtime_error("predict_moves_thinking: engine needs vocab_json");
    }
    const int N = (int)fens.size();
    if (N == 0) return {};
    const int B = cfg_.batch_size;
    const int E = cfg_.embed_dim;
    const int Mv = cfg_.move_vocab_size;
    const int Bv = cfg_.board_vocab_size;
    const int Kb = cfg_.n_buckets;
    const int H = cfg_.value_hidden_size;
    const int max_S = std::min(max_seq_len_cap, cfg_.max_seq_len);

    // Special-token IDs.
    const int wl_value_idx       = vocab_->wlValueIdx();
    const int d_value_idx        = vocab_->dValueIdx();
    const int start_think_idx    = vocab_->startThinkIdx();
    const int board_end_var_sub  = vocab_->boardEndVarIdx();
    const int board_end_think_sub= vocab_->boardEndThinkIdx();

    std::vector<float> wl_centers_h = d2h_fp32(w_.wl_bucket_centers, Kb);
    std::vector<float> d_centers_h  = d2h_fp32(w_.d_bucket_centers, Kb);

    std::vector<int> move_sub_to_full(Mv);
    for (int i = 0; i < Mv; ++i) move_sub_to_full[i] = vocab_->moveIdxToFullIdx(i);
    std::vector<int> board_sub_to_full(Bv);
    for (int i = 0; i < Bv; ++i) board_sub_to_full[i] = vocab_->boardIdxToFullIdx(i);

    std::vector<RolloutResult> all_results(N);

    // Per-slot host state.
    std::vector<std::vector<int32_t>> token_ids(B);
    std::vector<std::vector<int32_t>> block_ids(B);
    std::vector<int> next_block(B, 0);
    std::vector<int> orphan_ctr(B, 10000);
    std::vector<bool> active(B, false);
    std::vector<bool> in_variation(B, false);
    std::vector<bool> ended_thinking(B, false);
    std::vector<bool> truncated(B, false);
    std::vector<int> slot_to_fen(B, -1);
    std::vector<int> slot_iter_count(B, 0);
    std::vector<int32_t> slot_past_len_h(B, 0);  // host mirror of kv_.past_len()

    // For the refill path, we track which slots were just initialized and
    // skip them in the main step (so they only contribute to MOVE going forward).

    // Allocate a small slot-init scratch (per-slot 1 token) on host that we
    // reuse each step.
    std::vector<int32_t> step_ids(B, 0);
    std::vector<int32_t> step_pos(B, 0);
    std::vector<int32_t> step_active(B, 0);
    std::vector<uint8_t> step_wl_pos(B, 0);
    std::vector<uint8_t> step_d_pos(B, 0);
    std::vector<__half> step_wl_val(B, __float2half(0.0f));
    std::vector<__half> step_d_val(B, __float2half(0.0f));

    // Pending queue (LIFO over reversed inputs → submission-order pops).
    std::vector<int> pending;
    pending.reserve(N);
    for (int i = N - 1; i >= 0; --i) pending.push_back(i);
    auto pop_pending = [&]() -> int {
        if (pending.empty()) return -1;
        int fid = pending.back(); pending.pop_back(); return fid;
    };

    auto reset_slot = [&](int b, int fid) {
        token_ids[b].clear();
        block_ids[b].clear();
        next_block[b] = 0;
        orphan_ctr[b] = 10000;
        active[b] = false;
        in_variation[b] = false;
        ended_thinking[b] = false;
        truncated[b] = false;
        slot_iter_count[b] = 0;
        slot_past_len_h[b] = 0;
        if (fid < 0) {
            slot_to_fen[b] = -1;
            return;
        }
        const auto& fen = fens[fid];
        auto root = vocab_->fenToTokenIds(fen);
        if ((int)root.size() != 68) {
            throw std::runtime_error("init: expected 68 root tokens, got " +
                                     std::to_string(root.size()));
        }
        int bid = next_block[b]++;
        for (int t : root) {
            token_ids[b].push_back(t);
            block_ids[b].push_back(bid);
        }
        token_ids[b].push_back(start_think_idx);
        block_ids[b].push_back(orphan_ctr[b]++);
        active[b] = true;
        in_variation[b] = true;
        slot_to_fen[b] = fid;
    };

    // ---------- Init prefill helper ----------
    // Runs forward_prefill_block on slots flagged in `init_slots[B]`, writing
    // K/V into the causal cache for those slots only.  Other slots' active=0
    // for this pass; their cache and past_len are untouched.
    //
    // After: slot b's past_len is advanced by 69 (init length); the last
    // position's hidden state ([B, E]) is gathered into d_th_last_h_.
    auto run_init_prefill = [&](const std::vector<int>& init_slots) {
        if (init_slots.empty()) return;
        const int S = 69;  // 68 board + start_think

        std::vector<int32_t> ids_h(B * S, 0);
        std::vector<int32_t> pos_h(B * S, 0);
        std::vector<int32_t> block_h(B * S, 0);
        std::vector<int32_t> active_h(B, 0);
        for (int b : init_slots) {
            for (int s = 0; s < S; ++s) {
                ids_h[b * S + s] = token_ids[b][s];
                pos_h[b * S + s] = s;
                // E.5: kv_ is the CAUSAL cache. Use unique-per-token block_ids
                // so forward_prefill_block runs in causal mode (mask = j<=i),
                // populating causal-chain K/V at every layer.
                block_h[b * S + s] = 1000000 + s;
            }
            active_h[b] = 1;
        }

        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_ids_, ids_h.data(),
                                      B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_pos_, pos_h.data(),
                                      B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_block_, block_h.data(),
                                      B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(kv_.slot_active(), active_h.data(),
                                      B * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        // Set past_len[b] = 0 for the slots we're about to init.
        std::vector<int32_t> past_len_buf(B);
        CE_CUDA_CHECK(cudaMemcpyAsync(past_len_buf.data(), kv_.past_len(),
                                      B * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost, stream_.get()));
        stream_.sync();
        for (int b : init_slots) past_len_buf[b] = 0;
        CE_CUDA_CHECK(cudaMemcpyAsync(kv_.past_len(), past_len_buf.data(),
                                      B * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));

        // Zero wl/d injection (no overrides during init).
        CE_CUDA_CHECK(cudaMemsetAsync(d_th_wl_pos_, 0, B * S * sizeof(bool),
                                      stream_.get()));
        CE_CUDA_CHECK(cudaMemsetAsync(d_th_d_pos_, 0, B * S * sizeof(bool),
                                      stream_.get()));

        // Run prefill with cache write.
        model_.forward_prefill_block(
            d_th_ids_, d_th_pos_, d_th_block_, kv_.slot_active(),
            d_th_wl_pos_, d_th_d_pos_, d_th_wl_val_, d_th_d_val_,
            B, S, d_th_hidden_, stream_.get(), &kv_);

        // Gather hidden at position S-1 (start_think) for inited slots
        // into d_th_last_h_.  Other slots' last_h is untouched.
        for (int b : init_slots) {
            CE_CUDA_CHECK(cudaMemcpyAsync(
                d_th_last_h_ + b * E,
                d_th_hidden_ + (b * S + (S - 1)) * E,
                E * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        // Sync slot_past_len_h.
        for (int b : init_slots) slot_past_len_h[b] = S;
    };

    // ---------- Forward-decode helper (one step over all active slots) ----------
    // Inputs: per-slot new token id and (optional) wl/d injection flag+value.
    // Effects: K/V scattered into cache at past_len[b]; past_len[b] += 1 for
    // active slots; new hidden state at the input position written to
    // d_th_last_h_[b, :] for active slots.
    auto run_decode_step = [&](const std::vector<int32_t>& new_ids,
                               const std::vector<uint8_t>& wl_pos,
                               const std::vector<uint8_t>& d_pos,
                               const std::vector<__half>& wl_val,
                               const std::vector<__half>& d_val,
                               const std::vector<int32_t>& act_mask) {
        // Build per-slot pos = past_len[b].
        std::vector<int32_t> pos_h(B, 0);
        for (int b = 0; b < B; ++b) pos_h[b] = slot_past_len_h[b];

        // Upload per-slot scalars.  Strided as [B, 1].
        // We use d_th_ids_, d_th_pos_, etc., which are [B, max_S].  Since
        // S=1, layout is just [B] in those buffers' first column.
        // Build [B*1] buffer with one element per slot.
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_ids_, new_ids.data(),
                                      B * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_pos_, pos_h.data(),
                                      B * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_wl_pos_, wl_pos.data(),
                                      B * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_d_pos_, d_pos.data(),
                                      B * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_wl_val_, wl_val.data(),
                                      B * sizeof(__half),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_d_val_, d_val.data(),
                                      B * sizeof(__half),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(kv_.slot_active(), act_mask.data(),
                                      B * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));

        // Backup last_h before the forward (so we can preserve it for slots
        // that don't run this step — forward_decode writes zeros for inactive
        // slots, which would clobber their stale-but-valid last_h).
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_last_h_bkp_, d_th_last_h_,
                                      B * E * sizeof(__half),
                                      cudaMemcpyDeviceToDevice, stream_.get()));

        // Phase I: graph captures forward_decode reading d_th_full_idx_ as
        // the input id buffer.  For run_decode_step, we have new_ids in
        // d_th_ids_ — copy to d_th_full_idx_ before launching the graph,
        // OR just call forward_decode directly when ids come from a different
        // buffer.  Simplest correct path: copy ids → full_idx, then graph.
        if (decode_graph_ready_) {
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_full_idx_, d_th_ids_,
                                          B * sizeof(int32_t),
                                          cudaMemcpyDeviceToDevice, stream_.get()));
            CE_CUDA_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_.get()));
        } else {
            model_.forward_decode(
                d_th_ids_, d_th_pos_,
                d_th_wl_pos_, d_th_d_pos_,
                d_th_wl_val_, d_th_d_val_,
                kv_, d_th_last_h_, stream_.get());
        }

        // Restore last_h for inactive slots from the backup.
        restore_inactive_last_h(d_th_last_h_, d_th_last_h_bkp_,
                                kv_.slot_active(), B, E, stream_.get());

        // forward_decode internally calls past_len_increment after all layers,
        // so kv_.past_len()[b] is now slot_past_len_h[b] + 1 for active slots.
        for (int b = 0; b < B; ++b) {
            if (act_mask[b]) slot_past_len_h[b] += 1;
        }
    };

    // ---------- Helpers to run one head + argmax ----------
    // Two variants per head: one reading the CAUSAL last_h (used for
    // BOARD/AFTER_*), one reading the PREFIX last_h (used for MOVE/WL/D/FINAL).
    //
    // Phase K: the *_with_lp variants also compute the log-probability of
    // the chosen index under softmax(logits/temp) for GRPO `old_log_probs`.
    using IdxLp = std::pair<std::vector<int32_t>, std::vector<float>>;
    auto run_head_argmax_move_vocab_prefix = [&](const __half* head_w, const __half* head_b) {
        gemm_fp16(d_th_last_h_prefix_, head_w, head_b, d_logits_buf_,
                  B, Mv, E, nullptr, 0, stream_.get());
        argmax_fp16(d_logits_buf_, d_idx_out_, B, Mv, stream_.get());
        stream_.sync();
        return d2h_int32(d_idx_out_, B);
    };
    auto run_head_argmax_move_vocab_prefix_with_lp = [&](const __half* head_w,
                                                         const __half* head_b,
                                                         float temp) -> IdxLp {
        gemm_fp16(d_th_last_h_prefix_, head_w, head_b, d_logits_buf_,
                  B, Mv, E, nullptr, 0, stream_.get());
        argmax_fp16(d_logits_buf_, d_idx_out_, B, Mv, stream_.get());
        log_prob_at_idx_fp16(d_logits_buf_, d_idx_out_, temp, d_lp_out_,
                             B, Mv, stream_.get());
        stream_.sync();
        return {d2h_int32(d_idx_out_, B), d2h_fp32(d_lp_out_, B)};
    };
    auto run_head_argmax_board_vocab_causal = [&]() {
        gemm_fp16(d_th_last_h_, w_.board_head_w, w_.board_head_b, d_logits_buf_,
                  B, Bv, E, nullptr, 0, stream_.get());
        argmax_fp16(d_logits_buf_, d_idx_out_, B, Bv, stream_.get());
        stream_.sync();
        return d2h_int32(d_idx_out_, B);
    };
    auto run_value_head_argmax_prefix = [&](const __half* w1_w, const __half* w1_b,
                                            const __half* w2_w, const __half* w2_b) {
        gemm_fp16(d_th_last_h_prefix_, w1_w, w1_b, d_logits_buf_,
                  B, H, E, nullptr, 0, stream_.get());
        mish_inplace_fp16(d_logits_buf_, B * H, stream_.get());
        gemm_fp16(d_logits_buf_, w2_w, w2_b, d_logits_buf_ + B * H,
                  B, Kb, H, nullptr, 0, stream_.get());
        argmax_fp16(d_logits_buf_ + B * H, d_idx_out_, B, Kb, stream_.get());
        stream_.sync();
        return d2h_int32(d_idx_out_, B);
    };
    auto run_value_head_argmax_prefix_with_lp = [&](const __half* w1_w, const __half* w1_b,
                                                    const __half* w2_w, const __half* w2_b,
                                                    float temp) -> IdxLp {
        gemm_fp16(d_th_last_h_prefix_, w1_w, w1_b, d_logits_buf_,
                  B, H, E, nullptr, 0, stream_.get());
        mish_inplace_fp16(d_logits_buf_, B * H, stream_.get());
        gemm_fp16(d_logits_buf_, w2_w, w2_b, d_logits_buf_ + B * H,
                  B, Kb, H, nullptr, 0, stream_.get());
        argmax_fp16(d_logits_buf_ + B * H, d_idx_out_, B, Kb, stream_.get());
        log_prob_at_idx_fp16(d_logits_buf_ + B * H, d_idx_out_, temp, d_lp_out_,
                             B, Kb, stream_.get());
        stream_.sync();
        return {d2h_int32(d_idx_out_, B), d2h_fp32(d_lp_out_, B)};
    };

    // ---------- Prefix re-prefill helper (E.5 quality path) ----------
    // Build a [B, max_S_active] tensor from per-slot host state, run
    // forward_prefill_block in PREFIX mode (block-aware mask, real block_ids),
    // and gather hidden at gather_pos[b] into d_th_last_h_prefix_.
    //
    // gather_pos[b] = -1 means "last position of this slot" (size-1).
    auto run_prefix_prefill_and_gather = [&](const std::vector<int32_t>& gather_pos_slot) {
        // S = max active token_ids length (capped at max_S).
        int S = 0;
        for (int b = 0; b < B; ++b) {
            if (active[b]) S = std::max(S, (int)token_ids[b].size());
        }
        if (S == 0) return;
        if (S > max_S) S = max_S;

        std::vector<int32_t> ids_h(B * S, 0);
        std::vector<int32_t> pos_h(B * S, 0);
        std::vector<int32_t> block_h(B * S, 0);
        std::vector<uint8_t> wlpos_h(B * S, 0);
        std::vector<uint8_t> dpos_h(B * S, 0);
        std::vector<__half> wlval_h(B * S, __float2half(0.0f));
        std::vector<__half> dval_h(B * S, __float2half(0.0f));
        std::vector<int32_t> active_h(B, 0);

        for (int b = 0; b < B; ++b) {
            int slot_S = std::min((int)token_ids[b].size(), S);
            for (int s = 0; s < slot_S; ++s) {
                ids_h[b * S + s]   = token_ids[b][s];
                pos_h[b * S + s]   = s;
                block_h[b * S + s] = block_ids[b][s];
            }
            if (active[b]) active_h[b] = 1;

            int fid = slot_to_fen[b];
            if (fid >= 0) {
                const auto& r = all_results[fid];
                for (size_t i = 0; i < r.wl_positions.size(); ++i) {
                    int p = r.wl_positions[i];
                    if (p >= 0 && p < S) {
                        wlpos_h[b * S + p] = 1;
                        wlval_h[b * S + p] = __float2half(r.wl_values[i]);
                    }
                }
                for (size_t i = 0; i < r.d_positions.size(); ++i) {
                    int p = r.d_positions[i];
                    if (p >= 0 && p < S) {
                        dpos_h[b * S + p] = 1;
                        dval_h[b * S + p] = __float2half(r.d_values[i]);
                    }
                }
            }
        }

        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_ids_,   ids_h.data(),   B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_pos_,   pos_h.data(),   B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_block_, block_h.data(), B * S * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_wl_pos_, wlpos_h.data(), B * S * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_d_pos_,  dpos_h.data(),  B * S * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_wl_val_, wlval_h.data(), B * S * sizeof(__half),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_d_val_,  dval_h.data(),  B * S * sizeof(__half),
                                      cudaMemcpyHostToDevice, stream_.get()));

        // Use a stub KvCache (active mask only — no cache write).
        KvCache stub_kv;
        std::vector<int32_t> active_h_dev = active_h;  // upload to a temp device pointer
        // Simpler: reuse kv_.slot_active() (we'll restore later — actually the
        // BOARD path will re-upload active_mask at its start, so this is fine).
        CE_CUDA_CHECK(cudaMemcpyAsync(kv_.slot_active(), active_h.data(),
                                      B * sizeof(int32_t), cudaMemcpyHostToDevice,
                                      stream_.get()));

        model_.forward_prefill_block(
            d_th_ids_, d_th_pos_, d_th_block_, kv_.slot_active(),
            d_th_wl_pos_, d_th_d_pos_, d_th_wl_val_, d_th_d_val_,
            B, S, d_th_hidden_, stream_.get(), /*kv_for_write=*/nullptr);

        // Gather hidden[b, gather_pos[b], :] into d_th_last_h_prefix_.
        for (int b = 0; b < B; ++b) {
            int p = gather_pos_slot[b];
            if (p < 0) p = std::max(0, (int)token_ids[b].size() - 1);
            if (p >= S) p = S - 1;
            CE_CUDA_CHECK(cudaMemcpyAsync(
                d_th_last_h_prefix_ + b * E,
                d_th_hidden_ + (b * S + p) * E,
                E * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
    };

    // ---------- Initial fill + init prefill ----------
    auto refill_idle = [&]() {
        std::vector<int> just_filled;
        for (int b = 0; b < B; ++b) {
            if (!active[b]) {
                int fid = pop_pending();
                if (fid >= 0) {
                    reset_slot(b, fid);
                    just_filled.push_back(b);
                }
            }
        }
        if (!just_filled.empty()) run_init_prefill(just_filled);
    };

    // Initial fill.
    refill_idle();

    // Build helper-buffer factory.
    auto zero_uint8 = [&](std::vector<uint8_t>& v) { std::fill(v.begin(), v.end(), 0); };
    auto zero_half = [&](std::vector<__half>& v) {
        std::fill(v.begin(), v.end(), __float2half(0.0f));
    };

    auto active_mask = [&](std::function<bool(int)> pred) {
        std::vector<int32_t> m(B, 0);
        for (int b = 0; b < B; ++b) m[b] = pred(b) ? 1 : 0;
        return m;
    };

    // ---------- Main loop ----------
    while (true) {
        bool any_active = false;
        for (int b = 0; b < B; ++b) if (active[b]) { any_active = true; break; }
        if (!any_active && pending.empty()) break;
        if (!any_active) { refill_idle(); continue; }

        // ===== MOVE state (active * in_variation) =====
        // E.5: re-run prefix prefill over full sequence to get prefix-mode hidden
        // at last position (matches Python's prefix_forward).
        {
            std::vector<int32_t> gp(B, -1);  // last position
            run_prefix_prefill_and_gather(gp);
        }
        auto [move_sub_h, move_lp_h] = run_head_argmax_move_vocab_prefix_with_lp(
            w_.thinking_policy_head_w, w_.thinking_policy_head_b, think_t_);

        // For active*in_variation slots: append the sampled move, then
        // forward_decode it to advance the cache and get hidden at move pos.
        std::vector<int32_t> step_ids_v(B, 0);
        std::vector<int32_t> act_in_var(B, 0);
        zero_uint8(step_wl_pos); zero_uint8(step_d_pos);
        zero_half(step_wl_val); zero_half(step_d_val);
        for (int b = 0; b < B; ++b) {
            if (active[b] && in_variation[b]) {
                int sub = move_sub_h[b];
                int full = move_sub_to_full[sub];
                int move_pos = (int)token_ids[b].size();
                token_ids[b].push_back(full);
                block_ids[b].push_back(orphan_ctr[b]++);
                int fid = slot_to_fen[b];
                if (fid >= 0) {
                    auto& r = all_results[fid];
                    r.move_positions.push_back(move_pos);
                    r.move_log_probs.push_back(move_lp_h[b]);
                }
                step_ids_v[b] = full;
                act_in_var[b] = 1;
            }
        }
        run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                        act_in_var);

        // ===== WL state =====
        // E.5: prefix prefill at MOVE position (last appended).
        {
            std::vector<int32_t> gp(B, -1);
            for (int b = 0; b < B; ++b) {
                if (active[b] && in_variation[b])
                    gp[b] = (int)token_ids[b].size() - 1;  // move position
            }
            run_prefix_prefill_and_gather(gp);
        }
        auto [wl_idx_h, wl_lp_h] = run_value_head_argmax_prefix_with_lp(
            w_.wl_w1_w, w_.wl_w1_b, w_.wl_w2_w, w_.wl_w2_b, wl_t_);
        // Append wl_value placeholder + record entry, then forward_decode w/ fourier(wl).
        std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
        zero_uint8(step_wl_pos); zero_uint8(step_d_pos);
        zero_half(step_wl_val); zero_half(step_d_val);
        std::fill(act_in_var.begin(), act_in_var.end(), 0);
        for (int b = 0; b < B; ++b) {
            if (active[b] && in_variation[b]) {
                int wl_pos_idx = (int)token_ids[b].size();
                token_ids[b].push_back(wl_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                int fid = slot_to_fen[b];
                if (fid >= 0) {
                    auto& r = all_results[fid];
                    r.wl_positions.push_back(wl_pos_idx);
                    r.wl_indices.push_back(wl_idx_h[b]);
                    r.wl_values.push_back(wl_centers_h[wl_idx_h[b]]);
                    r.wl_log_probs.push_back(wl_lp_h[b]);
                }
                step_ids_v[b] = wl_value_idx;
                step_wl_pos[b] = 1;
                step_wl_val[b] = __float2half(wl_centers_h[wl_idx_h[b]]);
                act_in_var[b] = 1;
            }
        }
        run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                        act_in_var);

        // ===== D state =====
        // E.5: prefix prefill at WL_VALUE position (with WL fourier injected).
        {
            std::vector<int32_t> gp(B, -1);
            for (int b = 0; b < B; ++b) {
                if (active[b] && in_variation[b])
                    gp[b] = (int)token_ids[b].size() - 1;  // wl_value position
            }
            run_prefix_prefill_and_gather(gp);
        }
        auto [d_idx_h, d_lp_h] = run_value_head_argmax_prefix_with_lp(
            w_.d_w1_w, w_.d_w1_b, w_.d_w2_w, w_.d_w2_b, d_t_);
        std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
        zero_uint8(step_wl_pos); zero_uint8(step_d_pos);
        zero_half(step_wl_val); zero_half(step_d_val);
        std::fill(act_in_var.begin(), act_in_var.end(), 0);
        for (int b = 0; b < B; ++b) {
            if (active[b] && in_variation[b]) {
                int d_pos_idx = (int)token_ids[b].size();
                token_ids[b].push_back(d_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                int fid = slot_to_fen[b];
                if (fid >= 0) {
                    auto& r = all_results[fid];
                    r.d_positions.push_back(d_pos_idx);
                    r.d_indices.push_back(d_idx_h[b]);
                    r.d_values.push_back(d_centers_h[d_idx_h[b]]);
                    r.d_log_probs.push_back(d_lp_h[b]);
                }
                step_ids_v[b] = d_value_idx;
                step_d_pos[b] = 1;
                step_d_val[b] = __float2half(d_centers_h[d_idx_h[b]]);
                act_in_var[b] = 1;
            }
        }
        run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                        act_in_var);

        // ===== BOARD state (68 steps, chained on-device) =====
        // Each step: argmax + LUT-gather to write the full-vocab id directly
        // into d_th_full_idx_ (device).  Next step's forward_decode reads
        // from d_th_full_idx_ — no host roundtrip.  At end of BOARD, we
        // batch-d2h the 68 sub-vocab samples for host bookkeeping.
        std::vector<int> this_board_bid(B, 0);
        for (int b = 0; b < B; ++b) {
            if (active[b] && in_variation[b]) this_board_bid[b] = next_block[b]++;
        }
        // active mask for the entire BOARD loop: same for all 68 steps for
        // a given slot (truncation is checked only at the end).
        std::vector<int32_t> board_act(B, 0);
        for (int b = 0; b < B; ++b) {
            if (active[b] && in_variation[b]
                && (int)token_ids[b].size() + 68 <= max_S) {
                board_act[b] = 1;
            }
        }
        // Zero wl_pos / d_pos / wl_val / d_val for BOARD steps (no fourier).
        zero_uint8(step_wl_pos); zero_uint8(step_d_pos);
        zero_half(step_wl_val); zero_half(step_d_val);
        // Upload buffers that don't change across the 68 steps.
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_wl_pos_, step_wl_pos.data(),
                                      B * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_d_pos_, step_d_pos.data(),
                                      B * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_wl_val_, step_wl_val.data(),
                                      B * sizeof(__half),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(d_th_d_val_, step_d_val.data(),
                                      B * sizeof(__half),
                                      cudaMemcpyHostToDevice, stream_.get()));
        CE_CUDA_CHECK(cudaMemcpyAsync(kv_.slot_active(), board_act.data(),
                                      B * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream_.get()));

        for (int step = 0; step < 68; ++step) {
            // 1. Run board_head GEMM on current last_h.
            gemm_fp16(d_th_last_h_, w_.board_head_w, w_.board_head_b,
                      d_logits_buf_, B, Bv, E, nullptr, 0, stream_.get());
            // 2. Fused argmax + LUT scatter → writes both sub_idx and full_idx
            //    for this step.  full_idx becomes the next step's input token.
            argmax_lut_scatter_fp16(d_logits_buf_, d_board_sub_to_full_,
                                    d_th_sub_idx_log_ + step * B,    // [B] sub log
                                    d_th_full_idx_,                  // [B] full
                                    B, Bv, stream_.get());
            // 3. Build per-slot pos buffer (= slot_past_len_h[b]).
            // We don't need a host-side build: forward_decode reads pos from
            // d_th_pos_, which we update once per step using the per-slot
            // past_len (already in slot_past_len_h on host).  TODO: this is a
            // small H2D each step; could be eliminated by computing pos
            // on-device from past_len.  For now, accept it.
            std::vector<int32_t> pos_h(B, 0);
            for (int b = 0; b < B; ++b) pos_h[b] = slot_past_len_h[b];
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_pos_, pos_h.data(),
                                          B * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
            // 4. Backup last_h (so inactive slots' state is preserved).
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_last_h_bkp_, d_th_last_h_,
                                          B * E * sizeof(__half),
                                          cudaMemcpyDeviceToDevice, stream_.get()));
            // 5. Forward_decode reading d_th_full_idx_ as input.
            //    Phase I: replay the captured graph if available (saves ~135
            //    kernel launches per step over the BOARD loop's 68 iterations).
            if (decode_graph_ready_) {
                CE_CUDA_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_.get()));
            } else {
                model_.forward_decode(
                    d_th_full_idx_, d_th_pos_,
                    d_th_wl_pos_, d_th_d_pos_, d_th_wl_val_, d_th_d_val_,
                    kv_, d_th_last_h_, stream_.get());
            }
            // 6. Restore last_h for inactive slots.
            restore_inactive_last_h(d_th_last_h_, d_th_last_h_bkp_,
                                    kv_.slot_active(), B, E, stream_.get());
            // 7. Advance host past_len mirror.
            for (int b = 0; b < B; ++b) {
                if (board_act[b]) slot_past_len_h[b] += 1;
            }
        }
        // Batch d2h all 68 steps' sub-vocab samples for host bookkeeping.
        std::vector<int32_t> sub_log(B * 68);
        CE_CUDA_CHECK(cudaMemcpyAsync(sub_log.data(), d_th_sub_idx_log_,
                                      B * 68 * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost, stream_.get()));
        stream_.sync();
        // Append to per-slot token_ids.
        for (int step = 0; step < 68; ++step) {
            for (int b = 0; b < B; ++b) {
                if (!board_act[b]) {
                    if (step == 0 && active[b] && in_variation[b]
                        && (int)token_ids[b].size() + 68 > max_S) {
                        truncated[b] = true; active[b] = false;
                    }
                    continue;
                }
                int sub = sub_log[step * B + b];
                int full = board_sub_to_full[sub];
                token_ids[b].push_back(full);
                block_ids[b].push_back(this_board_bid[b]);
            }
        }

        // ===== AFTER_BOARD state =====
        // last_h is hidden at the 68th board token; sample via board_head.
        auto ab_sub_h = run_head_argmax_board_vocab_causal();
        std::vector<bool> ended_var(B, false);
        std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
        std::fill(act_in_var.begin(), act_in_var.end(), 0);
        for (int b = 0; b < B; ++b) {
            if (!(active[b] && in_variation[b])) continue;
            if (ab_sub_h[b] == board_end_var_sub) {
                ended_var[b] = true;
                int full = board_sub_to_full[ab_sub_h[b]];
                token_ids[b].push_back(full);
                block_ids[b].push_back(orphan_ctr[b]++);
                step_ids_v[b] = full;
                act_in_var[b] = 1;
            }
            // else: continue_var — don't append, don't forward_decode for this slot.
        }
        run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                        act_in_var);

        // ===== AFTER_END_VAR state (gated by ended_var) =====
        bool any_ended_var = false;
        for (bool e : ended_var) if (e) { any_ended_var = true; break; }
        if (any_ended_var) {
            auto aev_sub_h = run_head_argmax_board_vocab_causal();
            std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
            std::fill(act_in_var.begin(), act_in_var.end(), 0);
            for (int b = 0; b < B; ++b) {
                if (!ended_var[b]) continue;
                if (aev_sub_h[b] == board_end_think_sub) {
                    int full = board_sub_to_full[aev_sub_h[b]];
                    token_ids[b].push_back(full);
                    block_ids[b].push_back(orphan_ctr[b]++);
                    ended_thinking[b] = true;
                    in_variation[b] = false;
                    step_ids_v[b] = full;
                    act_in_var[b] = 1;
                }
                // else: new_variation — don't append, don't forward_decode.
            }
            run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                            act_in_var);
        }

        // ===== FINAL state (gated by ended_thinking) =====
        bool any_final = false;
        for (int b = 0; b < B; ++b) if (ended_thinking[b] && active[b]) { any_final = true; break; }
        if (any_final) {
            // E.5: prefix prefill at end_think position (last appended).
            {
                std::vector<int32_t> gp(B, -1);
                run_prefix_prefill_and_gather(gp);
            }
            auto [fm_sub_h, fm_lp_h] = run_head_argmax_move_vocab_prefix_with_lp(
                w_.policy_head_w, w_.policy_head_b, policy_t_);
            std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
            std::fill(act_in_var.begin(), act_in_var.end(), 0);
            for (int b = 0; b < B; ++b) {
                if (!(ended_thinking[b] && active[b])) continue;
                int sub = fm_sub_h[b];
                int full = move_sub_to_full[sub];
                int move_pos = (int)token_ids[b].size();
                token_ids[b].push_back(full);
                block_ids[b].push_back(orphan_ctr[b]++);
                int fid = slot_to_fen[b];
                if (fid >= 0) {
                    auto& r = all_results[fid];
                    std::string mp = vocab_->idxToToken(full);
                    r.move = decoder::DecoderVocab::pseudoToStandardUci(mp);
                    r.move_positions.push_back(move_pos);
                    r.move_log_probs.push_back(fm_lp_h[b]);
                }
                step_ids_v[b] = full;
                act_in_var[b] = 1;
            }
            run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                            act_in_var);

            // Final WL: prefix prefill at final-move position.
            {
                std::vector<int32_t> gp(B, -1);
                for (int b = 0; b < B; ++b) {
                    if (ended_thinking[b])
                        gp[b] = (int)token_ids[b].size() - 1;
                }
                run_prefix_prefill_and_gather(gp);
            }
            auto [fwl_h, fwl_lp_h] = run_value_head_argmax_prefix_with_lp(
                w_.wl_w1_w, w_.wl_w1_b, w_.wl_w2_w, w_.wl_w2_b, wl_t_);
            std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
            zero_uint8(step_wl_pos); zero_uint8(step_d_pos);
            zero_half(step_wl_val); zero_half(step_d_val);
            std::fill(act_in_var.begin(), act_in_var.end(), 0);
            for (int b = 0; b < B; ++b) {
                if (!(ended_thinking[b] && active[b])) continue;
                int wl_pos_idx = (int)token_ids[b].size();
                token_ids[b].push_back(wl_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                int fid = slot_to_fen[b];
                if (fid >= 0) {
                    auto& r = all_results[fid];
                    r.wl_positions.push_back(wl_pos_idx);
                    r.wl_indices.push_back(fwl_h[b]);
                    r.wl_values.push_back(wl_centers_h[fwl_h[b]]);
                    r.wl_log_probs.push_back(fwl_lp_h[b]);
                    r.final_wl_index = fwl_h[b];
                    r.final_wl_value = wl_centers_h[fwl_h[b]];
                }
                step_ids_v[b] = wl_value_idx;
                step_wl_pos[b] = 1;
                step_wl_val[b] = __float2half(wl_centers_h[fwl_h[b]]);
                act_in_var[b] = 1;
            }
            run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                            act_in_var);

            // Final D: prefix prefill at wl_value position (with WL fourier).
            {
                std::vector<int32_t> gp(B, -1);
                for (int b = 0; b < B; ++b) {
                    if (ended_thinking[b])
                        gp[b] = (int)token_ids[b].size() - 1;
                }
                run_prefix_prefill_and_gather(gp);
            }
            auto [fd_h, fd_lp_h] = run_value_head_argmax_prefix_with_lp(
                w_.d_w1_w, w_.d_w1_b, w_.d_w2_w, w_.d_w2_b, d_t_);
            std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
            zero_uint8(step_wl_pos); zero_uint8(step_d_pos);
            zero_half(step_wl_val); zero_half(step_d_val);
            std::fill(act_in_var.begin(), act_in_var.end(), 0);
            for (int b = 0; b < B; ++b) {
                if (!(ended_thinking[b] && active[b])) continue;
                int d_pos_idx = (int)token_ids[b].size();
                token_ids[b].push_back(d_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                int fid = slot_to_fen[b];
                if (fid >= 0) {
                    auto& r = all_results[fid];
                    r.d_positions.push_back(d_pos_idx);
                    r.d_indices.push_back(fd_h[b]);
                    r.d_values.push_back(d_centers_h[fd_h[b]]);
                    r.d_log_probs.push_back(fd_lp_h[b]);
                    r.final_d_index = fd_h[b];
                    r.final_d_value = d_centers_h[fd_h[b]];
                    r.ended_thinking = true;
                }
                step_ids_v[b] = d_value_idx;
                step_d_pos[b] = 1;
                step_d_val[b] = __float2half(d_centers_h[fd_h[b]]);
                act_in_var[b] = 1;

                active[b] = false;  // mark slot done
            }
            run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                            act_in_var);
        }

        // ===== Iter cap + truncation + commit + refill =====
        for (int b = 0; b < B; ++b) {
            if (slot_to_fen[b] < 0) continue;

            slot_iter_count[b]++;
            bool slot_done = false;

            if (active[b] && (int)token_ids[b].size() >= max_S) {
                truncated[b] = true; active[b] = false; slot_done = true;
            }
            if (!active[b]) slot_done = true;
            if (slot_iter_count[b] >= max_iters && active[b]) {
                truncated[b] = true; active[b] = false; slot_done = true;
            }

            if (slot_done) {
                int fid = slot_to_fen[b];
                if (fid >= 0 && fid < (int)all_results.size()) {
                    auto& r = all_results[fid];
                    r.token_ids = token_ids[b];
                    r.block_ids = block_ids[b];
                    r.truncated = truncated[b];
                }
                slot_to_fen[b] = -1;
            }
        }
        // Refill any newly-idle slots.
        refill_idle();
    }

    return all_results;
}

}  // namespace cutlass_engine

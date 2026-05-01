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
                block_h[b * S + s] = block_ids[b][s];
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

        model_.forward_decode(
            d_th_ids_, d_th_pos_,
            d_th_wl_pos_, d_th_d_pos_,
            d_th_wl_val_, d_th_d_val_,
            kv_, d_th_last_h_, stream_.get());

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
    auto run_head_argmax_move_vocab = [&](const __half* head_w, const __half* head_b) {
        // [B, E] @ [Mv, E]^T + bias -> [B, Mv], argmax -> [B] int32.
        gemm_fp16(d_th_last_h_, head_w, head_b, d_logits_buf_,
                  B, Mv, E, nullptr, 0, stream_.get());
        argmax_fp16(d_logits_buf_, d_idx_out_, B, Mv, stream_.get());
        stream_.sync();
        return d2h_int32(d_idx_out_, B);
    };
    auto run_head_argmax_board_vocab = [&]() {
        gemm_fp16(d_th_last_h_, w_.board_head_w, w_.board_head_b, d_logits_buf_,
                  B, Bv, E, nullptr, 0, stream_.get());
        argmax_fp16(d_logits_buf_, d_idx_out_, B, Bv, stream_.get());
        stream_.sync();
        return d2h_int32(d_idx_out_, B);
    };
    auto run_value_head_argmax = [&](const __half* w1_w, const __half* w1_b,
                                     const __half* w2_w, const __half* w2_b) {
        // [B, E] @ [H, E]^T + b -> [B, H] -> mish -> [B, H] @ [Kb, H]^T + b -> [B, Kb] -> argmax
        gemm_fp16(d_th_last_h_, w1_w, w1_b, d_logits_buf_,
                  B, H, E, nullptr, 0, stream_.get());
        mish_inplace_fp16(d_logits_buf_, B * H, stream_.get());
        gemm_fp16(d_logits_buf_, w2_w, w2_b, d_logits_buf_ + B * H,
                  B, Kb, H, nullptr, 0, stream_.get());
        argmax_fp16(d_logits_buf_ + B * H, d_idx_out_, B, Kb, stream_.get());
        stream_.sync();
        return d2h_int32(d_idx_out_, B);
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
        // Sample move from thinking_policy_head(last_h).
        auto move_sub_h = run_head_argmax_move_vocab(
            w_.thinking_policy_head_w, w_.thinking_policy_head_b);

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
                token_ids[b].push_back(full);
                block_ids[b].push_back(orphan_ctr[b]++);
                step_ids_v[b] = full;
                act_in_var[b] = 1;
            }
        }
        run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                        act_in_var);

        // ===== WL state =====
        // Predict WL at MOVE position from current last_h (which is move's hidden).
        auto wl_idx_h = run_value_head_argmax(w_.wl_w1_w, w_.wl_w1_b,
                                              w_.wl_w2_w, w_.wl_w2_b);
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
        // Predict D from current last_h (wl's hidden, with fourier injected).
        auto d_idx_h = run_value_head_argmax(w_.d_w1_w, w_.d_w1_b,
                                             w_.d_w2_w, w_.d_w2_b);
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
                }
                step_ids_v[b] = d_value_idx;
                step_d_pos[b] = 1;
                step_d_val[b] = __float2half(d_centers_h[d_idx_h[b]]);
                act_in_var[b] = 1;
            }
        }
        run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                        act_in_var);

        // ===== BOARD state (68 steps) =====
        // Each step: read last_h (hidden at d_value or previous board token),
        // sample board token via board_head, append, forward_decode that token.
        std::vector<int> this_board_bid(B, 0);
        for (int b = 0; b < B; ++b) {
            if (active[b] && in_variation[b]) this_board_bid[b] = next_block[b]++;
        }
        for (int step = 0; step < 68; ++step) {
            auto sub_h = run_head_argmax_board_vocab();

            std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
            zero_uint8(step_wl_pos); zero_uint8(step_d_pos);
            zero_half(step_wl_val); zero_half(step_d_val);
            std::fill(act_in_var.begin(), act_in_var.end(), 0);
            for (int b = 0; b < B; ++b) {
                if (!(active[b] && in_variation[b])) continue;
                if ((int)token_ids[b].size() >= max_S) {
                    truncated[b] = true; active[b] = false;
                    continue;
                }
                int sub = sub_h[b];
                int full = board_sub_to_full[sub];
                token_ids[b].push_back(full);
                block_ids[b].push_back(this_board_bid[b]);
                step_ids_v[b] = full;
                act_in_var[b] = 1;
            }
            run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                            act_in_var);
        }

        // ===== AFTER_BOARD state =====
        // last_h is hidden at the 68th board token; sample via board_head.
        auto ab_sub_h = run_head_argmax_board_vocab();
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
            auto aev_sub_h = run_head_argmax_board_vocab();
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
            // Sample final move from policy_head(last_h).
            auto fm_sub_h = run_head_argmax_move_vocab(
                w_.policy_head_w, w_.policy_head_b);
            std::fill(step_ids_v.begin(), step_ids_v.end(), 0);
            std::fill(act_in_var.begin(), act_in_var.end(), 0);
            for (int b = 0; b < B; ++b) {
                if (!(ended_thinking[b] && active[b])) continue;
                int sub = fm_sub_h[b];
                int full = move_sub_to_full[sub];
                token_ids[b].push_back(full);
                block_ids[b].push_back(orphan_ctr[b]++);
                int fid = slot_to_fen[b];
                if (fid >= 0) {
                    auto& r = all_results[fid];
                    std::string mp = vocab_->idxToToken(full);
                    r.move = decoder::DecoderVocab::pseudoToStandardUci(mp);
                }
                step_ids_v[b] = full;
                act_in_var[b] = 1;
            }
            run_decode_step(step_ids_v, step_wl_pos, step_d_pos, step_wl_val, step_d_val,
                            act_in_var);

            // Final WL.
            auto fwl_h = run_value_head_argmax(w_.wl_w1_w, w_.wl_w1_b,
                                               w_.wl_w2_w, w_.wl_w2_b);
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

            // Final D.
            auto fd_h = run_value_head_argmax(w_.d_w1_w, w_.d_w1_b,
                                              w_.d_w2_w, w_.d_w2_b);
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

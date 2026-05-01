// Full thinking-trace state machine for ThinkingEngine.
//
// Mirrors chessdecoder/inference/think.py:run_thinking() — same state
// machine (MOVE → WL_D → BOARD → AFTER_BOARD → AFTER_END_VAR → FINAL),
// but batched (lockstep over B slots) and on-GPU.
//
// Strategy: re-run forward_prefill_block over the full sequence at every
// state-machine step.  Matches Python's behavior exactly (no KV-cache
// shortcut).  Slow but correct.  Optimization (incremental cache, kernel
// fusion) is a follow-up.
//
// Each step's per-slot active mask gates work for slots that are still
// in-variation / awaiting end_var resolution / awaiting end_think.
// Inactive slots' tokens are ignored on output.

#include "cutlass_engine/engine.hpp"
#include "cutlass_engine/check.hpp"
#include "cutlass_engine/kernels.hpp"

#include "vocab.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cutlass_engine {

namespace {

// Pull a single FP16 value from device.
__half d2h_half(const __half* dptr) {
    __half h;
    cudaMemcpy(&h, dptr, sizeof(__half), cudaMemcpyDeviceToHost);
    return h;
}

// Copy a [B] FP32 vector to host.
std::vector<float> d2h_fp32(const float* dptr, int B) {
    std::vector<float> out(B);
    cudaMemcpy(out.data(), dptr, B * sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

// Copy a [B] int32 vector to host.
std::vector<int32_t> d2h_int32(const int32_t* dptr, int B) {
    std::vector<int32_t> out(B);
    cudaMemcpy(out.data(), dptr, B * sizeof(int32_t), cudaMemcpyDeviceToHost);
    return out;
}

}  // namespace

std::vector<RolloutResult> ThinkingEngine::predict_moves_thinking(
    const std::vector<std::string>& fens, float temperature,
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

    if (max_S > max_init_S_ * 64) {
        // Our scratch buffers are sized for max_init_S * batch_size; the
        // thinking path needs much larger scratch.  We reuse d_th_* below,
        // sized at construction for cfg_.max_seq_len.
    }

    // Per-process special token IDs (full-vocab + sub-vocab).
    const int wl_value_idx       = vocab_->wlValueIdx();
    const int d_value_idx        = vocab_->dValueIdx();
    const int start_think_idx    = vocab_->startThinkIdx();
    const int end_think_idx      = vocab_->endThinkIdx();
    const int end_var_idx        = vocab_->endVarIdx();
    const int board_end_var_sub  = vocab_->boardEndVarIdx();
    const int board_end_think_sub= vocab_->boardEndThinkIdx();

    // Pull bucket centers to host (for setting wl_entries / d_entries values).
    std::vector<float> wl_centers_h = d2h_fp32(w_.wl_bucket_centers, Kb);
    std::vector<float> d_centers_h  = d2h_fp32(w_.d_bucket_centers, Kb);

    // Pull move/board sub-to-full LUTs to host.
    std::vector<int> move_sub_to_full(Mv);
    for (int i = 0; i < Mv; ++i) move_sub_to_full[i] = vocab_->moveIdxToFullIdx(i);
    std::vector<int> board_sub_to_full(Bv);
    for (int i = 0; i < Bv; ++i) board_sub_to_full[i] = vocab_->boardIdxToFullIdx(i);

    std::vector<RolloutResult> all_results(N);

    // Process FENs in chunks of B.
    for (int chunk_start = 0; chunk_start < N; chunk_start += B) {
        const int chunk_n = std::min(B, N - chunk_start);

        // Per-slot state (host).
        std::vector<std::vector<int32_t>> token_ids(B);
        std::vector<std::vector<int32_t>> block_ids(B);
        std::vector<int> next_block(B, 0);
        std::vector<int> orphan_ctr(B, 10000);
        std::vector<bool> active(B, false);
        std::vector<bool> in_variation(B, false);
        std::vector<bool> ended_thinking(B, false);
        std::vector<bool> truncated(B, false);
        // Per-slot legal-move mask (host bool).
        std::vector<std::vector<uint8_t>> legal_masks(B, std::vector<uint8_t>(Mv, 0));

        // Init each slot.
        for (int b = 0; b < chunk_n; ++b) {
            const auto& fen = fens[chunk_start + b];
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
            auto legal = vocab_->legalMoveIndices(fen);
            for (int idx : legal) if (idx >= 0 && idx < Mv) legal_masks[b][idx] = 1;
        }
        // Padded slots beyond chunk_n: stay inactive.
        for (int b = chunk_n; b < B; ++b) {
            // Token vectors empty; padding handled in build_step().
        }

        // Helper: build [B, S] device tensors from per-slot state at length S.
        // Stores into d_th_ids_, d_th_pos_, d_th_block_, d_th_wl_pos_, etc.
        // Returns S (= max active slot length, capped at max_S).
        auto build_step = [&](bool use_block_aware,
                              std::vector<int32_t>& active_h_out) {
            // S = max of active token_ids lengths.
            int S = 0;
            for (int b = 0; b < B; ++b)
                if (active[b]) S = std::max(S, (int)token_ids[b].size());
            if (S == 0) S = 1;
            S = std::min(S, max_S);

            // Build CPU staging buffers (one alloc per step — small, B*S*~6 bytes).
            std::vector<int32_t> ids_h(B * S, 0);
            std::vector<int32_t> pos_h(B * S, 0);
            std::vector<int32_t> block_h(B * S, 0);
            std::vector<uint8_t> wlpos_h(B * S, 0);
            std::vector<uint8_t> dpos_h(B * S, 0);
            std::vector<__half> wlval_h(B * S, __float2half(0.0f));
            std::vector<__half> dval_h(B * S, __float2half(0.0f));
            active_h_out.assign(B, 0);

            for (int b = 0; b < B; ++b) {
                int slot_S = (int)token_ids[b].size();
                if (slot_S > S) slot_S = S;  // truncate (should not happen — S is max)
                for (int s = 0; s < slot_S; ++s) {
                    ids_h[b * S + s]   = token_ids[b][s];
                    pos_h[b * S + s]   = s;
                    block_h[b * S + s] = use_block_aware ? block_ids[b][s] : (1000000 + s);
                    // ↑ unique-per-token block ids → mask becomes purely causal
                }
                if (active[b]) active_h_out[b] = 1;
            }
            // Inject WL/D values from per-slot entries.  We store them on the
            // RolloutResult during sampling (hand off below).  For now
            // re-derive from results vector (we set them below in MOVE/WL_D).
            // We use all_results[fen_id] as the canonical store.
            for (int b = 0; b < chunk_n; ++b) {
                int fid = chunk_start + b;
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

            // Upload.
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_ids_, ids_h.data(),
                                          B * S * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_pos_, pos_h.data(),
                                          B * S * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_block_, block_h.data(),
                                          B * S * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_wl_pos_, wlpos_h.data(),
                                          B * S * sizeof(uint8_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_d_pos_, dpos_h.data(),
                                          B * S * sizeof(uint8_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_wl_val_, wlval_h.data(),
                                          B * S * sizeof(__half),
                                          cudaMemcpyHostToDevice, stream_.get()));
            CE_CUDA_CHECK(cudaMemcpyAsync(d_th_d_val_, dval_h.data(),
                                          B * S * sizeof(__half),
                                          cudaMemcpyHostToDevice, stream_.get()));
            CE_CUDA_CHECK(cudaMemcpyAsync(d_active_buf_, active_h_out.data(),
                                          B * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, stream_.get()));
            return S;
        };

        // Helper: run forward_prefill_block and gather hidden at per-slot pos.
        // gather_pos[b] = position whose hidden we want.  Out: d_th_last_h_ [B, E].
        auto run_forward_and_gather = [&](bool use_block_aware,
                                          const std::vector<int32_t>& gather_pos) {
            std::vector<int32_t> active_h;
            int S = build_step(use_block_aware, active_h);

            model_.forward_prefill_block(
                d_th_ids_, d_th_pos_, d_th_block_, d_active_buf_,
                d_th_wl_pos_, d_th_d_pos_, d_th_wl_val_, d_th_d_val_,
                B, S, d_th_hidden_, stream_.get());

            // Gather hidden[b, gather_pos[b], :] into d_th_last_h_[b, :].
            // We do this with B small cudaMemcpyAsyncs (B ≤ 256, so ~64 us total).
            for (int b = 0; b < B; ++b) {
                int p = (gather_pos[b] >= 0 && gather_pos[b] < S) ? gather_pos[b] : 0;
                CE_CUDA_CHECK(cudaMemcpyAsync(
                    d_th_last_h_ + b * E,
                    d_th_hidden_ + (b * S + p) * E,
                    E * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream_.get()));
            }
            return S;
        };

        // -------------- Main per-iteration loop --------------
        for (int iter = 0; iter < max_iters; ++iter) {
            // Bail out if no slots active.
            bool any_active = false;
            for (int b = 0; b < B; ++b) if (active[b]) any_active = true;
            if (!any_active) break;

            // ---- MOVE state ----
            // Sample move from thinking_policy_head at the LAST position
            // (current end of token_ids for active slots).
            std::vector<int32_t> gather_pos(B, 0);
            for (int b = 0; b < B; ++b) {
                gather_pos[b] = (int)token_ids[b].size() - 1;
                if (gather_pos[b] < 0) gather_pos[b] = 0;
            }
            run_forward_and_gather(/*use_block_aware=*/true, gather_pos);

            // Run thinking_policy_head GEMM: [B, E] @ [Mv, E]^T + bias
            gemm_fp16(d_th_last_h_, w_.thinking_policy_head_w, w_.thinking_policy_head_b,
                      d_logits_buf_, B, Mv, E, nullptr, 0, stream_.get());

            // No legal-mask filter for intermediate moves — matches
            // chessdecoder/inference/think.py:run_thinking, which trusts the
            // model to output legal-looking moves.

            // Argmax (temp=0).
            argmax_fp16(d_logits_buf_, d_idx_out_, B, Mv, stream_.get());
            stream_.sync();
            auto move_idx_h = d2h_int32(d_idx_out_, B);
            for (int b = 0; b < B; ++b) {
                if (!(active[b] && in_variation[b])) continue;
                int sub = move_idx_h[b];
                int full = move_sub_to_full[sub];
                token_ids[b].push_back(full);
                block_ids[b].push_back(orphan_ctr[b]++);
            }

            // ---- WL_D state ----
            // (1) predict WL from move position via wl_head.
            for (int b = 0; b < B; ++b) {
                gather_pos[b] = (int)token_ids[b].size() - 1;  // move position
                if (gather_pos[b] < 0) gather_pos[b] = 0;
            }
            run_forward_and_gather(true, gather_pos);
            // wl_head: [B, E] -> [B, H] via w1+bias, then mish, then [B, Kb] via w2+bias.
            gemm_fp16(d_th_last_h_, w_.wl_w1_w, w_.wl_w1_b,
                      d_logits_buf_, B, H, E, nullptr, 0, stream_.get());
            mish_inplace_fp16(d_logits_buf_, B * H, stream_.get());
            gemm_fp16(d_logits_buf_, w_.wl_w2_w, w_.wl_w2_b,
                      d_logits_buf_ + B * H, B, Kb, H, nullptr, 0, stream_.get());
            argmax_fp16(d_logits_buf_ + B * H, d_idx_out_, B, Kb, stream_.get());
            stream_.sync();
            auto wl_idx_h = d2h_int32(d_idx_out_, B);

            // Append wl_value placeholder + record entry.
            for (int b = 0; b < B; ++b) {
                if (!(active[b] && in_variation[b])) continue;
                int wl_pos_idx = (int)token_ids[b].size();
                token_ids[b].push_back(wl_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);

                int fid = chunk_start + b;
                if (fid < (int)all_results.size()) {
                    all_results[fid].wl_positions.push_back(wl_pos_idx);
                    all_results[fid].wl_indices.push_back(wl_idx_h[b]);
                    all_results[fid].wl_values.push_back(wl_centers_h[wl_idx_h[b]]);
                }
            }

            // (2) predict D from wl_value position via d_head (with WL injected).
            for (int b = 0; b < B; ++b) {
                gather_pos[b] = (int)token_ids[b].size() - 1;  // wl_value position
                if (gather_pos[b] < 0) gather_pos[b] = 0;
            }
            run_forward_and_gather(true, gather_pos);
            gemm_fp16(d_th_last_h_, w_.d_w1_w, w_.d_w1_b,
                      d_logits_buf_, B, H, E, nullptr, 0, stream_.get());
            mish_inplace_fp16(d_logits_buf_, B * H, stream_.get());
            gemm_fp16(d_logits_buf_, w_.d_w2_w, w_.d_w2_b,
                      d_logits_buf_ + B * H, B, Kb, H, nullptr, 0, stream_.get());
            argmax_fp16(d_logits_buf_ + B * H, d_idx_out_, B, Kb, stream_.get());
            stream_.sync();
            auto d_idx_h = d2h_int32(d_idx_out_, B);

            for (int b = 0; b < B; ++b) {
                if (!(active[b] && in_variation[b])) continue;
                int d_pos_idx = (int)token_ids[b].size();
                token_ids[b].push_back(d_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);

                int fid = chunk_start + b;
                if (fid < (int)all_results.size()) {
                    all_results[fid].d_positions.push_back(d_pos_idx);
                    all_results[fid].d_indices.push_back(d_idx_h[b]);
                    all_results[fid].d_values.push_back(d_centers_h[d_idx_h[b]]);
                }
            }

            // ---- BOARD state ----
            // 68 causal-mode iterations, each appending one board token.
            int board_bid;  // single block_id for this board's 68 tokens
            (void)board_bid;
            // Each slot gets its own board block_id:
            std::vector<int> this_board_bid(B, 0);
            for (int b = 0; b < B; ++b) {
                if (active[b] && in_variation[b]) this_board_bid[b] = next_block[b]++;
            }
            for (int step = 0; step < 68; ++step) {
                // Truncate check.
                bool any_can_continue = false;
                for (int b = 0; b < B; ++b) {
                    if (active[b] && in_variation[b] && (int)token_ids[b].size() < max_S) {
                        any_can_continue = true;
                    } else if (active[b] && (int)token_ids[b].size() >= max_S) {
                        truncated[b] = true; active[b] = false;
                    }
                }
                if (!any_can_continue) break;

                for (int b = 0; b < B; ++b) {
                    gather_pos[b] = (int)token_ids[b].size() - 1;
                    if (gather_pos[b] < 0) gather_pos[b] = 0;
                }
                run_forward_and_gather(/*use_block_aware=*/false, gather_pos);

                // board_head: [B, E] -> [B, Bv]
                gemm_fp16(d_th_last_h_, w_.board_head_w, w_.board_head_b,
                          d_logits_buf_, B, Bv, E, nullptr, 0, stream_.get());
                argmax_fp16(d_logits_buf_, d_idx_out_, B, Bv, stream_.get());
                stream_.sync();
                auto sub_h = d2h_int32(d_idx_out_, B);

                for (int b = 0; b < B; ++b) {
                    if (!(active[b] && in_variation[b])) continue;
                    int full = board_sub_to_full[sub_h[b]];
                    token_ids[b].push_back(full);
                    block_ids[b].push_back(this_board_bid[b]);
                }
            }

            // ---- AFTER_BOARD state ----
            // Run causal forward, board_head, sample, decide end_var per slot.
            for (int b = 0; b < B; ++b) {
                gather_pos[b] = (int)token_ids[b].size() - 1;
                if (gather_pos[b] < 0) gather_pos[b] = 0;
            }
            run_forward_and_gather(false, gather_pos);
            gemm_fp16(d_th_last_h_, w_.board_head_w, w_.board_head_b,
                      d_logits_buf_, B, Bv, E, nullptr, 0, stream_.get());
            argmax_fp16(d_logits_buf_, d_idx_out_, B, Bv, stream_.get());
            stream_.sync();
            auto ab_sub_h = d2h_int32(d_idx_out_, B);

            std::vector<bool> ended_var(B, false);
            for (int b = 0; b < B; ++b) {
                if (!(active[b] && in_variation[b])) continue;
                if (ab_sub_h[b] == board_end_var_sub) {
                    ended_var[b] = true;
                    int full = board_sub_to_full[ab_sub_h[b]];
                    token_ids[b].push_back(full);
                    block_ids[b].push_back(orphan_ctr[b]++);
                }
                // else: continue_var (or unexpected) — don't append; in_variation
                // stays true; will sample next move at next iter's MOVE state.
            }

            // ---- AFTER_END_VAR state ----
            // Only for slots with ended_var: probe end_think.
            bool any_ended_var = false;
            for (int b = 0; b < B; ++b) if (ended_var[b]) any_ended_var = true;
            if (any_ended_var) {
                for (int b = 0; b < B; ++b) {
                    gather_pos[b] = (int)token_ids[b].size() - 1;
                    if (gather_pos[b] < 0) gather_pos[b] = 0;
                }
                run_forward_and_gather(false, gather_pos);
                gemm_fp16(d_th_last_h_, w_.board_head_w, w_.board_head_b,
                          d_logits_buf_, B, Bv, E, nullptr, 0, stream_.get());
                argmax_fp16(d_logits_buf_, d_idx_out_, B, Bv, stream_.get());
                stream_.sync();
                auto aev_sub_h = d2h_int32(d_idx_out_, B);

                for (int b = 0; b < B; ++b) {
                    if (!ended_var[b]) continue;
                    if (aev_sub_h[b] == board_end_think_sub) {
                        // Append end_think, transition to FINAL.
                        int full = board_sub_to_full[aev_sub_h[b]];
                        token_ids[b].push_back(full);
                        block_ids[b].push_back(orphan_ctr[b]++);
                        ended_thinking[b] = true;
                        in_variation[b] = false;
                    } else {
                        // new_variation — don't append; in_variation stays true,
                        // next iter starts a new variation root_move.
                    }
                }
            }

            // ---- FINAL state (gated by ended_thinking) ----
            bool any_final = false;
            for (int b = 0; b < B; ++b) if (ended_thinking[b] && active[b]) any_final = true;
            if (any_final) {
                // (1) Sample final move via policy_head from last-pos hidden.
                for (int b = 0; b < B; ++b) {
                    gather_pos[b] = (int)token_ids[b].size() - 1;
                    if (gather_pos[b] < 0) gather_pos[b] = 0;
                }
                run_forward_and_gather(true, gather_pos);
                gemm_fp16(d_th_last_h_, w_.policy_head_w, w_.policy_head_b,
                          d_logits_buf_, B, Mv, E, nullptr, 0, stream_.get());
                // Match Python: no legal-mask filter on the final move either.
                argmax_fp16(d_logits_buf_, d_idx_out_, B, Mv, stream_.get());
                stream_.sync();
                auto final_idx_h = d2h_int32(d_idx_out_, B);

                for (int b = 0; b < B; ++b) {
                    if (!(ended_thinking[b] && active[b])) continue;
                    int sub = final_idx_h[b];
                    int full = move_sub_to_full[sub];
                    token_ids[b].push_back(full);
                    block_ids[b].push_back(orphan_ctr[b]++);
                    int fid = chunk_start + b;
                    auto& r = all_results[fid];
                    std::string mp = vocab_->idxToToken(full);
                    r.move = decoder::DecoderVocab::pseudoToStandardUci(mp);
                }

                // (2) Final WL via wl_head from final_move position.
                for (int b = 0; b < B; ++b) {
                    gather_pos[b] = (int)token_ids[b].size() - 1;
                    if (gather_pos[b] < 0) gather_pos[b] = 0;
                }
                run_forward_and_gather(true, gather_pos);
                gemm_fp16(d_th_last_h_, w_.wl_w1_w, w_.wl_w1_b,
                          d_logits_buf_, B, H, E, nullptr, 0, stream_.get());
                mish_inplace_fp16(d_logits_buf_, B * H, stream_.get());
                gemm_fp16(d_logits_buf_, w_.wl_w2_w, w_.wl_w2_b,
                          d_logits_buf_ + B * H, B, Kb, H, nullptr, 0, stream_.get());
                argmax_fp16(d_logits_buf_ + B * H, d_idx_out_, B, Kb, stream_.get());
                stream_.sync();
                auto fwl_h = d2h_int32(d_idx_out_, B);

                for (int b = 0; b < B; ++b) {
                    if (!(ended_thinking[b] && active[b])) continue;
                    int wl_pos_idx = (int)token_ids[b].size();
                    token_ids[b].push_back(wl_value_idx);
                    block_ids[b].push_back(orphan_ctr[b]++);
                    int fid = chunk_start + b;
                    auto& r = all_results[fid];
                    r.wl_positions.push_back(wl_pos_idx);
                    r.wl_indices.push_back(fwl_h[b]);
                    r.wl_values.push_back(wl_centers_h[fwl_h[b]]);
                    r.final_wl_index = fwl_h[b];
                    r.final_wl_value = wl_centers_h[fwl_h[b]];
                }

                // (3) Final D via d_head from wl position (with WL injected).
                for (int b = 0; b < B; ++b) {
                    gather_pos[b] = (int)token_ids[b].size() - 1;
                    if (gather_pos[b] < 0) gather_pos[b] = 0;
                }
                run_forward_and_gather(true, gather_pos);
                gemm_fp16(d_th_last_h_, w_.d_w1_w, w_.d_w1_b,
                          d_logits_buf_, B, H, E, nullptr, 0, stream_.get());
                mish_inplace_fp16(d_logits_buf_, B * H, stream_.get());
                gemm_fp16(d_logits_buf_, w_.d_w2_w, w_.d_w2_b,
                          d_logits_buf_ + B * H, B, Kb, H, nullptr, 0, stream_.get());
                argmax_fp16(d_logits_buf_ + B * H, d_idx_out_, B, Kb, stream_.get());
                stream_.sync();
                auto fd_h = d2h_int32(d_idx_out_, B);

                for (int b = 0; b < B; ++b) {
                    if (!(ended_thinking[b] && active[b])) continue;
                    int d_pos_idx = (int)token_ids[b].size();
                    token_ids[b].push_back(d_value_idx);
                    block_ids[b].push_back(orphan_ctr[b]++);
                    int fid = chunk_start + b;
                    auto& r = all_results[fid];
                    r.d_positions.push_back(d_pos_idx);
                    r.d_indices.push_back(fd_h[b]);
                    r.d_values.push_back(d_centers_h[fd_h[b]]);
                    r.final_d_index = fd_h[b];
                    r.final_d_value = d_centers_h[fd_h[b]];
                    r.ended_thinking = true;

                    // Mark slot done.
                    active[b] = false;
                }
            }

            // Truncation check.
            for (int b = 0; b < B; ++b) {
                if (active[b] && (int)token_ids[b].size() >= max_S) {
                    truncated[b] = true;
                    active[b] = false;
                }
            }
        }

        // Commit results: fill token_ids/block_ids/truncated for each slot.
        for (int b = 0; b < chunk_n; ++b) {
            int fid = chunk_start + b;
            auto& r = all_results[fid];
            r.token_ids = token_ids[b];
            r.block_ids = block_ids[b];
            r.truncated = truncated[b];
            // r.move was set in FINAL state (or stays empty if truncated before FINAL).
        }
    }

    return all_results;
}

}  // namespace cutlass_engine

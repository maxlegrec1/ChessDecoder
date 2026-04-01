#include "batched_engine.hpp"

#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace decoder
{

// ============================================================================
// Construction: load model, upload head weights to GPU
// ============================================================================

BatchedInferenceEngine::BatchedInferenceEngine(
    const std::string& backbone_pt_path,
    const std::string& weights_dir,
    const std::string& vocab_path,
    const std::string& config_path,
    int max_batch_size)
    : max_batch_size_(max_batch_size)
{
    // Load config (minimal JSON parser, same as decoder_engine.cpp)
    std::ifstream cf(config_path);
    if (!cf) throw std::runtime_error("Failed to open config: " + config_path);
    std::string config_json((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());

    auto readInt = [&](const std::string& key) -> int {
        std::string search = "\"" + key + "\"";
        size_t pos = config_json.find(search);
        if (pos == std::string::npos)
            throw std::runtime_error("Key not found in config: " + key);
        pos = config_json.find(':', pos) + 1;
        while (pos < config_json.size() && std::isspace(config_json[pos])) pos++;
        size_t end = pos;
        if (config_json[end] == '-') end++;
        while (end < config_json.size() && std::isdigit(config_json[end])) end++;
        return std::stoi(config_json.substr(pos, end - pos));
    };

    embed_dim_ = readInt("embed_dim");
    int num_layers = readInt("num_layers");
    int num_heads = readInt("num_heads");
    int head_dim = embed_dim_ / num_heads;
    max_seq_len_ = readInt("max_seq_len");
    int board_vocab = readInt("board_vocab_size");
    int move_vocab = readInt("move_vocab_size");
    int n_buckets = readInt("n_buckets");
    int val_hidden = readInt("value_hidden_size");

    int num_fourier_freq = readInt("num_fourier_freq");

    // Load vocab
    vocab_ = std::make_unique<DecoderVocab>(vocab_path);

    // Load head weights
    heads_ = std::make_unique<Heads>(weights_dir, embed_dim_, board_vocab,
                                     move_vocab, val_hidden, n_buckets, num_fourier_freq);

    // Load backbone
    backbone_ = std::make_unique<BatchedBackbone>(
        backbone_pt_path, num_layers, num_heads, head_dim,
        embed_dim_, max_seq_len_, max_batch_size);

    // Upload head weights to GPU as FP16 (transposed for matmul)
    auto uploadWeightT = [](const float* data, int out_dim, int in_dim) {
        auto w = torch::from_blob(const_cast<float*>(data),
            {out_dim, in_dim}, torch::kFloat32).to(torch::kFloat16).to(torch::kCUDA);
        return w.t().contiguous();
    };
    auto uploadBias = [](const float* data, int dim) {
        return torch::from_blob(const_cast<float*>(data),
            {dim}, torch::kFloat32).to(torch::kFloat16).to(torch::kCUDA);
    };

    // Board head
    board_w_t_ = uploadWeightT(heads_->boardWeightData(), board_vocab, embed_dim_);
    board_b_ = uploadBias(heads_->boardBiasData(), board_vocab);
    {
        auto lut_cpu = torch::zeros({board_vocab}, torch::kInt64);
        for (int i = 0; i < board_vocab; i++)
            lut_cpu[i] = vocab_->boardIdxToFullIdx(i);
        board_lut_ = lut_cpu.to(torch::kCUDA);
    }

    // Policy heads
    policy_w_t_ = uploadWeightT(heads_->policyWeightData(), move_vocab, embed_dim_);
    policy_b_ = uploadBias(heads_->policyBiasData(), move_vocab);
    think_w_t_ = uploadWeightT(heads_->thinkingPolicyWeightData(), move_vocab, embed_dim_);
    think_b_ = uploadBias(heads_->thinkingPolicyBiasData(), move_vocab);

    // Value heads (WL)
    {
        int H = val_hidden;
        wl_w1_t_ = uploadWeightT(heads_->wlW1WeightData(), H, embed_dim_);
        wl_b1_ = uploadBias(heads_->wlW1BiasData(), H);
        wl_w2_t_ = uploadWeightT(heads_->wlW2WeightData(), n_buckets, H);
        wl_b2_ = uploadBias(heads_->wlW2BiasData(), n_buckets);
        wl_centers_ = torch::from_blob(const_cast<float*>(heads_->wlBucketCentersData()),
            {n_buckets}, torch::kFloat32).to(torch::kCUDA);
    }

    // Value heads (D)
    {
        int H = val_hidden;
        d_w1_t_ = uploadWeightT(heads_->dW1WeightData(), H, embed_dim_);
        d_b1_ = uploadBias(heads_->dW1BiasData(), H);
        d_w2_t_ = uploadWeightT(heads_->dW2WeightData(), n_buckets, H);
        d_b2_ = uploadBias(heads_->dW2BiasData(), n_buckets);
        d_centers_ = torch::from_blob(const_cast<float*>(heads_->dBucketCentersData()),
            {n_buckets}, torch::kFloat32).to(torch::kCUDA);
    }

    std::cout << "[BatchedEngine] Ready: batch=" << max_batch_size
              << ", embed=" << embed_dim_ << ", max_seq=" << max_seq_len_
              << std::endl;
}

// ============================================================================
// Head evaluations (batched, GPU)
// ============================================================================

torch::Tensor BatchedInferenceEngine::sampleBatched(torch::Tensor logits, float temp)
{
    // logits: [B, V] FP16 on CUDA — cast to FP32 to avoid Inf/NaN from FP16 overflow
    auto logits_f32 = logits.to(torch::kFloat32);

    if (temp <= 0.0f)
        return torch::argmax(logits_f32, /*dim=*/1);  // [B]

    // Clamp to prevent NaN in softmax (FP16 matmul can produce ±Inf)
    logits_f32 = logits_f32.clamp(-1e4f, 1e4f);
    auto probs = torch::softmax(logits_f32 / temp, /*dim=*/1);
    return torch::multinomial(probs, /*num_samples=*/1).squeeze(1);  // [B]
}

torch::Tensor BatchedInferenceEngine::evalThinkingPolicyHead(torch::Tensor h, float temp)
{
    // h: [B, E] FP16 → logits [B, move_vocab] → sub_idx [B]
    auto logits = torch::mm(h, think_w_t_) + think_b_;
    return sampleBatched(logits, temp);  // [B] move sub-vocab indices
}

torch::Tensor BatchedInferenceEngine::evalBoardHead(torch::Tensor h, float temp)
{
    // h: [B, E] FP16 → logits [B, board_vocab]
    auto logits = torch::mm(h, board_w_t_) + board_b_;
    return sampleBatched(logits, temp);  // [B] board sub-vocab indices
}

torch::Tensor BatchedInferenceEngine::evalWlHead(torch::Tensor h, float temp)
{
    // h: [B, E] → MLP → [B, n_buckets] → sample → bucket center values [B]
    auto hidden = torch::mm(h, wl_w1_t_) + wl_b1_;
    hidden = torch::mish(hidden);
    auto logits = torch::mm(hidden, wl_w2_t_) + wl_b2_;
    auto idx = sampleBatched(logits, temp);  // [B]
    return wl_centers_.index({idx});  // [B] float values
}

torch::Tensor BatchedInferenceEngine::evalDHead(torch::Tensor h, float temp)
{
    auto hidden = torch::mm(h, d_w1_t_) + d_b1_;
    hidden = torch::mish(hidden);
    auto logits = torch::mm(hidden, d_w2_t_) + d_b2_;
    auto idx = sampleBatched(logits, temp);
    return d_centers_.index({idx});
}

torch::Tensor BatchedInferenceEngine::evalPolicyHead(
    torch::Tensor h, float temp,
    const std::vector<std::string>& fens)
{
    // h: [B, E] → logits [B, move_vocab]
    auto logits = torch::mm(h, policy_w_t_) + policy_b_;

    // Mask illegal moves per FEN
    int B = h.size(0);
    auto logits_a = logits.to(torch::kFloat32).cpu();
    auto logits_acc = logits_a.accessor<float, 2>();

    for (int b = 0; b < B; b++)
    {
        auto legal = vocab_->legalMoveIndices(fens[b]);
        std::set<int> legal_set(legal.begin(), legal.end());
        for (int j = 0; j < logits_a.size(1); j++)
        {
            if (legal_set.find(j) == legal_set.end())
                logits_acc[b][j] = -1e9f;
        }
    }
    logits_a = logits_a.cuda().to(torch::kFloat16);

    return sampleBatched(logits_a, temp);  // [B] move sub-vocab indices
}

// ============================================================================
// Main inference: predictMoves (lockstep state machine)
// ============================================================================

std::vector<BatchedInferenceEngine::Result>
BatchedInferenceEngine::predictMoves(
    const std::vector<std::string>& fens, float temperature)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    torch::NoGradGuard no_grad;

    int B_actual = static_cast<int>(fens.size());
    if (B_actual > max_batch_size_)
        throw std::runtime_error("Batch size exceeds max_batch_size");

    // Always work with max_batch_size internally — pad unused slots
    int B = max_batch_size_;

    // Pad fens list to B (reuse first FEN for padding, results discarded)
    std::vector<std::string> padded_fens = fens;
    while ((int)padded_fens.size() < B)
        padded_fens.push_back(fens[0]);

    // Resolve temperatures
    float think_temp = (think_temperature >= 0.0f) ? think_temperature : temperature;
    float policy_temp = (policy_temperature >= 0.0f) ? policy_temperature : temperature;
    float board_temp = board_temperature;
    float wl_temp = wl_temperature;
    float d_temp = d_temperature;

    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    // Per-sequence state
    std::vector<std::vector<int>> token_ids(B);
    std::vector<std::vector<int>> block_ids(B);
    std::vector<std::vector<std::pair<int, float>>> wl_entries(B), d_entries(B);
    std::vector<int> next_block(B, 0);
    std::vector<int> orphan_ctr(B, 10000);
    std::vector<std::string> first_root_move(B);

    // Move sub-vocab index to full vocab index lookup (on GPU)
    int mv = heads_->moveVocabSize();
    auto move_lut_cpu = torch::zeros({mv}, torch::kInt64);
    for (int i = 0; i < mv; i++)
        move_lut_cpu[i] = vocab_->moveIdxToFullIdx(i);
    auto move_lut = move_lut_cpu.cuda();

    int start_think_idx = vocab_->startThinkIdx();
    int end_think_idx = vocab_->endThinkIdx();
    int end_var_idx = vocab_->endVarIdx();
    int wl_value_idx = vocab_->wlValueIdx();
    int d_value_idx = vocab_->dValueIdx();
    int board_end_var_sub = vocab_->boardEndVarIdx();
    int board_end_think_sub = vocab_->boardEndThinkIdx();

    // All-true mask (all elements active at init)
    auto all_active = torch::ones({B}, opts_bool);

    // ================================================================
    // Phase 1: Root boards + start_think
    // ================================================================

    // Encode all FENs → [B, 68] token IDs
    int root_len = 68;
    auto root_ids = torch::zeros({B, root_len}, opts_int);
    for (int b = 0; b < B; b++)
    {
        auto ids = vocab_->fenToTokenIds(padded_fens[b]);
        for (int j = 0; j < root_len && j < (int)ids.size(); j++)
            root_ids[b][j] = ids[j];

        // Record in per-sequence state
        int bid = next_block[b]++;
        for (int j = 0; j < root_len; j++)
        {
            token_ids[b].push_back(ids[j]);
            block_ids[b].push_back(bid);
        }
    }

    // Append start_think
    int init_len = root_len + 1;
    auto init_ids = torch::zeros({B, init_len}, opts_int);
    init_ids.index({torch::indexing::Slice(), torch::indexing::Slice(0, root_len)}) = root_ids;
    init_ids.index({torch::indexing::Slice(), root_len}) = start_think_idx;

    for (int b = 0; b < B; b++)
    {
        token_ids[b].push_back(start_think_idx);
        block_ids[b].push_back(orphan_ctr[b]++);
    }

    auto init_pos = torch::arange(init_len, opts_int).unsqueeze(0).expand({B, init_len});
    auto init_ov = torch::zeros({B, init_len}, opts_fp16);
    auto init_om = torch::zeros({B, init_len}, opts_bool);

    // Causal prefill — all tokens are real for all elements
    backbone_->resetCausal();
    backbone_->resetPrefix();
    auto num_real_init = torch::full({B}, init_len, opts_int);
    backbone_->causalForward(init_ids, init_pos, init_ov, init_om,
                             all_active, num_real_init);
    backbone_->syncCausalToGraph();

    // Prefix init with block-aware mask
    auto block_ids_t = torch::zeros({B, init_len}, opts_int);
    for (int b = 0; b < B; b++)
        for (int j = 0; j < init_len; j++)
            block_ids_t[b][j] = block_ids[b][j];

    auto same_block = block_ids_t.unsqueeze(2) == block_ids_t.unsqueeze(1);  // [B, S, S]
    auto causal_mask = torch::tril(torch::ones({init_len, init_len}, opts_bool));
    auto prefix_mask = torch::where(
        same_block | causal_mask.unsqueeze(0),
        torch::tensor(0.0f, torch::kCUDA),
        torch::tensor(-1e9f, torch::kCUDA));
    prefix_mask = prefix_mask.unsqueeze(1);  // [B, 1, S, S]

    auto h_init = backbone_->prefixForward(init_ids, init_pos, prefix_mask,
                                           init_ov, init_om, all_active);
    backbone_->syncPrefixToGraph();
    auto saved_h = h_init.index({torch::indexing::Slice(), init_len - 1}).contiguous();

    // Per-sequence position tracking
    std::vector<int> seq_pos(B, init_len);
    int cur_len = init_len;

    // Causal hidden state saved at end of board gen loop
    torch::Tensor h_board_after_loop;

    // ================================================================
    // Phase 2: Variation loop
    // ================================================================

    auto active = torch::ones({B}, opts_bool);
    auto in_variation = torch::ones({B}, opts_bool);

    for (int iter = 0; iter < max_seq_len_; iter++)
    {
        if (!active.any().item<bool>())
            break;

        auto need_move = active & in_variation;  // [B]

        if (!need_move.any().item<bool>())
            break;

        // Sync cur_len from per-sequence positions
        cur_len = 0;
        for (int b = 0; b < B; b++)
            if (need_move[b].item<bool>() && seq_pos[b] > cur_len)
                cur_len = seq_pos[b];

        // Helper: build per-sequence position tensor from seq_pos + offset
        auto makePosT = [&](int off = 0) {
            auto p = torch::zeros({B, 1}, opts_int);
            for (int b = 0; b < B; b++)
                p[b][0] = seq_pos[b] + off;
            return p;
        };

        // ── Step 2a: MOVE ──────────────────────────────────────────────
        {
            auto logits = torch::mm(saved_h, think_w_t_) + think_b_;
            auto logits_f32 = logits.to(torch::kFloat32);
            logits_f32.index_put_({~need_move}, torch::tensor(-1e9f, torch::kCUDA));
            logits = logits_f32.to(torch::kFloat16);
            auto sub_idx = sampleBatched(logits, think_temp);
            auto full_idx = move_lut.index({sub_idx});

            auto full_idx_cpu = full_idx.cpu();
            auto full_idx_a = full_idx_cpu.accessor<int64_t, 1>();
            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                int tok = full_idx_a[b];
                token_ids[b].push_back(tok);
                block_ids[b].push_back(orphan_ctr[b]++);
                if (first_root_move[b].empty())
                    first_root_move[b] = vocab_->idxToToken(tok);
            }

            // Prefix incremental with active mask
            auto move_ids_t = full_idx.unsqueeze(1);
            auto move_pos = makePosT();
            auto move_ov = torch::zeros({B, 1}, opts_fp16);
            auto move_om = torch::zeros({B, 1}, opts_bool);
            auto h_move = backbone_->prefixIncremental(move_ids_t, move_pos,
                                                       move_ov, move_om, need_move);
            saved_h = h_move.squeeze(1);
            for (int b = 0; b < B; b++)
                if (need_move[b].item<bool>()) seq_pos[b]++;
            cur_len = *std::max_element(seq_pos.begin(), seq_pos.end());
        }

        // ── Step 2b: WL_D ──────────────────────────────────────────────
        {
            // WL
            auto wl_vals = evalWlHead(saved_h, wl_temp);
            auto wl_cpu = wl_vals.cpu();
            auto wl_a = wl_cpu.accessor<float, 1>();

            auto wl_ids_t = torch::full({B, 1}, wl_value_idx, opts_int);
            auto wl_pos = makePosT();
            auto wl_ov = wl_vals.to(torch::kFloat16).unsqueeze(1);
            auto wl_om = need_move.unsqueeze(1);
            auto h_wl = backbone_->prefixIncremental(wl_ids_t, wl_pos,
                                                     wl_ov, wl_om, need_move);
            saved_h = h_wl.squeeze(1);

            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                token_ids[b].push_back(wl_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                wl_entries[b].push_back({seq_pos[b], wl_a[b]});
                seq_pos[b]++;
            }
            cur_len = *std::max_element(seq_pos.begin(), seq_pos.end());

            // D
            auto d_vals = evalDHead(saved_h, d_temp);
            auto d_cpu = d_vals.cpu();
            auto d_a = d_cpu.accessor<float, 1>();

            auto d_ids_t = torch::full({B, 1}, d_value_idx, opts_int);
            auto d_pos = makePosT();
            auto d_ov = d_vals.to(torch::kFloat16).unsqueeze(1);
            auto d_om = need_move.unsqueeze(1);
            auto h_d = backbone_->prefixIncremental(d_ids_t, d_pos,
                                                    d_ov, d_om, need_move);
            saved_h = h_d.squeeze(1);

            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                token_ids[b].push_back(d_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                d_entries[b].push_back({seq_pos[b], d_a[b]});
                seq_pos[b]++;
            }
            cur_len = *std::max_element(seq_pos.begin(), seq_pos.end());
        }

        // ── Step 2c: BOARD (67 autoregressive steps) ────────────────────
        {
            // Causal catch-up: forward orphan tokens since last causal
            // update.  Sequences may have different numbers of new tokens.
            int causal_start = backbone_->causalLen();
            int max_new = 0;
            for (int b = 0; b < B; b++)
            {
                int my = seq_pos[b] - causal_start;
                if (my > max_new) max_new = my;
            }

            if (max_new > 0)
            {
                backbone_->syncGraphToCausal();
                auto catch_ids = torch::zeros({B, max_new}, opts_int);
                auto catch_pos = torch::zeros({B, max_new}, opts_int);
                auto catch_ov = torch::zeros({B, max_new}, opts_fp16);
                auto catch_om = torch::zeros({B, max_new}, opts_bool);

                // Build num_real tensor for per-element masking
                auto num_real = torch::zeros({B}, opts_int);

                for (int b = 0; b < B; b++)
                {
                    int my_new = seq_pos[b] - causal_start;
                    num_real[b] = my_new;
                    for (int j = 0; j < max_new; j++)
                    {
                        int pos = causal_start + j;
                        if (j < my_new && pos < (int)token_ids[b].size())
                        {
                            catch_ids[b][j] = token_ids[b][pos];
                            catch_pos[b][j] = pos;
                        }
                        else
                        {
                            // Pad: repeat last real token at same position
                            // (masked out by valid bitmap, so content doesn't matter)
                            catch_ids[b][j] = token_ids[b].back();
                            catch_pos[b][j] = seq_pos[b] - 1;
                        }
                    }
                }

                // Set override for WL/D positions
                for (int b = 0; b < B; b++)
                {
                    if (!need_move[b].item<bool>()) continue;
                    if (!wl_entries[b].empty())
                    {
                        auto& wl_e = wl_entries[b].back();
                        int rel = wl_e.first - causal_start;
                        if (rel >= 0 && rel < max_new)
                        {
                            catch_ov[b][rel] = static_cast<at::Half>(wl_e.second);
                            catch_om[b][rel] = true;
                        }
                    }
                    if (!d_entries[b].empty())
                    {
                        auto& d_e = d_entries[b].back();
                        int rel = d_e.first - causal_start;
                        if (rel >= 0 && rel < max_new)
                        {
                            catch_ov[b][rel] = static_cast<at::Half>(d_e.second);
                            catch_om[b][rel] = true;
                        }
                    }
                }

                // Per-element masking: num_real tells backbone how many tokens
                // are real per element, so padded positions get -inf in mask
                backbone_->causalForward(catch_ids, catch_pos, catch_ov, catch_om,
                                         need_move, num_real);

                backbone_->syncCausalToGraph();
            }

            // Board gen with per-sequence positions
            int start_pos_idx = vocab_->startPosIdx();
            auto first_tok = torch::full({B, 1}, start_pos_idx, opts_int);
            auto first_pos = makePosT();
            auto first_ov = torch::zeros({B, 1}, opts_fp16);
            auto first_om = torch::zeros({B, 1}, opts_bool);

            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                int bid = next_block[b]++;
                token_ids[b].push_back(start_pos_idx);
                block_ids[b].push_back(bid);
                seq_pos[b]++;
            }

            // Pre-mark all 68 board positions as valid in one kernel launch,
            // so the 68 causalIncremental calls below skip redundant marking.
            int board_cache_start = backbone_->causalLen();
            backbone_->markCausalValidRange(board_cache_start, 68, need_move);

            auto h_board = backbone_->causalIncremental(first_tok, first_pos,
                                                        first_ov, first_om, need_move);

            // Board gen: 67 autoregressive steps
            auto board_output = torch::zeros({B, 67}, opts_int);

            for (int step = 0; step < 67; step++)
            {
                auto h_last = h_board.squeeze(1);
                auto board_logits = torch::mm(h_last, board_w_t_) + board_b_;
                auto sub_idx = sampleBatched(board_logits, 0.0f);
                auto full_idx = board_lut_.index({sub_idx});
                board_output.index_put_({torch::indexing::Slice(), step}, full_idx);

                auto next_ids = full_idx.unsqueeze(1);
                auto next_pos = makePosT();
                for (int b = 0; b < B; b++)
                    if (need_move[b].item<bool>()) seq_pos[b]++;
                h_board = backbone_->causalIncremental(next_ids, next_pos,
                                                      first_ov, first_om, need_move);
            }

            h_board_after_loop = h_board.squeeze(1).contiguous();
            backbone_->syncGraphToCausal();

            cur_len = *std::max_element(seq_pos.begin(), seq_pos.end());

            // Copy board tokens to CPU
            auto board_cpu = board_output.cpu();
            auto board_acc = board_cpu.accessor<int64_t, 2>();
            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                int bid = next_block[b] - 1;
                for (int j = 0; j < 67; j++)
                {
                    token_ids[b].push_back(board_acc[b][j]);
                    block_ids[b].push_back(bid);
                }
            }

            // Prefix block forward for the 68 board tokens with per-seq positions
            auto board_ids_t = torch::zeros({B, 68}, opts_int);
            auto board_pos_t = torch::zeros({B, 68}, opts_int);
            auto board_ov_t = torch::zeros({B, 68}, opts_fp16);
            auto board_om_t = torch::zeros({B, 68}, opts_bool);

            for (int b = 0; b < B; b++)
            {
                int bstart = seq_pos[b] - 68;
                for (int j = 0; j < 68; j++)
                {
                    int idx = bstart + j;
                    board_ids_t[b][j] = (idx >= 0 && idx < (int)token_ids[b].size())
                                        ? token_ids[b][idx] : 0;
                    board_pos_t[b][j] = bstart + j;
                }
            }

            backbone_->syncGraphToPrefix();
            auto h_block = backbone_->prefixBlockForward(
                board_ids_t, board_pos_t, board_ov_t, board_om_t, need_move);
            backbone_->syncPrefixToGraph();
            saved_h = h_block.index({torch::indexing::Slice(), 67}).contiguous();
        }

        // ── Step 2d: AFTER_BOARD ────────────────────────────────────────
        {
            auto h_dec = h_board_after_loop;
            auto board_logits = torch::mm(h_dec, board_w_t_) + board_b_;
            auto sub_idx = sampleBatched(board_logits, board_temp);
            auto sub_idx_cpu = sub_idx.cpu();
            auto sub_a = sub_idx_cpu.accessor<int64_t, 1>();

            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;

                if (sub_a[b] == board_end_var_sub)
                {
                    token_ids[b].push_back(end_var_idx);
                    block_ids[b].push_back(orphan_ctr[b]++);
                    in_variation[b] = false;
                    seq_pos[b]++;
                }
                // else: continue PV — no token, no position advance
            }

            // Prefix for end_var sequences
            auto ended_var = need_move & ~in_variation;
            if (ended_var.any().item<bool>())
            {
                auto ev_ids = torch::full({B, 1}, end_var_idx, opts_int);
                auto ev_pos = makePosT(-1);
                auto ev_ov = torch::zeros({B, 1}, opts_fp16);
                auto ev_om = torch::zeros({B, 1}, opts_bool);
                auto h_ev = backbone_->prefixIncremental(ev_ids, ev_pos,
                                                         ev_ov, ev_om, ended_var);
                // Blend: use h_ev for ended_var sequences, keep saved_h for others
                auto mask_2d = ended_var.unsqueeze(1);
                saved_h = torch::where(mask_2d, h_ev.squeeze(1), saved_h);
            }
        }

        // ── Step 2e: AFTER_END_VAR ──────────────────────────────────────
        {
            auto between_vars = active & ~in_variation;
            if (between_vars.any().item<bool>())
            {
                auto ev_tok = torch::zeros({B, 1}, opts_int);
                for (int b = 0; b < B; b++)
                    ev_tok[b][0] = token_ids[b].back();
                auto ev_pos2 = makePosT(-1);
                auto ev_ov2 = torch::zeros({B, 1}, opts_fp16);
                auto ev_om2 = torch::zeros({B, 1}, opts_bool);

                // Probe: get hidden WITHOUT touching causal cache
                auto h_ev2 = backbone_->causalProbe(ev_tok, ev_pos2, ev_ov2, ev_om2);

                auto h_dec2 = h_ev2.squeeze(1);
                auto logits2 = torch::mm(h_dec2, board_w_t_) + board_b_;
                auto sub2 = sampleBatched(logits2, board_temp);
                auto sub2_cpu = sub2.cpu();
                auto sub2_a = sub2_cpu.accessor<int64_t, 1>();

                std::vector<int> end_think_seqs;
                for (int b = 0; b < B; b++)
                {
                    if (!between_vars[b].item<bool>()) continue;

                    if (sub2_a[b] == board_end_think_sub)
                    {
                        token_ids[b].push_back(end_think_idx);
                        block_ids[b].push_back(orphan_ctr[b]++);
                        active[b] = false;
                        seq_pos[b]++;
                        end_think_seqs.push_back(b);
                    }
                    else
                    {
                        // New variation
                        in_variation[b] = true;
                    }
                }

                // Prefix incremental for end_think sequences
                if (!end_think_seqs.empty())
                {
                    auto et_active = torch::zeros({B}, opts_bool);
                    for (int b : end_think_seqs) et_active[b] = true;

                    auto et_ids = torch::full({B, 1}, end_think_idx, opts_int);
                    auto et_pos = torch::zeros({B, 1}, opts_int);
                    for (int b : end_think_seqs)
                        et_pos[b][0] = seq_pos[b] - 1;
                    auto et_ov = torch::zeros({B, 1}, opts_fp16);
                    auto et_om = torch::zeros({B, 1}, opts_bool);

                    auto h_et = backbone_->prefixIncremental(et_ids, et_pos,
                                                             et_ov, et_om, et_active);
                    auto h_et1 = h_et.squeeze(1);

                    for (int b : end_think_seqs)
                        saved_h[b] = h_et1[b];
                }
            }
        }

        // Check max sequence length
        for (int b = 0; b < B; b++)
        {
            if (active[b].item<bool>() && (int)token_ids[b].size() >= max_seq_len_ - 5)
                active[b] = false;
        }
    }

    // ================================================================
    // Phase 3: FINAL — predict final moves
    // ================================================================

    auto final_sub_idx = evalPolicyHead(saved_h, policy_temp, padded_fens);
    auto final_full = move_lut.index({final_sub_idx});
    auto final_cpu = final_full.cpu();
    auto final_a = final_cpu.accessor<int64_t, 1>();

    // Build results
    std::vector<Result> results(B);
    for (int b = 0; b < B; b++)
    {
        int tok = final_a[b];
        std::string move = vocab_->pseudoToStandardUci(vocab_->idxToToken(tok));

        if (move.empty() && !first_root_move[b].empty())
            move = vocab_->pseudoToStandardUci(first_root_move[b]);

        results[b].move = move;
        results[b].token_ids = std::move(token_ids[b]);
        results[b].wl_entries = std::move(wl_entries[b]);
        results[b].d_entries = std::move(d_entries[b]);
    }

    results.resize(B_actual);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    total_time += elapsed;
    for (int b = 0; b < B_actual; b++)
        total_tokens += results[b].token_ids.size();

    return results;
}

} // namespace decoder

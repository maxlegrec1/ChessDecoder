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
    // Same pattern as decoder_engine.cpp constructor
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

    // Causal prefill, then sync to graph buffers for incremental mode
    backbone_->resetCausal();
    backbone_->resetPrefix();
    backbone_->causalForward(init_ids, init_pos, init_ov, init_om);
    backbone_->syncCausalToGraph();

    // Prefix init with block-aware mask
    // Build block_ids tensor [B, init_len]
    auto block_ids_t = torch::zeros({B, init_len}, opts_int);
    for (int b = 0; b < B; b++)
        for (int j = 0; j < init_len; j++)
            block_ids_t[b][j] = block_ids[b][j];

    // Build prefix mask [B, 1, init_len, init_len]
    auto same_block = block_ids_t.unsqueeze(2) == block_ids_t.unsqueeze(1);  // [B, S, S]
    auto causal_mask = torch::tril(torch::ones({init_len, init_len}, opts_bool));
    auto prefix_mask = torch::where(
        same_block | causal_mask.unsqueeze(0),
        torch::tensor(0.0f, torch::kCUDA),
        torch::tensor(-1e9f, torch::kCUDA));
    prefix_mask = prefix_mask.unsqueeze(1);  // [B, 1, S, S]

    auto h_init = backbone_->prefixForward(init_ids, init_pos, prefix_mask, init_ov, init_om);
    backbone_->syncPrefixToGraph();
    // Extract hidden at last position (start_think) → [B, E]
    auto saved_h = h_init.index({torch::indexing::Slice(), init_len - 1}).contiguous();  // [B, E]

    int cur_len = init_len;  // All sequences at same length after init

    // ================================================================
    // Phase 2: Variation loop
    // ================================================================

    auto active = torch::ones({B}, opts_bool);
    auto in_variation = torch::ones({B}, opts_bool);  // Start: all entering first variation

    for (int iter = 0; iter < max_seq_len_; iter++)
    {
        if (!active.any().item<bool>())
            break;

        auto need_move = active & in_variation;  // [B]

        if (!need_move.any().item<bool>())
        {
            // All active sequences are between variations (AFTER_END_VAR)
            // This shouldn't normally happen on the first iteration, but handle it
            break;
        }

        // ── Step 2a: MOVE ──────────────────────────────────────────────
        {
            auto logits = torch::mm(saved_h, think_w_t_) + think_b_;  // [B, move_vocab]
            // Mask inactive sequences
            // Mask inactive: cast logits to FP32 for masking, then back
            auto logits_f32 = logits.to(torch::kFloat32);
            logits_f32.index_put_({~need_move}, torch::tensor(-1e9f, torch::kCUDA));
            logits = logits_f32.to(torch::kFloat16);
            auto sub_idx = sampleBatched(logits, think_temp);  // [B]
            auto full_idx = move_lut.index({sub_idx});          // [B] full vocab

            // Record tokens (CPU)
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

            // Prefix incremental
            auto move_ids_t = full_idx.unsqueeze(1);  // [B, 1]
            auto move_pos = torch::full({B, 1}, cur_len, opts_int);
            auto move_ov = torch::zeros({B, 1}, opts_fp16);
            auto move_om = torch::zeros({B, 1}, opts_bool);
            auto h_move = backbone_->prefixIncremental(move_ids_t, move_pos, move_ov, move_om);
            saved_h = h_move.squeeze(1);  // [B, E]
            cur_len++;
        }

        // ── Step 2b: WL_D ──────────────────────────────────────────────
        {
            // WL
            auto wl_vals = evalWlHead(saved_h, wl_temp);  // [B] float
            auto wl_cpu = wl_vals.cpu();
            auto wl_a = wl_cpu.accessor<float, 1>();

            auto wl_ids_t = torch::full({B, 1}, wl_value_idx, opts_int);
            auto wl_pos = torch::full({B, 1}, cur_len, opts_int);
            auto wl_ov = wl_vals.to(torch::kFloat16).unsqueeze(1);  // [B, 1]
            auto wl_om = need_move.unsqueeze(1);  // override only for active sequences
            auto h_wl = backbone_->prefixIncremental(wl_ids_t, wl_pos, wl_ov, wl_om);
            saved_h = h_wl.squeeze(1);

            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                token_ids[b].push_back(wl_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                wl_entries[b].push_back({cur_len, wl_a[b]});
            }
            cur_len++;

            // D
            auto d_vals = evalDHead(saved_h, d_temp);  // [B] float
            auto d_cpu = d_vals.cpu();
            auto d_a = d_cpu.accessor<float, 1>();

            auto d_ids_t = torch::full({B, 1}, d_value_idx, opts_int);
            auto d_pos = torch::full({B, 1}, cur_len, opts_int);
            auto d_ov = d_vals.to(torch::kFloat16).unsqueeze(1);
            auto d_om = need_move.unsqueeze(1);
            auto h_d = backbone_->prefixIncremental(d_ids_t, d_pos, d_ov, d_om);
            saved_h = h_d.squeeze(1);

            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                token_ids[b].push_back(d_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                d_entries[b].push_back({cur_len, d_a[b]});
            }
            cur_len++;
        }

        // ── Step 2c: BOARD (67 autoregressive steps) ────────────────────
        {
            // Causal catch-up: forward orphan tokens (move + wl + d = 3 tokens)
            // since last causal update. All sequences have same number of new tokens.
            int causal_start = backbone_->causalLen();
            int new_tokens = cur_len - causal_start;

            if (new_tokens > 0)
            {
                // Sync graph → dynamic cache before dynamic forward
                backbone_->syncGraphToCausal();
                // Build catch-up input from token_ids
                auto catch_ids = torch::zeros({B, new_tokens}, opts_int);
                auto catch_pos = torch::zeros({B, new_tokens}, opts_int);
                auto catch_ov = torch::zeros({B, new_tokens}, opts_fp16);
                auto catch_om = torch::zeros({B, new_tokens}, opts_bool);

                for (int b = 0; b < B; b++)
                {
                    for (int j = 0; j < new_tokens; j++)
                    {
                        int pos = causal_start + j;
                        catch_ids[b][j] = (pos < (int)token_ids[b].size())
                                          ? token_ids[b][pos] : 0;
                        catch_pos[b][j] = pos;
                    }
                }

                // Set override for WL/D positions
                // WL is at position cur_len-3 (relative: new_tokens-3), D at cur_len-2 (rel: new_tokens-2)
                // relative to causal_start
                for (int b = 0; b < B; b++)
                {
                    if (!need_move[b].item<bool>()) continue;
                    // WL override
                    if (!wl_entries[b].empty())
                    {
                        auto& wl_e = wl_entries[b].back();
                        int rel = wl_e.first - causal_start;
                        if (rel >= 0 && rel < new_tokens)
                        {
                            catch_ov[b][rel] = static_cast<at::Half>(wl_e.second);
                            catch_om[b][rel] = true;
                        }
                    }
                    if (!d_entries[b].empty())
                    {
                        auto& d_e = d_entries[b].back();
                        int rel = d_e.first - causal_start;
                        if (rel >= 0 && rel < new_tokens)
                        {
                            catch_ov[b][rel] = static_cast<at::Half>(d_e.second);
                            catch_om[b][rel] = true;
                        }
                    }
                }

                backbone_->causalForward(catch_ids, catch_pos, catch_ov, catch_om);
                backbone_->syncCausalToGraph();
            }

            // Get first board token from board head on last causal hidden
            // For simplicity, do one causal incremental to get the hidden for the
            // first board token (start_pos)
            int start_pos_idx = vocab_->startPosIdx();
            auto first_tok = torch::full({B, 1}, start_pos_idx, opts_int);
            auto first_pos = torch::full({B, 1}, cur_len, opts_int);
            auto first_ov = torch::zeros({B, 1}, opts_fp16);
            auto first_om = torch::zeros({B, 1}, opts_bool);

            // Record start_pos
            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                int bid = next_block[b]++;
                token_ids[b].push_back(start_pos_idx);
                block_ids[b].push_back(bid);
            }

            auto h_board = backbone_->causalIncremental(first_tok, first_pos, first_ov, first_om);
            cur_len++;

            // Board gen: 67 autoregressive steps
            auto board_output = torch::zeros({B, 67}, opts_int);

            for (int step = 0; step < 67; step++)
            {
                // Eval board head
                auto h_last = h_board.squeeze(1);  // [B, E]
                auto board_logits = torch::mm(h_last, board_w_t_) + board_b_;  // [B, board_vocab]
                auto sub_idx = sampleBatched(board_logits, 0.0f);  // always argmax for board tokens
                auto full_idx = board_lut_.index({sub_idx});  // [B]
                board_output.index_put_({torch::indexing::Slice(), step}, full_idx);

                // Next step: causal incremental
                auto next_ids = full_idx.unsqueeze(1);  // [B, 1]
                auto next_pos = torch::full({B, 1}, cur_len, opts_int);
                h_board = backbone_->causalIncremental(next_ids, next_pos, first_ov, first_om);
                cur_len++;
            }

            // Sync graph tier back to dynamic causal cache
            // (needed before any subsequent dynamic causal forwards or tier switches)
            backbone_->syncGraphToCausal();

            // ONE sync: copy board tokens to CPU
            auto board_cpu = board_output.cpu();
            auto board_acc = board_cpu.accessor<int64_t, 2>();
            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;
                // The block ID for this board was set when start_pos was added
                int bid = next_block[b] - 1;
                for (int j = 0; j < 67; j++)
                {
                    token_ids[b].push_back(board_acc[b][j]);
                    block_ids[b].push_back(bid);
                }
            }

            // Prefix block forward for the 68 board tokens
            int board_start = cur_len - 68;
            auto board_ids_t = torch::zeros({B, 68}, opts_int);
            auto board_pos_t = torch::zeros({B, 68}, opts_int);
            auto board_ov_t = torch::zeros({B, 68}, opts_fp16);
            auto board_om_t = torch::zeros({B, 68}, opts_bool);

            for (int b = 0; b < B; b++)
            {
                for (int j = 0; j < 68; j++)
                {
                    int idx = board_start + j;
                    board_ids_t[b][j] = (idx < (int)token_ids[b].size())
                                        ? token_ids[b][idx] : 0;
                    board_pos_t[b][j] = board_start + j;
                }
            }

            backbone_->syncGraphToPrefix();
            auto h_block = backbone_->prefixBlockForward(
                board_ids_t, board_pos_t, board_ov_t, board_om_t);
            backbone_->syncPrefixToGraph();
            // Extract hidden at last board token (position 67 within block)
            saved_h = h_block.index({torch::indexing::Slice(), 67}).contiguous();  // [B, E]
        }

        // ── Step 2d: AFTER_BOARD ────────────────────────────────────────
        {
            // Causal incremental on last board token to get board head decision
            int last_tok_pos = cur_len - 1;
            // Get last board token per sequence
            auto last_tok = torch::zeros({B, 1}, opts_int);
            for (int b = 0; b < B; b++)
                last_tok[b][0] = token_ids[b].back();
            auto last_pos = torch::full({B, 1}, last_tok_pos, opts_int);
            auto last_ov = torch::zeros({B, 1}, opts_fp16);
            auto last_om = torch::zeros({B, 1}, opts_bool);

            auto h_after = backbone_->causalIncremental(last_tok, last_pos, last_ov, last_om);
            cur_len++;

            auto h_dec = h_after.squeeze(1);
            auto board_logits = torch::mm(h_dec, board_w_t_) + board_b_;
            auto sub_idx = sampleBatched(board_logits, board_temp);
            auto sub_idx_cpu = sub_idx.cpu();
            auto sub_a = sub_idx_cpu.accessor<int64_t, 1>();

            for (int b = 0; b < B; b++)
            {
                if (!need_move[b].item<bool>()) continue;

                if (sub_a[b] == board_end_var_sub)
                {
                    // End variation
                    token_ids[b].push_back(end_var_idx);
                    block_ids[b].push_back(orphan_ctr[b]++);
                    in_variation[b] = false;

                    // Prefix incremental for end_var
                    // (will be done batched below)
                }
                // else: continue PV (stay in_variation, loop back to MOVE)
            }

            // Prefix incremental for sequences that emitted end_var
            auto ended_var = need_move & ~in_variation;
            if (ended_var.any().item<bool>())
            {
                auto ev_ids = torch::full({B, 1}, end_var_idx, opts_int);
                auto ev_pos = torch::full({B, 1}, cur_len - 1, opts_int);
                auto ev_ov = torch::zeros({B, 1}, opts_fp16);
                auto ev_om = torch::zeros({B, 1}, opts_bool);
                auto h_ev = backbone_->prefixIncremental(ev_ids, ev_pos, ev_ov, ev_om);
                // Update saved_h for sequences that ended variation
                auto mask_2d = ended_var.unsqueeze(1);  // [B, 1]
                saved_h = torch::where(mask_2d, h_ev.squeeze(1), saved_h);
            }
        }

        // ── Step 2e: AFTER_END_VAR ──────────────────────────────────────
        {
            auto between_vars = active & ~in_variation;
            if (between_vars.any().item<bool>())
            {
                // Causal incremental to check end_think vs new_variation
                auto ev_tok = torch::zeros({B, 1}, opts_int);
                for (int b = 0; b < B; b++)
                    ev_tok[b][0] = token_ids[b].back();
                auto ev_pos2 = torch::full({B, 1}, cur_len, opts_int);
                auto ev_ov2 = torch::zeros({B, 1}, opts_fp16);
                auto ev_om2 = torch::zeros({B, 1}, opts_bool);

                auto h_ev2 = backbone_->causalIncremental(ev_tok, ev_pos2, ev_ov2, ev_om2);
                cur_len++;

                auto h_dec2 = h_ev2.squeeze(1);
                auto logits2 = torch::mm(h_dec2, board_w_t_) + board_b_;
                auto sub2 = sampleBatched(logits2, board_temp);
                auto sub2_cpu = sub2.cpu();
                auto sub2_a = sub2_cpu.accessor<int64_t, 1>();

                for (int b = 0; b < B; b++)
                {
                    if (!between_vars[b].item<bool>()) continue;

                    if (sub2_a[b] == board_end_think_sub)
                    {
                        token_ids[b].push_back(end_think_idx);
                        block_ids[b].push_back(orphan_ctr[b]++);
                        active[b] = false;

                        // Prefix incremental for end_think
                        auto et_ids = torch::full({B, 1}, end_think_idx, opts_int);
                        auto et_pos = torch::full({B, 1}, cur_len - 1, opts_int);
                        auto et_ov = torch::zeros({B, 1}, opts_fp16);
                        auto et_om = torch::zeros({B, 1}, opts_bool);
                        auto h_et = backbone_->prefixIncremental(et_ids, et_pos, et_ov, et_om);
                        auto mask_et = (~active & between_vars).unsqueeze(1);  // [B, 1]
                        saved_h = torch::where(mask_et, h_et.squeeze(1), saved_h);
                        break;  // Only one prefix incremental per step
                    }
                    else
                    {
                        // New variation
                        in_variation[b] = true;
                    }
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

    auto final_sub_idx = evalPolicyHead(saved_h, policy_temp, padded_fens);  // [B] move sub-vocab
    auto final_full = move_lut.index({final_sub_idx});  // [B] full vocab
    auto final_cpu = final_full.cpu();
    auto final_a = final_cpu.accessor<int64_t, 1>();

    // Build results
    std::vector<Result> results(B);
    for (int b = 0; b < B; b++)
    {
        int tok = final_a[b];
        std::string move = vocab_->pseudoToStandardUci(vocab_->idxToToken(tok));

        // If empty (sequence hit max_len without end_think), try fallback
        if (move.empty() && !first_root_move[b].empty())
            move = vocab_->pseudoToStandardUci(first_root_move[b]);

        results[b].move = move;
        results[b].token_ids = std::move(token_ids[b]);
        results[b].wl_entries = std::move(wl_entries[b]);
        results[b].d_entries = std::move(d_entries[b]);
    }

    // Truncate to actual batch size (discard padding)
    results.resize(B_actual);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    total_time += elapsed;
    for (int b = 0; b < B_actual; b++)
        total_tokens += results[b].token_ids.size();

    return results;
}

} // namespace decoder

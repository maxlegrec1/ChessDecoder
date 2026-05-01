#include "batched_engine.hpp"

#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace decoder
{

// ============================================================================
// Construction: load model, upload head weights to GPU
// ============================================================================

ThinkingBatchedInferenceEngine::ThinkingBatchedInferenceEngine(
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

torch::Tensor ThinkingBatchedInferenceEngine::sampleBatched(torch::Tensor logits, float temp)
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

torch::Tensor ThinkingBatchedInferenceEngine::evalThinkingPolicyHead(torch::Tensor h, float temp)
{
    // h: [B, E] FP16 → logits [B, move_vocab] → sub_idx [B]
    auto logits = torch::mm(h, think_w_t_) + think_b_;
    return sampleBatched(logits, temp);  // [B] move sub-vocab indices
}

torch::Tensor ThinkingBatchedInferenceEngine::evalBoardHead(torch::Tensor h, float temp)
{
    // h: [B, E] FP16 → logits [B, board_vocab]
    auto logits = torch::mm(h, board_w_t_) + board_b_;
    return sampleBatched(logits, temp);  // [B] board sub-vocab indices
}

ThinkingBatchedInferenceEngine::ValueSample
ThinkingBatchedInferenceEngine::evalWlHead(torch::Tensor h, float temp)
{
    // h: [B, E] → MLP → [B, n_buckets]. Sample with `temp`, but record the
    // log-prob from the UNSCALED log_softmax (matches Python's
    // chessdecoder/rl/log_probs.py — same convention as evalPolicyHead so
    // current/old log-probs stay consistent for the GRPO ratio).
    auto hidden = torch::mm(h, wl_w1_t_) + wl_b1_;
    hidden = torch::mish(hidden);
    auto logits = torch::mm(hidden, wl_w2_t_) + wl_b2_;
    auto log_probs = torch::log_softmax(logits.to(torch::kFloat32), /*dim=*/1);
    auto idx = sampleBatched(logits, temp);                              // [B]
    auto idx_lp = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1);      // [B]
    auto value = wl_centers_.index({idx});                                // [B]
    return {idx, value, idx_lp};
}

ThinkingBatchedInferenceEngine::ValueSample
ThinkingBatchedInferenceEngine::evalDHead(torch::Tensor h, float temp)
{
    auto hidden = torch::mm(h, d_w1_t_) + d_b1_;
    hidden = torch::mish(hidden);
    auto logits = torch::mm(hidden, d_w2_t_) + d_b2_;
    auto log_probs = torch::log_softmax(logits.to(torch::kFloat32), /*dim=*/1);
    auto idx = sampleBatched(logits, temp);
    auto idx_lp = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1);
    auto value = d_centers_.index({idx});
    return {idx, value, idx_lp};
}

std::pair<torch::Tensor, torch::Tensor>
ThinkingBatchedInferenceEngine::evalPolicyHead(
    torch::Tensor h, float temp,
    const std::vector<std::string>& fens)
{
    // h: [B, E] → logits [B, move_vocab]
    auto logits = torch::mm(h, policy_w_t_) + policy_b_;

    // Compute log-probs on the UNMASKED logits (matches PyTorch
    // chessdecoder/rl/log_probs.py which uses raw policy_head output).
    auto logits_f32 = logits.to(torch::kFloat32);
    auto log_probs = torch::log_softmax(logits_f32, /*dim=*/1);

    // Mask illegal moves per FEN (sampling copy only)
    int B = h.size(0);
    auto logits_a = logits_f32.cpu();
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
    auto masked_cuda = logits_a.cuda().to(torch::kFloat16);

    auto sub_idx = sampleBatched(masked_cuda, temp);           // [B]
    auto lp = log_probs.gather(1, sub_idx.unsqueeze(1)).squeeze(1);  // [B]
    return {sub_idx, lp};
}

// ============================================================================
// Main inference: predictMoves (lockstep state machine)
// ============================================================================

std::vector<ThinkingBatchedInferenceEngine::Result>
ThinkingBatchedInferenceEngine::predictMoves(
    const std::vector<std::string>& fens, float temperature)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    torch::NoGradGuard no_grad;

    int N = static_cast<int>(fens.size());
    if (N == 0) return {};

    // Continuous batching: B engine slots, N FENs (N may exceed B). Slot
    // refills from the pending queue when its rollout terminates. RoPE
    // position is decoupled from physical KV cache position, so a refilled
    // slot's logical seq_pos resets to init_len while its physical KV slots
    // sit wherever the global causal_len_ points.
    int B = max_batch_size_;
    int initial_fill = std::min(N, B);
    std::deque<int> pending;
    for (int i = B; i < N; i++) pending.push_back(i);

    // Per-slot FEN identity. Padded slots beyond initial_fill use fen[0] for
    // their unused prefill computation; we never read their results.
    std::vector<int> slot_to_fen(B, 0);
    for (int b = 0; b < initial_fill; b++) slot_to_fen[b] = b;

    // padded_fens is dynamic now: rebuilt whenever a slot is refilled.
    std::vector<std::string> padded_fens(B);
    for (int b = 0; b < B; b++) padded_fens[b] = fens[slot_to_fen[b]];

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
    std::vector<std::vector<std::pair<int, float>>> move_log_probs(B);
    std::vector<std::vector<std::pair<int, int>>>   wl_bucket_indices(B), d_bucket_indices(B);
    std::vector<std::vector<std::pair<int, float>>> wl_log_probs(B), d_log_probs(B);
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

    // Causal hidden state saved at end of board gen loop
    torch::Tensor h_board_after_loop;

    // ================================================================
    // Phase 2: Variation loop (with continuous-batched slot refill)
    // ================================================================

    // Per-FEN result storage (filled in submission order via slot_to_fen).
    std::vector<Result> all_results(N);

    // Slots beyond initial_fill are inactive from the start (no FEN to run
    // there). Their saved_h, KV, etc. carry the prefill from padded fen[0]
    // but we never read the result.
    auto active = torch::zeros({B}, opts_bool);
    if (initial_fill > 0) {
        active.slice(0, 0, initial_fill).fill_(true);
    }
    auto in_variation = active.clone();
    auto prev_active = active.clone();

    for (int iter = 0; iter < max_seq_len_; iter++)
    {
        // Exit when no active slots AND no pending FENs to refill with.
        if (!active.any().item<bool>() && pending.empty())
            break;

        auto need_move = active & in_variation;  // [B]

        if (!need_move.any().item<bool>() && pending.empty())
            break;

        // Sync need_move to host ONCE per iteration. All per-element checks
        // below use nm_a[b] instead of need_move[b].item<bool>() — turns B
        // syncs per loop into 1 per iter. Same pattern applied to active and
        // between_vars in their respective sections.
        auto need_move_cpu = need_move.cpu();
        auto nm_a = need_move_cpu.accessor<bool, 1>();

        // Helper: build per-sequence position tensor from seq_pos + offset.
        // Build on host (cheap vector op) and do ONE H2D transfer instead of B
        // per-element scalar writes to a CUDA tensor (each was ~10us — called
        // 67×B per board-gen plus several per outer iter).
        auto makePosT = [&](int off = 0) {
            auto p_cpu = torch::empty({B, 1}, torch::TensorOptions().dtype(torch::kInt64));
            auto p_a = p_cpu.accessor<int64_t, 2>();
            for (int b = 0; b < B; b++)
                p_a[b][0] = seq_pos[b] + off;
            return p_cpu.to(torch::kCUDA);
        };

        // ── Step 2a: MOVE ──────────────────────────────────────────────
        {
            auto logits = torch::mm(saved_h, think_w_t_) + think_b_;
            auto logits_f32 = logits.to(torch::kFloat32);
            // Compute log-probs on RAW logits (no temperature, no padding
            // mask) — matches PyTorch chessdecoder/rl/log_probs.py.
            auto log_probs = torch::log_softmax(logits_f32, /*dim=*/1);

            // Mask inactive batch rows for the sampling copy only.
            logits_f32.index_put_({~need_move}, torch::tensor(-1e9f, torch::kCUDA));
            logits = logits_f32.to(torch::kFloat16);
            auto sub_idx = sampleBatched(logits, think_temp);
            auto full_idx = move_lut.index({sub_idx});

            // Gather per-element log-prob at the sampled sub-vocab index.
            auto lp = log_probs.gather(1, sub_idx.unsqueeze(1)).squeeze(1);  // [B]
            auto lp_cpu = lp.cpu();
            auto lp_a = lp_cpu.accessor<float, 1>();

            auto full_idx_cpu = full_idx.cpu();
            auto full_idx_a = full_idx_cpu.accessor<int64_t, 1>();
            for (int b = 0; b < B; b++)
            {
                if (!nm_a[b]) continue;
                int tok = full_idx_a[b];
                // seq_pos[b] is the position about to receive the move token.
                // The hidden state that produced the move lives at seq_pos[b] - 1
                // (the last token already in token_ids[b]), which is the position
                // marked by thinking_move_mask in sequence.py::parse_rollout.
                int pred_pos = seq_pos[b] - 1;
                move_log_probs[b].push_back({pred_pos, lp_a[b]});

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
                if (nm_a[b]) seq_pos[b]++;
        }

        // ── Step 2b: WL_D ──────────────────────────────────────────────
        {
            // WL
            auto wl_sample = evalWlHead(saved_h, wl_temp);
            auto wl_cpu = wl_sample.value.cpu();
            auto wl_a = wl_cpu.accessor<float, 1>();
            auto wl_idx_cpu = wl_sample.idx.cpu();
            auto wl_idx_a = wl_idx_cpu.accessor<int64_t, 1>();
            auto wl_lp_cpu = wl_sample.log_prob.cpu();
            auto wl_lp_a = wl_lp_cpu.accessor<float, 1>();

            auto wl_ids_t = torch::full({B, 1}, wl_value_idx, opts_int);
            auto wl_pos = makePosT();
            auto wl_ov = wl_sample.value.to(torch::kFloat16).unsqueeze(1);
            auto wl_om = need_move.unsqueeze(1);
            auto h_wl = backbone_->prefixIncremental(wl_ids_t, wl_pos,
                                                     wl_ov, wl_om, need_move);
            saved_h = h_wl.squeeze(1);

            for (int b = 0; b < B; b++)
            {
                if (!nm_a[b]) continue;
                token_ids[b].push_back(wl_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                wl_entries[b].push_back({seq_pos[b], wl_a[b]});
                wl_bucket_indices[b].push_back({seq_pos[b], (int)wl_idx_a[b]});
                wl_log_probs[b].push_back({seq_pos[b], wl_lp_a[b]});
                seq_pos[b]++;
            }

            // D
            auto d_sample = evalDHead(saved_h, d_temp);
            auto d_cpu = d_sample.value.cpu();
            auto d_a = d_cpu.accessor<float, 1>();
            auto d_idx_cpu = d_sample.idx.cpu();
            auto d_idx_a = d_idx_cpu.accessor<int64_t, 1>();
            auto d_lp_cpu = d_sample.log_prob.cpu();
            auto d_lp_a = d_lp_cpu.accessor<float, 1>();

            auto d_ids_t = torch::full({B, 1}, d_value_idx, opts_int);
            auto d_pos = makePosT();
            auto d_ov = d_sample.value.to(torch::kFloat16).unsqueeze(1);
            auto d_om = need_move.unsqueeze(1);
            auto h_d = backbone_->prefixIncremental(d_ids_t, d_pos,
                                                    d_ov, d_om, need_move);
            saved_h = h_d.squeeze(1);

            for (int b = 0; b < B; b++)
            {
                if (!nm_a[b]) continue;
                token_ids[b].push_back(d_value_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
                d_entries[b].push_back({seq_pos[b], d_a[b]});
                d_bucket_indices[b].push_back({seq_pos[b], (int)d_idx_a[b]});
                d_log_probs[b].push_back({seq_pos[b], d_lp_a[b]});
                seq_pos[b]++;
            }
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
                    if (!nm_a[b]) continue;
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
                if (!nm_a[b]) continue;
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

            // Board gen: 67 autoregressive steps. Maintain `cur_pos` on GPU and
            // increment it via add_() each step instead of rebuilding via
            // makePosT (saves 67 H2D transfers per board cycle).
            auto board_output = torch::zeros({B, 67}, opts_int);
            auto cur_pos = first_pos.clone();
            auto inc_active = need_move.to(torch::kInt64).unsqueeze(1);  // [B, 1]

            for (int step = 0; step < 67; step++)
            {
                auto h_last = h_board.squeeze(1);
                auto board_logits = torch::mm(h_last, board_w_t_) + board_b_;
                auto sub_idx = sampleBatched(board_logits, 0.0f);
                auto full_idx = board_lut_.index({sub_idx});
                board_output.index_put_({torch::indexing::Slice(), step}, full_idx);

                auto next_ids = full_idx.unsqueeze(1);
                cur_pos = cur_pos + inc_active;  // GPU op, no sync, no H2D
                for (int b = 0; b < B; b++)
                    if (nm_a[b]) seq_pos[b]++;
                h_board = backbone_->causalIncremental(next_ids, cur_pos,
                                                      first_ov, first_om, need_move);
            }

            h_board_after_loop = h_board.squeeze(1).contiguous();
            backbone_->syncGraphToCausal();


            // Copy board tokens to CPU
            auto board_cpu = board_output.cpu();
            auto board_acc = board_cpu.accessor<int64_t, 2>();
            for (int b = 0; b < B; b++)
            {
                if (!nm_a[b]) continue;
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
                if (!nm_a[b]) continue;

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
                // Sync between_vars to host once for the per-element loops
                auto bv_cpu = between_vars.cpu();
                auto bv_a = bv_cpu.accessor<bool, 1>();

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
                    if (!bv_a[b]) continue;

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

        // Check max sequence length. Sync `active` to host once for the
        // B-element scan instead of per-element .item<bool>() syncs.
        auto active_cpu_for_maxcheck = active.cpu();
        auto act_a = active_cpu_for_maxcheck.accessor<bool, 1>();
        for (int b = 0; b < B; b++)
        {
            if (act_a[b] && (int)token_ids[b].size() >= max_seq_len_ - 5)
                active[b] = false;
        }

        // Global buffer-overflow safeguard. In CB, the global causal_len_ /
        // prefix_len_ grow with iter count regardless of any single slot's
        // logical seq_pos. Once they approach max_seq_len_, the next iter
        // would write beyond the pre-allocated cache and the slice clamp
        // produces a shape mismatch with the model's S-sized output. Force
        // all active slots to terminate; they'll get final-predicted in the
        // just_finished branch below.
        if (backbone_->causalLen() + 80 > max_seq_len_ ||
            backbone_->prefixLen() + 80 > max_seq_len_)
        {
            active.zero_();
            in_variation.zero_();
        }

        // ── CB: per-slot final predict for newly-finished slots, then refill ──
        auto prev_act_cpu = prev_active.cpu();
        auto pa_a = prev_act_cpu.accessor<bool, 1>();
        auto cur_act_cpu = active.cpu();
        auto ca_a = cur_act_cpu.accessor<bool, 1>();

        std::vector<int> just_finished;
        for (int b = 0; b < B; b++)
            if (pa_a[b] && !ca_a[b]) just_finished.push_back(b);

        if (!just_finished.empty())
        {
            // Batched final predict on the full B saved_h. Wasted compute on
            // non-finished slots is one matmul + softmax — cheap. For finished
            // slots, saved_h holds the end_think hidden state (set in step 2e).
            auto final_pair = evalPolicyHead(saved_h, policy_temp, padded_fens);
            auto final_full = move_lut.index({final_pair.first});
            auto ff_cpu = final_full.cpu();
            auto ff_a = ff_cpu.accessor<int64_t, 1>();
            auto flp_cpu = final_pair.second.cpu();
            auto flp_a = flp_cpu.accessor<float, 1>();

            for (int b : just_finished)
            {
                int tok = ff_a[b];
                std::string move = vocab_->pseudoToStandardUci(vocab_->idxToToken(tok));
                if (move.empty() && !first_root_move[b].empty())
                    move = vocab_->pseudoToStandardUci(first_root_move[b]);

                int pred_pos = seq_pos[b] - 1;
                move_log_probs[b].push_back({pred_pos, flp_a[b]});
                token_ids[b].push_back(tok);

                // Copy (not move) — step 2c's catch-up loop in subsequent
                // iters reads token_ids[b].back() for ALL slots regardless of
                // need_move (mask handles correctness). Moving would empty
                // the vector and segfault that read. Slot's accumulators are
                // cleared on refill (below) if a new FEN is queued; otherwise
                // they sit unread but valid until end of call.
                int fen_id = slot_to_fen[b];
                all_results[fen_id].move = std::move(move);
                all_results[fen_id].token_ids = token_ids[b];
                all_results[fen_id].wl_entries = wl_entries[b];
                all_results[fen_id].d_entries = d_entries[b];
                all_results[fen_id].move_log_probs = move_log_probs[b];
                all_results[fen_id].wl_bucket_indices = wl_bucket_indices[b];
                all_results[fen_id].d_bucket_indices = d_bucket_indices[b];
                all_results[fen_id].wl_log_probs = wl_log_probs[b];
                all_results[fen_id].d_log_probs = d_log_probs[b];

                total_tokens += all_results[fen_id].token_ids.size();
            }
        }

        // Buffer-overflow safety: a refilled slot will continue generating
        // for ~1500 tokens (avg rollout length); if the global causal_len_ /
        // prefix_len_ would push past max_seq_len_ during that, stop refilling
        // and enter drain mode. Remaining FENs in pending are skipped — the
        // Python wrapper detects empty results in all_results and re-calls.
        const int rollout_budget = 1500;
        bool buffer_full = (backbone_->causalLen() + rollout_budget > max_seq_len_) ||
                           (backbone_->prefixLen() + rollout_budget > max_seq_len_);

        // Refill any inactive slot that has a queued FEN waiting.
        std::vector<int> refill_idxs;
        if (!buffer_full)
        {
            for (int b = 0; b < B; b++)
            {
                if (!ca_a[b] && !pending.empty())
                {
                    int new_fen_id = pending.front();
                    pending.pop_front();
                    slot_to_fen[b] = new_fen_id;
                    padded_fens[b] = fens[new_fen_id];
                    refill_idxs.push_back(b);
                }
            }
        }
        else if (!pending.empty())
        {
            // Drain mode entered. Stop accepting new FENs; let active slots finish.
            // Caller will re-submit pending FENs in a fresh predictMoves call
            // (which resets causal_len_/prefix_len_ via Phase 1 resetCausal+resetPrefix).
            pending.clear();
        }

        if (!refill_idxs.empty())
        {
            // Reset host-side per-slot accumulators for refilled slots.
            for (int b : refill_idxs)
            {
                token_ids[b].clear();
                block_ids[b].clear();
                wl_entries[b].clear();
                d_entries[b].clear();
                move_log_probs[b].clear();
                wl_bucket_indices[b].clear();
                d_bucket_indices[b].clear();
                wl_log_probs[b].clear();
                d_log_probs[b].clear();
                next_block[b] = 0;
                orphan_ctr[b] = 10000;
                first_root_move[b].clear();
                seq_pos[b] = init_len;

                // Encode new FEN's 68 root tokens + start_think into accumulators.
                auto ids = vocab_->fenToTokenIds(padded_fens[b]);
                int bid = next_block[b]++;
                for (int j = 0; j < root_len; j++)
                {
                    token_ids[b].push_back(ids[j]);
                    block_ids[b].push_back(bid);
                }
                token_ids[b].push_back(start_think_idx);
                block_ids[b].push_back(orphan_ctr[b]++);
            }

            // Build refill batch tensors on host first, single H2D each.
            // Per-element writes to CUDA tensors trigger sync per write —
            // a refill cycle of K slots × init_len would produce K*69*4
            // (tensors) sync ops, dominating refill cost.
            auto opts_int64_cpu = torch::TensorOptions().dtype(torch::kInt64);
            auto opts_fp16_cpu = torch::TensorOptions().dtype(torch::kFloat16);
            auto opts_bool_cpu = torch::TensorOptions().dtype(torch::kBool);

            auto refill_active_cpu = torch::zeros({B}, opts_bool_cpu);
            auto rf_ids_cpu = torch::zeros({B, init_len}, opts_int64_cpu);
            auto rf_block_ids_cpu = torch::zeros({B, init_len}, opts_int64_cpu);
            auto ra_a = refill_active_cpu.accessor<bool, 1>();
            auto rids_a = rf_ids_cpu.accessor<int64_t, 2>();
            auto rblk_a = rf_block_ids_cpu.accessor<int64_t, 2>();

            for (int b : refill_idxs)
            {
                ra_a[b] = true;
                for (int j = 0; j < init_len; j++)
                {
                    rids_a[b][j] = token_ids[b][j];
                    rblk_a[b][j] = block_ids[b][j];
                }
            }
            auto refill_active = refill_active_cpu.cuda();
            auto rf_ids = rf_ids_cpu.cuda();
            auto rf_block_ids = rf_block_ids_cpu.cuda();
            auto rf_pos = torch::arange(init_len, opts_int).unsqueeze(0)
                              .expand({B, init_len}).contiguous();
            auto rf_ov = torch::zeros({B, init_len}, opts_fp16);
            auto rf_om = torch::zeros({B, init_len}, opts_bool);

            // Block-aware prefix mask (same construction as Phase 1).
            auto rf_same_block = rf_block_ids.unsqueeze(2) == rf_block_ids.unsqueeze(1);
            auto rf_causal = torch::tril(torch::ones({init_len, init_len}, opts_bool));
            auto rf_prefix_mask = torch::where(
                rf_same_block | rf_causal.unsqueeze(0),
                torch::tensor(0.0f, torch::kCUDA),
                torch::tensor(-1e9f, torch::kCUDA))
                .unsqueeze(1);  // [B, 1, init_len, init_len]

            auto new_saved = backbone_->resetSlotsForRefill(
                refill_active, rf_ids, rf_pos, rf_prefix_mask, rf_ov, rf_om);

            // Update saved_h only for refilled slots.
            saved_h = torch::where(refill_active.unsqueeze(1), new_saved, saved_h);

            // Reactivate refilled slots in active/in_variation tensors.
            active = active | refill_active;
            in_variation = in_variation | refill_active;
        }

        // Snapshot active for next iter's just-finished detection.
        prev_active = active.clone();
    }

    // ================================================================
    // Loop-exit fallback: predict finals for any slots still active.
    // (Triggers only if the for-loop iter budget runs out before all
    //  rollouts finish — unreachable in practice given seq budget caps.)
    // ================================================================
    if (active.any().item<bool>())
    {
        auto final_pair = evalPolicyHead(saved_h, policy_temp, padded_fens);
        auto final_full = move_lut.index({final_pair.first});
        auto ff_cpu = final_full.cpu();
        auto ff_a = ff_cpu.accessor<int64_t, 1>();
        auto flp_cpu = final_pair.second.cpu();
        auto flp_a = flp_cpu.accessor<float, 1>();
        auto act_cpu = active.cpu();
        auto act_a = act_cpu.accessor<bool, 1>();

        for (int b = 0; b < B; b++)
        {
            if (!act_a[b]) continue;
            int tok = ff_a[b];
            std::string move = vocab_->pseudoToStandardUci(vocab_->idxToToken(tok));
            if (move.empty() && !first_root_move[b].empty())
                move = vocab_->pseudoToStandardUci(first_root_move[b]);
            int pred_pos = seq_pos[b] - 1;
            move_log_probs[b].push_back({pred_pos, flp_a[b]});
            token_ids[b].push_back(tok);

            int fen_id = slot_to_fen[b];
            all_results[fen_id].move = std::move(move);
            all_results[fen_id].token_ids = std::move(token_ids[b]);
            all_results[fen_id].wl_entries = std::move(wl_entries[b]);
            all_results[fen_id].d_entries = std::move(d_entries[b]);
            all_results[fen_id].move_log_probs = std::move(move_log_probs[b]);
            all_results[fen_id].wl_bucket_indices = std::move(wl_bucket_indices[b]);
            all_results[fen_id].d_bucket_indices = std::move(d_bucket_indices[b]);
            all_results[fen_id].wl_log_probs = std::move(wl_log_probs[b]);
            all_results[fen_id].d_log_probs = std::move(d_log_probs[b]);
            total_tokens += all_results[fen_id].token_ids.size();
        }
    }
    // After the post-loop fallback, no further reads of per-slot accumulators
    // happen, so std::move there is safe.

    auto t1 = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration<double>(t1 - t0).count();

    return all_results;
}

} // namespace decoder

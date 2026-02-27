#include "decoder_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <c10/cuda/CUDAStream.h>

namespace {
inline double profNow() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
} // anonymous namespace

namespace decoder
{

namespace
{

int readJsonInt(const std::string& json, const std::string& key)
{
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos)
        throw std::runtime_error("Key not found in config: " + key);
    pos = json.find(':', pos);
    pos++;
    while (pos < json.size() && std::isspace(json[pos])) pos++;
    size_t end = pos;
    if (json[end] == '-') end++;
    while (end < json.size() && std::isdigit(json[end])) end++;
    return std::stoi(json.substr(pos, end - pos));
}

} // anonymous namespace

// ============================================================================
// Constructor
// ============================================================================

ThinkingInferenceEngine::ThinkingInferenceEngine(
    const std::string& backbone_pt_path,
    const std::string& weights_dir,
    const std::string& vocab_path,
    const std::string& config_path)
{
    std::ifstream cf(config_path);
    if (!cf) throw std::runtime_error("Failed to open config: " + config_path);
    std::string config_json((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());

    embed_dim_ = readJsonInt(config_json, "embed_dim");
    num_layers_ = readJsonInt(config_json, "num_layers");
    num_heads_ = readJsonInt(config_json, "num_heads");
    head_dim_ = readJsonInt(config_json, "head_dim");
    max_seq_len_ = readJsonInt(config_json, "max_seq_len");

    int board_vocab_size = readJsonInt(config_json, "board_vocab_size");
    int move_vocab_size = readJsonInt(config_json, "move_vocab_size");
    int n_buckets = readJsonInt(config_json, "n_buckets");
    int value_hidden_size = readJsonInt(config_json, "value_hidden_size");
    int num_fourier_freq = readJsonInt(config_json, "num_fourier_freq");

    vocab_ = std::make_unique<DecoderVocab>(vocab_path);
    heads_ = std::make_unique<Heads>(weights_dir, embed_dim_, board_vocab_size, move_vocab_size,
                                     value_hidden_size, n_buckets, num_fourier_freq);
    backbone_ = std::make_unique<TorchCausalBackbone>(backbone_pt_path, num_layers_, num_heads_,
                                                       head_dim_, embed_dim_, max_seq_len_);

    // Upload all head weights to GPU (pre-transpose, FP16).
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
    {
        board_head_w_gpu_t_ = uploadWeightT(heads_->boardWeightData(), heads_->boardVocabSize(), embed_dim_);
        board_head_b_gpu_ = uploadBias(heads_->boardBiasData(), heads_->boardVocabSize());

        int bvs = heads_->boardVocabSize();
        auto lut_cpu = torch::zeros({bvs}, torch::kInt64);
        for (int i = 0; i < bvs; i++)
            lut_cpu[i] = vocab_->boardIdxToFullIdx(i);
        board_lut_gpu_ = lut_cpu.to(torch::kCUDA);
    }

    // Policy heads
    policy_w_gpu_t_ = uploadWeightT(heads_->policyWeightData(), heads_->moveVocabSize(), embed_dim_);
    policy_b_gpu_ = uploadBias(heads_->policyBiasData(), heads_->moveVocabSize());
    think_policy_w_gpu_t_ = uploadWeightT(heads_->thinkingPolicyWeightData(), heads_->moveVocabSize(), embed_dim_);
    think_policy_b_gpu_ = uploadBias(heads_->thinkingPolicyBiasData(), heads_->moveVocabSize());

    // WL head (two-layer MLP)
    {
        int H = heads_->valueHiddenSize();
        int B = heads_->nBuckets();
        wl_w1_gpu_t_ = uploadWeightT(heads_->wlW1WeightData(), H, embed_dim_);
        wl_b1_gpu_ = uploadBias(heads_->wlW1BiasData(), H);
        wl_w2_gpu_t_ = uploadWeightT(heads_->wlW2WeightData(), B, H);
        wl_b2_gpu_ = uploadBias(heads_->wlW2BiasData(), B);
        wl_centers_gpu_ = torch::from_blob(const_cast<float*>(heads_->wlBucketCentersData()),
            {B}, torch::kFloat32).to(torch::kCUDA);
    }

    // D head (two-layer MLP)
    {
        int H = heads_->valueHiddenSize();
        int B = heads_->nBuckets();
        d_w1_gpu_t_ = uploadWeightT(heads_->dW1WeightData(), H, embed_dim_);
        d_b1_gpu_ = uploadBias(heads_->dW1BiasData(), H);
        d_w2_gpu_t_ = uploadWeightT(heads_->dW2WeightData(), B, H);
        d_b2_gpu_ = uploadBias(heads_->dW2BiasData(), B);
        d_centers_gpu_ = torch::from_blob(const_cast<float*>(heads_->dBucketCentersData()),
            {B}, torch::kFloat32).to(torch::kCUDA);
    }

    std::cout << "[DecoderEngine] Loaded: " << num_layers_ << " layers, "
              << embed_dim_ << "d, max_seq=" << max_seq_len_ << std::endl;
}

// ============================================================================
// Causal forward helpers
// ============================================================================

std::vector<float> ThinkingInferenceEngine::causalPrefill()
{
    int S = static_cast<int>(token_ids_.size());
    std::vector<int64_t> input_ids(S);
    std::vector<int64_t> input_pos(S);
    for (int i = 0; i < S; i++)
    {
        input_ids[i] = token_ids_[i];
        input_pos[i] = i;
    }

    std::vector<float> override_values(S, 0.0f);
    std::vector<uint8_t> override_flags(S, 0);

    for (const auto& [pos, val] : wl_entries_)
    {
        override_values[pos] = val;
        override_flags[pos] = 1;
    }
    for (const auto& [pos, val] : d_entries_)
    {
        override_values[pos] = val;
        override_flags[pos] = 1;
    }

    backbone_->resetCache();

    std::vector<float> hidden(S * embed_dim_);
    backbone_->forward(input_ids.data(), input_pos.data(), S, 0,
                       override_values.data(), override_flags.data(),
                       hidden.data());
    return hidden;
}

// ============================================================================
// Prefix mask & sampling
// ============================================================================

void ThinkingInferenceEngine::buildPrefixMask(std::vector<float>& mask) const
{
    int S = static_cast<int>(token_ids_.size());

    for (int i = 0; i < S; i++)
    {
        for (int j = 0; j < S; j++)
        {
            bool causal = (j <= i);
            bool same_block = (block_ids_[i] == block_ids_[j]);
            mask[i * S + j] = (causal || same_block) ? 0.0f : -1e9f;
        }
    }
}

int ThinkingInferenceEngine::sampleToken(const float* logits, int vocab_size, float temperature) const
{
    if (temperature <= 0.0f)
    {
        return static_cast<int>(std::max_element(logits, logits + vocab_size) - logits);
    }

    std::vector<float> probs(vocab_size);
    float max_val = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++)
    {
        probs[i] = std::exp((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++)
        probs[i] /= sum;

    static thread_local std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// ============================================================================
// Fallback move (no thinking)
// ============================================================================

std::string ThinkingInferenceEngine::fallbackMove(const std::string& fen, float temperature)
{
    token_ids_.clear();
    block_ids_.clear();
    wl_entries_.clear();
    d_entries_.clear();
    next_block_ = 0;
    orphan_ctr_ = 10000;

    auto root_ids = vocab_->fenToTokenIds(fen);
    int bid = next_block_++;
    for (int id : root_ids)
    {
        token_ids_.push_back(id);
        block_ids_.push_back(bid);
    }

    int S = static_cast<int>(token_ids_.size());
    std::vector<int64_t> input_ids(S);
    std::vector<int64_t> input_pos(S);
    for (int i = 0; i < S; i++)
    {
        input_ids[i] = token_ids_[i];
        input_pos[i] = i;
    }
    std::vector<float> mask(S * S);
    buildPrefixMask(mask);
    std::vector<float> override_values(S, 0.0f);
    std::vector<uint8_t> override_flags(S, 0);

    std::vector<float> hidden(S * embed_dim_);
    backbone_->forwardPrefix(input_ids.data(), input_pos.data(), S,
                             mask.data(), override_values.data(), override_flags.data(),
                             hidden.data());

    int last_pos = S - 1;
    std::vector<float> logits(heads_->moveVocabSize());
    heads_->evalPolicyHead(hidden.data() + last_pos * embed_dim_, logits.data());

    auto legal_indices = vocab_->legalMoveIndices(fen);
    if (!legal_indices.empty())
    {
        std::vector<float> masked_logits(heads_->moveVocabSize(), -1e9f);
        for (int idx : legal_indices)
            masked_logits[idx] = logits[idx];
        logits = masked_logits;
    }

    int move_sub_idx = sampleToken(logits.data(), heads_->moveVocabSize(), temperature);
    int full_idx = vocab_->moveIdxToFullIdx(move_sub_idx);
    return DecoderVocab::pseudoToStandardUci(vocab_->idxToToken(full_idx));
}

// ============================================================================
// Main inference: predictMove
// ============================================================================

std::string ThinkingInferenceEngine::predictMove(const std::string& fen, float temperature)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // Reset state
    token_ids_.clear();
    block_ids_.clear();
    wl_entries_.clear();
    d_entries_.clear();
    next_block_ = 0;
    orphan_ctr_ = 10000;
    backbone_->resetCache();
    backbone_->resetPrefixCache();

    // Resolve per-head temperatures.
    // think/policy: -1.0 means "use the temperature arg". >= 0 overrides.
    float think_temp = (think_temperature >= 0.0f) ? think_temperature : temperature;
    float policy_temp = (policy_temperature >= 0.0f) ? policy_temperature : temperature;
    float board_temp = board_temperature;
    float wl_temp = wl_temperature;
    float d_temp = d_temperature;

    auto isFull = [&]() { return static_cast<int>(token_ids_.size()) >= max_seq_len_; };
    auto orphan = [&]() { return ++orphan_ctr_; };
    auto append = [&](int tok_id, int bid) {
        token_ids_.push_back(tok_id);
        block_ids_.push_back(bid);
    };
    auto done = [&](const std::string& move) -> std::string {
        total_tokens += static_cast<int64_t>(token_ids_.size());
        auto t1 = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double>(t1 - t0).count();
        last_token_ids_ = token_ids_;
        last_wl_entries_ = wl_entries_;
        last_d_entries_ = d_entries_;
        return move;
    };

    // Helper: prefix incremental via CUDA graph, saves hidden state on GPU
    auto prefixIncrGraph = [&](int64_t full_idx, int64_t pos, float ov, uint8_t of) {
        auto h_gpu = backbone_->prefixIncrementalGraph(full_idx, pos, ov, of);
        saved_prefix_hidden_gpu_ = h_gpu.index({0, 0}).clone();  // [E] FP16 CUDA
    };

    // 1. Root board (68 tokens)
    auto root_ids = vocab_->fenToTokenIds(fen);
    int bid = next_block_++;
    for (int id : root_ids)
        append(id, bid);

    if (isFull())
        return done(fallbackMove(fen, temperature));

    // 2. start_think
    append(vocab_->startThinkIdx(), orphan());

    // 3. Initialize prefix KV cache with full prefix forward
    {
        double _pt0 = 0; if (profiling) { c10::cuda::getCurrentCUDAStream().synchronize(); _pt0 = profNow(); }
        int S = static_cast<int>(token_ids_.size());
        std::vector<int64_t> input_ids(S);
        std::vector<int64_t> input_pos(S);
        for (int i = 0; i < S; i++)
        {
            input_ids[i] = token_ids_[i];
            input_pos[i] = i;
        }
        std::vector<float> mask(S * S);
        buildPrefixMask(mask);
        std::vector<float> override_values(S, 0.0f);
        std::vector<uint8_t> override_flags(S, 0);

        std::vector<float> hidden_cpu(embed_dim_);
        backbone_->prefixInit(input_ids.data(), input_pos.data(), S,
                              mask.data(), override_values.data(), override_flags.data(),
                              S - 1, hidden_cpu.data());

        saved_prefix_hidden_gpu_ = torch::from_blob(hidden_cpu.data(),
            {embed_dim_}, torch::kFloat32).to(torch::kFloat16).to(torch::kCUDA);

        backbone_->syncPrefixCacheToGraph();
        if (profiling) { c10::cuda::getCurrentCUDAStream().synchronize(); prof_prefix_init += profNow() - _pt0; }
    }

    // Profiling helper lambda
    auto psync = [&]() { if (profiling) c10::cuda::getCurrentCUDAStream().synchronize(); };
    auto pnow = [&]() -> double { psync(); return profNow(); };

    // 4. Autoregressive thinking loop
    State state = State::MOVE;
    std::string first_root_move;

    while (!isFull())
    {
        switch (state)
        {
        case State::MOVE:
        {
            double _t0 = pnow();
            int move_sub_idx = evalThinkingPolicyHeadGpu(think_temp);
            if (profiling) { psync(); prof_head_eval += profNow() - _t0; }
            int full_idx = vocab_->moveIdxToFullIdx(move_sub_idx);
            const std::string& tok = vocab_->idxToToken(full_idx);

            int move_pos = static_cast<int>(token_ids_.size());
            append(full_idx, orphan());

            if (first_root_move.empty() && DecoderVocab::isMoveToken(tok))
                first_root_move = DecoderVocab::pseudoToStandardUci(tok);

            double _t1 = pnow();
            prefixIncrGraph(static_cast<int64_t>(full_idx),
                            static_cast<int64_t>(move_pos), 0.0f, 0);
            if (profiling) { psync(); prof_prefix_incr += profNow() - _t1; }

            state = State::WL_D;
            break;
        }
        case State::WL_D:
        {
            if (isFull()) goto exit_loop;

            double _t0 = pnow();
            float wl = predictWlGpu(wl_temp);
            if (profiling) { psync(); prof_head_eval += profNow() - _t0; }
            int wl_pos = static_cast<int>(token_ids_.size());
            append(vocab_->wlValueIdx(), orphan());
            wl_entries_.emplace_back(wl_pos, wl);
            double _t1 = pnow();
            prefixIncrGraph(static_cast<int64_t>(vocab_->wlValueIdx()),
                            static_cast<int64_t>(wl_pos), wl, 1);
            if (profiling) { psync(); prof_prefix_incr += profNow() - _t1; }

            double _t2 = pnow();
            float d = predictDGpu(d_temp);
            if (profiling) { psync(); prof_head_eval += profNow() - _t2; }
            int d_pos = static_cast<int>(token_ids_.size());
            append(vocab_->dValueIdx(), orphan());
            d_entries_.emplace_back(d_pos, d);
            double _t3 = pnow();
            prefixIncrGraph(static_cast<int64_t>(vocab_->dValueIdx()),
                            static_cast<int64_t>(d_pos), d, 1);
            if (profiling) { psync(); prof_prefix_incr += profNow() - _t3; }

            state = State::BOARD;
            break;
        }
        case State::BOARD:
        {
            if (isFull()) goto exit_loop;

            int board_bid = next_block_++;
            int board_start_pos = static_cast<int>(token_ids_.size());
            int cached = backbone_->cacheLen();

            // Phase 1: Ensure causal cache covers all tokens up to board start
            if (cached == 0)
            {
                double _t0 = pnow();
                // First board: full causal prefill + sync to graph
                auto prefill_hidden = causalPrefill();
                backbone_->syncCausalCacheToGraph();

                // First board token from prefill hidden
                int last_pos = board_start_pos - 1;
                auto h_prefill = torch::from_blob(
                    prefill_hidden.data() + last_pos * embed_dim_,
                    {1, embed_dim_}, torch::kFloat32
                ).to(torch::kFloat16).to(torch::kCUDA);
                auto logits_gpu = torch::mm(h_prefill, board_head_w_gpu_t_) + board_head_b_gpu_;
                int board_sub_idx = torch::argmax(logits_gpu, 1).item<int>();
                int full_idx = vocab_->boardIdxToFullIdx(board_sub_idx);
                append(full_idx, board_bid);
                if (profiling) { psync(); prof_board_prefill += profNow() - _t0; }
            }
            else
            {
                double _t0 = pnow();
                // Subsequent boards: catch up via non-graph forward to ensure
                // exact numerical match (CUDA graph incremental can accumulate
                // tiny FP16 precision differences in the padded KV buffer).
                backbone_->syncGraphToCausalCache();

                int new_count = board_start_pos - cached;
                if (new_count <= 0 || board_start_pos >= max_seq_len_)
                    goto exit_loop;

                std::vector<int64_t> catch_ids(new_count);
                std::vector<int64_t> catch_pos(new_count);
                std::vector<float> catch_ov(new_count, 0.0f);
                std::vector<uint8_t> catch_of(new_count, 0);

                for (int i = 0; i < new_count; i++)
                {
                    int abs_pos = cached + i;
                    catch_ids[i] = token_ids_[abs_pos];
                    catch_pos[i] = abs_pos;
                    for (const auto& [p, val] : wl_entries_)
                        if (p == abs_pos) { catch_ov[i] = val; catch_of[i] = 1; break; }
                    if (!catch_of[i])
                        for (const auto& [p, val] : d_entries_)
                            if (p == abs_pos) { catch_ov[i] = val; catch_of[i] = 1; break; }
                }

                std::vector<float> catch_hidden(new_count * embed_dim_);
                backbone_->forward(catch_ids.data(), catch_pos.data(), new_count, cached,
                                   catch_ov.data(), catch_of.data(), catch_hidden.data());
                backbone_->syncCausalCacheToGraph();

                // First board token from last hidden in catch-up
                int last_idx = new_count - 1;
                auto h_catchup = torch::from_blob(
                    catch_hidden.data() + last_idx * embed_dim_,
                    {1, embed_dim_}, torch::kFloat32
                ).to(torch::kFloat16).to(torch::kCUDA);
                auto logits_gpu = torch::mm(h_catchup, board_head_w_gpu_t_) + board_head_b_gpu_;
                int board_sub_idx = torch::argmax(logits_gpu, 1).item<int>();
                int full_idx = vocab_->boardIdxToFullIdx(board_sub_idx);
                append(full_idx, board_bid);
                if (profiling) { psync(); prof_board_catchup += profNow() - _t0; }
            }

            // Phase 2: Generate 67 board tokens via GPU-only loop
            // Use tiered CUDA graphs with smaller KV buffers when possible
            {
                double _t0 = pnow();

                auto output_ids = torch::zeros({67},
                    torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

                // Limit generation to not exceed max_seq_len (positions must stay < max_seq_len for RoPE)
                int max_board_tokens = std::min(67, max_seq_len_ - board_start_pos - 1);
                if (max_board_tokens < 0) max_board_tokens = 0;

                int needed = board_start_pos + max_board_tokens;
                int tier = backbone_->selectBoardTier(needed);

                if (tier >= 0)
                {
                    // Use smaller tier graph (less KV bandwidth)
                    backbone_->prepareBoardGen(tier, token_ids_.back());
                    for (int i = 0; i < max_board_tokens; i++)
                    {
                        output_ids[i] = backbone_->boardGenStep(tier,
                            board_head_w_gpu_t_, board_head_b_gpu_, board_lut_gpu_,
                            board_start_pos + i);
                    }
                    // One sync to collect all token IDs
                    auto ids_cpu = output_ids.cpu();
                    for (int i = 0; i < max_board_tokens; i++)
                        append(static_cast<int>(ids_cpu[i].item<int64_t>()), board_bid);
                    backbone_->finishBoardGen(tier);
                }
                else
                {
                    // Fallback: use full-size graph (positions > 2048)
                    backbone_->setGraphInput(token_ids_.back());
                    for (int i = 0; i < max_board_tokens; i++)
                    {
                        output_ids[i] = backbone_->causalBoardStep(
                            board_head_w_gpu_t_, board_head_b_gpu_, board_lut_gpu_,
                            board_start_pos + i);
                    }
                    auto ids_cpu = output_ids.cpu();
                    for (int i = 0; i < max_board_tokens; i++)
                        append(static_cast<int>(ids_cpu[i].item<int64_t>()), board_bid);
                }

                if (profiling) { psync(); prof_board_gen += profNow() - _t0; }
            }

            // Phase 3: Prefix block forward for the board tokens
            {
                double _t0 = pnow();
                backbone_->syncGraphToPrefixCache();
                int board_len = static_cast<int>(token_ids_.size()) - board_start_pos;
                std::vector<int64_t> board_ids(board_len);
                std::vector<int64_t> board_pos(board_len);
                std::vector<float> board_ov(board_len, 0.0f);
                std::vector<uint8_t> board_of(board_len, 0);

                for (int i = 0; i < board_len; i++)
                {
                    board_ids[i] = token_ids_[board_start_pos + i];
                    board_pos[i] = board_start_pos + i;
                }

                std::vector<float> hidden_cpu(embed_dim_);
                backbone_->prefixBlockForward(
                    board_ids.data(), board_pos.data(), board_len,
                    board_ov.data(), board_of.data(),
                    board_len - 1, hidden_cpu.data());

                saved_prefix_hidden_gpu_ = torch::from_blob(hidden_cpu.data(),
                    {embed_dim_}, torch::kFloat32).to(torch::kFloat16).to(torch::kCUDA);

                backbone_->syncPrefixCacheToGraphAfterBlock();
                if (profiling) { psync(); prof_prefix_block += profNow() - _t0; }
            }

            state = State::AFTER_BOARD;
            break;
        }
        case State::AFTER_BOARD:
        {
            if (isFull()) goto exit_loop;

            int pos = static_cast<int>(token_ids_.size()) - 1;
            double _t0 = pnow();
            auto h_gpu = backbone_->causalIncrementalGraph(
                token_ids_[pos], pos, 0.0f, 0);
            if (profiling) { psync(); prof_causal_incr += profNow() - _t0; }

            double _t1 = pnow();
            auto logits_gpu = torch::mm(
                h_gpu.view({1, embed_dim_}), board_head_w_gpu_t_
            ) + board_head_b_gpu_;
            int board_sub_idx;
            if (board_temp > 0.0f) {
                auto probs = torch::softmax(logits_gpu / board_temp, 1);
                board_sub_idx = torch::multinomial(probs, 1).item<int>();
            } else {
                board_sub_idx = torch::argmax(logits_gpu, 1).item<int>();
            }
            if (profiling) { psync(); prof_head_eval += profNow() - _t1; }

            if (board_sub_idx == vocab_->boardEndVarIdx())
            {
                int ev_pos = static_cast<int>(token_ids_.size());
                int full_idx = vocab_->boardIdxToFullIdx(board_sub_idx);
                append(full_idx, orphan());
                prefixIncrGraph(static_cast<int64_t>(full_idx),
                                static_cast<int64_t>(ev_pos), 0.0f, 0);
                state = State::AFTER_END_VAR;
            }
            else
            {
                state = State::MOVE;
            }
            break;
        }
        case State::AFTER_END_VAR:
        {
            if (isFull()) goto exit_loop;

            int pos = static_cast<int>(token_ids_.size()) - 1;
            double _t0 = pnow();
            auto h_gpu = backbone_->causalIncrementalGraph(
                token_ids_[pos], pos, 0.0f, 0);
            if (profiling) { psync(); prof_causal_incr += profNow() - _t0; }

            double _t1 = pnow();
            auto logits_gpu = torch::mm(
                h_gpu.view({1, embed_dim_}), board_head_w_gpu_t_
            ) + board_head_b_gpu_;
            int board_sub_idx;
            if (board_temp > 0.0f) {
                auto probs = torch::softmax(logits_gpu / board_temp, 1);
                board_sub_idx = torch::multinomial(probs, 1).item<int>();
            } else {
                board_sub_idx = torch::argmax(logits_gpu, 1).item<int>();
            }
            if (profiling) { psync(); prof_head_eval += profNow() - _t1; }

            if (board_sub_idx == vocab_->boardEndThinkIdx())
            {
                int et_pos = static_cast<int>(token_ids_.size());
                int full_idx = vocab_->boardIdxToFullIdx(board_sub_idx);
                append(full_idx, orphan());
                prefixIncrGraph(static_cast<int64_t>(full_idx),
                                static_cast<int64_t>(et_pos), 0.0f, 0);
                state = State::FINAL;
            }
            else
            {
                state = State::MOVE;
            }
            break;
        }
        case State::FINAL:
        {
            if (isFull()) goto exit_loop;

            auto legal_indices = vocab_->legalMoveIndices(fen);
            int move_sub_idx = evalPolicyHeadGpu(policy_temp, legal_indices);
            int full_idx = vocab_->moveIdxToFullIdx(move_sub_idx);
            return done(DecoderVocab::pseudoToStandardUci(vocab_->idxToToken(full_idx)));
        }
        }
    }

exit_loop:
    if (!first_root_move.empty())
    {
        auto legal = vocab_->legalMoveIndices(fen);
        for (int idx : legal)
        {
            int full_idx = vocab_->moveIdxToFullIdx(idx);
            if (DecoderVocab::pseudoToStandardUci(vocab_->idxToToken(full_idx)) == first_root_move)
                return done(first_root_move);
        }
    }
    return done(fallbackMove(fen, temperature));
}

// ============================================================================
// GPU head evaluation
// ============================================================================

int ThinkingInferenceEngine::evalThinkingPolicyHeadGpu(float temperature)
{
    auto h = saved_prefix_hidden_gpu_.view({1, embed_dim_});
    auto logits = torch::mm(h, think_policy_w_gpu_t_) + think_policy_b_gpu_;
    if (temperature <= 0.0f)
        return torch::argmax(logits, 1).item<int>();

    auto logits_cpu = logits.to(torch::kFloat32).cpu().contiguous();
    return sampleToken(logits_cpu.data_ptr<float>(), heads_->moveVocabSize(), temperature);
}

int ThinkingInferenceEngine::evalPolicyHeadGpu(float temperature, const std::vector<int>& legal_indices)
{
    auto h = saved_prefix_hidden_gpu_.view({1, embed_dim_});
    auto logits_gpu = torch::mm(h, policy_w_gpu_t_) + policy_b_gpu_;

    auto logits_cpu = logits_gpu.to(torch::kFloat32).cpu().contiguous();
    float* logits_ptr = logits_cpu.data_ptr<float>();
    int mvs = heads_->moveVocabSize();

    if (!legal_indices.empty())
    {
        std::vector<float> masked(mvs, -1e9f);
        for (int idx : legal_indices)
            masked[idx] = logits_ptr[idx];
        return sampleToken(masked.data(), mvs, temperature);
    }

    return sampleToken(logits_ptr, mvs, temperature);
}

float ThinkingInferenceEngine::predictWlGpu(float temperature)
{
    auto h = saved_prefix_hidden_gpu_.view({1, embed_dim_});
    auto hidden = torch::mm(h, wl_w1_gpu_t_) + wl_b1_gpu_;
    hidden = torch::mish(hidden);
    auto logits = torch::mm(hidden, wl_w2_gpu_t_) + wl_b2_gpu_;
    int idx;
    if (temperature > 0.0f) {
        auto probs = torch::softmax(logits / temperature, 1);
        idx = torch::multinomial(probs, 1).item<int>();
    } else {
        idx = torch::argmax(logits, 1).item<int>();
    }
    return wl_centers_gpu_[idx].item<float>();
}

float ThinkingInferenceEngine::predictDGpu(float temperature)
{
    auto h = saved_prefix_hidden_gpu_.view({1, embed_dim_});
    auto hidden = torch::mm(h, d_w1_gpu_t_) + d_b1_gpu_;
    hidden = torch::mish(hidden);
    auto logits = torch::mm(hidden, d_w2_gpu_t_) + d_b2_gpu_;
    int idx;
    if (temperature > 0.0f) {
        auto probs = torch::softmax(logits / temperature, 1);
        idx = torch::multinomial(probs, 1).item<int>();
    } else {
        idx = torch::argmax(logits, 1).item<int>();
    }
    return d_centers_gpu_[idx].item<float>();
}

} // namespace decoder

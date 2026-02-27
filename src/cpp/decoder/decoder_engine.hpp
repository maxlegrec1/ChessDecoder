#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "heads.hpp"
#include "torch_backbone.hpp"
#include "vocab.hpp"

namespace decoder
{

/// Full thinking inference engine.
/// Implements the same state machine as ThinkingModelWrapper.predict_move() in Python.
///
/// Uses two KV caches:
/// - Causal KV cache (inside TorchCausalBackbone): for autoregressive board generation
/// - Prefix KV cache (inside TorchCausalBackbone): for incremental move/value prediction
class ThinkingInferenceEngine
{
public:
    /// Construct engine from exported TorchScript backbone + weights.
    ThinkingInferenceEngine(
        const std::string& backbone_pt_path,
        const std::string& weights_dir,
        const std::string& vocab_path,
        const std::string& config_path);

    /// Predict the best move for a FEN position using thinking inference.
    std::string predictMove(const std::string& fen, float temperature = 0.0f);

    /// Get token IDs from the last predictMove() call.
    const std::vector<int>& lastTokenIds() const { return last_token_ids_; }

    /// Get WL entries from the last predictMove() call: (position, value) pairs.
    const std::vector<std::pair<int, float>>& lastWlEntries() const { return last_wl_entries_; }

    /// Get D entries from the last predictMove() call: (position, value) pairs.
    const std::vector<std::pair<int, float>>& lastDEntries() const { return last_d_entries_; }

    /// Convert a token ID to its string name.
    const std::string& idxToToken(int idx) const { return vocab_->idxToToken(idx); }

    // Per-head temperature overrides.
    // board: controls structural decisions (end_var, end_think). NOT board piece tokens.
    // think/policy: -1.0 means "use the temperature arg to predictMove()". >= 0 overrides.
    // wl/d: sample bucket instead of argmax when > 0.
    float board_temperature{0.0f};
    float think_temperature{-1.0f};
    float policy_temperature{-1.0f};
    float wl_temperature{0.0f};
    float d_temperature{0.0f};

    // Stats
    int64_t total_tokens{0};
    double total_time{0.0};

    // Profiling counters (accumulated across predict_move calls)
    bool profiling{false};
    double prof_prefix_init{0}, prof_board_prefill{0}, prof_board_catchup{0};
    double prof_board_gen{0}, prof_prefix_block{0}, prof_prefix_incr{0};
    double prof_causal_incr{0}, prof_head_eval{0}, prof_sync_ops{0};
    void resetProfile() {
        prof_prefix_init = prof_board_prefill = prof_board_catchup = 0;
        prof_board_gen = prof_prefix_block = prof_prefix_incr = 0;
        prof_causal_incr = prof_head_eval = prof_sync_ops = 0;
    }

private:
    enum class State
    {
        MOVE,
        WL_D,
        BOARD,
        AFTER_BOARD,
        AFTER_END_VAR,
        FINAL,
    };

    /// Run causal prefill over all current tokens, populating causal KV cache.
    std::vector<float> causalPrefill();

    /// Build prefix attention mask from block IDs.
    void buildPrefixMask(std::vector<float>& mask) const;

    /// Sample from logits (argmax if temperature <= 0).
    int sampleToken(const float* logits, int vocab_size, float temperature) const;

    /// Fallback: direct policy head on root board (no thinking).
    std::string fallbackMove(const std::string& fen, float temperature);

    /// GPU head evaluation
    int evalThinkingPolicyHeadGpu(float temperature);
    int evalPolicyHeadGpu(float temperature, const std::vector<int>& legal_indices);
    float predictWlGpu(float temperature);
    float predictDGpu(float temperature);

    // Components
    std::unique_ptr<TorchCausalBackbone> backbone_;
    std::unique_ptr<Heads> heads_;
    std::unique_ptr<DecoderVocab> vocab_;

    // Model config
    int embed_dim_;
    int num_layers_;
    int num_heads_;
    int head_dim_;
    int max_seq_len_;

    // Current sequence state (rebuilt for each predict_move call)
    std::vector<int> token_ids_;
    std::vector<int> block_ids_;
    std::vector<std::pair<int, float>> wl_entries_;
    std::vector<std::pair<int, float>> d_entries_;
    int next_block_;
    int orphan_ctr_;

    // Saved from last predictMove() call for inspection
    std::vector<int> last_token_ids_;
    std::vector<std::pair<int, float>> last_wl_entries_;
    std::vector<std::pair<int, float>> last_d_entries_;

    // GPU head weights (pre-transposed FP16 CUDA)
    torch::Tensor board_head_w_gpu_t_;  // [E, board_vocab_size] FP16 CUDA (transposed)
    torch::Tensor board_head_b_gpu_;    // [board_vocab_size] FP16 CUDA
    torch::Tensor board_lut_gpu_;       // [board_vocab_size] int64 CUDA

    torch::Tensor policy_w_gpu_t_, policy_b_gpu_;
    torch::Tensor think_policy_w_gpu_t_, think_policy_b_gpu_;
    torch::Tensor wl_w1_gpu_t_, wl_b1_gpu_, wl_w2_gpu_t_, wl_b2_gpu_;
    torch::Tensor d_w1_gpu_t_, d_b1_gpu_, d_w2_gpu_t_, d_b2_gpu_;
    torch::Tensor wl_centers_gpu_, d_centers_gpu_;  // FP32 on CUDA

    // Saved prefix hidden state on GPU
    torch::Tensor saved_prefix_hidden_gpu_;  // [E] FP16 CUDA
};

} // namespace decoder

#pragma once

#include "batched_backbone.hpp"
#include "heads.hpp"
#include "vocab.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace decoder
{

/// Batched thinking inference engine.
///
/// Processes B FENs simultaneously through a lockstep state machine.
/// All sequences advance through the same phases together (MOVE → WL_D →
/// BOARD → decision), with active masks for finished sequences.
///
/// No CUDA graphs — batching provides sufficient GPU utilization.
/// No subprocess workers — runs in the main process.
class ThinkingBatchedInferenceEngine
{
public:
    ThinkingBatchedInferenceEngine(const std::string& backbone_pt_path,
                           const std::string& weights_dir,
                           const std::string& vocab_path,
                           const std::string& config_path,
                           int max_batch_size);

    struct Result
    {
        std::string move;
        std::vector<int> token_ids;
        std::vector<std::pair<int, float>> wl_entries;
        std::vector<std::pair<int, float>> d_entries;
        // (prediction_position, log_prob) — one entry per sampled move token.
        // Position is the index in token_ids of the hidden-state token that
        // predicted the move (same position marked by thinking_move_mask /
        // final_move_mask in chessdecoder/rl/sequence.py).
        std::vector<std::pair<int, float>> move_log_probs;
    };

    /// Process up to max_batch_size FENs. Pads internally if fewer.
    std::vector<Result> predictMoves(const std::vector<std::string>& fens,
                                     float temperature);

    // Temperature controls (same semantics as ThinkingSingleInferenceEngine)
    float board_temperature = 0.0f;
    float think_temperature = -1.0f;
    float policy_temperature = -1.0f;
    float wl_temperature = 0.0f;
    float d_temperature = 0.0f;

    // Statistics
    int64_t total_tokens = 0;
    double total_time = 0.0;

private:
    // ── Head evaluation (batched, on GPU) ──────────────────────────────
    // All return GPU tensors. Masking of inactive sequences is caller's job.

    /// [B, E] → [B] sampled move token indices (full vocab)
    torch::Tensor evalThinkingPolicyHead(torch::Tensor h, float temp);
    /// [B, E] → ([B] sampled sub-vocab indices, [B] log-probs under UNMASKED
    /// log_softmax). Legal-move masking is applied only to the sampling copy.
    std::pair<torch::Tensor, torch::Tensor>
        evalPolicyHead(torch::Tensor h, float temp,
                       const std::vector<std::string>& fens);
    torch::Tensor evalBoardHead(torch::Tensor h, float temp);

    /// [B, E] → [B] float values (WL or D)
    torch::Tensor evalWlHead(torch::Tensor h, float temp);
    torch::Tensor evalDHead(torch::Tensor h, float temp);

    /// Sample from logits [B, V] → [B] indices. Argmax if temp <= 0.
    torch::Tensor sampleBatched(torch::Tensor logits, float temp);

    std::unique_ptr<BatchedBackbone> backbone_;
    std::unique_ptr<Heads> heads_;
    std::unique_ptr<DecoderVocab> vocab_;

    int max_batch_size_;
    int embed_dim_, max_seq_len_;

    // GPU head weights (pre-transposed, FP16, shared across batch)
    torch::Tensor board_w_t_, board_b_, board_lut_;
    torch::Tensor policy_w_t_, policy_b_;
    torch::Tensor think_w_t_, think_b_;
    torch::Tensor wl_w1_t_, wl_b1_, wl_w2_t_, wl_b2_, wl_centers_;
    torch::Tensor d_w1_t_, d_b1_, d_w2_t_, d_b2_, d_centers_;
};

} // namespace decoder

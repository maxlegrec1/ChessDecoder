#pragma once

#include <string>
#include <vector>

namespace decoder
{

/// Head weight storage and CPU fallback evaluation.
/// Primary use: loads weights from disk and provides raw data pointers for GPU upload.
/// CPU eval is only used for fallback (evalPolicyHead in fallbackMove).
class Heads
{
public:
    /// Load all head weights from the weights directory.
    explicit Heads(const std::string& weights_dir, int embed_dim,
                   int board_vocab_size, int move_vocab_size,
                   int value_hidden_size, int n_buckets, int num_fourier_freq);

    /// Evaluate policy_head on CPU: hidden[embed_dim] -> logits[move_vocab_size]
    /// Used only in fallbackMove (no-thinking path).
    void evalPolicyHead(const float* hidden, float* logits) const;

    int embedDim() const { return embed_dim_; }
    int boardVocabSize() const { return board_vocab_size_; }
    int moveVocabSize() const { return move_vocab_size_; }

    // Raw data access for GPU upload
    const float* boardWeightData() const { return board_weight_.data(); }
    const float* boardBiasData() const { return board_bias_.data(); }

    const float* policyWeightData() const { return policy_weight_.data(); }
    const float* policyBiasData() const { return policy_bias_.data(); }

    const float* thinkingPolicyWeightData() const { return thinking_policy_weight_.data(); }
    const float* thinkingPolicyBiasData() const { return thinking_policy_bias_.data(); }

    const float* wlW1WeightData() const { return wl_w1_weight_.data(); }
    const float* wlW1BiasData() const { return wl_w1_bias_.data(); }
    const float* wlW2WeightData() const { return wl_w2_weight_.data(); }
    const float* wlW2BiasData() const { return wl_w2_bias_.data(); }

    const float* dW1WeightData() const { return d_w1_weight_.data(); }
    const float* dW1BiasData() const { return d_w1_bias_.data(); }
    const float* dW2WeightData() const { return d_w2_weight_.data(); }
    const float* dW2BiasData() const { return d_w2_bias_.data(); }

    const float* wlBucketCentersData() const { return wl_bucket_centers_.data(); }
    const float* dBucketCentersData() const { return d_bucket_centers_.data(); }

    int valueHiddenSize() const { return value_hidden_size_; }
    int nBuckets() const { return n_buckets_; }

private:
    /// Gemv: out = W * x + b, where W is [out_dim, in_dim], x is [in_dim]
    static void gemv(const float* W, const float* x, const float* b,
                     int out_dim, int in_dim, float* out);

    int embed_dim_;
    int board_vocab_size_;
    int move_vocab_size_;
    int value_hidden_size_;
    int n_buckets_;

    // Board head: Linear(E -> board_vocab_size)
    std::vector<float> board_weight_;  // [board_vocab_size, E]
    std::vector<float> board_bias_;    // [board_vocab_size]

    // Policy head: Linear(E -> move_vocab_size)
    std::vector<float> policy_weight_;
    std::vector<float> policy_bias_;

    // Thinking policy head: Linear(E -> move_vocab_size)
    std::vector<float> thinking_policy_weight_;
    std::vector<float> thinking_policy_bias_;

    // WL head: Linear(E -> H) -> Mish -> Linear(H -> n_buckets)
    std::vector<float> wl_w1_weight_;  // [H, E]
    std::vector<float> wl_w1_bias_;    // [H]
    std::vector<float> wl_w2_weight_;  // [n_buckets, H]
    std::vector<float> wl_w2_bias_;    // [n_buckets]

    // D head: same structure
    std::vector<float> d_w1_weight_;
    std::vector<float> d_w1_bias_;
    std::vector<float> d_w2_weight_;
    std::vector<float> d_w2_bias_;

    // Bucket centers
    std::vector<float> wl_bucket_centers_;  // [n_buckets]
    std::vector<float> d_bucket_centers_;   // [n_buckets]
};

} // namespace decoder

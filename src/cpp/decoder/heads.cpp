#include "heads.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace decoder
{

namespace
{

// ======================== FP16 conversion helpers ========================
// Used ONLY at output boundaries (not in inner loops) to match cuBLAS HGEMM
// behavior: FP32 accumulation internally, FP16 store on output.

// Convert FP16 (uint16_t) to FP32 — exact, no rounding
float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            f32 = sign << 31;
        }
        else
        {
            exponent = 1;
            while (!(mantissa & 0x400))
            {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    }
    else if (exponent == 31)
    {
        f32 = (sign << 31) | 0x7F800000 | (mantissa << 13);
    }
    else
    {
        f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

// Convert FP32 to FP16 — round to nearest even (IEEE 754)
uint16_t fp32_to_fp16(float f)
{
    uint32_t fi;
    std::memcpy(&fi, &f, sizeof(fi));

    uint32_t sign = (fi >> 31) & 1;
    int32_t exp = static_cast<int32_t>((fi >> 23) & 0xFF) - 127;
    uint32_t mant = fi & 0x7FFFFF;

    if (exp == 128)
    {
        if (mant == 0)
            return static_cast<uint16_t>((sign << 15) | 0x7C00);
        return static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant >> 13));
    }

    if (exp > -127)
        mant |= 0x800000;
    else
    {
        exp = -127;
    }

    int32_t h_exp = exp + 15;

    if (h_exp >= 31)
        return static_cast<uint16_t>((sign << 15) | 0x7C00);

    if (h_exp <= 0)
    {
        int shift = 14 - h_exp;
        if (shift > 24) return static_cast<uint16_t>(sign << 15);

        uint32_t h_mant = mant >> shift;
        uint32_t remainder = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (h_mant & 1)))
            h_mant++;

        return static_cast<uint16_t>((sign << 15) | h_mant);
    }

    // Strip implicit 1-bit: only keep the 10 stored mantissa bits
    uint16_t h_mant = static_cast<uint16_t>((mant >> 13) & 0x3FF);
    uint32_t remainder = mant & 0x1FFF;
    if (remainder > 0x1000 || (remainder == 0x1000 && (h_mant & 1)))
        h_mant++;

    if (h_mant >= 0x400)
    {
        h_mant = 0;
        h_exp++;
        if (h_exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }

    return static_cast<uint16_t>((sign << 15) | (h_exp << 10) | h_mant);
}

// Round FP32 to FP16 precision (FP32 → FP16 → FP32)
float round_fp16(float f)
{
    return fp16_to_fp32(fp32_to_fp16(f));
}

// ======================== File loaders ========================

std::vector<float> loadFP16(const std::string& path, size_t expected_elements)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open weight file: " + path);

    f.seekg(0, std::ios::end);
    size_t fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    size_t num_elements = fsize / 2;
    if (expected_elements > 0 && num_elements != expected_elements)
    {
        throw std::runtime_error("Weight file " + path + " has " + std::to_string(num_elements)
                                 + " elements, expected " + std::to_string(expected_elements));
    }

    std::vector<uint16_t> fp16_data(num_elements);
    f.read(reinterpret_cast<char*>(fp16_data.data()), fsize);

    std::vector<float> fp32_data(num_elements);
    for (size_t i = 0; i < num_elements; i++)
    {
        fp32_data[i] = fp16_to_fp32(fp16_data[i]);
    }
    return fp32_data;
}

std::vector<float> loadFP32(const std::string& path, size_t expected_elements)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open weight file: " + path);

    f.seekg(0, std::ios::end);
    size_t fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    size_t num_elements = fsize / 4;
    if (expected_elements > 0 && num_elements != expected_elements)
    {
        throw std::runtime_error("Weight file " + path + " has " + std::to_string(num_elements)
                                 + " elements, expected " + std::to_string(expected_elements));
    }

    std::vector<float> data(num_elements);
    f.read(reinterpret_cast<char*>(data.data()), fsize);
    return data;
}

// Softmax-weighted sum for value prediction.
// Python: probs = F.softmax(logits.float(), dim=-1); return (probs * centers).sum()
// .float() means softmax in FP32 on FP16-precision logits — which is what we have.
float softmax_sum(const float* logits, const float* centers, int n)
{
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum_exp = 0.0f;
    float weighted_sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float e = std::exp(logits[i] - max_val);
        sum_exp += e;
        weighted_sum += e * centers[i];
    }
    return weighted_sum / sum_exp;
}

} // anonymous namespace

Heads::Heads(const std::string& weights_dir, int embed_dim,
             int board_vocab_size, int move_vocab_size,
             int value_hidden_size, int n_buckets, int num_fourier_freq)
    : embed_dim_(embed_dim)
    , board_vocab_size_(board_vocab_size)
    , move_vocab_size_(move_vocab_size)
    , value_hidden_size_(value_hidden_size)
    , n_buckets_(n_buckets)
    , num_fourier_freq_(num_fourier_freq)
{
    std::string d = weights_dir;
    if (d.back() != '/') d += '/';

    board_weight_ = loadFP16(d + "board_head_weight.bin", board_vocab_size * embed_dim);
    board_bias_ = loadFP16(d + "board_head_bias.bin", board_vocab_size);
    policy_weight_ = loadFP16(d + "policy_head_weight.bin", move_vocab_size * embed_dim);
    policy_bias_ = loadFP16(d + "policy_head_bias.bin", move_vocab_size);
    thinking_policy_weight_ = loadFP16(d + "thinking_policy_head_weight.bin", move_vocab_size * embed_dim);
    thinking_policy_bias_ = loadFP16(d + "thinking_policy_head_bias.bin", move_vocab_size);
    wl_w1_weight_ = loadFP16(d + "wl_head_w1_weight.bin", value_hidden_size * embed_dim);
    wl_w1_bias_ = loadFP16(d + "wl_head_w1_bias.bin", value_hidden_size);
    wl_w2_weight_ = loadFP16(d + "wl_head_w2_weight.bin", n_buckets * value_hidden_size);
    wl_w2_bias_ = loadFP16(d + "wl_head_w2_bias.bin", n_buckets);
    d_w1_weight_ = loadFP16(d + "d_head_w1_weight.bin", value_hidden_size * embed_dim);
    d_w1_bias_ = loadFP16(d + "d_head_w1_bias.bin", value_hidden_size);
    d_w2_weight_ = loadFP16(d + "d_head_w2_weight.bin", n_buckets * value_hidden_size);
    d_w2_bias_ = loadFP16(d + "d_head_w2_bias.bin", n_buckets);
    fourier_frequencies_ = loadFP16(d + "fourier_frequencies.bin", num_fourier_freq);
    fourier_proj_weight_ = loadFP16(d + "fourier_proj_weight.bin", embed_dim * 2 * num_fourier_freq);
    fourier_proj_bias_ = loadFP16(d + "fourier_proj_bias.bin", embed_dim);
    wl_bucket_centers_ = loadFP32(d + "wl_bucket_centers.bin", n_buckets);
    d_bucket_centers_ = loadFP32(d + "d_bucket_centers.bin", n_buckets);
}

// FP16 gemv: FP32 accumulation, FP16 store on output (matches cuBLAS HGEMM).
void Heads::gemv(const float* W, const float* x, const float* b,
                 int out_dim, int in_dim, float* out)
{
    for (int i = 0; i < out_dim; i++)
    {
        float acc = 0.0f;
        const float* row = W + i * in_dim;
        for (int j = 0; j < in_dim; j++)
            acc += row[j] * x[j];
        if (b != nullptr)
            acc += b[i];
        out[i] = round_fp16(acc);
    }
}

// Mish activation: x * tanh(softplus(x)), FP16 output.
float Heads::mish(float x)
{
    float sp = std::log(1.0f + std::exp(x));
    return round_fp16(x * std::tanh(sp));
}

// Hidden states from TRT --fp16 are already FP16-precision.
// No need to pre-round them.
void Heads::evalBoardHead(const float* hidden, float* logits) const
{
    gemv(board_weight_.data(), hidden, board_bias_.data(),
         board_vocab_size_, embed_dim_, logits);
}

void Heads::evalPolicyHead(const float* hidden, float* logits) const
{
    gemv(policy_weight_.data(), hidden, policy_bias_.data(),
         move_vocab_size_, embed_dim_, logits);
}

void Heads::evalThinkingPolicyHead(const float* hidden, float* logits) const
{
    gemv(thinking_policy_weight_.data(), hidden, thinking_policy_bias_.data(),
         move_vocab_size_, embed_dim_, logits);
}

void Heads::evalWlHead(const float* hidden, float* logits) const
{
    // Layer 1: Linear(E -> H), output FP16
    std::vector<float> h(value_hidden_size_);
    gemv(wl_w1_weight_.data(), hidden, wl_w1_bias_.data(),
         value_hidden_size_, embed_dim_, h.data());
    // Mish: FP32 compute, FP16 output
    for (int i = 0; i < value_hidden_size_; i++)
        h[i] = mish(h[i]);
    // Layer 2: Linear(H -> n_buckets), output FP16
    gemv(wl_w2_weight_.data(), h.data(), wl_w2_bias_.data(),
         n_buckets_, value_hidden_size_, logits);
}

void Heads::evalDHead(const float* hidden, float* logits) const
{
    std::vector<float> h(value_hidden_size_);
    gemv(d_w1_weight_.data(), hidden, d_w1_bias_.data(),
         value_hidden_size_, embed_dim_, h.data());
    for (int i = 0; i < value_hidden_size_; i++)
        h[i] = mish(h[i]);
    gemv(d_w2_weight_.data(), h.data(), d_w2_bias_.data(),
         n_buckets_, value_hidden_size_, logits);
}

void Heads::evalFourier(float value, float* embedding) const
{
    // FP16 Fourier encoding: matches PyTorch .half() behavior.
    //   f = 2 * pi * value * frequencies
    //   features = [cos(f), sin(f)]
    //   embedding = proj(features)
    int F = num_fourier_freq_;
    float val = round_fp16(value);
    float two_pi = round_fp16(2.0f * static_cast<float>(M_PI));
    float two_pi_val = round_fp16(two_pi * val);

    std::vector<float> features(2 * F);
    for (int i = 0; i < F; i++)
    {
        float angle = round_fp16(two_pi_val * fourier_frequencies_[i]);
        features[i] = round_fp16(std::cos(angle));
        features[F + i] = round_fp16(std::sin(angle));
    }

    gemv(fourier_proj_weight_.data(), features.data(), fourier_proj_bias_.data(),
         embed_dim_, 2 * F, embedding);
}

float Heads::predictWl(const float* hidden) const
{
    std::vector<float> logits(n_buckets_);
    evalWlHead(hidden, logits.data());
    // Argmax to bucket center (matches training data generation and model.predict_move)
    int best = 0;
    for (int i = 1; i < n_buckets_; i++)
        if (logits[i] > logits[best]) best = i;
    return wl_bucket_centers_[best];
}

float Heads::predictD(const float* hidden) const
{
    std::vector<float> logits(n_buckets_);
    evalDHead(hidden, logits.data());
    int best = 0;
    for (int i = 1; i < n_buckets_; i++)
        if (logits[i] > logits[best]) best = i;
    return d_bucket_centers_[best];
}

} // namespace decoder

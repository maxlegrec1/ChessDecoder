#include "heads.hpp"

#include <cstring>
#include <fstream>
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

} // anonymous namespace

Heads::Heads(const std::string& weights_dir, int embed_dim,
             int board_vocab_size, int move_vocab_size,
             int value_hidden_size, int n_buckets, int num_fourier_freq)
    : embed_dim_(embed_dim)
    , board_vocab_size_(board_vocab_size)
    , move_vocab_size_(move_vocab_size)
    , value_hidden_size_(value_hidden_size)
    , n_buckets_(n_buckets)
{
    (void)num_fourier_freq;  // Fourier is baked into TorchScript backbone

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

void Heads::evalPolicyHead(const float* hidden, float* logits) const
{
    gemv(policy_weight_.data(), hidden, policy_bias_.data(),
         move_vocab_size_, embed_dim_, logits);
}


} // namespace decoder

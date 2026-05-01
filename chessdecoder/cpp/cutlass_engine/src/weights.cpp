#include "cutlass_engine/weights.hpp"
#include "cutlass_engine/check.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace cutlass_engine {

namespace {

namespace fs = std::filesystem;

std::vector<char> read_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("could not open " + path);
    auto size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}

// Upload a flat FP16 .bin file to a device half buffer. Asserts byte count.
void upload_fp16(const std::string& path, __half* dst, std::size_t expected_bytes) {
    auto buf = read_binary(path);
    if (buf.size() != expected_bytes) {
        char err[512];
        std::snprintf(err, sizeof(err),
                      "weights: %s expected %zu bytes, got %zu",
                      path.c_str(), expected_bytes, buf.size());
        throw std::runtime_error(err);
    }
    CE_CUDA_CHECK(cudaMemcpy(dst, buf.data(), buf.size(), cudaMemcpyHostToDevice));
}

void upload_fp32(const std::string& path, float* dst, std::size_t expected_bytes) {
    auto buf = read_binary(path);
    if (buf.size() != expected_bytes) {
        char err[512];
        std::snprintf(err, sizeof(err),
                      "weights: %s expected %zu bytes, got %zu",
                      path.c_str(), expected_bytes, buf.size());
        throw std::runtime_error(err);
    }
    CE_CUDA_CHECK(cudaMemcpy(dst, buf.data(), buf.size(), cudaMemcpyHostToDevice));
}

// Convert a FP32 .bin to FP16 via a host-side cast and upload. Some exported
// tensors (frequencies, fourier proj) come from FP32 dumps in older exports.
void upload_fp32_as_fp16(const std::string& path, __half* dst,
                        std::size_t expected_count) {
    auto buf = read_binary(path);
    if (buf.size() != expected_count * 4) {
        char err[512];
        std::snprintf(err, sizeof(err),
                      "weights: %s expected %zu fp32 elems, got %zu bytes",
                      path.c_str(), expected_count, buf.size());
        throw std::runtime_error(err);
    }
    std::vector<__half> h(expected_count);
    const float* src = reinterpret_cast<const float*>(buf.data());
    for (std::size_t i = 0; i < expected_count; ++i) {
        h[i] = __float2half_rn(src[i]);
    }
    CE_CUDA_CHECK(cudaMemcpy(dst, h.data(), h.size() * sizeof(__half),
                             cudaMemcpyHostToDevice));
}

bool path_exists(const std::string& p) { return fs::exists(p); }

// Try fp16 first, then fp32 (some older exports were fp32).
void upload_fp16_or_fp32(const std::string& base, __half* dst,
                        std::size_t expected_fp16_bytes) {
    const std::string fp16 = base + ".bin";
    if (path_exists(fp16)) {
        // Heuristic: file is fp16 iff size matches. If size doubles, treat as fp32.
        auto sz = fs::file_size(fp16);
        if (sz == expected_fp16_bytes) {
            upload_fp16(fp16, dst, expected_fp16_bytes);
        } else if (sz == expected_fp16_bytes * 2) {
            upload_fp32_as_fp16(fp16, dst, expected_fp16_bytes / 2);
        } else {
            throw std::runtime_error("weights: " + fp16 + " has unexpected size " +
                                     std::to_string(sz));
        }
        return;
    }
    throw std::runtime_error("weights: missing file " + fp16);
}

// Compute the cos/sin RoPE table on the host (matches torchtune's
// RotaryPositionalEmbeddings). Uses base=10000 by default.
void compute_rope_table(int max_seq_len, int head_dim,
                        std::vector<float>& cos, std::vector<float>& sin,
                        float base = 10000.0f) {
    const int half = head_dim / 2;
    cos.assign(static_cast<std::size_t>(max_seq_len) * half, 0.0f);
    sin.assign(static_cast<std::size_t>(max_seq_len) * half, 0.0f);
    for (int p = 0; p < max_seq_len; ++p) {
        for (int j = 0; j < half; ++j) {
            float theta = std::pow(base, -float(2 * j) / float(head_dim));
            float angle = float(p) * theta;
            cos[p * half + j] = std::cos(angle);
            sin[p * half + j] = std::sin(angle);
        }
    }
}

}  // namespace

ModelWeights load_weights(const std::string& weights_dir,
                          const ModelConfig& cfg, Arena& arena) {
    ModelWeights w;
    const int E = cfg.embed_dim;
    const int V = cfg.vocab_size;
    const int H = cfg.value_hidden_size;
    const int B_v = cfg.board_vocab_size;
    const int M_v = cfg.move_vocab_size;
    const int K_b = cfg.n_buckets;
    const int F = cfg.num_fourier_freq;
    const int d_ff = cfg.d_ff;

    const std::size_t E_bytes = E * sizeof(__half);

    // ---------------- Backbone (per-layer) ----------------
    w.layers.resize(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; ++i) {
        auto& L = w.layers[i];
        L.sa_norm = arena.allocT<__half>(E);
        L.mlp_norm = arena.allocT<__half>(E);
        L.qkv_w = arena.allocT<__half>(3 * E * E);
        L.out_w = arena.allocT<__half>(E * E);
        L.gate_up_w = arena.allocT<__half>(2 * d_ff * E);
        L.down_w = arena.allocT<__half>(E * d_ff);

        const std::string base = weights_dir + "/backbone/layer_" + std::to_string(i);
        upload_fp16_or_fp32(base + "_sa_norm",   L.sa_norm,   E_bytes);
        upload_fp16_or_fp32(base + "_mlp_norm",  L.mlp_norm,  E_bytes);
        upload_fp16_or_fp32(base + "_qkv_w",     L.qkv_w,     3 * E * E_bytes);
        upload_fp16_or_fp32(base + "_out_w",     L.out_w,     E * E_bytes);
        upload_fp16_or_fp32(base + "_gate_up_w", L.gate_up_w, 2 * d_ff * E_bytes);
        upload_fp16_or_fp32(base + "_down_w",    L.down_w,    E * d_ff * sizeof(__half));
    }

    // ---------------- Embedding + final norm ----------------
    w.tok_embedding = arena.allocT<__half>(V * E);
    w.final_norm = arena.allocT<__half>(E);
    upload_fp16_or_fp32(weights_dir + "/backbone/tok_embedding", w.tok_embedding,
                       V * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/backbone/final_norm", w.final_norm, E_bytes);

    // ---------------- Heads (existing exporter layout) ----------------
    w.board_head_w = arena.allocT<__half>(B_v * E);
    w.board_head_b = arena.allocT<__half>(B_v);
    upload_fp16_or_fp32(weights_dir + "/board_head_weight", w.board_head_w, B_v * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/board_head_bias",   w.board_head_b, B_v * sizeof(__half));

    w.policy_head_w = arena.allocT<__half>(M_v * E);
    w.policy_head_b = arena.allocT<__half>(M_v);
    upload_fp16_or_fp32(weights_dir + "/policy_head_weight", w.policy_head_w, M_v * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/policy_head_bias",   w.policy_head_b, M_v * sizeof(__half));

    w.thinking_policy_head_w = arena.allocT<__half>(M_v * E);
    w.thinking_policy_head_b = arena.allocT<__half>(M_v);
    upload_fp16_or_fp32(weights_dir + "/thinking_policy_head_weight",
                       w.thinking_policy_head_w, M_v * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/thinking_policy_head_bias",
                       w.thinking_policy_head_b, M_v * sizeof(__half));

    w.wl_w1_w = arena.allocT<__half>(H * E);
    w.wl_w1_b = arena.allocT<__half>(H);
    w.wl_w2_w = arena.allocT<__half>(K_b * H);
    w.wl_w2_b = arena.allocT<__half>(K_b);
    upload_fp16_or_fp32(weights_dir + "/wl_head_w1_weight", w.wl_w1_w, H * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/wl_head_w1_bias",   w.wl_w1_b, H * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/wl_head_w2_weight", w.wl_w2_w, K_b * H * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/wl_head_w2_bias",   w.wl_w2_b, K_b * sizeof(__half));

    w.d_w1_w = arena.allocT<__half>(H * E);
    w.d_w1_b = arena.allocT<__half>(H);
    w.d_w2_w = arena.allocT<__half>(K_b * H);
    w.d_w2_b = arena.allocT<__half>(K_b);
    upload_fp16_or_fp32(weights_dir + "/d_head_w1_weight", w.d_w1_w, H * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/d_head_w1_bias",   w.d_w1_b, H * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/d_head_w2_weight", w.d_w2_w, K_b * H * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/d_head_w2_bias",   w.d_w2_b, K_b * sizeof(__half));

    // ---------------- Bucket centers (FP32 .bin) ----------------
    w.wl_bucket_centers = arena.allocT<float>(K_b);
    w.d_bucket_centers  = arena.allocT<float>(K_b);
    upload_fp32(weights_dir + "/wl_bucket_centers.bin", w.wl_bucket_centers, K_b * sizeof(float));
    upload_fp32(weights_dir + "/d_bucket_centers.bin",  w.d_bucket_centers,  K_b * sizeof(float));

    // ---------------- Fourier encoder ----------------
    w.fourier_freq    = arena.allocT<__half>(F);
    w.fourier_proj_w  = arena.allocT<__half>(E * 2 * F);
    w.fourier_proj_b  = arena.allocT<__half>(E);
    upload_fp16_or_fp32(weights_dir + "/backbone/fourier_freq",   w.fourier_freq,   F * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/backbone/fourier_proj_w", w.fourier_proj_w, E * 2 * F * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/backbone/fourier_proj_b", w.fourier_proj_b, E_bytes);

    // ---------------- RoPE precomputed table ----------------
    const int half = cfg.head_dim / 2;
    std::vector<float> cos_h, sin_h;
    compute_rope_table(cfg.max_seq_len, cfg.head_dim, cos_h, sin_h);
    w.rope_cos = arena.allocT<float>(cfg.max_seq_len * half);
    w.rope_sin = arena.allocT<float>(cfg.max_seq_len * half);
    CE_CUDA_CHECK(cudaMemcpy(w.rope_cos, cos_h.data(), cos_h.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    CE_CUDA_CHECK(cudaMemcpy(w.rope_sin, sin_h.data(), sin_h.size() * sizeof(float),
                             cudaMemcpyHostToDevice));

    return w;
}

void reupload_weights(const std::string& weights_dir, const ModelConfig& cfg,
                      ModelWeights& w) {
    // Same as load_weights but preserves the existing pointer addresses.
    const int E = cfg.embed_dim;
    const int V = cfg.vocab_size;
    const int H = cfg.value_hidden_size;
    const int B_v = cfg.board_vocab_size;
    const int M_v = cfg.move_vocab_size;
    const int K_b = cfg.n_buckets;
    const int F = cfg.num_fourier_freq;
    const int d_ff = cfg.d_ff;
    const std::size_t E_bytes = E * sizeof(__half);

    for (int i = 0; i < cfg.num_layers; ++i) {
        auto& L = w.layers[i];
        const std::string base = weights_dir + "/backbone/layer_" + std::to_string(i);
        upload_fp16_or_fp32(base + "_sa_norm",   L.sa_norm,   E_bytes);
        upload_fp16_or_fp32(base + "_mlp_norm",  L.mlp_norm,  E_bytes);
        upload_fp16_or_fp32(base + "_qkv_w",     L.qkv_w,     3 * E * E_bytes);
        upload_fp16_or_fp32(base + "_out_w",     L.out_w,     E * E_bytes);
        upload_fp16_or_fp32(base + "_gate_up_w", L.gate_up_w, 2 * d_ff * E_bytes);
        upload_fp16_or_fp32(base + "_down_w",    L.down_w,    E * d_ff * sizeof(__half));
    }
    upload_fp16_or_fp32(weights_dir + "/backbone/tok_embedding", w.tok_embedding, V * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/backbone/final_norm",    w.final_norm,    E_bytes);

    upload_fp16_or_fp32(weights_dir + "/board_head_weight",         w.board_head_w,           B_v * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/board_head_bias",           w.board_head_b,           B_v * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/policy_head_weight",        w.policy_head_w,          M_v * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/policy_head_bias",          w.policy_head_b,          M_v * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/thinking_policy_head_weight", w.thinking_policy_head_w, M_v * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/thinking_policy_head_bias",   w.thinking_policy_head_b, M_v * sizeof(__half));

    upload_fp16_or_fp32(weights_dir + "/wl_head_w1_weight", w.wl_w1_w, H * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/wl_head_w1_bias",   w.wl_w1_b, H * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/wl_head_w2_weight", w.wl_w2_w, K_b * H * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/wl_head_w2_bias",   w.wl_w2_b, K_b * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/d_head_w1_weight",  w.d_w1_w,  H * E_bytes);
    upload_fp16_or_fp32(weights_dir + "/d_head_w1_bias",    w.d_w1_b,  H * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/d_head_w2_weight",  w.d_w2_w,  K_b * H * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/d_head_w2_bias",    w.d_w2_b,  K_b * sizeof(__half));

    upload_fp32(weights_dir + "/wl_bucket_centers.bin", w.wl_bucket_centers, K_b * sizeof(float));
    upload_fp32(weights_dir + "/d_bucket_centers.bin",  w.d_bucket_centers,  K_b * sizeof(float));

    upload_fp16_or_fp32(weights_dir + "/backbone/fourier_freq",   w.fourier_freq,   F * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/backbone/fourier_proj_w", w.fourier_proj_w, E * 2 * F * sizeof(__half));
    upload_fp16_or_fp32(weights_dir + "/backbone/fourier_proj_b", w.fourier_proj_b, E_bytes);
    // RoPE table is shape-only, not weight-derived, so it doesn't need re-upload.
}

}  // namespace cutlass_engine

#pragma once

#include <cstdint>
#include <string>

namespace cutlass_engine {

// All shapes the engine knows about. Frozen at engine construction; the
// rest of the code never re-derives or queries them.
struct ModelConfig {
    int vocab_size{0};
    int embed_dim{0};
    int num_heads{0};
    int num_layers{0};
    int head_dim{0};         // = embed_dim / num_heads
    int d_ff{0};             // SwiGLU expansion (default 4*embed_dim)
    int max_seq_len{0};
    int board_vocab_size{0};
    int move_vocab_size{0};
    int n_buckets{0};        // value head buckets (100)
    int value_hidden_size{0};// value head MLP hidden (256)
    int num_fourier_freq{0};
    float wl_sigma{0.4f};

    int batch_size{0};       // engine batch (set in ctor)

    // Sub-vocab → full-vocab mappings (loaded from vocab.json).
    // Sizes: board_vocab_size and move_vocab_size respectively.
    // We store these so the engine never needs to call into Python for them.
    // (Owned externally; just pointers here for serialization-free access.)
};

// Parses config.json (the file written by chessdecoder/rl/rollout.py:export_model)
// and returns a populated ModelConfig.  Throws on missing keys.
ModelConfig load_model_config(const std::string& config_json_path, int batch_size);

}  // namespace cutlass_engine

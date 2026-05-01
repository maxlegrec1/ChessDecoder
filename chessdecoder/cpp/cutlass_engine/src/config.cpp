#include "cutlass_engine/config.hpp"

#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cutlass_engine {

namespace {

// A regex-based JSON-int extractor — same trick the existing decoder_engine.cpp
// uses (no JSON-library dep). config.json is flat; values are int-only.
int read_json_int(const std::string& src, const std::string& key) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch m;
    if (!std::regex_search(src, m, re)) {
        throw std::runtime_error("config.json: missing key '" + key + "'");
    }
    return std::stoi(m[1].str());
}

float read_json_float(const std::string& src, const std::string& key,
                     float default_value) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)");
    std::smatch m;
    if (!std::regex_search(src, m, re)) {
        return default_value;
    }
    return std::stof(m[1].str());
}

}  // namespace

ModelConfig load_model_config(const std::string& path, int batch_size) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("could not open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();

    ModelConfig c;
    c.embed_dim = read_json_int(src, "embed_dim");
    c.num_heads = read_json_int(src, "num_heads");
    c.num_layers = read_json_int(src, "num_layers");
    c.max_seq_len = read_json_int(src, "max_seq_len");
    c.d_ff = read_json_int(src, "d_ff");
    c.vocab_size = read_json_int(src, "vocab_size");
    c.head_dim = read_json_int(src, "head_dim");
    c.board_vocab_size = read_json_int(src, "board_vocab_size");
    c.move_vocab_size = read_json_int(src, "move_vocab_size");
    c.n_buckets = read_json_int(src, "n_buckets");
    c.value_hidden_size = read_json_int(src, "value_hidden_size");
    c.num_fourier_freq = read_json_int(src, "num_fourier_freq");
    c.wl_sigma = read_json_float(src, "wl_sigma", 0.4f);
    c.batch_size = batch_size;

    if (c.head_dim * c.num_heads != c.embed_dim) {
        throw std::runtime_error("config.json: embed_dim != num_heads * head_dim");
    }
    return c;
}

}  // namespace cutlass_engine

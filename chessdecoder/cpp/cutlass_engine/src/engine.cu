#include "cutlass_engine/engine.hpp"
#include "cutlass_engine/check.hpp"

#include <cstdio>
#include <stdexcept>

namespace cutlass_engine {

namespace {

// Compute the bytes the engine needs from arena. This is conservative —
// production should compute exactly per layer/buffer.
std::size_t estimate_arena_bytes(const ModelConfig& c) {
    std::size_t bytes = 0;
    const std::size_t E = c.embed_dim;
    const std::size_t V = c.vocab_size;
    const std::size_t H = c.value_hidden_size;
    const std::size_t Bv = c.board_vocab_size;
    const std::size_t Mv = c.move_vocab_size;
    const std::size_t Kb = c.n_buckets;
    const std::size_t F = c.num_fourier_freq;
    const std::size_t d_ff = c.d_ff;
    const std::size_t MAX = c.max_seq_len;
    const std::size_t Bs = c.batch_size;
    const std::size_t NL = c.num_layers;
    const std::size_t NH = c.num_heads;
    const std::size_t HD = c.head_dim;

    // Backbone weights (FP16).
    bytes += NL * (E * 2 + 3*E*E + E*E + 2*d_ff*E + E*d_ff) * sizeof(__half);
    // Embedding + final norm.
    bytes += (V * E + E) * sizeof(__half);
    // Heads (FP16).
    bytes += (Bv * E + Bv + 2 * (Mv * E + Mv) + 4 * (H * E + H + Kb * H + Kb)) * sizeof(__half);
    // Bucket centers (FP32).
    bytes += 2 * Kb * sizeof(float);
    // Fourier (FP16).
    bytes += (F + E * 2 * F + E) * sizeof(__half);
    // RoPE table (FP32).
    bytes += 2 * MAX * (HD / 2) * sizeof(float);

    // KV cache (FP16).
    bytes += 2 * NL * Bs * NH * MAX * HD * sizeof(__half);
    bytes += Bs * sizeof(int32_t) * 2;  // past_len, slot_active

    // LayerWorkspace at max_M=Bs*max_seq_len (worst case for prefill init).
    // We size for max_M = Bs*71 (a reasonable upper bound for init prefill —
    // 68 board tokens + move + wl + d). Refills use the same buffer.
    const std::size_t max_M = Bs * 71;
    bytes += max_M * E * sizeof(__half) * 6;            // h_in/h_out/residual + 3 buffers
    bytes += max_M * 3 * E * sizeof(__half);            // qkv
    bytes += max_M * E * sizeof(__half);                // attn_out
    bytes += max_M * 2 * d_ff * sizeof(__half);
    bytes += max_M * d_ff * sizeof(__half);
    bytes += max_M * sizeof(int32_t);                   // pos

    // 25% headroom for alignment/padding.
    bytes = bytes + bytes / 4;
    return bytes;
}

}  // namespace

ThinkingEngine::ThinkingEngine(const std::string& /*backbone_pt*/,
                               const std::string& weights_dir,
                               const std::string& /*vocab_json*/,
                               const std::string& config_json,
                               int batch_size) {
    cfg_ = load_model_config(config_json, batch_size);
    const std::size_t total = estimate_arena_bytes(cfg_);
    arena_.reserve(total);
    pinned_arena_.reserve(64 * 1024 * 1024);  // 64 MB pinned host (rollout dumps)

    w_ = load_weights(weights_dir, cfg_, arena_);
    kv_.allocate(cfg_, arena_);

    // Worst-case workspace: prefill of 71 tokens × B.
    const int max_M = cfg_.batch_size * 71;
    model_.initialize(cfg_, w_, arena_, max_M);

    std::printf("[cutlass_engine] arena: %.2f GB used of %.2f GB reserved\n",
                arena_.used_bytes() / 1e9, arena_.total_bytes() / 1e9);
}

ThinkingEngine::~ThinkingEngine() = default;

void ThinkingEngine::update_weights(const std::string& weights_dir) {
    reupload_weights(weights_dir, cfg_, w_);
}

std::vector<RolloutResult> ThinkingEngine::predict_moves(
    const std::vector<std::string>& fens, float /*fallback_temperature*/) {
    sched_.initialize(cfg_.batch_size, (int)fens.size());
    sched_.enqueue_all();

    // SCAFFOLD: this is the entry point that drives the per-state batched
    // engine loop. The kernels and model are in place, but the per-state
    // dispatch + sampler-to-state-machine wiring is the bulk of Phase E.
    //
    // What's in place (wired): model.forward_decode / model.forward_prefill_block,
    // kv_cache management, FMHA decode/prefill, all sampler kernels, head GEMMs.
    //
    // What remains: a host-side dispatcher that for each step (a) collects
    // active slots by state, (b) populates per-state input tensors (ids, pos,
    // legal_mask), (c) launches forward_decode + the head + sampler for that
    // state, (d) reads back sampled idx + log_p (or buffers them on-device for
    // batched D2H at end-of-rollout), (e) updates SlotInfo per slot. This
    // dispatcher is the next session's work.
    //
    // For now we throw so callers see the engine is partial, not silently
    // producing nonsense.
    throw std::runtime_error(
        "cutlass_engine.predict_moves: per-state dispatch not yet wired. "
        "Phases A–C complete; Phase E (engine driver) is the next milestone. "
        "See chessdecoder/cpp/decoder/ for the working libtorch-based engine.");
}

}  // namespace cutlass_engine

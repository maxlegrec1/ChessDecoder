#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "cutlass_engine/allocator.hpp"
#include "cutlass_engine/config.hpp"

namespace cutlass_engine {

// Per-slot KV cache. Layout:
//   K_cache, V_cache : [num_layers, B, NH, max_seq_len, HD] FP16
// (contiguous; num_layers is outermost so a layer's slab is one contiguous
// chunk for cache-friendly streaming.)
//
// Per-slot logical position `past_len[B] int32` is the source of truth for
// "how many cache positions are valid". Decoupled from physical position —
// when a slot is reset (continuous batching refill), we just zero past_len
// and the existing physical cache is overwritten on next write.
class KvCache {
public:
    void allocate(const ModelConfig& cfg, Arena& arena);

    __half* K() { return K_; }
    __half* V() { return V_; }
    int32_t* past_len() { return past_len_; }
    int32_t* slot_active() { return slot_active_; }
    void set_slot_active_ptr(int32_t* p) { slot_active_ = p; }

    // Reset slot to logical zero. Caller must zero past_len[b] and active[b]
    // can be flipped accordingly.
    void mark_reset_pending(int slot);

    // Issue past_len[active]=0 on the engine stream.
    void reset_pending_slots(cudaStream_t stream);

private:
    __half* K_{nullptr};
    __half* V_{nullptr};
    int32_t* past_len_{nullptr};
    int32_t* slot_active_{nullptr};

    int B_{0};
    int max_seq_len_{0};
    int num_layers_{0};
    int num_heads_{0};
    int head_dim_{0};

    // Bitmap of slots pending reset (host-side; flushed in reset_pending_slots).
    std::uint64_t pending_mask_{0};
};

}  // namespace cutlass_engine

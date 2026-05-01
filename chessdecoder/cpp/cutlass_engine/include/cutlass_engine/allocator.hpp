#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace cutlass_engine {

// One cudaMalloc, hand out aligned slices. Used for every device tensor in the
// engine. No free() per slice — the whole arena is freed at engine destruction.
//
// 256B alignment is enough for cp.async.bulk and TMA; CUTLASS kernels expect
// at least 128B for tensor-op aligned loads.
class Arena {
public:
    Arena() = default;
    ~Arena();

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Reserve `total_bytes`. Must be called before any alloc().
    void reserve(std::size_t total_bytes);

    // Allocate `bytes` bytes aligned to `align`. Returns a device pointer
    // inside the arena. Throws std::runtime_error on overflow.
    void* alloc(std::size_t bytes, std::size_t align = 256);

    // Slice a typed device pointer of `count` elements.
    template <typename T>
    T* allocT(std::size_t count, std::size_t align = 256) {
        return static_cast<T*>(alloc(count * sizeof(T), align));
    }

    std::size_t used_bytes() const { return used_; }
    std::size_t total_bytes() const { return total_; }

    // Reset the bump cursor. Memory contents are not cleared; caller must
    // re-init what they need. Intended for debugging only — production engine
    // does not reset.
    void reset() { used_ = 0; }

private:
    void* base_{nullptr};
    std::size_t used_{0};
    std::size_t total_{0};
};

// Pinned host arena for end-of-rollout dumps. Same bump-allocator semantics.
class PinnedArena {
public:
    PinnedArena() = default;
    ~PinnedArena();

    PinnedArena(const PinnedArena&) = delete;
    PinnedArena& operator=(const PinnedArena&) = delete;

    void reserve(std::size_t total_bytes);
    void* alloc(std::size_t bytes, std::size_t align = 64);

    template <typename T>
    T* allocT(std::size_t count, std::size_t align = 64) {
        return static_cast<T*>(alloc(count * sizeof(T), align));
    }

private:
    void* base_{nullptr};
    std::size_t used_{0};
    std::size_t total_{0};
};

}  // namespace cutlass_engine

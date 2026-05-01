#include "cutlass_engine/allocator.hpp"
#include "cutlass_engine/check.hpp"

#include <stdexcept>
#include <string>

namespace cutlass_engine {

namespace {
inline std::size_t align_up(std::size_t x, std::size_t a) {
    return (x + a - 1) & ~(a - 1);
}
}

// ---------- Arena ----------

Arena::~Arena() {
    if (base_) {
        cudaFree(base_);
    }
}

void Arena::reserve(std::size_t total_bytes) {
    if (base_) {
        throw std::runtime_error("Arena: reserve() called twice");
    }
    CE_CUDA_CHECK(cudaMalloc(&base_, total_bytes));
    total_ = total_bytes;
    used_ = 0;
}

void* Arena::alloc(std::size_t bytes, std::size_t align) {
    used_ = align_up(used_, align);
    if (used_ + bytes > total_) {
        throw std::runtime_error(
            "Arena overflow: requested " + std::to_string(bytes) +
            " bytes at offset " + std::to_string(used_) +
            " (capacity " + std::to_string(total_) + ")");
    }
    void* p = static_cast<char*>(base_) + used_;
    used_ += bytes;
    return p;
}

// ---------- PinnedArena ----------

PinnedArena::~PinnedArena() {
    if (base_) {
        cudaFreeHost(base_);
    }
}

void PinnedArena::reserve(std::size_t total_bytes) {
    if (base_) {
        throw std::runtime_error("PinnedArena: reserve() called twice");
    }
    CE_CUDA_CHECK(cudaHostAlloc(&base_, total_bytes, cudaHostAllocDefault));
    total_ = total_bytes;
    used_ = 0;
}

void* PinnedArena::alloc(std::size_t bytes, std::size_t align) {
    used_ = align_up(used_, align);
    if (used_ + bytes > total_) {
        throw std::runtime_error(
            "PinnedArena overflow: requested " + std::to_string(bytes) +
            " bytes at offset " + std::to_string(used_) +
            " (capacity " + std::to_string(total_) + ")");
    }
    void* p = static_cast<char*>(base_) + used_;
    used_ += bytes;
    return p;
}

}  // namespace cutlass_engine

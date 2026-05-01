#include "cutlass_engine/kv_cache.hpp"
#include "cutlass_engine/check.hpp"

#include <cstdint>
#include <cstring>

namespace cutlass_engine {

namespace {

__global__ void reset_past_len_kernel(int32_t* past_len, int32_t* slot_active,
                                      const int32_t* slot_idx, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    int b = slot_idx[i];
    past_len[b] = 0;
    slot_active[b] = 1;
}

}  // namespace

void KvCache::allocate(const ModelConfig& cfg, Arena& arena) {
    B_ = cfg.batch_size;
    max_seq_len_ = cfg.max_seq_len;
    num_layers_ = cfg.num_layers;
    num_heads_ = cfg.num_heads;
    head_dim_ = cfg.head_dim;

    const std::size_t per_layer =
        (std::size_t)B_ * num_heads_ * max_seq_len_ * head_dim_;
    K_ = arena.allocT<__half>(num_layers_ * per_layer);
    V_ = arena.allocT<__half>(num_layers_ * per_layer);
    past_len_ = arena.allocT<int32_t>(B_);
    slot_active_ = arena.allocT<int32_t>(B_);

    CE_CUDA_CHECK(cudaMemset(past_len_, 0, B_ * sizeof(int32_t)));
    CE_CUDA_CHECK(cudaMemset(slot_active_, 0, B_ * sizeof(int32_t)));
    // K/V are zeroed lazily — first write overwrites; intermediate reads use
    // past_len as the bound, so unused regions are never observed.
}

void KvCache::mark_reset_pending(int slot) {
    pending_mask_ |= (std::uint64_t(1) << slot);
}

void KvCache::reset_pending_slots(cudaStream_t stream) {
    if (pending_mask_ == 0) return;
    int slot_buf[64];
    int n = 0;
    for (int b = 0; b < B_ && b < 64; ++b) {
        if (pending_mask_ & (std::uint64_t(1) << b)) {
            slot_buf[n++] = b;
        }
    }
    pending_mask_ = 0;
    if (n == 0) return;
    // Stage to device.
    int32_t* d_idx = nullptr;
    cudaMallocAsync(&d_idx, n * sizeof(int32_t), stream);
    cudaMemcpyAsync(d_idx, slot_buf, n * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    int blocks = (n + 31) / 32;
    reset_past_len_kernel<<<blocks, 32, 0, stream>>>(past_len_, slot_active_,
                                                      d_idx, n);
    cudaFreeAsync(d_idx, stream);
    CE_CUDA_LAST();
}

}  // namespace cutlass_engine

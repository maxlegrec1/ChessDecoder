#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace cutlass_engine {

enum class Precision : uint8_t {
    FP16 = 0,
    FP8_E4M3 = 1,
};

// Compile-time alias for "the activation type the engine uses on the hot path".
// We start with FP16; FP8 is a follow-up phase that swaps this template.
using act_t = __half;

constexpr float FP16_LOWEST = -65504.0f;
constexpr float FP16_HIGHEST = 65504.0f;

// FP8 e4m3 dynamic range (S/E4/M3): max 448, min normal ~2e-3.
constexpr float E4M3_MAX = 448.0f;

}  // namespace cutlass_engine

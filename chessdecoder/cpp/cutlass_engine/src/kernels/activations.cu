#include "cutlass_engine/kernels.hpp"
#include "cutlass_engine/check.hpp"

#include <cuda_fp16.h>

namespace cutlass_engine {

// Activations live with kernels.hpp's swiglu/mish; this file is a placeholder
// so the build system has a target if we later add more (e.g. tanh, gelu).
// Currently empty — mish lives in swiglu.cu next to silu.

}  // namespace cutlass_engine

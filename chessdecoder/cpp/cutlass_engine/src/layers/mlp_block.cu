// MLP block forward implementation lives in attention_block.cu (kept together
// for one-translation-unit-per-layer-block compilation). This file is here so
// the build system's source list compiles; the symbol is defined elsewhere.
namespace cutlass_engine { /* see attention_block.cu */ }

// transformer_layer_forward is defined in attention_block.cu alongside the
// attention block, since it just composes attention + MLP. This file exists
// so the build system's source list compiles cleanly.
namespace cutlass_engine { /* see attention_block.cu */ }

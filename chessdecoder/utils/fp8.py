"""FP8 (Phase 1) — swap `nn.Linear` in the hot path for torchao Float8Linear.

Transformer-Engine isn't installable on this stack (Python 3.13 + torch 2.9 +
CUDA 12.8 has no prebuilt wheels and source builds run out of /tmp), so we use
torchao.float8 — a dynamic-scaling FP8 layer with the same forward/backward
signature as nn.Linear. Each Float8Linear casts its inputs/weights to
``e4m3fn`` per call (no persistent scale state -> resumable from any fp32/bf16
checkpoint with no extra buffers).

What gets converted: every nn.Linear whose `in_features` AND `out_features` are
multiples of 16 AND >= 256 — i.e. all of the V2 hot path
(attn q/k/v/output, SwiGLU gate/up/down, PerceiverPool q/k/v/o, fourier proj).
What stays in bf16: tiny per-head linears (transition sq/stm/cas heads, WDLHead
output 1024->405, policy_head 1024->1924 — the latter two have non-div-by-16
outputs anyway), nn.Embedding, and RMSNorm.

Pair with `torch.autocast("cuda", dtype=torch.bfloat16)` in the training forward
— no GradScaler in bf16/fp8 mode (bf16 has the same range as fp32).
"""
from __future__ import annotations

import torch.nn as nn


def _is_fp8_friendly(m: nn.Module, fqn: str) -> bool:
    """True iff this Linear's (in,out) both divide by 16 and >=256.

    The 16-divisibility is an FP8-tensorcore alignment requirement on Hopper.
    The >=256 floor excludes the small per-head output linears (which would
    spend more time on the cast/scale than on the GEMM).
    """
    if not isinstance(m, nn.Linear):
        return False
    if m.in_features % 16 or m.out_features % 16:
        return False
    return min(m.in_features, m.out_features) >= 256


def convert_model_to_fp8(model: nn.Module, recipe: str = "tensorwise") -> nn.Module:
    """In-place swap of hot-path Linears for Float8Linear.

    recipe: ``tensorwise`` (one scale per tensor — fastest), ``rowwise`` (per
    row, better numerics, ~5% slower), or ``rowwise_with_gw_hp`` (rowwise but
    high-precision grad-weight matmul, safer for long pretraining).
    Returns the same model object (mutated)."""
    from torchao.float8 import (Float8LinearConfig,
                                convert_to_float8_training)

    cfg = Float8LinearConfig.from_recipe_name(recipe)
    # Pad the GEMM's inner (K) dim to a multiple of 16 at runtime. This is the
    # safety net for the backward grad-weight matmul, where K = B*S — with a
    # dynamic batch B (e.g. partial last batch of a parquet shard), inductor
    # cannot prove K % 16 == 0 at compile time and the FP8 meta-check raises.
    cfg = type(cfg)(**{**cfg.__dict__, "pad_inner_dim": True})
    convert_to_float8_training(model, module_filter_fn=_is_fp8_friendly,
                               config=cfg)
    return model


def compile_fp8_hot_path(model) -> None:
    """``torch.compile`` the board_encoder + decoder in place.

    Without compile, eager Float8Linear costs 3 extra kernel launches per
    layer (cast input fp32->bf16, amax+cast bf16->fp8 for input/weight, then
    scaled_mm). bf16 autocast fuses those into one cublasLt call -> eager FP8
    is ~2× slower than bf16. Compiling the two big subgraphs (the encoder's 8
    layers + the decoder's 10 layers) lets inductor fuse cast+matmul +
    backward into ~1 kernel per Linear, turning the on-paper FP8 speedup into
    actual wall-clock. Compile *whole-model* with fullgraph=False doesn't help
    (torchtune control flow breaks the graph); per-submodule compile keeps the
    biggest contiguous segments compile-able while everything else stays eager.
    """
    import torch
    # dynamic=False: each recompile sees concrete shapes (no symbolic dims).
    # Dynamic shapes break FP8's K-divisibility-by-16 check on the backward
    # grad-weight matmul (K = B*S). With dynamic=False, inductor specializes
    # on each new shape — a few minutes of recompile cost up front, then
    # cached. Pair with a fixed batch size in the train loop.
    if hasattr(model, "board_encoder"):
        model.board_encoder = torch.compile(model.board_encoder,
                                            mode="default", dynamic=False)
    if hasattr(model, "decoder"):
        model.decoder = torch.compile(model.decoder, mode="default",
                                      dynamic=False)


def count_fp8_linears(model: nn.Module) -> tuple[int, int]:
    """(#fp8_linears, #total_linears) — for a sanity print after conversion."""
    from torchao.float8.float8_linear import Float8Linear
    n_fp8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    # Float8Linear subclasses nn.Linear so it counts in nn.Linear too —
    # subtract the FP8 count to get the *remaining* bf16/fp32 Linears.
    n_lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return n_fp8, n_lin - n_fp8

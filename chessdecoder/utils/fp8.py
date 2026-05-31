"""FP8 — swap ``nn.Linear`` in the hot path for torchao ``Float8Linear``.

Transformer-Engine isn't installable on this stack (Python 3.13 + torch 2.9 +
CUDA 12.8 has no prebuilt wheels and source builds run out of /tmp), so we use
torchao.float8 — a dynamic-scaling FP8 layer with the same forward/backward
signature as nn.Linear. Each Float8Linear casts its inputs/weights to
``e4m3fn`` per call (no persistent scale state -> resumable from any fp32/bf16
checkpoint with no extra buffers).

What gets converted: every ``nn.Linear`` whose ``in_features`` AND
``out_features`` are multiples of 16 AND >= 256 — i.e. all of the encoder's hot
path (attn q/k/v/output, SwiGLU gate/up/down). What stays in bf16: small
heads (``policy_head`` 1024->1924, ``wdl_head`` 1024->N_CELLS), nn.Embedding,
and RMSNorm.

Pair with ``torch.autocast("cuda", dtype=torch.bfloat16)`` in the training
forward — no GradScaler in bf16/fp8 mode (bf16 has the same range as fp32).
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
    """``torch.compile`` the encoder stack in place.

    Without compile, eager Float8Linear costs 3 extra kernel launches per layer
    (cast input fp32->bf16, amax+cast bf16->fp8 for input/weight, then
    scaled_mm). bf16 autocast fuses those into one cublasLt call -> eager FP8
    is ~2× slower than bf16. Compiling the encoder ``nn.Sequential`` lets
    inductor fuse cast+matmul + backward into ~1 kernel per Linear, turning
    the on-paper FP8 speedup into actual wall-clock.

    Compiles ``model.encoder`` (the encoder layer stack) — the embeddings,
    final norm, and per-head Linears stay eager.
    """
    import torch
    # dynamic=False: each recompile sees concrete shapes (no symbolic dims).
    # Dynamic shapes break FP8's K-divisibility-by-16 check on the backward
    # grad-weight matmul (K = B*S). With dynamic=False, inductor specializes
    # on each new shape — a few minutes of recompile cost up front, then
    # cached. Pair with a fixed batch size in the train loop.
    if hasattr(model, "encoder"):
        model.encoder = torch.compile(model.encoder, mode="default",
                                      dynamic=False)


def convert_moe_experts_to_fp8(model) -> int:
    """Wrap MoE expert weights (the 3-D gate/up/down params) with torchao's
    ``ScaledGroupedMMTensor`` so their ``torch._grouped_mm`` calls run in rowwise
    float8 — the MoE analogue of the Float8Linear swap for the dense path. The
    small bf16 router Linear is left alone. Returns the number of MoE layers
    converted (0 for a dense model)."""
    from chessdecoder.models.moe import MoEFeedForward
    from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor
    from torchao.prototype.moe_training.conversion_utils import MoEScalingType
    n = 0
    for m in model.modules():
        if isinstance(m, MoEFeedForward):
            for name in ("gate_w", "up_w", "down_w"):
                p = getattr(m, name)
                if not isinstance(p.data, ScaledGroupedMMTensor):
                    setattr(m, name, nn.Parameter(
                        ScaledGroupedMMTensor(p.data, MoEScalingType.FP8_ROWWISE),
                        requires_grad=p.requires_grad))
            n += 1
    return n


def count_fp8_linears(model: nn.Module) -> tuple[int, int]:
    """(#fp8_linears, #total_linears) — for a sanity print after conversion."""
    from torchao.float8.float8_linear import Float8Linear
    n_fp8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    # Float8Linear subclasses nn.Linear so it counts in nn.Linear too —
    # subtract the FP8 count to get the *remaining* bf16/fp32 Linears.
    n_lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return n_fp8, n_lin - n_fp8

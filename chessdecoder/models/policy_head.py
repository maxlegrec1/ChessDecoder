"""Policy-head variants for the chess encoder.

Two variants, selectable via ``model.policy_head``:

- ``linear`` (default): a single ``nn.Linear`` from the readout vector to the
  full move-sub-vocab. The original setup — fast, no spatial bias.
- ``cross_attn``: LC0-style attention policy. Per-square Q and K projections
  produce a pairwise ``[B, 64, 64]`` score table; the logit for each of the
  1924 move tokens is looked up as ``scores[b, from_sq, to_sq]`` plus a small
  learned per-promotion bias for the ``r/b/q`` underpromotion variants
  (knight is the no-suffix default in our vocab — promo_idx 0).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _file_id(c: str) -> int:
    return ord(c) - ord("a")


def _rank_id(c: str) -> int:
    return ord(c) - ord("1")


def _square_id(s: str) -> int:
    """``'a1'`` -> 0 ... ``'h8'`` -> 63, matching ``chess.SQUARES`` order
    (file-major within each rank, ranks bottom-to-top)."""
    return _rank_id(s[1]) * 8 + _file_id(s[0])


def _move_components(move: str):
    """``(from_sq, to_sq, promo_idx)`` for a ``policy_index`` move string.

    promo_idx: 0 = no suffix (knight under our vocab's convention, or
    non-promotion for non-promotion moves); 1 = q; 2 = r; 3 = b.
    """
    from_sq = _square_id(move[:2])
    to_sq = _square_id(move[2:4])
    if len(move) == 4:
        promo = 0
    else:
        promo = {"q": 1, "r": 2, "b": 3}[move[4]]
    return from_sq, to_sq, promo


class CrossAttnPolicyHead(nn.Module):
    """LC0-style attention policy from the 64 board squares.

    Input  : ``x_squares: [B, 64, d_model]`` — the encoder's output at the
             64 board-square positions.
    Output : ``[B, move_vocab_size]`` move logits.

    Implementation:
        Q = x @ q_proj.T      [B, 64, d_pol]
        K = x @ k_proj.T      [B, 64, d_pol]
        S = Q @ K.T / sqrt(d) [B, 64, 64]
        logit[b, m] = S[b, from[m], to[m]] + promo_bias[promo_idx[m]]
    """

    def __init__(self, d_model: int, move_vocab_size: int,
                 policy_index, d_pol: Optional[int] = None):
        super().__init__()
        if d_pol is None:
            d_pol = d_model
        self.d_pol = d_pol
        self.q_proj = nn.Linear(d_model, d_pol, bias=False)
        self.k_proj = nn.Linear(d_model, d_pol, bias=False)
        self.scale = 1.0 / math.sqrt(d_pol)
        # Pre-compute per-move (from, to, promo) lookup tensors.
        from_sq = torch.empty(move_vocab_size, dtype=torch.long)
        to_sq = torch.empty(move_vocab_size, dtype=torch.long)
        promo_idx = torch.empty(move_vocab_size, dtype=torch.long)
        for i, mv in enumerate(policy_index):
            f, t, p = _move_components(mv)
            from_sq[i] = f
            to_sq[i] = t
            promo_idx[i] = p
        self.register_buffer("from_sq", from_sq, persistent=False)
        self.register_buffer("to_sq", to_sq, persistent=False)
        self.register_buffer("promo_idx", promo_idx, persistent=False)
        # Underpromotion bias (knight/q/r/b). Init to zero so behaviour at
        # step 0 is content-only (the score table alone determines moves).
        self.promo_bias = nn.Parameter(torch.zeros(4))

    def forward(self, x_squares: torch.Tensor) -> torch.Tensor:
        # x_squares: [B, 64, d_model]
        q = self.q_proj(x_squares)                              # [B, 64, d_pol]
        k = self.k_proj(x_squares)                              # [B, 64, d_pol]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, 64, 64]
        # Gather per-move scores: [B, M] = scores[B, from_sq, to_sq].
        logits = scores[:, self.from_sq, self.to_sq]
        return logits + self.promo_bias[self.promo_idx]

"""Policy-head variants for the chess encoder.

Two variants, selectable via ``model.policy_head``:

- ``linear`` (default): a single ``nn.Linear`` from the readout vector to
  the move sub-vocab. Fast, no spatial structure.

- ``cross_attn``: LC0's attention-policy head. Per-square Q and K
  projections produce a pairwise ``[B, 64, 64]`` score table — the move
  logit for each of the 1924 (from, to)-indexed tokens is gathered as
  ``scores[from_sq, to_sq]``. Queen promotions and 4-char "non-pawn move"
  tokens share the same ``(from, to)`` slot (LC0 main-path convention; the
  model disambiguates from board context). Rook and bishop underpromotions
  get an additional *content-dependent* offset from a small
  ``Linear(d_pol, 2)`` over the K vectors of the promotion-target squares
  — so the model can say "in *this* board, promote to a rook here because
  of the discovered check".

  Differences from LC0:
    - LC0 has explicit knight underpromotion (``n`` suffix); our vocab
      omits it entirely (verified empirically — 50/50 sampled 4-char
      "promotion-pair" best_moves were actual *non-pawn moves* on
      promotion-rank squares, not knight underpromotions). So we have 2
      underpromotion pieces (r, b) instead of LC0's 3 (n, r, b).
    - LC0 flips the board to side-to-move, so it only handles rank-8
      promotions (8 target squares). We don't flip, so we handle both
      rank-1 (black) and rank-8 (white) promotions — 16 target squares.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# Promotion-target square indices in chess.SQUARES order (a1=0 .. h8=63):
# - rank 1 squares: 0..7   (black promotes here)
# - rank 8 squares: 56..63 (white promotes here)
_PROMO_SQUARES = list(range(0, 8)) + list(range(56, 64))
_PROMO_SQ_TO_IDX = {sq: i for i, sq in enumerate(_PROMO_SQUARES)}

# Underpromotion piece indices (queen is implicit in the main path; knight
# is absent from our vocab — see module docstring).
_UNDERPROMO_PIECE = {"r": 0, "b": 1}
_N_UNDERPROMO_PIECES = 2


def _file_id(c: str) -> int:
    return ord(c) - ord("a")


def _rank_id(c: str) -> int:
    return ord(c) - ord("1")


def _square_id(s: str) -> int:
    """``'a1'`` -> 0 ... ``'h8'`` -> 63, matching ``chess.SQUARES`` order
    (file-major within each rank, ranks bottom-to-top)."""
    return _rank_id(s[1]) * 8 + _file_id(s[0])


def _is_promo_capable(from_sq: int, to_sq: int) -> bool:
    """True iff ``(from_sq, to_sq)`` is a pawn promotion move (rank 7 -> 8
    or rank 2 -> 1)."""
    from_rank = from_sq // 8                      # 0..7
    to_rank = to_sq // 8
    return ((from_rank == 6 and to_rank == 7) or
            (from_rank == 1 and to_rank == 0))


def _classify_move(move: str):
    """Return ``(from_sq, to_sq, is_underpromo, promo_target_idx, piece_idx)``
    for one ``policy_index`` move.

    Only 5-char ``r`` and ``b`` suffix moves are underpromotions in our
    vocab. 4-char moves are always non-promotion (regular piece moves);
    5-char ``q`` moves share the ``(from, to)`` slot with the corresponding
    4-char non-pawn move (LC0 main-path convention). ``promo_target_idx``
    and ``piece_idx`` are 0 placeholders for non-underpromotion moves —
    the forward-pass gather is masked out via ``is_underpromo_mask``.
    """
    from_sq = _square_id(move[:2])
    to_sq = _square_id(move[2:4])
    suffix = move[4] if len(move) == 5 else ""
    if suffix in _UNDERPROMO_PIECE:
        return (from_sq, to_sq, True,
                _PROMO_SQ_TO_IDX[to_sq], _UNDERPROMO_PIECE[suffix])
    # Either 4-char (non-pawn move) or 5-char-"q" (queen promotion, shares
    # slot with the 4-char non-pawn move): both use base score only.
    return from_sq, to_sq, False, 0, 0


class CrossAttnPolicyHead(nn.Module):
    """LC0-style attention policy with content-dependent underpromotion offsets.

    Input  : ``x_squares: [B, 64, d_model]`` — encoder output for the 64
             board squares.
    Output : ``[B, move_vocab_size]`` move logits.

    Forward:
        q = Wq @ x_squares                                   [B, 64, d_pol]
        k = Wk @ x_squares                                   [B, 64, d_pol]
        scores = q @ k.T / sqrt(d_pol)                       [B, 64, 64]
        promo_k = k[:, promo_target_squares, :]              [B, 16, d_pol]
        promo_offsets = Wp @ promo_k                         [B, 16, 3]
        base[b, m] = scores[b, from[m], to[m]]               [B, M]
        offset[b, m] = promo_offsets[b, tgt[m], piece[m]] * is_underpromo[m]
        return base + offset
    """

    def __init__(self, d_model: int, move_vocab_size: int,
                 policy_index, d_pol: Optional[int] = None,
                 policy_embedding: bool = False):
        super().__init__()
        if d_pol is None:
            d_pol = d_model
        self.d_pol = d_pol
        # Optional BT4-style pre-projection of the encoder output before the
        # from/to attention map: policy_tokens = mish(policy_embedding(x)).
        # This is the only structural difference between BT4's policy head and
        # ours (+1.05M params).
        self.policy_embedding = (nn.Sequential(nn.Linear(d_model, d_model),
                                               nn.Mish())
                                 if policy_embedding else None)
        self.q_proj = nn.Linear(d_model, d_pol, bias=False)
        self.k_proj = nn.Linear(d_model, d_pol, bias=False)
        self.scale = 1.0 / math.sqrt(d_pol)

        # Per-promotion-target key -> underpromotion-piece offsets.
        self.promo_offset_proj = nn.Linear(d_pol, _N_UNDERPROMO_PIECES,
                                           bias=False)
        # Zero init: at step 0 the underpromotion offset is exactly 0, so
        # the policy matches the static-bias (or no-bias) baseline exactly
        # in its early behavior — avoids the first-step loss spike from a
        # default-Xavier promo projection randomly amplifying underpromo
        # logits before any gradient has flowed.
        nn.init.zeros_(self.promo_offset_proj.weight)

        # Pre-compute per-move metadata.
        from_sq = torch.empty(move_vocab_size, dtype=torch.long)
        to_sq = torch.empty(move_vocab_size, dtype=torch.long)
        is_underpromo = torch.zeros(move_vocab_size, dtype=torch.bool)
        promo_target_idx = torch.zeros(move_vocab_size, dtype=torch.long)
        piece_idx = torch.zeros(move_vocab_size, dtype=torch.long)
        for i, mv in enumerate(policy_index):
            f, t, up, tgt, p = _classify_move(mv)
            from_sq[i] = f
            to_sq[i] = t
            is_underpromo[i] = up
            promo_target_idx[i] = tgt
            piece_idx[i] = p
        self.register_buffer("from_sq", from_sq, persistent=False)
        self.register_buffer("to_sq", to_sq, persistent=False)
        # Float mask for vectorized application of the underpromo offset.
        self.register_buffer("is_underpromo_mask",
                             is_underpromo.float(), persistent=False)
        self.register_buffer("promo_target_idx", promo_target_idx, persistent=False)
        self.register_buffer("piece_idx", piece_idx, persistent=False)
        # 16 promotion-target square indices in chess.SQUARES order (rank 1
        # squares 0..7 for black, rank 8 squares 56..63 for white).
        self.register_buffer("promo_target_squares",
                             torch.tensor(_PROMO_SQUARES, dtype=torch.long),
                             persistent=False)

    def forward(self, x_squares: torch.Tensor) -> torch.Tensor:
        if self.policy_embedding is not None:
            x_squares = self.policy_embedding(x_squares)       # [B, 64, d_model]
        # Standard (from, to) attention scores.
        q = self.q_proj(x_squares)                             # [B, 64, d_pol]
        k = self.k_proj(x_squares)                             # [B, 64, d_pol]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, 64, 64]
        base = scores[:, self.from_sq, self.to_sq]             # [B, M]

        # Content-dependent underpromotion offsets — gathered from the K
        # vectors of the 16 promotion-target squares.
        promo_k = k[:, self.promo_target_squares, :]           # [B, 16, d_pol]
        promo_offsets = self.promo_offset_proj(promo_k)        # [B, 16, 3]
        # Gather per-move offsets: [B, M].
        underpromo = promo_offsets[:, self.promo_target_idx, self.piece_idx]
        return base + underpromo * self.is_underpromo_mask

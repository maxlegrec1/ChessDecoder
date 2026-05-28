"""Positional / attention-bias schemes for the chess encoder.

Five variants, selectable via ``model.attention_variant``:

- ``baseline``: learned absolute position embedding added before layer 0,
  plain bidirectional MHA. (No work for this module — handled in ChessEncoder.)
- ``rope1d``: torchtune's RotaryPositionalEmbeddings rotating Q/K by 1D
  position 0..seq_len-1. No token-level pos embedding.
- ``rope2d``: split head_dim in two — first half rotates by ``rank``, second by
  ``file``. Board squares get (rank+1, file+1) in [1,8]^2; specials get unique
  out-of-board coords so they still have distinct rotations.
- ``relpos2d``: ALiBi/T5-style learned scalar bias per (Δrank, Δfile) bucket
  added to the attention logits. 15x15 board-square buckets + N_KIND x N_KIND
  buckets for special-involving pairs. Bias is shared across layers.
- ``geom``: like ``relpos2d`` but the square-square bias is produced by a
  small MLP over a feature vector — Δrank, Δfile, chebyshev distance, knight
  flag, diagonal flag, same-rank/same-file flags — so the model can express
  the geometry chess pieces actually use. Specials still use the per-kind table.

Notes
-----
- Bias modules return ``[1, num_heads, seq_len, seq_len]`` and are broadcast
  into SDPA's ``attn_mask`` (float-mask path — added to logits before softmax).
- Token layout is described by a ``TokenLayout`` so the same bias machinery
  works for the 68-token default, the 67-token (no end_pos), 66-token (no CLS
  no end_pos), and 64-token (squares only) variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings

# Token-kind codes used to bucket special-involving pairs.
KIND_SQUARE, KIND_CLS, KIND_END, KIND_CASTLING, KIND_STM = 0, 1, 2, 3, 4
N_KINDS = 5


@dataclass(frozen=True)
class TokenLayout:
    """Describes which sequence positions are board squares vs. specials,
    so the geometric bias machinery can work for any input mode.

    ``square_positions[i]`` is the chess-square index (0..63 in
    ``chess.SQUARES`` order: a1..h8) that lives at sequence position
    ``squares[i]``. ``cls_pos``/``end_pos``/``castling_pos``/``stm_pos``
    are the sequence positions of those special tokens, or ``None`` if
    that special doesn't appear in this layout.
    """
    seq_len: int
    squares: Tuple[int, ...]         # sequence positions of the 64 squares
    cls_pos: Optional[int]
    end_pos: Optional[int]
    castling_pos: Optional[int]
    stm_pos: Optional[int]

    def kind_at(self, pos: int) -> int:
        if pos == self.cls_pos: return KIND_CLS
        if pos == self.end_pos: return KIND_END
        if pos == self.castling_pos: return KIND_CASTLING
        if pos == self.stm_pos: return KIND_STM
        return KIND_SQUARE

    def square_id(self, pos: int) -> int:
        """The chess.SQUARES index (0..63) for sequence position ``pos``."""
        return self._sq_lookup[pos]

    def __post_init__(self):
        # Build a reverse lookup: pos -> chess-square index.
        # (object.__setattr__ because the dataclass is frozen.)
        lookup = {sp: i for i, sp in enumerate(self.squares)}
        object.__setattr__(self, "_sq_lookup", lookup)


# Canonical layouts for each input mode.
LAYOUT_68 = TokenLayout(
    seq_len=68,
    squares=tuple(range(1, 65)),
    cls_pos=0, end_pos=65, castling_pos=66, stm_pos=67)
LAYOUT_67 = TokenLayout(   # drop end_pos
    seq_len=67,
    squares=tuple(range(1, 65)),
    cls_pos=0, end_pos=None, castling_pos=65, stm_pos=66)
LAYOUT_66 = TokenLayout(   # drop CLS + end_pos
    seq_len=66,
    squares=tuple(range(0, 64)),
    cls_pos=None, end_pos=None, castling_pos=64, stm_pos=65)
LAYOUT_64 = TokenLayout(   # squares only (stm/castling as broadcast embeds)
    seq_len=64,
    squares=tuple(range(0, 64)),
    cls_pos=None, end_pos=None, castling_pos=None, stm_pos=None)

INPUT_MODE_TO_LAYOUT = {
    "default": LAYOUT_68,
    "no_end": LAYOUT_67,
    "no_cls_no_end": LAYOUT_66,
    "lc0_64": LAYOUT_64,
}


def _square_rank_file(sq_id: int) -> tuple[int, int]:
    """For a chess-square index (0..63), return ``(rank, file)`` in
    ``chess.SQUARES`` order (file-major within rank)."""
    return sq_id // 8, sq_id % 8


def _build_coords(layout: TokenLayout) -> torch.Tensor:
    """``[seq_len, 2]`` long tensor: ``(rank, file)`` for each token.

    Board squares live in ``[1, 8]^2`` so they don't collide with the
    specials, which get unique out-of-board ``(rank, file)`` pairs.
    """
    coords = torch.zeros(layout.seq_len, 2, dtype=torch.long)
    # Specials at rank 0 (CLS) or rank 10 (end/castling/stm) on distinct files.
    special_file_counter = 0
    for pos, file_offset in zip(
            (layout.cls_pos, layout.end_pos,
             layout.castling_pos, layout.stm_pos),
            (0, 0, 1, 2)):
        if pos is None:
            continue
        rank = 0 if pos == layout.cls_pos else 10
        coords[pos] = torch.tensor([rank, file_offset])
    for pos in layout.squares:
        r, f = _square_rank_file(layout.square_id(pos))
        coords[pos] = torch.tensor([r + 1, f + 1])
    return coords


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE half-rotation along the last dim: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------------------
# 2D RoPE
# ---------------------------------------------------------------------------

class Rope2D(nn.Module):
    """2D RoPE — first half of head_dim rotates by rank, second by file."""

    def __init__(self, head_dim: int, layout: TokenLayout,
                 base: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim={head_dim} must be divisible by 4 "
                             "(splits into two halves, each rotates pairwise)")
        half = head_dim // 2
        freqs = (1.0 / (base ** (torch.arange(0, half, 2).float() / half)))
        coords = _build_coords(layout)
        ranks = coords[:, 0].float()
        files = coords[:, 1].float()
        ang_r = ranks[:, None] * freqs[None, :]
        ang_f = files[:, None] * freqs[None, :]
        ang_r = torch.cat([ang_r, ang_r], dim=-1)
        ang_f = torch.cat([ang_f, ang_f], dim=-1)
        self.register_buffer("cos_r", ang_r.cos(), persistent=False)
        self.register_buffer("sin_r", ang_r.sin(), persistent=False)
        self.register_buffer("cos_f", ang_f.cos(), persistent=False)
        self.register_buffer("sin_f", ang_f.sin(), persistent=False)

    def forward(self, x: torch.Tensor, *,
                input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        half = x.shape[-1] // 2
        x_r, x_f = x[..., :half], x[..., half:]
        cos_r = self.cos_r.to(x.dtype)[None, :, None, :]
        sin_r = self.sin_r.to(x.dtype)[None, :, None, :]
        cos_f = self.cos_f.to(x.dtype)[None, :, None, :]
        sin_f = self.sin_f.to(x.dtype)[None, :, None, :]
        out_r = x_r * cos_r + _rotate_half(x_r) * sin_r
        out_f = x_f * cos_f + _rotate_half(x_f) * sin_f
        return torch.cat([out_r, out_f], dim=-1)


# ---------------------------------------------------------------------------
# Pair-bucketing helpers for relpos / geom
# ---------------------------------------------------------------------------

def _build_pair_buckets(layout: TokenLayout):
    """For every ``(i, j) ∈ [0, seq_len)^2``:

    - ``is_sq[i, j]``: whether both ends are board squares (geometric path).
    - ``sq_bucket[i, j]``: ``(Δrank + 7, Δfile + 7)`` flat index in [0, 224]
      for square-square pairs (Δ = ``square_i`` rank/file - ``square_j``).
    - ``sp_bucket[i, j]``: per-kind-pair flat index in ``[0, N_KINDS²)``
      for pairs involving at least one special.
    """
    S = layout.seq_len
    is_sq = torch.zeros(S, S, dtype=torch.bool)
    sq_bucket = torch.zeros(S, S, dtype=torch.long)
    sp_bucket = torch.zeros(S, S, dtype=torch.long)
    for i in range(S):
        ki = layout.kind_at(i)
        for j in range(S):
            kj = layout.kind_at(j)
            if ki == KIND_SQUARE and kj == KIND_SQUARE:
                ri, fi = _square_rank_file(layout.square_id(i))
                rj, fj = _square_rank_file(layout.square_id(j))
                dr, df = ri - rj, fi - fj
                is_sq[i, j] = True
                sq_bucket[i, j] = (dr + 7) * 15 + (df + 7)
            else:
                sp_bucket[i, j] = ki * N_KINDS + kj
    return is_sq, sq_bucket, sp_bucket


# ---------------------------------------------------------------------------
# Relative-position bias (T5/ALiBi-style)
# ---------------------------------------------------------------------------

class RelPos2DBias(nn.Module):
    """Learned scalar bias per (Δrank, Δfile) bucket, added to attention
    logits. Shared across layers. Returns ``[1, num_heads, seq_len, seq_len]``."""

    N_SQ_BUCKETS = 15 * 15
    N_SP_BUCKETS = N_KINDS * N_KINDS

    def __init__(self, num_heads: int, layout: TokenLayout,
                 init_std: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
        self.sq_bias = nn.Parameter(
            torch.zeros(num_heads, self.N_SQ_BUCKETS))
        self.sp_bias = nn.Parameter(
            torch.zeros(num_heads, self.N_SP_BUCKETS))
        nn.init.normal_(self.sq_bias, std=init_std)
        nn.init.normal_(self.sp_bias, std=init_std)
        is_sq, sq_bucket, sp_bucket = _build_pair_buckets(layout)
        self.register_buffer("is_sq", is_sq, persistent=False)
        self.register_buffer("sq_bucket", sq_bucket, persistent=False)
        self.register_buffer("sp_bucket", sp_bucket, persistent=False)

    def forward(self) -> torch.Tensor:
        b_sq = self.sq_bias[:, self.sq_bucket]
        b_sp = self.sp_bias[:, self.sp_bucket]
        return torch.where(self.is_sq[None], b_sq, b_sp).unsqueeze(0)


# ---------------------------------------------------------------------------
# Geometric attention bias
# ---------------------------------------------------------------------------

class GeomAttnBias(nn.Module):
    """Square-square bias from a small MLP over chess-specific geometric
    features; special-involving pairs use the per-kind-pair table.

    Features per ordered pair (i, j) of board squares (7-dim, [-1, 1]-ish):
      0: Δrank / 7            (signed, normalized)
      1: Δfile / 7            (signed, normalized)
      2: chebyshev dist / 7   (max-norm)
      3: knight-flag          ({|Δr|, |Δf|} == {1, 2})
      4: diagonal flag        (|Δr| == |Δf|, |Δr| > 0)
      5: same-rank flag       (Δr == 0)
      6: same-file flag       (Δf == 0)
    """

    N_FEATURES = 7
    N_SP_BUCKETS = N_KINDS * N_KINDS

    def __init__(self, num_heads: int, layout: TokenLayout,
                 hidden: int = 64, init_std: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = layout.seq_len
        self.mlp = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, num_heads, bias=True),
        )
        self.sp_bias = nn.Parameter(torch.zeros(num_heads, self.N_SP_BUCKETS))
        nn.init.normal_(self.sp_bias, std=init_std)
        # Bias-zero init on the last MLP layer so the model starts close to
        # the relpos-zero behaviour (the MLP doesn't slam attention right
        # out of the gate before the geometric features mean anything).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        features, is_sq, sp_bucket = self._build_features(layout)
        self.register_buffer("features", features, persistent=False)
        self.register_buffer("is_sq", is_sq, persistent=False)
        self.register_buffer("sp_bucket", sp_bucket, persistent=False)

    @staticmethod
    def _build_features(layout: TokenLayout):
        is_sq, _, sp_bucket = _build_pair_buckets(layout)
        S = layout.seq_len
        feats = torch.zeros(S, S, GeomAttnBias.N_FEATURES, dtype=torch.float32)
        for i in range(S):
            for j in range(S):
                if not is_sq[i, j]:
                    continue
                ri, fi = _square_rank_file(layout.square_id(i))
                rj, fj = _square_rank_file(layout.square_id(j))
                dr, df = ri - rj, fi - fj
                cheby = max(abs(dr), abs(df))
                knight = 1.0 if {abs(dr), abs(df)} == {1, 2} else 0.0
                diag = 1.0 if abs(dr) == abs(df) and (dr != 0) else 0.0
                same_r = 1.0 if dr == 0 else 0.0
                same_f = 1.0 if df == 0 else 0.0
                feats[i, j, 0] = dr / 7.0
                feats[i, j, 1] = df / 7.0
                feats[i, j, 2] = cheby / 7.0
                feats[i, j, 3] = knight
                feats[i, j, 4] = diag
                feats[i, j, 5] = same_r
                feats[i, j, 6] = same_f
        return feats, is_sq, sp_bucket

    def forward(self) -> torch.Tensor:
        flat = self.features.view(-1, self.N_FEATURES)
        b_mlp = self.mlp(flat).view(self.seq_len, self.seq_len, -1).permute(2, 0, 1)
        b_sp = self.sp_bias[:, self.sp_bucket]
        return torch.where(self.is_sq[None], b_mlp, b_sp).unsqueeze(0)


# ---------------------------------------------------------------------------
# Smolgen — content-conditional attention bias (Leela BT style)
# ---------------------------------------------------------------------------

class Smolgen(nn.Module):
    """Per-layer Smolgen attention bias generator (Leela BT style).

    Currently hardcoded to the 68-token layout (slices ``x[:, 1:65]`` for
    the 64 board squares). For other input modes, set attention_variant to
    something other than 'smolgen'. (The refactor for arbitrary layouts is
    straightforward but unused for now.)
    """

    def __init__(self, d_model: int, num_heads: int, d1: int = 32,
                 d_hidden: int = 256, gen_size: int = 256,
                 init_std: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
        self.gen_size = gen_size
        self.sm1 = nn.Linear(d_model, d1, bias=False)
        self.sm2 = nn.Linear(64 * d1, d_hidden, bias=False)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.sm3 = nn.Linear(d_hidden, num_heads * gen_size, bias=False)
        self.ln2 = nn.LayerNorm(num_heads * gen_size)
        self.pos_enc_weight = nn.Parameter(
            torch.empty(64 * 64, gen_size))
        nn.init.normal_(self.pos_enc_weight, std=init_std)
        self.sp_bias = nn.Parameter(torch.zeros(num_heads, N_KINDS * N_KINDS))
        nn.init.normal_(self.sp_bias, std=init_std)
        is_sq, _, sp_bucket = _build_pair_buckets(LAYOUT_68)
        self.register_buffer("is_sq", is_sq, persistent=False)
        self.register_buffer("sp_bucket", sp_bucket, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        squares = x[:, 1:65]
        y = self.sm1(squares)
        y = y.reshape(B, -1)
        y = torch.nn.functional.gelu(self.sm2(y))
        y = self.ln1(y)
        y = torch.nn.functional.gelu(self.sm3(y))
        y = self.ln2(y).view(B, self.num_heads, self.gen_size)
        b_sq = torch.einsum("bhi,oi->bho", y, self.pos_enc_weight)
        b_sq = b_sq.view(B, self.num_heads, 64, 64)
        b_sq_padded = torch.nn.functional.pad(b_sq, (1, 3, 1, 3))
        sp_full = self.sp_bias[:, self.sp_bucket]
        return torch.where(self.is_sq[None, None],
                           b_sq_padded,
                           sp_full.unsqueeze(0))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pos_modules(variant: str, embed_dim: int, num_heads: int,
                      layout: TokenLayout):
    """Return ``(pos_embeddings, bias_module)``.

    ``pos_embeddings`` is the per-layer module passed to MHA (rotates Q/K).
    ``bias_module`` is a parameter-bearing module whose ``__call__()`` returns
    ``[1, num_heads, seq_len, seq_len]`` added to attention logits each forward.
    Either may be ``None``. The ``smolgen`` variant doesn't use either —
    per-layer Smolgen instances live inside each ``EncoderLayer`` instead.
    """
    head_dim = embed_dim // num_heads
    if variant in ("baseline", "smolgen"):
        return None, None
    if variant == "rope1d":
        return RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=layout.seq_len), None
    if variant == "rope2d":
        return Rope2D(head_dim=head_dim, layout=layout), None
    if variant == "relpos2d":
        return None, RelPos2DBias(num_heads=num_heads, layout=layout)
    if variant == "geom":
        return None, GeomAttnBias(num_heads=num_heads, layout=layout)
    raise ValueError(f"unknown attention_variant {variant!r} "
                     f"(expected baseline/rope1d/rope2d/relpos2d/geom/smolgen)")

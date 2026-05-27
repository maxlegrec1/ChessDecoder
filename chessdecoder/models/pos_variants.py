"""Positional / attention-bias schemes for the chess encoder.

Five variants, selectable via ``model.attention_variant``:

- ``baseline``: learned absolute position embedding added before layer 0,
  plain bidirectional MHA. (No work for this module — handled in ChessEncoder.)
- ``rope1d``: torchtune's RotaryPositionalEmbeddings rotating Q/K by 1D
  position 0..67. No token-level pos embedding.
- ``rope2d``: split head_dim in two — first half rotates by ``rank``, second by
  ``file``. Board squares get (rank+1, file+1) in [1,8]^2; specials get unique
  out-of-board coords so they still have distinct rotations.
- ``relpos2d``: ALiBi/T5-style learned scalar bias per (Δrank, Δfile) bucket
  added to the attention logits. 15x15 board-square buckets + 5x5 token-kind
  buckets for special-involving pairs. Bias is shared across layers.
- ``geom``: like ``relpos2d`` but the square-square bias is produced by a
  small MLP over a feature vector — Δrank, Δfile, chebyshev distance, knight
  flag, diagonal flag, same-rank/same-file flags — so the model can express
  the geometry chess pieces actually use. Specials still use the 5x5 table.

Notes
-----
- Bias modules return ``[num_heads, seq_len, seq_len]`` and are broadcast into
  SDPA's ``attn_mask`` (float-mask path — added to logits before softmax).
- The bias modules are computed *outside* the FP8-compiled encoder stack
  each forward, so the param gradients flow normally without fighting FP8's
  K-divisibility checks.
- Token layout (0..67): ``0=CLS``, ``1..64=squares (a1..h8 in chess.SQUARES
  order)``, ``65=end_pos``, ``66=castling``, ``67=stm``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings

SEQ_LEN = 68
# token-kind ids used to bucket special-involving pairs
KIND_CLS, KIND_SQUARE, KIND_END, KIND_CASTLING, KIND_STM = 0, 1, 2, 3, 4
N_KINDS = 5


def _token_kind(i: int) -> int:
    if i == 0: return KIND_CLS
    if i == 65: return KIND_END
    if i == 66: return KIND_CASTLING
    if i == 67: return KIND_STM
    return KIND_SQUARE


def _square_rank_file(i: int) -> tuple[int, int]:
    """For i in 1..64, return (rank, file) ∈ [0,7]^2 in chess.SQUARES order."""
    assert 1 <= i <= 64
    return (i - 1) // 8, (i - 1) % 8


def _build_coords() -> torch.Tensor:
    """[seq_len, 2] long tensor: (rank, file) for each token.

    Board squares live in [1, 8]^2 so they don't collide with the specials,
    which occupy rank-0 (CLS) and rank-10 (end/castling/stm).
    """
    coords = torch.zeros(SEQ_LEN, 2, dtype=torch.long)
    coords[0] = torch.tensor([0, 0])               # CLS
    for i in range(1, 65):
        r, f = _square_rank_file(i)
        coords[i] = torch.tensor([r + 1, f + 1])    # in [1, 8]^2
    coords[65] = torch.tensor([10, 0])             # end_pos
    coords[66] = torch.tensor([10, 1])             # castling
    coords[67] = torch.tensor([10, 2])             # stm
    return coords


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE half-rotation along the last dim: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------------------
# 2D RoPE
# ---------------------------------------------------------------------------

class Rope2D(nn.Module):
    """2D RoPE — first half of head_dim rotates by rank, second by file.

    Compatible with ``torchtune.modules.MultiHeadAttention``'s
    ``pos_embeddings`` slot: ``forward(x, *, input_pos=None)`` where
    ``x`` is ``[b, s, n_h, h_d]``.
    """

    def __init__(self, head_dim: int, max_seq_len: int = SEQ_LEN,
                 base: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim={head_dim} must be divisible by 4 "
                             "(splits into two halves, each rotates pairwise)")
        half = head_dim // 2
        # Per-half inverse frequencies — standard RoPE schedule on dim=half.
        freqs = (1.0 / (base ** (torch.arange(0, half, 2).float() / half)))
        coords = _build_coords()
        ranks = coords[:, 0].float()                          # [seq_len]
        files = coords[:, 1].float()
        # angle[i, k] = coord[i] * freqs[k]  ->  [seq_len, half/2]
        ang_r = ranks[:, None] * freqs[None, :]
        ang_f = files[:, None] * freqs[None, :]
        # Duplicate to get the full half-dim: [seq_len, half]
        ang_r = torch.cat([ang_r, ang_r], dim=-1)
        ang_f = torch.cat([ang_f, ang_f], dim=-1)
        self.register_buffer("cos_r", ang_r.cos(), persistent=False)
        self.register_buffer("sin_r", ang_r.sin(), persistent=False)
        self.register_buffer("cos_f", ang_f.cos(), persistent=False)
        self.register_buffer("sin_f", ang_f.sin(), persistent=False)

    def forward(self, x: torch.Tensor, *,
                input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [b, s, n_h, h_d]
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

def _build_pair_buckets():
    """Precompute, for every (i, j) ∈ [0,seq_len)^2:
      - whether the pair is square-square (geometric path)
      - the (Δrank+7, Δfile+7) bucket for square-square pairs (15x15 = 225)
      - the (kind_i, kind_j) bucket for special-involving pairs (5x5 = 25)
    """
    is_sq = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.bool)
    sq_bucket = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.long)
    sp_bucket = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.long)
    for i in range(SEQ_LEN):
        ki = _token_kind(i)
        for j in range(SEQ_LEN):
            kj = _token_kind(j)
            if ki == KIND_SQUARE and kj == KIND_SQUARE:
                ri, fi = _square_rank_file(i)
                rj, fj = _square_rank_file(j)
                dr = ri - rj                                  # in [-7, 7]
                df = fi - fj                                  # in [-7, 7]
                is_sq[i, j] = True
                sq_bucket[i, j] = (dr + 7) * 15 + (df + 7)    # in [0, 224]
            else:
                sp_bucket[i, j] = ki * N_KINDS + kj            # in [0, 24]
    return is_sq, sq_bucket, sp_bucket


# ---------------------------------------------------------------------------
# Relative-position bias (T5/ALiBi-style)
# ---------------------------------------------------------------------------

class RelPos2DBias(nn.Module):
    """Learned scalar bias per (Δrank, Δfile) bucket, added to attention
    logits. Shared across layers. Returns ``[num_heads, seq_len, seq_len]``."""

    N_SQ_BUCKETS = 15 * 15                                    # 225
    N_SP_BUCKETS = N_KINDS * N_KINDS                          # 25

    def __init__(self, num_heads: int, init_std: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
        self.sq_bias = nn.Parameter(
            torch.zeros(num_heads, self.N_SQ_BUCKETS))
        self.sp_bias = nn.Parameter(
            torch.zeros(num_heads, self.N_SP_BUCKETS))
        nn.init.normal_(self.sq_bias, std=init_std)
        nn.init.normal_(self.sp_bias, std=init_std)
        is_sq, sq_bucket, sp_bucket = _build_pair_buckets()
        self.register_buffer("is_sq", is_sq, persistent=False)
        self.register_buffer("sq_bucket", sq_bucket, persistent=False)
        self.register_buffer("sp_bucket", sp_bucket, persistent=False)

    def forward(self) -> torch.Tensor:
        # Gather per-pair bias from both tables, blend by is_sq.
        # Returns [1, h, s, s] — explicit B=1 so SDPA / dynamo see it as
        # (batch-broadcast, heads, L, L) and don't misread [h, s, s] as
        # [B=h, H=1, s, s].
        b_sq = self.sq_bias[:, self.sq_bucket]                # [h, s, s]
        b_sp = self.sp_bias[:, self.sp_bucket]                # [h, s, s]
        return torch.where(self.is_sq[None], b_sq, b_sp).unsqueeze(0)


# ---------------------------------------------------------------------------
# Geometric attention bias
# ---------------------------------------------------------------------------

class GeomAttnBias(nn.Module):
    """Square-square bias from a small MLP over chess-specific geometric
    features; special-involving pairs use the same 5x5 table as relpos2d.

    Features per ordered pair (i, j) of board squares (7-dim, [-1, 1]-ish):
      0: Δrank / 7            (signed, normalized)
      1: Δfile / 7            (signed, normalized)
      2: chebyshev dist / 7   (max-norm)
      3: knight-flag          ({|Δr|, |Δf|} == {1, 2})
      4: diagonal flag        (|Δr| == |Δf|)
      5: same-rank flag       (Δr == 0)
      6: same-file flag       (Δf == 0)
    """

    N_FEATURES = 7
    N_SP_BUCKETS = N_KINDS * N_KINDS

    def __init__(self, num_heads: int, hidden: int = 64, init_std: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
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

        features, is_sq, sp_bucket = self._build_features()
        self.register_buffer("features", features, persistent=False)
        self.register_buffer("is_sq", is_sq, persistent=False)
        self.register_buffer("sp_bucket", sp_bucket, persistent=False)

    @staticmethod
    def _build_features():
        is_sq, _, sp_bucket = _build_pair_buckets()
        feats = torch.zeros(SEQ_LEN, SEQ_LEN, GeomAttnBias.N_FEATURES,
                            dtype=torch.float32)
        for i in range(SEQ_LEN):
            for j in range(SEQ_LEN):
                if not is_sq[i, j]:
                    continue
                ri, fi = _square_rank_file(i)
                rj, fj = _square_rank_file(j)
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
        # MLP everywhere (cheap: 68*68 = 4624 forward passes through a
        # tiny network) then overwrite special-involving pairs with the
        # learned 5x5 table. Returns [1, h, s, s] (see RelPos2DBias).
        flat = self.features.view(-1, self.N_FEATURES)
        b_mlp = self.mlp(flat).view(SEQ_LEN, SEQ_LEN, -1).permute(2, 0, 1)
        b_sp = self.sp_bias[:, self.sp_bucket]                # [h, s, s]
        return torch.where(self.is_sq[None], b_mlp, b_sp).unsqueeze(0)


# ---------------------------------------------------------------------------
# Smolgen — content-conditional attention bias (Leela BT style)
# ---------------------------------------------------------------------------

class Smolgen(nn.Module):
    """Per-layer Smolgen attention bias generator (Leela BT style).

    Takes ``x`` of shape ``(B, 68, d_model)`` — the *current layer's* input
    activations — and emits a ``(B, num_heads, 68, 68)`` attention bias.
    The 64x64 board-square sub-bias is generated *content-conditionally*
    from the input via a small per-square -> pooled-board -> per-head MLP,
    then "decoded" into 4096 pair logits through a shared
    ``pos_enc_weight: (64*64, gen_size)``. Special-involving pairs (CLS,
    end_pos, castling, stm) use a learned 5x5 table per head — same idea
    as RelPos2DBias's special path.

    The motivation vs ``geom`` / ``relpos2d``: those biases are
    *static* (depend only on position). Smolgen lets the bias depend on
    what's *on* the board right now — so the model can express patterns
    like "this diagonal is hot for this position".

    Param budget at defaults (d1=32, d_hidden=256, gen_size=256, H=8,
    d_model=512): ~2M params per layer; per 6-layer model ~12M.
    """

    def __init__(self, d_model: int, num_heads: int, d1: int = 32,
                 d_hidden: int = 256, gen_size: int = 256,
                 init_std: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
        self.gen_size = gen_size
        # Per-square content compression (small Linear, stays bf16 under
        # the FP8 filter since d1 < 256).
        self.sm1 = nn.Linear(d_model, d1, bias=False)
        # Board summary -> per-head intent. Both linears are big enough to
        # hit the FP8 conversion path.
        self.sm2 = nn.Linear(64 * d1, d_hidden, bias=False)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.sm3 = nn.Linear(d_hidden, num_heads * gen_size, bias=False)
        self.ln2 = nn.LayerNorm(num_heads * gen_size)
        # Shared "decoder" — turns a (H, gen_size) intent into (H, 64*64)
        # board-square bias logits.
        self.pos_enc_weight = nn.Parameter(
            torch.empty(64 * 64, gen_size))
        nn.init.normal_(self.pos_enc_weight, std=init_std)
        # Per-head learned 5x5 special-token bias table.
        self.sp_bias = nn.Parameter(torch.zeros(num_heads, N_KINDS * N_KINDS))
        nn.init.normal_(self.sp_bias, std=init_std)
        _, _, sp_bucket = _build_pair_buckets()
        is_sq, _, _ = _build_pair_buckets()
        self.register_buffer("is_sq", is_sq, persistent=False)
        self.register_buffer("sp_bucket", sp_bucket, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 68, d_model)
        B = x.shape[0]
        squares = x[:, 1:65]                                      # (B, 64, d_model)
        y = self.sm1(squares)                                     # (B, 64, d1)
        y = y.reshape(B, -1)                                      # (B, 64*d1)
        y = torch.nn.functional.gelu(self.sm2(y))                 # (B, d_hidden)
        y = self.ln1(y)
        y = torch.nn.functional.gelu(self.sm3(y))                 # (B, H*gen)
        y = self.ln2(y).view(B, self.num_heads, self.gen_size)    # (B, H, gen)
        # einsum: "bhi,oi->bho" — same decoder used across heads.
        b_sq = torch.einsum("bhi,oi->bho", y, self.pos_enc_weight)
        b_sq = b_sq.view(B, self.num_heads, 64, 64)
        # Pad the 64x64 board-square bias to 68x68 (zeros in the special
        # rows/cols), then ``where`` selects the special-pair bias for
        # those slots — avoids in-place assignment under torch.compile.
        b_sq_padded = torch.nn.functional.pad(b_sq, (1, 3, 1, 3))     # (B, H, 68, 68)
        sp_full = self.sp_bias[:, self.sp_bucket]                     # (H, 68, 68)
        return torch.where(self.is_sq[None, None],
                           b_sq_padded,
                           sp_full.unsqueeze(0))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pos_modules(variant: str, embed_dim: int, num_heads: int,
                      seq_len: int = SEQ_LEN):
    """Return ``(pos_embeddings, bias_module)``.

    ``pos_embeddings`` is the per-layer module passed to MHA (rotates Q/K).
    ``bias_module`` is a parameter-bearing module whose ``__call__()`` returns
    ``[num_heads, seq_len, seq_len]`` added to attention logits each forward.
    Either may be ``None``. The ``smolgen`` variant doesn't use either —
    per-layer Smolgen instances live inside each ``EncoderLayer`` instead.
    """
    head_dim = embed_dim // num_heads
    if variant in ("baseline", "smolgen"):
        return None, None
    if variant == "rope1d":
        return RotaryPositionalEmbeddings(dim=head_dim,
                                          max_seq_len=seq_len), None
    if variant == "rope2d":
        return Rope2D(head_dim=head_dim, max_seq_len=seq_len), None
    if variant == "relpos2d":
        return None, RelPos2DBias(num_heads=num_heads)
    if variant == "geom":
        return None, GeomAttnBias(num_heads=num_heads)
    raise ValueError(f"unknown attention_variant {variant!r} "
                     f"(expected baseline/rope1d/rope2d/relpos2d/geom/smolgen)")

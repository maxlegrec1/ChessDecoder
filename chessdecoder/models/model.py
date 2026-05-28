"""ChessEncoder — transformer encoder over the board tokens.

Two input modes selectable via ``input_mode``:

  lc0_64  64 tokens: just the board squares; side-to-move and castling
          rights are added as broadcast embeddings to every square. Value
          via mean-pool (no CLS slot).
  cls_65  65 tokens: a dedicated CLS slot + 64 squares. stm/castling are
          still broadcast embeddings (added to every token, CLS included).
          Value reads off the CLS slot.

Both work with any ``attention_variant`` (baseline / rope1d / rope2d /
relpos2d / geom). Policy head selectable via ``policy_head``: ``linear``
(single Linear from the readout) or ``cross_attn`` (LC0-style attention
from per-square Q/K — works in both input modes since it always slices
out the 64 square positions from the encoder output).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchtune.modules import RMSNorm

from chessdecoder.models.layers import EncoderLayer, EncoderStack
from chessdecoder.models.policy_head import CrossAttnPolicyHead
from chessdecoder.models.pos_variants import (
    INPUT_MODE_TO_LAYOUT, TokenLayout, build_pos_modules)
from chessdecoder.models.value_buckets import CELL_WDL, N_CELLS, mean_wdl as _mean_wdl
from chessdecoder.models.vocab import move_vocab_size, policy_index


# Column indices in the raw cached 68-token board_ids:
# 0 = CLS, 1..64 = squares (chess.SQUARES order), 65 = end_pos,
# 66 = castling, 67 = stm.
_RAW_CLS_POS = 0
_RAW_SQ_START, _RAW_SQ_END = 1, 65
_RAW_CASTLING_POS = 66
_RAW_STM_POS = 67


def _slice_inputs(board_ids: torch.Tensor, input_mode: str):
    """Carve the raw cached ``[B, 68]`` ids into the per-input-mode inputs.

    Returns ``(model_board_ids, stm_id, castling_id)``. stm/castling are
    always pulled out separately for broadcast at embed time.
    """
    stm_id = board_ids[:, _RAW_STM_POS]
    castling_id = board_ids[:, _RAW_CASTLING_POS]
    if input_mode == "lc0_64":
        return board_ids[:, _RAW_SQ_START:_RAW_SQ_END], stm_id, castling_id
    if input_mode == "cls_65":
        # CLS (col 0) + 64 squares (cols 1..64).
        return board_ids[:, _RAW_CLS_POS:_RAW_SQ_END], stm_id, castling_id
    raise ValueError(f"unknown input_mode {input_mode!r} "
                     "(expected 'lc0_64' or 'cls_65')")


class ChessEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 1024,
                 num_heads: int = 16, num_layers: int = 12,
                 seq_len: int = 64, d_ff: int = 1536,
                 attention_variant: str = "geom",
                 input_mode: str = "lc0_64",
                 policy_head: str = "cross_attn"):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_mode = input_mode
        self.policy_head_type = policy_head
        self.attention_variant = attention_variant

        layout: TokenLayout = INPUT_MODE_TO_LAYOUT[input_mode]
        self.layout = layout
        self.seq_len = layout.seq_len

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)

        # Token-level pos embedding: baseline has no positional info inside
        # attention, so it always needs one. Bias-based + RoPE variants
        # encode position inside attention — but for ``lc0_64`` /  ``cls_65``
        # without dedicated stm/castling-with-pos-embedding tokens, we still
        # need *some* absolute pos embedding to distinguish the squares.
        # Keep it for all variants here for simplicity.
        if attention_variant == "baseline" or input_mode in ("lc0_64", "cls_65"):
            self.pos_embedding = nn.Embedding(layout.seq_len, embed_dim)
        else:
            self.pos_embedding = None

        pos_module, bias_module = build_pos_modules(
            attention_variant, embed_dim=embed_dim,
            num_heads=num_heads, layout=layout)
        self.bias_module = bias_module

        self.encoder = EncoderStack([
            EncoderLayer(embed_dim, num_heads, d_ff,
                         max_seq_len=layout.seq_len,
                         pos_embeddings=pos_module)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim=embed_dim)

        # Heads.
        self.wdl_head = nn.Linear(embed_dim, N_CELLS)
        if policy_head == "cross_attn":
            self.policy_head = CrossAttnPolicyHead(
                d_model=embed_dim, move_vocab_size=move_vocab_size,
                policy_index=policy_index)
        elif policy_head == "linear":
            self.policy_head = nn.Linear(embed_dim, move_vocab_size)
        else:
            raise ValueError(f"unknown policy_head {policy_head!r}")

        self.register_buffer("cell_wdl", CELL_WDL.clone())
        self.register_buffer("_pos_ids",
                             torch.arange(layout.seq_len), persistent=False)
        # Sequence positions of the 64 board squares — used by cross_attn
        # policy to read off square features regardless of input_mode.
        self.register_buffer("_square_positions",
                             torch.tensor(list(layout.squares),
                                          dtype=torch.long),
                             persistent=False)

    def forward(self, board_ids_raw: torch.Tensor) -> dict:
        # board_ids_raw is always [N, 68] (loader doesn't know input_mode).
        board_ids, stm_id, castling_id = _slice_inputs(board_ids_raw,
                                                       self.input_mode)
        x = self.tok_embedding(board_ids)                       # [N, S, D]
        # stm/castling broadcast to every token (CLS, when present, also
        # picks them up — the model can learn to use or ignore).
        x = x + self.tok_embedding(stm_id).unsqueeze(1)
        x = x + self.tok_embedding(castling_id).unsqueeze(1)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(self._pos_ids)

        mask = self.bias_module() if self.bias_module is not None else None
        x = self.encoder(x, mask=mask)
        x = self.norm(x)

        # Value readout: CLS slot when present, mean-pool otherwise.
        if self.layout.cls_pos is not None:
            cls = x[:, self.layout.cls_pos, :]
        else:
            cls = x.mean(dim=1)

        if self.policy_head_type == "cross_attn":
            squares_out = x[:, self._square_positions, :]       # [N, 64, D]
            pol_logits = self.policy_head(squares_out)
        else:
            pol_logits = self.policy_head(cls)
        return {"policy": pol_logits,
                "wdl": self.wdl_head(cls)}

    def mean_wdl(self, wdl_logits: torch.Tensor) -> torch.Tensor:
        return _mean_wdl(wdl_logits, self.cell_wdl)

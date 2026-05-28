"""ChessEncoder — single classical transformer encoder over the board tokens.

Input modes (selectable via ``input_mode``):

  default       68 tokens: CLS | a1..h8 (64) | end_pos | castling | stm.
                Value reads off CLS.
  no_end        67 tokens: drop end_pos. CLS | a1..h8 | castling | stm.
                Value reads off CLS.
  no_cls_no_end 66 tokens: drop CLS + end_pos. a1..h8 | castling | stm.
                Value via mean-pool over all 66 tokens (no CLS slot).
  lc0_64        64 tokens: just the board squares; side-to-move and
                castling rights are added as broadcast embeddings to every
                square. Value via mean-pool.

All four work with any ``attention_variant`` (baseline / rope1d / rope2d /
relpos2d / geom). ``smolgen`` is hardcoded to the 68-token layout for now.

Policy head: ``linear`` (single Linear from the readout vector) or
``cross_attn`` (LC0 attention-style policy from per-square Q/K — works in
every input mode by slicing out the 64 square positions).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchtune.modules import RMSNorm

from chessdecoder.models.layers import EncoderLayer, EncoderStack
from chessdecoder.models.policy_head import CrossAttnPolicyHead
from chessdecoder.models.pos_variants import (
    INPUT_MODE_TO_LAYOUT, Smolgen, TokenLayout, build_pos_modules)
from chessdecoder.models.value_buckets import CELL_WDL, N_CELLS, mean_wdl as _mean_wdl
from chessdecoder.models.vocab import move_vocab_size, policy_index


# Slot offsets in the *raw 68-token* cached layout (chess.SQUARES order).
_RAW_CLS_POS = 0
_RAW_SQ_START, _RAW_SQ_END = 1, 65
_RAW_END_POS = 65
_RAW_CASTLING_POS = 66
_RAW_STM_POS = 67


def _slice_inputs(board_ids: torch.Tensor, input_mode: str
                  ) -> tuple[torch.Tensor, Optional[torch.Tensor],
                             Optional[torch.Tensor]]:
    """Carve the raw cached ``[B, 68]`` ids into ``(board_ids, stm_id,
    castling_id)`` per input mode.

    For ``lc0_64`` the stm/castling are *separate* (broadcast at embed
    time). For the other modes stm/castling are columns inside the
    returned ``board_ids`` already (None for those two extras).
    """
    if input_mode == "default":                          # all 68 columns
        return board_ids, None, None
    if input_mode == "no_end":                           # drop col 65
        cols = list(range(0, _RAW_END_POS)) + [_RAW_CASTLING_POS, _RAW_STM_POS]
        return board_ids[:, cols], None, None
    if input_mode == "no_cls_no_end":                    # drop cols 0 and 65
        cols = list(range(_RAW_SQ_START, _RAW_SQ_END)) + \
               [_RAW_CASTLING_POS, _RAW_STM_POS]
        return board_ids[:, cols], None, None
    if input_mode == "lc0_64":                           # squares only
        return (board_ids[:, _RAW_SQ_START:_RAW_SQ_END],
                board_ids[:, _RAW_STM_POS],
                board_ids[:, _RAW_CASTLING_POS])
    raise ValueError(f"unknown input_mode {input_mode!r}")


class ChessEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 1024,
                 num_heads: int = 16, num_layers: int = 12,
                 seq_len: int = 68, d_ff: int = 1536,
                 attention_variant: str = "baseline",
                 input_mode: str = "default",
                 policy_head: str = "linear"):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_mode = input_mode
        self.policy_head_type = policy_head
        self.attention_variant = attention_variant

        if attention_variant == "smolgen" and input_mode != "default":
            raise ValueError(
                "smolgen currently hardcodes the 68-token layout; "
                "pair it with input_mode='default'")

        layout: TokenLayout = INPUT_MODE_TO_LAYOUT[input_mode]
        self.layout = layout
        self.seq_len = layout.seq_len

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)

        # Token-level pos embedding: needed for baseline / smolgen (no
        # in-attention positional info) and lc0_64 (otherwise the 64
        # squares are indistinguishable). For rope variants and the
        # bias-based variants, position info comes from inside attention.
        if attention_variant in ("baseline", "smolgen") or input_mode == "lc0_64":
            self.pos_embedding = nn.Embedding(layout.seq_len, embed_dim)
        else:
            self.pos_embedding = None

        pos_module, bias_module = build_pos_modules(
            attention_variant, embed_dim=embed_dim,
            num_heads=num_heads, layout=layout)
        self.bias_module = bias_module

        def _smolgen():
            return (Smolgen(d_model=embed_dim, num_heads=num_heads)
                    if attention_variant == "smolgen" else None)
        self.encoder = EncoderStack([
            EncoderLayer(embed_dim, num_heads, d_ff,
                         max_seq_len=layout.seq_len,
                         pos_embeddings=pos_module, smolgen=_smolgen())
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
        # Sequence positions of the 64 board squares — used to read off
        # squares for the cross_attn policy regardless of input_mode.
        self.register_buffer("_square_positions",
                             torch.tensor(list(layout.squares),
                                          dtype=torch.long),
                             persistent=False)

    def forward(self, board_ids_raw: torch.Tensor) -> dict:
        # board_ids_raw is always [N, 68] (loader doesn't know input_mode).
        board_ids, stm_id, castling_id = _slice_inputs(board_ids_raw,
                                                       self.input_mode)
        x = self.tok_embedding(board_ids)
        if self.input_mode == "lc0_64":
            # stm/castling get broadcast over every square.
            x = x + self.tok_embedding(stm_id).unsqueeze(1)
            x = x + self.tok_embedding(castling_id).unsqueeze(1)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(self._pos_ids)

        mask = self.bias_module() if self.bias_module is not None else None
        x = self.encoder(x, mask=mask)
        x = self.norm(x)

        # Readout for value (and linear policy):
        # CLS slot if the layout has one, else mean-pool over all tokens.
        if self.layout.cls_pos is not None:
            cls = x[:, self.layout.cls_pos, :]
        else:
            cls = x.mean(dim=1)

        if self.policy_head_type == "cross_attn":
            squares_out = x[:, self._square_positions, :]        # [N, 64, D]
            pol_logits = self.policy_head(squares_out)
        else:
            pol_logits = self.policy_head(cls)
        return {"policy": pol_logits,
                "wdl": self.wdl_head(cls)}

    def mean_wdl(self, wdl_logits: torch.Tensor) -> torch.Tensor:
        return _mean_wdl(wdl_logits, self.cell_wdl)

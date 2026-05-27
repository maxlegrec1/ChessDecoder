"""ChessEncoder — single classical transformer encoder over the board tokens.

Two input modes selectable via ``input_mode``:

- ``default`` (68 tokens): start_pos | 64 squares | end_pos | castling | stm.
  Policy/value read off the start_pos CLS slot.
- ``lc0_64`` (64 tokens): 64 board squares only. side-to-move and castling
  rights are added as broadcast embeddings to every square (so each square
  still "knows" them). Value reads via mean-pool over the 64 squares.

Positional information for the attention path is selectable via
``attention_variant`` (see ``pos_variants``). For now, the bias-based
variants (``relpos2d``, ``geom``, ``smolgen``) and the RoPE variants only
support ``input_mode='default'`` — switching the input layout to 64 squares
would change their internal seq_len bookkeeping. ``baseline`` works in both.

Policy head: ``linear`` (single Linear from the CLS / mean-pool readout) or
``cross_attn`` (LC0-style: per-square Q/K projections produce a pairwise
[64, 64] score, indexed by each move's (from_sq, to_sq) with an
under-promotion bias).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchtune.modules import RMSNorm

from chessdecoder.models.layers import EncoderLayer, EncoderStack
from chessdecoder.models.policy_head import CrossAttnPolicyHead
from chessdecoder.models.pos_variants import Smolgen, build_pos_modules
from chessdecoder.models.value_buckets import CELL_WDL, N_CELLS, mean_wdl as _mean_wdl
from chessdecoder.models.vocab import move_vocab_size, policy_index


# Slot offsets in the 68-token layout (chess.SQUARES order).
_CLS_POS = 0
_SQ_START, _SQ_END = 1, 65         # squares at [1, 65) — 64 entries
_END_POS = 65
_CASTLING_POS = 66
_STM_POS = 67


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

        if input_mode == "lc0_64":
            if attention_variant != "baseline":
                raise ValueError(
                    f"input_mode='lc0_64' currently only supports "
                    f"attention_variant='baseline'; got {attention_variant!r}. "
                    "(The bias / RoPE variants hardcode the 68-token layout.)")
            internal_seq_len = 64
        else:
            internal_seq_len = seq_len
        self.seq_len = internal_seq_len

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)

        # Token-level pos embedding: baseline / smolgen / lc0_64 all use it.
        if input_mode == "lc0_64" or attention_variant in ("baseline", "smolgen"):
            self.pos_embedding = nn.Embedding(internal_seq_len, embed_dim)
        else:
            self.pos_embedding = None

        pos_module, bias_module = build_pos_modules(
            attention_variant, embed_dim=embed_dim,
            num_heads=num_heads, seq_len=internal_seq_len)
        self.bias_module = bias_module

        # Smolgen is per-layer (each layer's input is different).
        def _smolgen():
            return (Smolgen(d_model=embed_dim, num_heads=num_heads)
                    if attention_variant == "smolgen" else None)
        self.encoder = EncoderStack([
            EncoderLayer(embed_dim, num_heads, d_ff,
                         max_seq_len=internal_seq_len,
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
                             torch.arange(internal_seq_len), persistent=False)

    def forward(self, board_ids: torch.Tensor) -> dict:
        # board_ids: always [N, 68] (the loader doesn't know about input_mode).
        if self.input_mode == "lc0_64":
            squares = board_ids[:, _SQ_START:_SQ_END]            # [N, 64]
            stm_id = board_ids[:, _STM_POS]                      # [N]
            castling_id = board_ids[:, _CASTLING_POS]            # [N]
            x = self.tok_embedding(squares)                      # [N, 64, D]
            # Add stm / castling as broadcast embeddings (same vec to all 64).
            x = x + self.tok_embedding(stm_id).unsqueeze(1)
            x = x + self.tok_embedding(castling_id).unsqueeze(1)
            x = x + self.pos_embedding(self._pos_ids)
        else:
            x = self.tok_embedding(board_ids)
            if self.pos_embedding is not None:
                x = x + self.pos_embedding(self._pos_ids)

        mask = self.bias_module() if self.bias_module is not None else None
        x = self.encoder(x, mask=mask)
        x = self.norm(x)

        # Readout for value (and linear policy): start_pos CLS for default,
        # mean-pool for lc0_64.
        if self.input_mode == "lc0_64":
            cls = x.mean(dim=1)                                  # [N, D]
            squares_out = x                                      # [N, 64, D]
        else:
            cls = x[:, _CLS_POS, :]                              # [N, D]
            squares_out = x[:, _SQ_START:_SQ_END, :]             # [N, 64, D]

        if self.policy_head_type == "cross_attn":
            pol_logits = self.policy_head(squares_out)           # [N, move_vocab]
        else:
            pol_logits = self.policy_head(cls)
        return {"policy": pol_logits,
                "wdl": self.wdl_head(cls)}

    def mean_wdl(self, wdl_logits: torch.Tensor) -> torch.Tensor:
        """[..., N_CELLS] -> [..., 3] expected WDL under the predicted simplex
        categorical."""
        return _mean_wdl(wdl_logits, self.cell_wdl)

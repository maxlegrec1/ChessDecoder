"""ChessEncoder — single classical transformer encoder over the 68 board
tokens. Predicts policy and WDL directly from the CLS (``start_pos``) token.

No decoder, no perceiver latents, no transition head. The encoder receives
``board_ids [N, 68]`` (start_pos | 64 squares | end_pos | castling | stm),
adds a learned absolute position embedding, runs N bidirectional layers, and
reads both heads off ``h[:, 0, :]``.

A/B-testable knobs live entirely in the config: depth (``num_layers``), width
(``embed_dim``), heads, FFN size, sequence length (currently 68 — leave room
to experiment with the LC0-style 64-token variant where stm/castling are
folded into the per-square embeddings).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchtune.modules import RMSNorm

from chessdecoder.models.layers import EncoderLayer
from chessdecoder.models.value_buckets import CELL_WDL, N_CELLS, mean_wdl as _mean_wdl
from chessdecoder.models.vocab import move_vocab_size


class ChessEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 1024,
                 num_heads: int = 16, num_layers: int = 12,
                 seq_len: int = 68, d_ff: int = 1536):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        # Sequential makes the whole stack a single compile target — we feed
        # it to torch.compile in the FP8 path.
        self.encoder = nn.Sequential(*[
            EncoderLayer(embed_dim, num_heads, d_ff, max_seq_len=seq_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim=embed_dim)

        self.policy_head = nn.Linear(embed_dim, move_vocab_size)
        self.wdl_head = nn.Linear(embed_dim, N_CELLS)

        self.register_buffer("cell_wdl", CELL_WDL.clone())
        self.register_buffer("_pos_ids", torch.arange(seq_len), persistent=False)

    def forward(self, board_ids: torch.Tensor) -> dict:
        # board_ids: [N, seq_len]
        x = self.tok_embedding(board_ids) + self.pos_embedding(self._pos_ids)
        x = self.encoder(x)
        x = self.norm(x)
        cls = x[:, 0, :]                                # start_pos = CLS
        return {"policy": self.policy_head(cls),        # [N, move_vocab_size]
                "wdl": self.wdl_head(cls)}              # [N, N_CELLS]

    def mean_wdl(self, wdl_logits: torch.Tensor) -> torch.Tensor:
        """[..., N_CELLS] -> [..., 3] expected WDL under the predicted simplex
        categorical."""
        return _mean_wdl(wdl_logits, self.cell_wdl)

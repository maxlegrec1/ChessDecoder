"""ChessEncoder — single classical transformer encoder over the 68 board
tokens. Predicts policy and WDL directly from the CLS (``start_pos``) token.

No decoder, no perceiver latents, no transition head. The encoder receives
``board_ids [N, 68]`` (start_pos | 64 squares | end_pos | castling | stm),
runs N bidirectional layers, and reads both heads off ``h[:, 0, :]``.

Positional information is selectable via ``attention_variant``:
  - ``baseline``: add a learned absolute pos embedding to the tokens (no
    in-layer machinery).
  - ``rope1d`` / ``rope2d``: rotate Q/K per head; no token-level pos embed.
  - ``relpos2d`` / ``geom``: learned attention-logit bias shared across layers;
    no token-level pos embed.

A/B-testable knobs live entirely in the config: depth (``num_layers``), width
(``embed_dim``), heads, FFN size, attention variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchtune.modules import RMSNorm

from chessdecoder.models.layers import EncoderLayer, EncoderStack
from chessdecoder.models.pos_variants import build_pos_modules
from chessdecoder.models.value_buckets import CELL_WDL, N_CELLS, mean_wdl as _mean_wdl
from chessdecoder.models.vocab import move_vocab_size


class ChessEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 1024,
                 num_heads: int = 16, num_layers: int = 12,
                 seq_len: int = 68, d_ff: int = 1536,
                 attention_variant: str = "baseline"):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.attention_variant = attention_variant

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)

        # Only ``baseline`` uses a token-level learned pos embedding;
        # everything else encodes position inside attention.
        if attention_variant == "baseline":
            self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        else:
            self.pos_embedding = None

        pos_module, bias_module = build_pos_modules(
            attention_variant, embed_dim=embed_dim,
            num_heads=num_heads, seq_len=seq_len)
        # Bias module owns learnable params -> attribute on the model.
        # pos_module is shared across layers (the RoPE cos/sin buffers are
        # identical across layers, so sharing is functionally equivalent).
        self.bias_module = bias_module

        self.encoder = EncoderStack([
            EncoderLayer(embed_dim, num_heads, d_ff, max_seq_len=seq_len,
                         pos_embeddings=pos_module)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim=embed_dim)

        self.policy_head = nn.Linear(embed_dim, move_vocab_size)
        self.wdl_head = nn.Linear(embed_dim, N_CELLS)

        self.register_buffer("cell_wdl", CELL_WDL.clone())
        self.register_buffer("_pos_ids", torch.arange(seq_len), persistent=False)

    def forward(self, board_ids: torch.Tensor) -> dict:
        # board_ids: [N, seq_len]
        x = self.tok_embedding(board_ids)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(self._pos_ids)
        mask = self.bias_module() if self.bias_module is not None else None
        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        cls = x[:, 0, :]                                # start_pos = CLS
        return {"policy": self.policy_head(cls),        # [N, move_vocab_size]
                "wdl": self.wdl_head(cls)}              # [N, N_CELLS]

    def mean_wdl(self, wdl_logits: torch.Tensor) -> torch.Tensor:
        """[..., N_CELLS] -> [..., 3] expected WDL under the predicted simplex
        categorical."""
        return _mean_wdl(wdl_logits, self.cell_wdl)

"""Encoder building blocks.

Pre-norm transformer encoder layer (bidirectional self-attention + SwiGLU MLP,
RMSNorm). The layer takes an optional ``pos_embeddings`` module that rotates
Q/K (used by RoPE variants) and an optional float ``mask`` added to attention
logits (used by relative-position / geometric-bias variants).
"""
from typing import Optional

import torch
import torch.nn as nn
from torchtune.modules import (MultiHeadAttention, FeedForward, RMSNorm)


class EncoderLayer(nn.Module):
    """Bidirectional self-attention + SwiGLU MLP, pre-RMSNorm."""

    def __init__(self, embed_dim: int, num_heads: int, d_ff: int,
                 max_seq_len: int = 128,
                 pos_embeddings: Optional[nn.Module] = None):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            k_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            v_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=pos_embeddings,
            max_seq_len=max_seq_len, is_causal=False)
        self.mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, d_ff, bias=False),
            down_proj=nn.Linear(d_ff, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, d_ff, bias=False))
        self.sa_norm = RMSNorm(dim=embed_dim)
        self.mlp_norm = RMSNorm(dim=embed_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.sa_norm(x)
        h = x + self.attn(n, n, mask=mask)
        return h + self.mlp(self.mlp_norm(h))


class EncoderStack(nn.Module):
    """Sequential application of ``EncoderLayer``s with an optional shared
    attention bias passed to every layer.

    Replaces ``nn.Sequential`` so we can thread a single ``mask`` tensor (the
    learned relpos / geometric bias) through the stack without recomputing it
    per layer. ``torch.compile`` traces through this for-loop fine as long as
    ``layers`` is a static ``nn.ModuleList``.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

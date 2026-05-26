"""Encoder building blocks.

A classical pre-norm transformer encoder layer (bidirectional self-attention +
SwiGLU MLP, RMSNorm) — no positional embeddings in the layer itself; the model
adds a learned absolute position embedding to the token embeddings before the
first layer.
"""
import torch
import torch.nn as nn
from torchtune.modules import (MultiHeadAttention, FeedForward, RMSNorm)


class EncoderLayer(nn.Module):
    """Bidirectional self-attention + SwiGLU MLP, pre-RMSNorm. No RoPE."""

    def __init__(self, embed_dim: int, num_heads: int, d_ff: int,
                 max_seq_len: int = 128):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            k_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            v_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=None, max_seq_len=max_seq_len, is_causal=False)
        self.mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, d_ff, bias=False),
            down_proj=nn.Linear(d_ff, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, d_ff, bias=False))
        self.sa_norm = RMSNorm(dim=embed_dim)
        self.mlp_norm = RMSNorm(dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.sa_norm(x)
        h = x + self.attn(n, n)
        return h + self.mlp(self.mlp_norm(h))

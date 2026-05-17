"""Shared building blocks for the V2 stack.

Relocated here so the V2 architecture is self-contained and the V1 model
(``models/model.py``, ``models/encoder.py``) can be removed. These are the
exact V1 implementations — behaviour unchanged.
"""
import math

import torch
import torch.nn as nn
from torchtune.modules import (MultiHeadAttention, FeedForward, RMSNorm,
                               RotaryPositionalEmbeddings)


class TransformerEncoderLayer(nn.Module):
    """Bidirectional self-attention + SwiGLU MLP, pre-norm, RoPE."""

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int,
                 rope: RotaryPositionalEmbeddings, max_seq_len: int = 128,
                 d_ff: int = None):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * embed_dim
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            k_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            v_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope, max_seq_len=max_seq_len, is_causal=False)
        self.mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, d_ff, bias=False),
            down_proj=nn.Linear(d_ff, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, d_ff, bias=False))
        self.sa_norm = RMSNorm(dim=embed_dim)
        self.mlp_norm = RMSNorm(dim=embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                input_pos: torch.Tensor = None):
        h = x + self.attn(self.sa_norm(x), self.sa_norm(x), mask=mask,
                          input_pos=input_pos)
        h = h + self.mlp(self.mlp_norm(h))
        return h


def make_wl_buckets(n_buckets=100, sigma=0.4):
    """Center-concentrated bucket centers in [-1, 1] via Gaussian-CDF quantiles."""
    t = torch.linspace(0.5 / n_buckets, 1 - 0.5 / n_buckets, n_buckets)
    centers = sigma * math.sqrt(2) * torch.erfinv(2 * t - 1)
    return centers.clamp(-1.0, 1.0)


def make_d_buckets(n_buckets=100):
    """Uniform bucket centers in [0, 1]."""
    return torch.linspace(0.5 / n_buckets, 1 - 0.5 / n_buckets, n_buckets)


class FourierEncoder(nn.Module):
    """Scalar -> embed_dim via learned Fourier features (Moondream-3 style)."""

    def __init__(self, embed_dim, num_frequencies=128):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(1, num_frequencies))
        self.proj = nn.Linear(2 * num_frequencies, embed_dim)

    def forward(self, x):
        f = 2 * math.pi * x.unsqueeze(-1) @ self.frequencies   # (N,1)@(1,F)->(N,F)
        features = torch.cat([f.cos(), f.sin()], dim=-1)        # (N,2F)
        return self.proj(features)                              # (N,embed_dim)


class ValueHead(nn.Module):
    """MLP head mapping hidden states to bucket logits."""

    def __init__(self, embed_dim, n_buckets=100, hidden_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, n_buckets))

    def forward(self, hidden_state):
        return self.mlp(hidden_state)

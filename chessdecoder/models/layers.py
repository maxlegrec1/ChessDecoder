"""Encoder building blocks.

Pre-norm transformer encoder layer (bidirectional self-attention + SwiGLU MLP,
RMSNorm). The layer takes an optional ``pos_embeddings`` module that rotates
Q/K (used by RoPE variants) and an optional float ``mask`` added to attention
logits (used by relative-position / geometric-bias variants).

Note: we don't use ``torchtune.modules.MultiHeadAttention`` here because its
SDPA path collapses the per-head dim — it does ``mask[:, None, :, :]``,
expecting a ``[B, S, S]`` mask that broadcasts over heads. T5/ALiBi-style
biases are *per-head*, so we'd lose the whole point of those variants. The
custom ``BidirAttn`` below is a thin Q/K/V/SDPA wrapper that accepts a
``[1, H, S, S]`` (or any broadcastable 4D) bias and plumbs it straight to
``F.scaled_dot_product_attention``.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import FeedForward, RMSNorm


class BidirAttn(nn.Module):
    """Bidirectional multi-head attention with optional RoPE + per-head bias.

    All projections are ``bias=False`` (matches the prior torchtune setup so
    the FP8 conversion still finds the same Linears). ``pos_embeddings`` is
    expected to expose ``forward(x, *, input_pos=None)`` over a
    ``[B, S, H, D]`` tensor (rotates Q/K) — same contract as torchtune's RoPE.
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 pos_embeddings: Optional[nn.Module] = None):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must divide num_heads "
                             f"{num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pos_embeddings = pos_embeddings

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q)
            k = self.pos_embeddings(k)
        # SDPA wants [B, H, S, D].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # mask: None or float [B', H', S, S] broadcastable to [B, H, S, S].
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, s, self.embed_dim)
        return self.output_proj(out)


class EncoderLayer(nn.Module):
    """Bidirectional self-attention + SwiGLU MLP, pre-RMSNorm."""

    def __init__(self, embed_dim: int, num_heads: int, d_ff: int,
                 max_seq_len: int = 128,
                 pos_embeddings: Optional[nn.Module] = None):
        super().__init__()
        self.attn = BidirAttn(embed_dim=embed_dim, num_heads=num_heads,
                              pos_embeddings=pos_embeddings)
        self.mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, d_ff, bias=False),
            down_proj=nn.Linear(d_ff, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, d_ff, bias=False))
        self.sa_norm = RMSNorm(dim=embed_dim)
        self.mlp_norm = RMSNorm(dim=embed_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.sa_norm(x)
        h = x + self.attn(n, mask=mask)
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

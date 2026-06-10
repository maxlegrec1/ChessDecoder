"""AgentDecoder — the search agent's decoder-only transformer.

Plain causal LM over the agent vocab (~31k): tied input/output embeddings,
RoPE (native max 8192 so Stage-A's random-offset training covers the whole
window), pre-RMSNorm, SwiGLU, flash attention via is_causal=True.

forward() returns HIDDEN STATES, not logits — the 31k-way head is applied
only at loss/generation positions (logits_at) to keep memory sane at long
stream lengths.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import (FeedForward, MultiHeadAttention, RMSNorm,
                               RotaryPositionalEmbeddings,
                               TransformerSelfAttentionLayer)

from chessdecoder.agent.patch_vocab import VOCAB_SIZE


class AgentDecoder(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 d_ff: int = 2048, max_seq_len: int = 8192):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        head_dim = embed_dim // num_heads

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.tok_embedding.weight, std=0.02)

        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = MultiHeadAttention(
                embed_dim=embed_dim, num_heads=num_heads,
                num_kv_heads=num_heads, head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                k_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                v_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=rope, max_seq_len=max_seq_len, is_causal=True)
            mlp = FeedForward(
                gate_proj=nn.Linear(embed_dim, d_ff, bias=False),
                down_proj=nn.Linear(d_ff, embed_dim, bias=False),
                up_proj=nn.Linear(embed_dim, d_ff, bias=False))
            self.layers.append(TransformerSelfAttentionLayer(
                attn=attn, mlp=mlp,
                sa_norm=RMSNorm(dim=embed_dim), mlp_norm=RMSNorm(dim=embed_dim)))
        self.norm = RMSNorm(dim=embed_dim)

    def forward(self, ids: torch.Tensor,
                input_pos: torch.Tensor | None = None) -> torch.Tensor:
        """ids [B,S] int64, input_pos [B,S] (RoPE positions; defaults to
        arange). Returns hidden [B,S,E]."""
        b, s = ids.shape
        if input_pos is None:
            input_pos = torch.arange(s, device=ids.device).unsqueeze(0).expand(b, -1)
        h = self.tok_embedding(ids)
        for layer in self.layers:
            h = layer(h, mask=None, input_pos=input_pos)   # is_causal -> flash
        return self.norm(h)

    def logits_at(self, hidden_flat: torch.Tensor) -> torch.Tensor:
        """Tied unembedding on selected positions only. hidden_flat [N,E] ->
        [N,V]."""
        return F.linear(hidden_flat, self.tok_embedding.weight)

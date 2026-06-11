"""AgentDecoder — the search agent's decoder-only transformer.

Plain causal LM over the agent vocab (~31k): tied input/output embeddings,
RoPE (native max 8192 so Stage-A's random-offset training covers the whole
window), pre-RMSNorm, SwiGLU, flash attention via is_causal=True.

forward() returns HIDDEN STATES, not logits — the 31k-way head is applied
only at loss/generation positions (logits_at) to keep memory sane at long
stream lengths.
"""
from __future__ import annotations

from contextlib import contextmanager

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
                input_pos: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """ids [B,S] int64, input_pos [B,S] (RoPE positions; defaults to
        arange). Returns hidden [B,S,E]. With KV caches enabled (see
        setup_caches) an explicit bool mask [B,S,cache_len] is mandatory:
        the cache buffer is full-length and zero-padded, and attending the
        zeros dilutes attention weights."""
        b, s = ids.shape
        if input_pos is None:
            input_pos = torch.arange(s, device=ids.device).unsqueeze(0).expand(b, -1)
        h = self.tok_embedding(ids)
        for layer in self.layers:
            h = layer(h, mask=mask, input_pos=input_pos)   # is_causal -> flash
        return self.norm(h)

    # -- KV-cache management (rollout engine) --------------------------------
    def setup_caches(self, batch_size: int, dtype: torch.dtype,
                     max_seq_len: int) -> None:
        dev = self.tok_embedding.weight.device
        for layer in self.layers:       # torchtune skips setup if a cache
            layer.attn.kv_cache = None  # exists — force a rebuild
            layer.attn.cache_enabled = False
        with dev:                       # torchtune allocates on default device
            for layer in self.layers:
                layer.setup_caches(batch_size, dtype,
                                   encoder_max_seq_len=None,
                                   decoder_max_seq_len=max_seq_len)

    def caches_are_enabled(self) -> bool:
        return self.layers[0].caches_are_enabled()

    def reset_caches(self) -> None:
        for layer in self.layers:
            layer.reset_cache()

    @contextmanager
    def caches_disabled(self):
        """Plain causal forwards (teacher-forced) while caches stay set up.

        kv_cache must be nulled, not just disabled: torchtune gates is_causal
        on ``kv_cache is None and mask is None`` — with caches merely set up,
        a mask=None forward silently runs BIDIRECTIONAL."""
        saved = []
        for layer in self.layers:
            saved.append((layer.attn.kv_cache, layer.attn.cache_enabled))
            layer.attn.kv_cache = None
            layer.attn.cache_enabled = False
        try:
            yield
        finally:
            for layer, (kc, en) in zip(self.layers, saved):
                layer.attn.kv_cache = kc
                layer.attn.cache_enabled = en

    def logits_at(self, hidden_flat: torch.Tensor) -> torch.Tensor:
        """Tied unembedding on selected positions only. hidden_flat [N,E] ->
        [N,V]."""
        return F.linear(hidden_flat, self.tok_embedding.weight)

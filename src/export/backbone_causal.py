"""
Exportable causal backbone for ONNX/TRT.

Standalone module that replicates the ChessDecoder backbone with:
- Explicit KV cache I/O (past/present per layer, stacked)
- Explicit attention mask input (constructed by caller)
- Manual RoPE via precomputed cos/sin cache + index_select
- No torchtune dependencies — all standard PyTorch ops

Includes Fourier encoder for WL/D value injection (runs on GPU in TRT
to exactly match PyTorch FP16 behavior).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneCausal(nn.Module):
    """Causal backbone with KV cache for autoregressive generation."""

    def __init__(self, num_layers, num_heads, head_dim, embed_dim, d_ff,
                 vocab_size, max_seq_len, num_fourier_freq=128):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)

        # RoPE cache: [max_seq_len, head_dim//2, 2] (cos, sin)
        self.register_buffer('rope_cache', torch.zeros(max_seq_len, head_dim // 2, 2))

        # Fourier encoder for WL/D value injection
        self.fourier_frequencies = nn.Parameter(torch.randn(1, num_fourier_freq))
        self.fourier_proj = nn.Linear(2 * num_fourier_freq, embed_dim)

        # Per-layer weights (stored as ModuleLists for clean state_dict)
        self.q_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
        self.k_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
        self.v_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
        self.o_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])

        self.sa_norm_weights = nn.ParameterList([nn.Parameter(torch.ones(embed_dim)) for _ in range(num_layers)])
        self.mlp_norm_weights = nn.ParameterList([nn.Parameter(torch.ones(embed_dim)) for _ in range(num_layers)])

        self.gate_projs = nn.ModuleList([nn.Linear(embed_dim, d_ff, bias=False) for _ in range(num_layers)])
        self.up_projs = nn.ModuleList([nn.Linear(embed_dim, d_ff, bias=False) for _ in range(num_layers)])
        self.down_projs = nn.ModuleList([nn.Linear(d_ff, embed_dim, bias=False) for _ in range(num_layers)])

        self.final_norm_weight = nn.Parameter(torch.ones(embed_dim))

    def _fourier_encode(self, values):
        """Encode scalar values via Fourier features. values: (N,) -> (N, E)"""
        f = 2 * math.pi * values.unsqueeze(-1) * self.fourier_frequencies  # (N, F)
        features = torch.cat([f.cos(), f.sin()], dim=-1)  # (N, 2F)
        return self.fourier_proj(features)  # (N, E)

    def _rms_norm(self, x, weight):
        x_float = x.float()
        rms = torch.sqrt(x_float.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x_float / rms * weight.float()).to(x.dtype)

    def _apply_rope(self, x, input_pos):
        """Apply RoPE. x: [B, S, NH, HD], input_pos: [B, S] -> [B, S, NH, HD]"""
        rope_vals = self.rope_cache[input_pos]
        rope_vals = rope_vals.unsqueeze(2)
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_out = torch.stack([
            xshaped[..., 0] * rope_vals[..., 0] - xshaped[..., 1] * rope_vals[..., 1],
            xshaped[..., 1] * rope_vals[..., 0] + xshaped[..., 0] * rope_vals[..., 1],
        ], dim=-1)
        return x_out.flatten(3).to(x.dtype)

    def forward(self, input_ids, input_pos, attention_mask,
                past_keys, past_values,
                override_values, override_mask):
        """
        Args:
            input_ids:       [1, S] int64
            input_pos:       [1, S] int64
            attention_mask:  [1, 1, S, S+past_len] float32 (0 or -1e9)
            past_keys:       [NL, 1, NH, past_len, HD] float32
            past_values:     [NL, 1, NH, past_len, HD] float32
            override_values: [1, S] float32 — scalar values for Fourier encoding
            override_mask:   [1, S] bool — True at positions to override with Fourier

        Returns:
            hidden_states:   [1, S, E] float32
            present_keys:    [NL, 1, NH, S+past_len, HD] float32
            present_values:  [NL, 1, NH, S+past_len, HD] float32
        """
        B, S = input_ids.shape

        # Token embedding with Fourier override
        h = self.tok_embedding(input_ids)
        # Always compute Fourier for all positions, then blend with mask
        # (avoids data-dependent branching for ONNX export)
        fourier_embs = self._fourier_encode(override_values.reshape(-1)).reshape(B, S, -1).to(h.dtype)
        mask_expanded = override_mask.unsqueeze(-1)  # [B, S, 1]
        h = torch.where(mask_expanded, fourier_embs, h)

        all_present_keys = []
        all_present_values = []

        for i in range(self.num_layers):
            residual = h
            h_norm = self._rms_norm(h, self.sa_norm_weights[i])

            q = self.q_projs[i](h_norm).view(B, S, self.num_heads, self.head_dim)
            k = self.k_projs[i](h_norm).view(B, S, self.num_heads, self.head_dim)
            v = self.v_projs[i](h_norm).view(B, S, self.num_heads, self.head_dim)

            q = self._apply_rope(q, input_pos)
            k = self._apply_rope(k, input_pos)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            past_k = past_keys[i]
            past_v = past_values[i]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

            all_present_keys.append(k)
            all_present_values.append(v)

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(h.dtype)
            attn_out = torch.matmul(attn_weights, v)

            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
            h = self.o_projs[i](attn_out) + residual

            residual = h
            h_norm = self._rms_norm(h, self.mlp_norm_weights[i])
            gate = F.silu(self.gate_projs[i](h_norm))
            up = self.up_projs[i](h_norm)
            h = self.down_projs[i](gate * up) + residual

        h = self._rms_norm(h, self.final_norm_weight)

        present_keys = torch.stack(all_present_keys)
        present_values = torch.stack(all_present_values)

        return h, present_keys, present_values


def from_chess_decoder(model):
    """Create BackboneCausal from a trained ChessDecoder, copying all weights."""
    num_layers = len(model.layers)
    layer0_attn = model.layers[0].attn
    num_heads = layer0_attn.num_heads
    head_dim = layer0_attn.head_dim
    embed_dim = num_heads * head_dim
    d_ff = model.layers[0].mlp.w1.out_features
    vocab_size = model.tok_embedding.num_embeddings
    rope = layer0_attn.pos_embeddings
    max_seq_len = rope.max_seq_len
    num_fourier_freq = model.fourier_encoder.frequencies.shape[1]

    backbone = BackboneCausal(num_layers, num_heads, head_dim, embed_dim, d_ff,
                              vocab_size, max_seq_len, num_fourier_freq)

    backbone.tok_embedding.weight.data.copy_(model.tok_embedding.weight.data)
    backbone.rope_cache.copy_(rope.cache)
    backbone.fourier_frequencies.data.copy_(model.fourier_encoder.frequencies.data)
    backbone.fourier_proj.weight.data.copy_(model.fourier_encoder.proj.weight.data)
    backbone.fourier_proj.bias.data.copy_(model.fourier_encoder.proj.bias.data)

    for i, layer in enumerate(model.layers):
        attn = layer.attn
        backbone.q_projs[i].weight.data.copy_(attn.q_proj.weight.data)
        backbone.k_projs[i].weight.data.copy_(attn.k_proj.weight.data)
        backbone.v_projs[i].weight.data.copy_(attn.v_proj.weight.data)
        backbone.o_projs[i].weight.data.copy_(attn.output_proj.weight.data)
        backbone.sa_norm_weights[i].data.copy_(layer.sa_norm.scale.data)
        backbone.mlp_norm_weights[i].data.copy_(layer.mlp_norm.scale.data)
        backbone.gate_projs[i].weight.data.copy_(layer.mlp.w1.weight.data)
        backbone.up_projs[i].weight.data.copy_(layer.mlp.w3.weight.data)
        backbone.down_projs[i].weight.data.copy_(layer.mlp.w2.weight.data)

    backbone.final_norm_weight.data.copy_(model.norm.scale.data)
    return backbone

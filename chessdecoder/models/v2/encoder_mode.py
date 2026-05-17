"""ChessEncoderV2 — the V2 architecture (Perceiver latents + causal decoder)
run in the *degenerate encoder special case* for a strict comparative benchmark.

This is V2 with no game history and no thinking trace: a single board in,
best-move out, cross-entropy on best_move — exactly the task/loss/eval the
current `ChessEncoder` runs use. The ONLY variable changed vs that baseline is
the architecture:

    68 board tokens
      → bidirectional encoder layers            (same as ChessEncoder backbone)
      → Perceiver pooling to k learned latents  (the V2 bottleneck, k=16)
      → causal self-attention over the latents  (the V2 causal decoder block)
      → mean-pool latents → policy head

Positional handling is kept RoPE-only (no extra learned/2-D spatial PE) so the
benchmark isolates "latents + causal" and does not confound it with the
spatial-awareness change discussed separately in markdowns/11.
"""
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings

from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.vocab import token_to_idx, policy_index
from chessdecoder.models.v2.layers import TransformerEncoderLayer


class PerceiverPool(nn.Module):
    """Cross-attention pooling: k learned query latents attend over the encoded
    board tokens (padding-masked). One block, residual on the latent stream."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm_q = RMSNorm(dim=embed_dim)
        self.norm_kv = RMSNorm(dim=embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _split(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, latents, ctx, key_valid_mask=None):
        # latents: [B,k,E]  ctx: [B,S,E]  key_valid_mask: [B,S] (True = valid)
        q = self._split(self.q_proj(self.norm_q(latents)))      # [B,H,k,D]
        kv = self.norm_kv(ctx)
        k = self._split(self.k_proj(kv))                         # [B,H,S,D]
        v = self._split(self.v_proj(kv))
        attn_mask = None
        if key_valid_mask is not None:
            # [B,1,1,S] additive (-inf on padded keys)
            attn_mask = torch.zeros(
                key_valid_mask.shape[0], 1, 1, key_valid_mask.shape[1],
                dtype=q.dtype, device=q.device,
            )
            attn_mask.masked_fill_(~key_valid_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        b, h, n, d = out.shape
        out = out.transpose(1, 2).reshape(b, n, h * d)
        return latents + self.o_proj(out)


class ChessEncoderV2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_policy_tokens: int = len(policy_index),
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_encoder_layers: int = 10,
        num_decoder_layers: int = 2,
        num_latents: int = 16,
        max_seq_len: int = 68,
        d_ff: int = 1536,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_policy_tokens = num_policy_tokens
        self.num_latents = num_latents
        self.max_seq_len = max_seq_len
        head_dim = embed_dim // num_heads
        d_ff = d_ff if d_ff is not None else 4 * embed_dim

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)

        # Shared RoPE for the bidirectional board encoder.
        enc_rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, head_dim, enc_rope,
                                    max_seq_len, d_ff)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = RMSNorm(dim=embed_dim)

        # Perceiver pooling: k learned latents.
        self.latent_queries = nn.Parameter(torch.randn(num_latents, embed_dim) * 0.02)
        self.pool = PerceiverPool(embed_dim, num_heads)

        # Causal decoder over the k latents (separate RoPE over latent positions).
        dec_rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=num_latents)
        self.decoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, head_dim, dec_rope,
                                    num_latents, d_ff)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = RMSNorm(dim=embed_dim)

        self.policy_head = nn.Linear(embed_dim, num_policy_tokens)

        self.pad_id = token_to_idx["pad"]
        self.start_pos_id = token_to_idx["start_pos"]

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None,
                padded: bool = True):
        """x: [B,S] token ids. attention_mask: [B,S] bool, True = valid token.

        ``padded=False`` asserts the batch has NO padding (every sample is a
        full-length sequence) and skips the dense [B,S,S] mask entirely, which
        lets attention use the fused/flash SDPA kernel. Only valid when there
        is genuinely no padding (true for ChessFENS: every FEN -> exactly 68
        tokens). Default ``padded=True`` keeps the original masked path so the
        contract for padded/general use (and predict_move) is unchanged."""
        bsz, seq_len = x.shape
        device = x.device

        enc_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        if padded and attention_mask is not None:
            enc_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            pool_key_mask = attention_mask
        else:
            enc_mask = None          # fused, unmasked attention
            pool_key_mask = None

        h = self.tok_embedding(x)
        for layer in self.encoder_layers:
            h = layer(h, mask=enc_mask, input_pos=enc_pos)
        h = self.encoder_norm(h)

        # Pool to k latents.
        latents = self.latent_queries.unsqueeze(0).expand(bsz, -1, -1)
        latents = self.pool(latents, h, key_valid_mask=pool_key_mask)

        # Causal self-attention over latents (latent i attends to latents <= i).
        k = self.num_latents
        causal = torch.tril(torch.ones(k, k, dtype=torch.bool, device=device))
        dec_mask = causal.unsqueeze(0).expand(bsz, -1, -1)
        dec_pos = torch.arange(k, device=device).unsqueeze(0).expand(bsz, -1)
        for layer in self.decoder_layers:
            latents = layer(latents, mask=dec_mask, input_pos=dec_pos)
        latents = self.decoder_norm(latents)

        pooled = latents.mean(dim=1)  # [B,E]
        return self.policy_head(pooled)

    @torch.no_grad()
    def predict_move(self, fen: str, temperature: float = 1.0,
                     force_legal: bool = True) -> str:
        """Identical semantics to ChessEncoder.predict_move so it drops into
        PytorchModelAdapter / elo_eval unchanged (strict eval comparability)."""
        self.eval()
        device = next(self.parameters()).device
        tokens = fen_to_position_tokens(fen)
        input_ids = torch.tensor([token_to_idx[t] for t in tokens],
                                 dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)

        logits = self(input_ids, attention_mask=attention_mask)[0]

        if force_legal:
            board = chess.Board(fen)
            vocab_legal_moves = []
            for move in board.legal_moves:
                uci = move.uci()
                if board.is_castling(move):
                    if uci == 'e1g1': uci = 'e1h1'
                    elif uci == 'e1c1': uci = 'e1a1'
                    elif uci == 'e8g8': uci = 'e8h8'
                    elif uci == 'e8c8': uci = 'e8a8'
                if uci in policy_index:
                    vocab_legal_moves.append(policy_index.index(uci))
            if vocab_legal_moves:
                mask = torch.full_like(logits, float('-inf'))
                mask[vocab_legal_moves] = 0
                logits = logits + mask

        if temperature == 0.0:
            idx = torch.argmax(logits).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()

        move_str = policy_index[idx]
        replacements = {'e1h1': 'e1g1', 'e1a1': 'e1c1',
                        'e8h8': 'e8g8', 'e8a8': 'e8c8'}
        return replacements.get(move_str, move_str)

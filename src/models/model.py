import math
import torch
import torch.nn as nn
import chess
from torchtune.modules import TransformerSelfAttentionLayer, MultiHeadAttention, FeedForward, RMSNorm, RotaryPositionalEmbeddings
from src.dataloader.data import fen_to_position_tokens
from src.models.vocab import token_to_idx, idx_to_token


def make_wl_buckets(n_buckets=100, sigma=0.4):
    """Center-concentrated bucket centers in [-1, 1] via Gaussian CDF quantiles."""
    t = torch.linspace(0.5 / n_buckets, 1 - 0.5 / n_buckets, n_buckets)
    centers = sigma * math.sqrt(2) * torch.erfinv(2 * t - 1)
    return centers.clamp(-1.0, 1.0)


def make_d_buckets(n_buckets=100):
    """Uniform bucket centers in [0, 1]."""
    return torch.linspace(0.5 / n_buckets, 1 - 0.5 / n_buckets, n_buckets)


class FourierEncoder(nn.Module):
    """Encodes a scalar value to embed_dim via learned Fourier features (Moondream-3 style)."""

    def __init__(self, embed_dim, num_frequencies=128):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(1, num_frequencies))
        self.proj = nn.Linear(2 * num_frequencies, embed_dim)

    def forward(self, x):
        # x: (N,)
        f = 2 * math.pi * x.unsqueeze(-1) @ self.frequencies  # (N,1) @ (1,F) -> (N,F)
        features = torch.cat([f.cos(), f.sin()], dim=-1)       # (N, 2F)
        return self.proj(features)  # (N, embed_dim)


class ValueHead(nn.Module):
    """MLP head that maps hidden states to bucket logits."""

    def __init__(self, embed_dim, n_buckets=100, hidden_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, n_buckets)
        )

    def forward(self, hidden_state):
        # hidden_state: (N, embed_dim) -> (N, n_buckets)
        return self.mlp(hidden_state)


class ChessDecoder(nn.Module):
    """
    Decoder chess model with causal attention and RoPE.

    Board representation is fixed 68 tokens per position:
        - start_pos (1)
        - 64 board tokens (each square: empty or color_piece)
        - end_pos (1)
        - castling (1)
        - side_to_move (1)

    Sequence per position: [board_68][move][WL][D] = 71 tokens.

    Heads:
        - board_head: nn.Linear for causal board generation
        - policy_head: nn.Linear for move prediction (prefix pass)
        - wl_head: ValueHead for WL prediction (100 buckets)
        - d_head: ValueHead for D prediction (100 buckets)
    """

    def __init__(self, vocab_size, embed_dim=768, num_heads=12, num_layers=12,
                 max_seq_len=2048, d_ff=None, n_buckets=100, value_hidden_size=256,
                 num_fourier_freq=128, wl_sigma=0.4):
        super().__init__()

        head_dim = embed_dim // num_heads
        d_ff = d_ff if d_ff is not None else 4 * embed_dim

        # Token embedding only (RoPE handles positional info)
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)

        # RoPE - shared across all layers
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                k_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                v_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=rope,
                max_seq_len=max_seq_len,
                is_causal=True
            )
            mlp = FeedForward(
                gate_proj=nn.Linear(embed_dim, d_ff, bias=False),
                down_proj=nn.Linear(d_ff, embed_dim, bias=False),
                up_proj=nn.Linear(embed_dim, d_ff, bias=False)
            )
            self.layers.append(TransformerSelfAttentionLayer(
                attn=attn,
                mlp=mlp,
                sa_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim)
            ))

        self.norm = RMSNorm(dim=embed_dim)

        # Board generation head (causal pass) — inherits old policy_head weights
        self.board_head = nn.Linear(embed_dim, vocab_size)

        # Move prediction head (prefix pass) — new, separate
        self.policy_head = nn.Linear(embed_dim, vocab_size)

        # Value heads
        self.wl_head = ValueHead(embed_dim, n_buckets, value_hidden_size)
        self.d_head = ValueHead(embed_dim, n_buckets, value_hidden_size)

        # Fourier encoder for value injection
        self.fourier_encoder = FourierEncoder(embed_dim, num_fourier_freq)

        # Bucket center buffers (not trainable)
        self.register_buffer('wl_bucket_centers', make_wl_buckets(n_buckets, wl_sigma))
        self.register_buffer('d_bucket_centers', make_d_buckets(n_buckets))

    def forward(self, x, input_pos=None, mask_type="causal", block_id=None,
                wl_values=None, d_values=None, wl_positions=None, d_positions=None):
        """
        Args:
            x (torch.Tensor): input tokens [B, S]
            input_pos (torch.Tensor): position ids [B, S]
            mask_type (str): "causal" or "prefix"
            block_id (torch.Tensor): block IDs for prefix masking [B, S]
            wl_values (torch.Tensor): fourier input values at WL positions [B, S]
            d_values (torch.Tensor): fourier input values at D positions [B, S]
            wl_positions (torch.BoolTensor): mask for WL placeholder positions [B, S]
            d_positions (torch.BoolTensor): mask for D placeholder positions [B, S]

        Returns:
            h: hidden states [B, S, E] — heads applied outside
        """
        bsz, seq_len = x.shape

        if input_pos is None:
            input_pos = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # 1. Mask Generation
        if mask_type == "causal":
            mask = None
        else:
            if block_id is None:
                raise ValueError("block_id is required for prefix mask mode")

            # Causal base mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

            # Vectorized: same_block[b,i,j] = (block_id[b,i] == block_id[b,j])
            same_block = block_id.unsqueeze(-1) == block_id.unsqueeze(-2)  # [B, S, S]
            mask = causal_mask.unsqueeze(0) | same_block

        # 2. Token embedding
        h = self.tok_embedding(x)

        # 3. Override embeddings at WL/D placeholder positions with fourier encodings
        if wl_positions is not None and wl_positions.any():
            h[wl_positions] = self.fourier_encoder(wl_values[wl_positions]).to(h.dtype)
        if d_positions is not None and d_positions.any():
            h[d_positions] = self.fourier_encoder(d_values[d_positions]).to(h.dtype)

        # 4. Transformer Layers
        for layer in self.layers:
            h = layer(h, mask=mask, input_pos=input_pos)

        h = self.norm(h)
        return h

    def discretize_to_bucket(self, values, bucket_centers):
        """Map continuous values to nearest bucket center."""
        # values: (N,), bucket_centers: (B,)
        diffs = (values.unsqueeze(-1) - bucket_centers.unsqueeze(0)).abs()  # (N, B)
        indices = diffs.argmin(dim=-1)  # (N,)
        return bucket_centers[indices]  # (N,)

    @torch.no_grad()
    def predict_move(self, fen: str, temperature: float = 1.0, force_legal: bool = True) -> str:
        """Predicts the next move given a FEN string."""
        self.eval()
        device = next(self.parameters()).device

        # Convert FEN to tokens
        tokens = fen_to_position_tokens(fen)
        input_ids = torch.tensor([token_to_idx[t] for t in tokens], dtype=torch.long).unsqueeze(0).to(device)

        seq_len = input_ids.shape[1]
        block_id = torch.zeros(1, seq_len, dtype=torch.long, device=device)

        # Forward pass using "prefix" mask to get full bidirectional board context
        h = self(input_ids, mask_type="prefix", block_id=block_id)

        # Policy head at the last position (stm token)
        policy_logits = self.policy_head(h)
        last_logits = policy_logits[0, -1, :]

        if force_legal:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            vocab_legal_moves = []

            for move in legal_moves:
                uci = move.uci()
                if board.is_castling(move):
                    if uci == 'e1g1': uci = 'e1h1'
                    elif uci == 'e1c1': uci = 'e1a1'
                    elif uci == 'e8g8': uci = 'e8h8'
                    elif uci == 'e8c8': uci = 'e8a8'

                if uci in token_to_idx:
                    vocab_legal_moves.append(token_to_idx[uci])

            if vocab_legal_moves:
                legal_mask = torch.full_like(last_logits, float('-inf'))
                legal_mask[vocab_legal_moves] = 0
                last_logits = last_logits + legal_mask

        if temperature == 0.0:
            idx = torch.argmax(last_logits).item()
        else:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()

        move_str = idx_to_token[idx]

        replacements = {
            'e1h1': 'e1g1', 'e1a1': 'e1c1',
            'e8h8': 'e8g8', 'e8a8': 'e8c8'
        }
        if move_str in replacements:
            move_str = replacements[move_str]

        return move_str

    @torch.no_grad()
    def predict_move_and_value(self, fen: str, temperature: float = 1.0, force_legal: bool = True):
        """
        Predicts the next move and WDL values given a FEN string.

        Returns:
            move_str: UCI move string
            wdl: dict with keys 'win', 'draw', 'loss'
        """
        self.eval()
        device = next(self.parameters()).device

        # Convert FEN to tokens
        tokens = fen_to_position_tokens(fen)
        input_ids = torch.tensor([token_to_idx[t] for t in tokens], dtype=torch.long).unsqueeze(0).to(device)

        seq_len = input_ids.shape[1]
        block_id = torch.zeros(1, seq_len, dtype=torch.long, device=device)

        # Forward pass 1: prefix mask for move prediction
        h = self(input_ids, mask_type="prefix", block_id=block_id)

        # Policy head at stm position (last token of board)
        policy_logits = self.policy_head(h)
        last_logits = policy_logits[0, -1, :]

        if force_legal:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            vocab_legal_moves = []

            for move in legal_moves:
                uci = move.uci()
                if board.is_castling(move):
                    if uci == 'e1g1': uci = 'e1h1'
                    elif uci == 'e1c1': uci = 'e1a1'
                    elif uci == 'e8g8': uci = 'e8h8'
                    elif uci == 'e8c8': uci = 'e8a8'

                if uci in token_to_idx:
                    vocab_legal_moves.append(token_to_idx[uci])

            if vocab_legal_moves:
                legal_mask = torch.full_like(last_logits, float('-inf'))
                legal_mask[vocab_legal_moves] = 0
                last_logits = last_logits + legal_mask

        if temperature == 0.0:
            idx = torch.argmax(last_logits).item()
        else:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()

        move_str = idx_to_token[idx]

        # WL prediction: feed move token, get WL from hidden state at move position
        # Append move token + WL placeholder + D placeholder
        move_token_id = torch.tensor([[idx]], dtype=torch.long, device=device)
        wl_token_id = torch.tensor([[token_to_idx["wl_value"]]], dtype=torch.long, device=device)
        d_token_id = torch.tensor([[token_to_idx["d_value"]]], dtype=torch.long, device=device)

        extended_ids = torch.cat([input_ids, move_token_id, wl_token_id, d_token_id], dim=1)
        ext_seq_len = extended_ids.shape[1]

        # Block id: board tokens in block 0, move/wl/d are orphans
        ext_block_id = torch.zeros(1, ext_seq_len, dtype=torch.long, device=device)
        # Orphan IDs for move, wl, d positions
        ext_block_id[0, seq_len] = 1      # move
        ext_block_id[0, seq_len + 1] = 2  # wl
        ext_block_id[0, seq_len + 2] = 3  # d

        # WL prediction: run with no fourier override to get WL logits at move position
        # Then get WL bucket center, inject fourier for D prediction
        # For simplicity, run full forward with fourier injection for both
        # Step 1: Get WL prediction
        wl_positions = torch.zeros(1, ext_seq_len, dtype=torch.bool, device=device)
        d_positions = torch.zeros(1, ext_seq_len, dtype=torch.bool, device=device)
        wl_positions[0, seq_len + 1] = True
        d_positions[0, seq_len + 2] = True

        # First pass: no fourier injection, get WL from move position hidden state
        h_ext = self(extended_ids, mask_type="prefix", block_id=ext_block_id)

        # WL prediction at move token position (seq_len)
        h_at_move = h_ext[0, seq_len, :]  # (E,)
        wl_logits = self.wl_head(h_at_move.unsqueeze(0))  # (1, 100)
        wl_idx = torch.argmax(wl_logits, dim=-1)  # (1,)
        wl_value = self.wl_bucket_centers[wl_idx].item()

        # D prediction at WL placeholder position (seq_len + 1)
        # Need to re-run with fourier(WL) injected at WL position
        wl_fourier_input = torch.zeros(1, ext_seq_len, device=device)
        d_fourier_input = torch.zeros(1, ext_seq_len, device=device)
        wl_fourier_input[0, seq_len + 1] = self.wl_bucket_centers[wl_idx]

        h_ext2 = self(extended_ids, mask_type="prefix", block_id=ext_block_id,
                      wl_values=wl_fourier_input, d_values=d_fourier_input,
                      wl_positions=wl_positions, d_positions=d_positions)

        h_at_wl = h_ext2[0, seq_len + 1, :]  # (E,)
        d_logits = self.d_head(h_at_wl.unsqueeze(0))  # (1, 100)
        d_idx = torch.argmax(d_logits, dim=-1)
        d_value = self.d_bucket_centers[d_idx].item()

        # Reconstruct W, D, L
        wl = wl_value
        d = d_value
        w = (1 - d + wl) / 2
        l = (1 - d - wl) / 2
        # Clamp to valid range
        w = max(0.0, min(1.0, w))
        l = max(0.0, min(1.0, l))
        d = max(0.0, min(1.0, d))

        # Post-processing: castling conversion
        replacements = {
            'e1h1': 'e1g1', 'e1a1': 'e1c1',
            'e8h8': 'e8g8', 'e8a8': 'e8c8'
        }
        if move_str in replacements:
            move_str = replacements[move_str]

        return move_str, {"win": w, "draw": d, "loss": l}

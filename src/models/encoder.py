import torch
import torch.nn as nn
import chess
from torchtune.modules import MultiHeadAttention, FeedForward, RMSNorm, RotaryPositionalEmbeddings
from src.dataloader.data import fen_to_position_tokens
from src.models.vocab import token_to_idx, policy_index


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with bidirectional self-attention and RoPE."""
    
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, rope: RotaryPositionalEmbeddings, max_seq_len: int = 128):
        super().__init__()
        
        self.attn = MultiHeadAttention(
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
            is_causal=False  # Bidirectional attention
        )
        
        self.mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            down_proj=nn.Linear(4 * embed_dim, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, 4 * embed_dim, bias=False)
        )
        
        self.sa_norm = RMSNorm(dim=embed_dim)
        self.mlp_norm = RMSNorm(dim=embed_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, input_pos: torch.Tensor = None):
        # Pre-norm architecture
        h = x + self.attn(self.sa_norm(x), self.sa_norm(x), mask=mask, input_pos=input_pos)
        h = h + self.mlp(self.mlp_norm(h))
        return h


class ChessEncoder(nn.Module):
    """
    Encoder-only chess model with bidirectional attention and RoPE.
    
    Takes a FEN position tokenized and predicts the best move (policy only).
    Uses full bidirectional attention with proper padding masking.
    
    Board representation is fixed 68 tokens:
        - start_pos (1)
        - 64 board tokens (each square: empty or color_piece)
        - end_pos (1)
        - castling (1)
        - side_to_move (1)
    
    Uses RoPE for positional encoding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_policy_tokens: int = len(policy_index),
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        max_seq_len: int = 128
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_policy_tokens = num_policy_tokens
        self.max_seq_len = max_seq_len
        head_dim = embed_dim // num_heads
        
        # Token embedding only (RoPE handles positional info)
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # RoPE - shared across all layers
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, head_dim, rope, max_seq_len)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(dim=embed_dim)
        
        # Policy head only - predicts move from pooled representation
        self.policy_head = nn.Linear(embed_dim, num_policy_tokens)
        
        # Store special token ids
        self.pad_id = token_to_idx["pad"]
        self.start_pos_id = token_to_idx["start_pos"]
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            x: Input token ids [batch_size, seq_len]
            attention_mask: Boolean mask [batch_size, seq_len] where True = valid token, False = padding
            
        Returns:
            policy_logits: [batch_size, num_policy_tokens]
        """
        bsz, seq_len = x.shape
        device = x.device
        
        # Position ids for RoPE
        input_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        
        # Build attention mask for padding
        # torchtune expects mask shape [batch_size, seq_len, seq_len] 
        # where True = attend, False = mask out
        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, S] -> [B, S, S]
            # Each position can attend to all non-padded positions
            attn_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            attn_mask = None
        
        # Token embeddings only (RoPE is applied in attention)
        h = self.tok_embedding(x)
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h, mask=attn_mask, input_pos=input_pos)
        
        h = self.norm(h)
        
        # Pool: use the start_pos token representation (first token)
        # Since all sequences start with start_pos, we use position 0
        pooled = h[:, 0, :]  # [batch_size, embed_dim]
        
        # Policy prediction
        policy_logits = self.policy_head(pooled)  # [batch_size, num_policy_tokens]
        
        return policy_logits
    
    @torch.no_grad()
    def predict_move(self, fen: str, temperature: float = 1.0, force_legal: bool = True) -> str:
        """
        Predicts the best move given a FEN string.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Convert FEN to tokens
        tokens = fen_to_position_tokens(fen)
        input_ids = torch.tensor(
            [token_to_idx[t] for t in tokens], 
            dtype=torch.long
        ).unsqueeze(0).to(device)
        
        # Create attention mask (all True, no padding for single inference)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
        
        # Forward pass
        policy_logits = self(input_ids, attention_mask=attention_mask)
        logits = policy_logits[0]  # [num_policy_tokens]
        
        if force_legal:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            vocab_legal_moves = []
            
            for move in legal_moves:
                uci = move.uci()
                # Handle castling conversion
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
        
        # Apply temperature
        if temperature == 0.0:
            idx = torch.argmax(logits).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()
        
        move_str = policy_index[idx]
        
        # Post-processing: Convert model's castling to standard UCI
        replacements = {
            'e1h1': 'e1g1',
            'e1a1': 'e1c1',
            'e8h8': 'e8g8',
            'e8a8': 'e8c8'
        }
        if move_str in replacements:
            move_str = replacements[move_str]
        
        return move_str

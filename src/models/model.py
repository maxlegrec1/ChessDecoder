import torch
import torch.nn as nn
import chess
from torchtune.modules import TransformerSelfAttentionLayer, MultiHeadAttention, FeedForward, RMSNorm, RotaryPositionalEmbeddings
from src.dataloader.data import fen_to_position_tokens
from src.models.vocab import token_to_idx, idx_to_token


class ChessDecoder(nn.Module):
    """
    Decoder chess model with causal attention and RoPE.
    
    Board representation is fixed 68 tokens per position:
        - start_pos (1)
        - 64 board tokens (each square: empty or color_piece)
        - end_pos (1)
        - castling (1)
        - side_to_move (1)
    
    Uses RoPE for positional encoding.
    """
    
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, num_layers=12, max_seq_len=2048):
        super().__init__()
        
        head_dim = embed_dim // num_heads
        
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
                gate_proj=nn.Linear(embed_dim, 4 * embed_dim, bias=False),
                down_proj=nn.Linear(4 * embed_dim, embed_dim, bias=False),
                up_proj=nn.Linear(embed_dim, 4 * embed_dim, bias=False)
            )
            self.layers.append(TransformerSelfAttentionLayer(
                attn=attn, 
                mlp=mlp,
                sa_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim)
            ))
            
        self.norm = RMSNorm(dim=embed_dim)
        self.policy_head = nn.Linear(embed_dim, vocab_size)
        self.value_head = nn.Linear(embed_dim, 3)
        
    def forward(self, x, input_pos=None, mask_type="causal", block_id=None):
        """
        Args:
            x (torch.Tensor): input tokens [b x s]
            input_pos (torch.Tensor): position ids [b x s]
            mask_type (str): "causal" or "prefix"
            block_id (torch.Tensor): block IDs for each position [b x s]
                Required for prefix mask mode. Positions with the same block_id
                can attend to each other bidirectionally.
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

        # 2. Token embedding only (RoPE is applied in attention)
        h = self.tok_embedding(x)

        # 3. Transformer Layers
        for layer in self.layers:
            h = layer(h, mask=mask, input_pos=input_pos)

        h = self.norm(h)

        policy_logits = self.policy_head(h)
        value_logits = self.value_head(h)

        return policy_logits, value_logits

    @torch.no_grad()
    def predict_move(self, fen: str, temperature: float = 1.0, force_legal: bool = True) -> str:
        """
        Predicts the next move given a FEN string.
        """
        self.eval()
        device = next(self.parameters()).device

        # Convert FEN to tokens
        tokens = fen_to_position_tokens(fen)
        input_ids = torch.tensor([token_to_idx[t] for t in tokens], dtype=torch.long).unsqueeze(0).to(device)

        # Create block_id: all tokens belong to the same block (block 0)
        # since we have a single board position
        seq_len = input_ids.shape[1]
        block_id = torch.zeros(1, seq_len, dtype=torch.long, device=device)

        # Forward pass using "prefix" mask to get full bidirectional board context
        policy_logits, _ = self(input_ids, mask_type="prefix", block_id=block_id)
        
        # Get logits for the last token
        last_logits = policy_logits[0, -1, :]
        
        if force_legal:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            vocab_legal_moves = []
            
            for move in legal_moves:
                uci = move.uci()
                # Handle castling conversion for the mask
                if board.is_castling(move):
                    # e1g1 -> e1h1
                    if uci == 'e1g1': uci = 'e1h1'
                    elif uci == 'e1c1': uci = 'e1a1'
                    elif uci == 'e8g8': uci = 'e8h8'
                    elif uci == 'e8c8': uci = 'e8a8'
                
                if uci in token_to_idx:
                    vocab_legal_moves.append(token_to_idx[uci])
            
            if vocab_legal_moves:
                mask = torch.full_like(last_logits, float('-inf'))
                mask[vocab_legal_moves] = 0
                last_logits = last_logits + mask
        
        # Apply temperature
        if temperature == 0.0:
            idx = torch.argmax(last_logits).item()
        else:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            
        move_str = idx_to_token[idx]
        
        # Post-processing: Convert model's castling (e1h1) to standard UCI (e1g1)
        replacements = {
            'e1h1': 'e1g1',
            'e1a1': 'e1c1',
            'e8h8': 'e8g8',
            'e8a8': 'e8c8'
        }
        if move_str in replacements:
            move_str = replacements[move_str]
            
        return move_str

    @torch.no_grad()
    def predict_move_and_value(self, fen: str, temperature: float = 1.0, force_legal: bool = True) -> str:
        """
        Predicts the next move given a FEN string.
        """
        self.eval()
        device = next(self.parameters()).device

        # Convert FEN to tokens
        tokens = fen_to_position_tokens(fen)
        input_ids = torch.tensor([token_to_idx[t] for t in tokens], dtype=torch.long).unsqueeze(0).to(device)

        # Create block_id: all tokens belong to the same block (block 0)
        # since we have a single board position
        seq_len = input_ids.shape[1]
        block_id = torch.zeros(1, seq_len, dtype=torch.long, device=device)

        # Forward pass using "prefix" mask to get full bidirectional board context
        policy_logits, value_logits = self(input_ids, mask_type="prefix", block_id=block_id)
        
        # Get logits for the last token
        last_logits = policy_logits[0, -1, :]
        last_value_logits = value_logits[0, -1, :]
        value = torch.softmax(last_value_logits, dim=-1)
        if force_legal:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            vocab_legal_moves = []
            
            for move in legal_moves:
                uci = move.uci()
                # Handle castling conversion for the mask
                if board.is_castling(move):
                    # e1g1 -> e1h1
                    if uci == 'e1g1': uci = 'e1h1'
                    elif uci == 'e1c1': uci = 'e1a1'
                    elif uci == 'e8g8': uci = 'e8h8'
                    elif uci == 'e8c8': uci = 'e8a8'
                
                if uci in token_to_idx:
                    vocab_legal_moves.append(token_to_idx[uci])
            
            if vocab_legal_moves:
                mask = torch.full_like(last_logits, float('-inf'))
                mask[vocab_legal_moves] = 0
                last_logits = last_logits + mask
        
        # Apply temperature
        if temperature == 0.0:
            idx = torch.argmax(last_logits).item()
        else:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            
        move_str = idx_to_token[idx]
        
        # Post-processing: Convert model's castling (e1h1) to standard UCI (e1g1)
        replacements = {
            'e1h1': 'e1g1',
            'e1a1': 'e1c1',
            'e8h8': 'e8g8',
            'e8a8': 'e8c8'
        }
        if move_str in replacements:
            move_str = replacements[move_str]
            
        return move_str,value

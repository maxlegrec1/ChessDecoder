import torch
import torch.nn as nn
import chess
from torchtune.modules import TransformerSelfAttentionLayer, MultiHeadAttention, FeedForward, RMSNorm
from src.dataloader.data import fen_to_position_tokens
from src.models.vocab import token_to_idx, idx_to_token, vocab, squares

class ChessDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, num_layers=12, max_seq_len=2048):
        super().__init__()
        # Reworked Embedding: Token + Absolute Position + Square
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Precompute mapping from token_id to square_id (0-63)
        token_to_square = torch.full((vocab_size,), 64, dtype=torch.long) # 64 = No Square
        square_to_idx = {s: i for i, s in enumerate(squares)}
        for i, token in enumerate(vocab):
            # piece_square tokens are formatted as "color_piece_square"
            parts = token.split('_')
            if len(parts) >= 3 and parts[-1] in square_to_idx:
                token_to_square[i] = square_to_idx[parts[-1]]
        self.register_buffer("token_to_square_map", token_to_square)
        self.square_embedding = nn.Embedding(65, embed_dim) # 64 squares + 1 null

        from src.models.vocab import policy_index
        self.start_pos_id = token_to_idx["start_pos"]
        self.num_policy_tokens = len(policy_index)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=embed_dim // num_heads,
                q_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                k_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                v_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=None, # RoPE is removed in favor of absolute+square embeddings
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
        
    def forward(self, x, input_pos=None, mask_type="causal"):
        """
        Args:
            x (torch.Tensor): input tokens [b x s]
            input_pos (torch.Tensor): position ids [b x s]
            mask_type (str): "causal" or "prefix"
        """
        bsz, seq_len = x.shape
        
        if input_pos is None:
            input_pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # 1. Mask Generation
        if mask_type == "causal":
            mask = None
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            mask = mask.unsqueeze(0).repeat(bsz, 1, 1)
            
            is_start = (x == self.start_pos_id)
            is_move = (x < self.num_policy_tokens)
            
            for b in range(bsz):
                starts = is_start[b].nonzero().flatten()
                moves = is_move[b].nonzero().flatten()
                for s in starts:
                    s_idx = s.item()
                    moves_after = moves[moves > s_idx]
                    if len(moves_after) > 0:
                        m_idx = moves_after[0].item()
                        mask[b, s_idx:m_idx, s_idx:m_idx] = True
                    else:
                        mask[b, s_idx:, s_idx:] = True
        
        # 2. Embedding Injection
        # Token embedding + Global sequence position + Explicit Square embedding
        h = self.tok_embedding(x)
        h = h + self.pos_embedding(input_pos)
        
        sq_ids = self.token_to_square_map[x]
        h = h + self.square_embedding(sq_ids)
        
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
        
        # Forward pass using "prefix" mask to get full bidirectional board context
        policy_logits, _ = self(input_ids, mask_type="prefix")
        
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

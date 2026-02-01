import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn_map import apm_map, apm_out
import math
from .encoding_simple import encode_fen_to_tensor, encode_moves_to_tensor
from .vocab import policy_index
from typing import Union, List, Optional
import bulletchess

class Gating(nn.Module):
    def __init__(self, features_shape, additive=True, init_value=None):
        super(Gating, self).__init__()
        self.additive = additive
        if init_value is None:
            init_value = 0 if self.additive else 1
        
        self.gate = nn.Parameter(torch.full(features_shape, float(init_value)))
        if not self.additive:
            self.gate.register_hook(lambda grad: torch.clamp(grad, min=0))

    def forward(self, x):
        if self.additive:
            return x + self.gate
        else:
            return x * self.gate

def ma_gating(x, in_features):
    x = Gating(in_features, additive=False)(x)
    x = Gating(in_features, additive=True)(x)
    return x

class RMSNorm(nn.Module):
    def __init__(self, in_features, scale=True):
        super(RMSNorm, self).__init__()
        self.scale = scale
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
        x_normalized = x / rms
        if self.scale:
            return x_normalized * self.gamma
        return x_normalized

class ApplyAttentionPolicyMap(nn.Module):
    def __init__(self):
        super(ApplyAttentionPolicyMap, self).__init__()
        # Register as buffers so they move with the model when .to(device) is called
        # Use same names as before for backward compatibility with saved models
        self.register_buffer('fc1', torch.from_numpy(apm_map).float())
        self.register_buffer('idx', torch.from_numpy(apm_out).long())

    def forward(self, logits, pp_logits):
        logits = torch.cat([logits.reshape(-1, 64 * 64),
                            pp_logits.reshape(-1, 8 * 24)],
                           dim=1)
        
        batch_size = logits.size(0)
        idx = self.idx.unsqueeze(0).expand(batch_size, -1)
        
        return torch.gather(logits, 1, idx)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class CustomMHA(nn.Module):
    def __init__(self, emb_size, d_model, num_heads, dropout=0.0, use_bias_qkv=True, use_bias_out=True,
                 use_smolgen=True, smol_hidden_channels=32, smol_hidden_sz=256, smol_gen_sz=256, smol_activation='swish'):
        super(CustomMHA, self).__init__()
        assert d_model % num_heads == 0
        self.emb_size = emb_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(emb_size, d_model, bias=use_bias_qkv)
        self.wk = nn.Linear(emb_size, d_model, bias=use_bias_qkv)
        self.wv = nn.Linear(emb_size, d_model, bias=use_bias_qkv)
        self.out_proj = nn.Linear(d_model, emb_size, bias=use_bias_out)
        self.attn_dropout = nn.Dropout(dropout)
        # Optional Smolgen components
        self.smol_compress = None
        self.smol_hidden1 = None
        self.smol_hidden1_ln = None
        self.smol_gen_from = None
        self.smol_gen_from_ln = None
        self.smol_weight_gen = None
        if use_smolgen:
            self.smol_compress = nn.Linear(emb_size, smol_hidden_channels, bias=False)
            self.smol_hidden1 = nn.Linear(64 * smol_hidden_channels, smol_hidden_sz, bias=True)
            self.smol_hidden1_ln = nn.LayerNorm(smol_hidden_sz, eps=1e-3)
            self.smol_gen_from = nn.Linear(smol_hidden_sz, num_heads * smol_gen_sz, bias=True)
            self.smol_gen_from_ln = nn.LayerNorm(num_heads * smol_gen_sz, eps=1e-3)
            self.smol_weight_gen = nn.Linear(smol_gen_sz, 64 * 64, bias=False)
        self.smol_activation = smol_activation

    def _shape(self, x):
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, return_attn=False):
        # x: (B, L, emb_size)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self._shape(q)  # (B, H, L, D)
        k = self._shape(k)
        v = self._shape(v)
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=x.dtype, device=x.device))
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale
        # Add Smolgen weights if present
        smol_w = None
        if self.smol_compress is not None:
            b, l, _ = x.shape
            compressed = self.smol_compress(x)  # (B, L, hidden_channels)
            compressed = compressed.reshape(b, l * compressed.shape[-1])  # (B, 64*hidden_channels)
            hidden_pre = self.smol_hidden1(compressed)
            hidden = F.silu(hidden_pre) if self.smol_activation == 'swish' else F.silu(hidden_pre)
            hidden_ln = self.smol_hidden1_ln(hidden)
            gen_from_pre = self.smol_gen_from(hidden_ln)
            gen_from_act = F.silu(gen_from_pre) if self.smol_activation == 'swish' else F.silu(gen_from_pre)
            gen_from = self.smol_gen_from_ln(gen_from_act)
            gen_from = gen_from.view(b, self.num_heads, -1)  # (B, H, gen_sz)
            smol_w = self.smol_weight_gen(gen_from)  # (B, H, 64*64)
            smol_w = smol_w.view(b, self.num_heads, l, l)
            attn_logits = attn_logits + smol_w
        # Numerically stable softmax matching TF (float32, subtract max)
        attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]
        attn_weights = torch.exp(attn_logits)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        out = self.out_proj(attn_output)
        if return_attn:
            return out, attn_weights, smol_w, attn_logits
        return out

class FFN(nn.Module):
    def __init__(self, emb_size, dff, activation=Mish(), omit_other_biases=False):
        super(FFN, self).__init__()
        self.dense1 = nn.Linear(emb_size, dff, bias=not omit_other_biases)
        self.activation = activation
        self.dense2 = nn.Linear(dff, emb_size, bias=not omit_other_biases)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, emb_size, d_model, num_heads, dff, dropout_rate, encoder_layers, skip_first_ln=False, encoder_rms_norm=False, omit_qkv_biases=False, omit_other_biases=False,
                 use_smolgen=True, smol_hidden_channels=32, smol_hidden_sz=256, smol_gen_sz=256, smol_activation='swish'):
        super(EncoderLayer, self).__init__()
        self.mha = CustomMHA(emb_size, d_model, num_heads, dropout=dropout_rate, use_bias_qkv=not omit_qkv_biases, use_bias_out=not omit_other_biases,
                             use_smolgen=use_smolgen, smol_hidden_channels=smol_hidden_channels, smol_hidden_sz=smol_hidden_sz, smol_gen_sz=smol_gen_sz, smol_activation=smol_activation)
        self.ffn = FFN(emb_size, dff, omit_other_biases=omit_other_biases)
        
        self.norm1 = RMSNorm(emb_size) if encoder_rms_norm else nn.LayerNorm(emb_size, eps=0.001)
        self.norm2 = RMSNorm(emb_size) if encoder_rms_norm else nn.LayerNorm(emb_size, eps=0.001)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.alpha = (2. * encoder_layers)**-0.25
        self.skip_first_ln = skip_first_ln

    def forward(self, x):
        attn_output = self.mha(x)
        attn_output = self.dropout1(attn_output)
        
        out1 = x + attn_output * self.alpha
        if not self.skip_first_ln:
            out1 = self.norm1(out1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        
        out2 = self.norm2(out1 + ffn_output * self.alpha)
        return out2

class PolicyHead(nn.Module):
    def __init__(self, pol_embedding_size, policy_d_model, opponent=False):
        super(PolicyHead, self).__init__()
        self.opponent = opponent
        self.wq = nn.Linear(pol_embedding_size, policy_d_model)
        self.wk = nn.Linear(pol_embedding_size, policy_d_model)
        self.ppo = nn.Linear(policy_d_model, 4, bias=False)
        self.apply_map = ApplyAttentionPolicyMap()

    def forward(self, x):
        if self.opponent:
            x = torch.flip(x, [1])

        queries = self.wq(x)
        keys = self.wk(x)

        matmul_qk = torch.matmul(queries, keys.transpose(-2, -1))
        
        dk = torch.sqrt(torch.tensor(keys.shape[-1], dtype=keys.dtype, device=keys.device))
        promotion_keys = keys[:, -8:, :]
        promotion_offsets = self.ppo(promotion_keys).transpose(-2,-1) * dk
        promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4, :]

        n_promo_logits = matmul_qk[:, -16:-8, -8:]
        q_promo_logits = (n_promo_logits + promotion_offsets[:, 0:1, :]).unsqueeze(3)
        r_promo_logits = (n_promo_logits + promotion_offsets[:, 1:2, :]).unsqueeze(3)
        b_promo_logits = (n_promo_logits + promotion_offsets[:, 2:3, :]).unsqueeze(3)
        promotion_logits = torch.cat([q_promo_logits, r_promo_logits, b_promo_logits], axis=3).reshape(-1, 8, 24)

        promotion_logits = promotion_logits / dk
        policy_attn_logits = matmul_qk / dk

        return self.apply_map(policy_attn_logits, promotion_logits)

class ValueHead(nn.Module):
    def __init__(self, embedding_size, val_embedding_size, default_activation=Mish()):
        super(ValueHead, self).__init__()
        self.embedding = nn.Linear(embedding_size, val_embedding_size)
        self.activation = default_activation
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(val_embedding_size * 64, 128)
        self.dense2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x

class BT4(nn.Module):
    def __init__(self, embedding_size=1024, embedding_dense_sz=512, encoder_layers=15, encoder_d_model=1024, encoder_heads=32, encoder_dff=1536, dropout_rate=0.0, pol_embedding_size=1024, policy_d_model=1024, val_embedding_size=128, default_activation=Mish(),
                 use_smolgen=True, smol_hidden_channels=32, smol_hidden_sz=256, smol_gen_sz=256, smol_activation='swish'):
        super(BT4, self).__init__()
        self.embedding_dense_sz = embedding_dense_sz
        # DeepNorm alpha used in embedding residual; default uses provided encoder_layers
        self.deepnorm_alpha = (2. * encoder_layers) ** -0.25
        
        self.embedding_preprocess = nn.Linear(64*12, 64*self.embedding_dense_sz)
        self.embedding = nn.Linear(112 + self.embedding_dense_sz, embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight) # Explicitly set initializer
        nn.init.zeros_(self.embedding.bias)

        self.embedding_ln = nn.LayerNorm(embedding_size, eps=0.001)
        
        self.gating_mult = Gating((64, embedding_size), additive=False)
        self.gating_add = Gating((64, embedding_size), additive=True)

        self.embedding_ffn = FFN(embedding_size, encoder_dff)
        self.embedding_ffn_ln = nn.LayerNorm(embedding_size, eps=0.001)
        
        self.encoder_layers_list = nn.ModuleList([
            EncoderLayer(embedding_size, encoder_d_model, encoder_heads, encoder_dff, dropout_rate, encoder_layers,
                         use_smolgen=use_smolgen, smol_hidden_channels=smol_hidden_channels, smol_hidden_sz=smol_hidden_sz, smol_gen_sz=smol_gen_sz, smol_activation=smol_activation)
            for _ in range(encoder_layers)
        ])
        
        self.policy_embedding = nn.Linear(embedding_size, pol_embedding_size)
        self.policy_head = PolicyHead(pol_embedding_size, policy_d_model)
        self.value_head_winner = ValueHead(embedding_size, val_embedding_size)
        self.value_head_q = ValueHead(embedding_size, val_embedding_size)
        self.activation = default_activation
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Keras' glorot_normal is equivalent to PyTorch's xavier_normal_
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: (batch, 112, 8, 8)
        flow = x.permute(0, 2, 3, 1).reshape(-1, 64, 112)
        
        pos_info = flow[..., :12]
        pos_info_flat = pos_info.reshape(-1, 64 * 12)
        
        pos_info_processed = self.embedding_preprocess(pos_info_flat)
        pos_info = pos_info_processed.reshape(-1, 64, self.embedding_dense_sz)
        
        flow = torch.cat([flow, pos_info], dim=-1)
        
        flow = self.embedding(flow)
        
        flow = self.activation(flow)
        
        flow = self.embedding_ln(flow)

        flow = self.gating_mult(flow)
        flow = self.gating_add(flow)
        
        ffn_dense1_pre = self.embedding_ffn.dense1(flow)
        ffn_dense1 = self.embedding_ffn.activation(ffn_dense1_pre)
        ffn_output = self.embedding_ffn.dense2(ffn_dense1)
        
        residual = flow + ffn_output * self.deepnorm_alpha
        flow = self.embedding_ffn_ln(residual)
        
        for i, layer in enumerate(self.encoder_layers_list):
            flow = layer(flow)
        
        policy_tokens = self.policy_embedding(flow)
        policy_tokens = self.activation(policy_tokens)
        
        policy_logits = self.policy_head(policy_tokens)
        
        value_winner = self.value_head_winner(flow)
        value_q = self.value_head_q(flow)

        return policy_logits, value_winner, value_q
    
    def get_move_from_fen_no_thinking(self, fen_or_moves: Union[str, List[str]], T: float, device: str = None, **kwargs) -> str:
        """
        Predict a move from a FEN position or move history without thinking/search.
        
        Args:
            fen_or_moves: Either a FEN string representing the chess position, or a list of UCI moves
            T: Temperature for sampling (0.0 = deterministic/argmax, >0.0 = stochastic)
            device: Device to run the model on (if None, uses model's device)
        
        Returns:
            UCI move string (e.g., 'e2e4')
        """
        # Detect device from model if not provided
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
        
        # Determine if input is FEN string or list of moves
        if isinstance(fen_or_moves, str):
            # FEN string input
            fen = fen_or_moves
            is_black_to_move = fen.split()[1] == 'b'
            input_tensor_112, legal_moves_mask = encode_fen_to_tensor(fen)
            castling_rights = fen.split()[2] if len(fen.split()) > 2 else ""
        elif isinstance(fen_or_moves, list):
            # List of UCI moves input
            move_history = fen_or_moves
            input_tensor_112, legal_moves_mask = encode_moves_to_tensor(move_history)
            # Create board to check if black is to move and for castling rights
            board = bulletchess.Board()
            for mv in move_history:
                move = bulletchess.Move.from_uci(mv)
                board.apply(move)
            is_black_to_move = (board.turn == bulletchess.BLACK)
            fen_parts = board.fen().split()
            castling_rights = fen_parts[2] if len(fen_parts) > 2 else ""
        else:
            raise ValueError("Input must be a FEN string or a list of UCI moves")
        
        input_tensor_112 = input_tensor_112.to(device, non_blocking=True)
        
        self.eval()
        with torch.inference_mode():
            policy_logits,_,_ = self.forward(input_tensor_112)
        
        # Apply legal moves mask without in-place ops (inference tensor)
        logits0 = policy_logits[0] + torch.from_numpy(legal_moves_mask).to(policy_logits.device)
        
        if T == 0.0:
            # Deterministic: return best move
            best_move_idx = torch.argmax(logits0).item()
            uci_move = policy_index[best_move_idx]
        else:
            # Stochastic sampling with temperature
            # Apply temperature scaling
            scaled_logits = logits0 / T
            # Apply softmax to get probabilities
            probs = F.softmax(scaled_logits, dim=0)
            # Sample from the distribution
            move_idx = torch.multinomial(probs, 1).item()
            uci_move = policy_index[move_idx]
        
        # If black is to move, the board was mirrored during encoding, so we need to mirror the move back
        # Mirror ranks: 1↔8, 2↔7, 3↔6, 4↔5 (keep file letters the same)
        if is_black_to_move:
            def mirror_rank(rank_char):
                rank = int(rank_char)
                return str(9 - rank)
            
            # UCI format: e2e4, e7e8q, etc.
            if len(uci_move) >= 4:
                from_file = uci_move[0]
                from_rank = uci_move[1]
                to_file = uci_move[2]
                to_rank = uci_move[3]
                promo = uci_move[4:] if len(uci_move) > 4 else ""
                
                uci_move = from_file + mirror_rank(from_rank) + to_file + mirror_rank(to_rank) + promo
        
        # Convert castling moves from king-to-rook-square format to standard castling format
        # Only if castling rights are available (check FEN castling rights)
        # Check and convert white castling moves
        if uci_move == "e1h1" and "K" in castling_rights:
            uci_move = "e1g1"
        elif uci_move == "e1a1" and "Q" in castling_rights:
            uci_move = "e1c1"
        # Check and convert black castling moves
        elif uci_move == "e8h8" and "k" in castling_rights:
            uci_move = "e8g8"
        elif uci_move == "e8a8" and "q" in castling_rights:
            uci_move = "e8c8"
        
        return uci_move
    
    def batch_get_moves_from_fens(self, fens: List[str], T: float, device: str = None, use_fp16: bool = False) -> List[str]:
        """
        Get moves for multiple FEN positions using batched inference.
        
        Args:
            fens: List of FEN strings representing chess positions
            T: Temperature for sampling (0.0 = deterministic/argmax, >0.0 = stochastic)
            device: Device to run the model on (if None, uses model's device)
        
        Returns:
            List of UCI move strings
        """
        if not fens:
            return []
        
        # Detect device from model if not provided
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
        
        batch_size = len(fens)
        
        # Batch encode all FENs
        input_tensors = []
        legal_moves_masks = []
        is_black_to_move_list = []
        castling_rights_list = []
        
        for fen in fens:
            input_tensor, legal_mask = encode_fen_to_tensor(fen)
            input_tensors.append(input_tensor.squeeze(0))  # Remove batch dim
            legal_moves_masks.append(legal_mask)
            is_black_to_move_list.append(fen.split()[1] == 'b')
            castling_rights_list.append(fen.split()[2] if len(fen.split()) > 2 else "")
        
        # Stack into batch tensor: (batch_size, 112, 8, 8)
        batch_tensor = torch.stack(input_tensors).to(device, non_blocking=True)
        if use_fp16 and device.type == 'cuda':
            batch_tensor = batch_tensor.half()
        
        # Run batched inference
        self.eval()
        with torch.inference_mode():
            if use_fp16 and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    policy_logits,_,_ = self.forward(batch_tensor)
            else:
                policy_logits,_,_ = self.forward(batch_tensor)
        
        # Process each position in the batch
        moves = []
        for i in range(batch_size):
            # Apply legal moves mask
            logits = policy_logits[i] + torch.from_numpy(legal_moves_masks[i]).to(policy_logits.device, dtype=policy_logits.dtype)
            
            # Sample move
            if T == 0.0:
                best_move_idx = torch.argmax(logits).item()
                uci_move = policy_index[best_move_idx]
            else:
                scaled_logits = logits / T
                probs = F.softmax(scaled_logits, dim=0)
                move_idx = torch.multinomial(probs, 1).item()
                uci_move = policy_index[move_idx]
            
            # Mirror move if black is to move
            if is_black_to_move_list[i]:
                def mirror_rank(rank_char):
                    rank = int(rank_char)
                    return str(9 - rank)
                
                if len(uci_move) >= 4:
                    from_file = uci_move[0]
                    from_rank = uci_move[1]
                    to_file = uci_move[2]
                    to_rank = uci_move[3]
                    promo = uci_move[4:] if len(uci_move) > 4 else ""
                    
                    uci_move = from_file + mirror_rank(from_rank) + to_file + mirror_rank(to_rank) + promo
            
            # Convert castling moves
            castling_rights = castling_rights_list[i]
            if uci_move == "e1h1" and "K" in castling_rights:
                uci_move = "e1g1"
            elif uci_move == "e1a1" and "Q" in castling_rights:
                uci_move = "e1c1"
            elif uci_move == "e8h8" and "k" in castling_rights:
                uci_move = "e8g8"
            elif uci_move == "e8a8" and "q" in castling_rights:
                uci_move = "e8c8"
            
            moves.append(uci_move)
        
        return moves
    
    def batch_get_moves_from_move_lists(self, move_lists: List[List[str]], T: float, device: str = None, use_fp16: bool = False, fens: Optional[List[str]] = None):
        """
        Get moves for multiple move histories using batched inference.
        
        Args:
            move_lists: List of move sequences, where each sequence is a list of UCI moves
            T: Temperature for sampling (0.0 = deterministic/argmax, >0.0 = stochastic)
            device: Device to run the model on (if None, uses model's device)
            fens: Optional list of FEN strings that represent the board state prior to 
                  applying the corresponding move list. When provided, each move history
                  is applied starting from the supplied FEN instead of the standard initial position.
        
        Returns:
            List of UCI move strings
        """
        if not move_lists:
            return []
        
        # Detect device from model if not provided
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
        
        batch_size = len(move_lists)
        
        if fens is not None and len(fens) != len(move_lists):
            raise ValueError("Length of fens must match length of move_lists when provided.")
        
        # Batch encode all move histories
        input_tensors = []
        legal_moves_masks = []
        is_black_to_move_list = []
        castling_rights_list = []
        
        for idx, move_history in enumerate(move_lists):
            starting_fen = fens[idx] if fens is not None else None
            input_tensor, legal_mask = encode_moves_to_tensor(move_history, starting_fen=starting_fen)
            input_tensors.append(input_tensor.squeeze(0))  # Remove batch dim
            legal_moves_masks.append(legal_mask)
            
            board = bulletchess.Board.from_fen(starting_fen) if starting_fen is not None else bulletchess.Board()
            for mv in move_history:
                move = bulletchess.Move.from_uci(mv)
                board.apply(move)
            is_black_to_move_list.append(board.turn == bulletchess.BLACK)
            fen_parts = board.fen().split()
            castling_rights_list.append(fen_parts[2] if len(fen_parts) > 2 else "")
        
        # Stack into batch tensor: (batch_size, 112, 8, 8)
        batch_tensor = torch.stack(input_tensors).to(device, non_blocking=True)
        if use_fp16 and device.type == 'cuda':
            batch_tensor = batch_tensor.half()
        
        # Run batched inference
        self.eval()
        with torch.inference_mode():
            if use_fp16 and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    policy_logits,_,_ = self.forward(batch_tensor)
            else:
                policy_logits,_,_ = self.forward(batch_tensor)
        
        # Process each position in the batch
        moves = []
        for i in range(batch_size):
            # Apply legal moves mask
            logits = policy_logits[i] + torch.from_numpy(legal_moves_masks[i]).to(policy_logits.device, dtype=policy_logits.dtype)
            
            # Sample move
            if T == 0.0:
                best_move_idx = torch.argmax(logits).item()
                uci_move = policy_index[best_move_idx]
            else:
                scaled_logits = logits / T
                probs = F.softmax(scaled_logits, dim=0)
                move_idx = torch.multinomial(probs, 1).item()
                uci_move = policy_index[move_idx]
            
            # Mirror move if black is to move
            if is_black_to_move_list[i]:
                def mirror_rank(rank_char):
                    rank = int(rank_char)
                    return str(9 - rank)
                
                if len(uci_move) >= 4:
                    from_file = uci_move[0]
                    from_rank = uci_move[1]
                    to_file = uci_move[2]
                    to_rank = uci_move[3]
                    promo = uci_move[4:] if len(uci_move) > 4 else ""
                    
                    uci_move = from_file + mirror_rank(from_rank) + to_file + mirror_rank(to_rank) + promo
            
            # Convert castling moves
            castling_rights = castling_rights_list[i]
            if uci_move == "e1h1" and "K" in castling_rights:
                uci_move = "e1g1"
            elif uci_move == "e1a1" and "Q" in castling_rights:
                uci_move = "e1c1"
            elif uci_move == "e8h8" and "k" in castling_rights:
                uci_move = "e8g8"
            elif uci_move == "e8a8" and "q" in castling_rights:
                uci_move = "e8c8"
            
            moves.append(uci_move)
        return moves

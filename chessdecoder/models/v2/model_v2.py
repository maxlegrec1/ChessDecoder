"""ChessDecoderV2 — the full V2 architecture (Phase A).

V2 = V1's training regime with two internal swaps (see markdowns/11):

  board_i (68 tok) ─► BoardEncoder (bidir, full attn) ─► z_i  (16 latents)
  causal stream:  [z_0(16), m1, wl1, d1, z_1(16), m2, wl2, d2, ...]
                    │  CausalLatentDecoder  (one clean causal mask -> flash)
                    │  heads: policy / thinking_policy / wl / d
                    ▼
  (z_i, move_{i+1}) ─► TransitionHead (parallel, all 64 squares + stm + castle)
                    ▼  board_{i+1}  (engine-free, no chess library at rollout)

Token routing (markdowns/11 §12.0):
  - the 68-token board block (incl. castling + side-to-move) goes through the
    encoder -> 16 latent vectors;
  - move / wl / d / control tokens stay single embeddings in the causal stream.

Note vs the plan: ``fen_to_position_tokens`` does **not** tokenize en-passant
(only start_pos + 64 squares + end_pos + castling + stm). So the absolute
transition head predicts 64 squares (13-way) + stm (2) + castling (16); there
is no ep target to predict. This keeps the transition head exactly invertible
to the model's own board tokenization.
"""
from __future__ import annotations

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import (MultiHeadAttention, FeedForward, RMSNorm,
                               RotaryPositionalEmbeddings,
                               TransformerSelfAttentionLayer)

from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.v2.layers import (
    TransformerEncoderLayer, FourierEncoder)
from chessdecoder.models.v2.encoder_mode import PerceiverPool
from chessdecoder.models.v2.value_buckets import CELL_WDL, N_CELLS
from chessdecoder.models.vocab import (
    token_to_idx, idx_to_token, policy_index,
    move_vocab_size, move_idx_to_full_idx, move_token_to_idx,
    piece_tokens, castling_tokens,
)

# ---------------------------------------------------------------------------
# Transition (world-model) class spaces — derived from the model's own
# board tokenization so the head is an exact inverse of fen_to_position_tokens.
# ---------------------------------------------------------------------------
SQUARE_VOCAB = ["empty"] + list(piece_tokens)          # 13 classes / square
STM_VOCAB = ["white_to_move", "black_to_move"]          # 2 classes
CASTLING_VOCAB = list(castling_tokens)                  # 16 classes
N_SQUARE_CLASSES = len(SQUARE_VOCAB)                    # 13
N_STM_CLASSES = len(STM_VOCAB)                          # 2
N_CASTLING_CLASSES = len(CASTLING_VOCAB)                # 16

# full-vocab token id -> transition class id (used by Phase B to build targets)
SQUARE_TOKID_TO_CLASS = {token_to_idx[t]: i for i, t in enumerate(SQUARE_VOCAB)}
STM_TOKID_TO_CLASS = {token_to_idx[t]: i for i, t in enumerate(STM_VOCAB)}
CASTLING_TOKID_TO_CLASS = {token_to_idx[t]: i for i, t in enumerate(CASTLING_VOCAB)}

# inverse: transition class id -> full-vocab token id (engine-free rollout:
# decode the parallel head back into the model's own 68-token board layout —
# the exact inverse of fen_to_position_tokens, no chess library involved)
SQUARE_CLASS_TO_TOKID = [token_to_idx[t] for t in SQUARE_VOCAB]
STM_CLASS_TO_TOKID = [token_to_idx[t] for t in STM_VOCAB]
CASTLING_CLASS_TO_TOKID = [token_to_idx[t] for t in CASTLING_VOCAB]

NUM_LATENTS_DEFAULT = 16


def board_tokens_to_transition_targets(board_ids: torch.Tensor):
    """board_ids: [B,68] full-vocab ids (output of fen_to_position_tokens).

    Returns (sq[B,64] long, stm[B] long, cas[B] long) class targets — the
    exact supervision the transition head must reproduce for board_{t+1}.
    Layout: idx 0 = start_pos, 1..64 = squares a1..h8, 65 = end_pos,
    66 = castling, 67 = side_to_move.
    """
    sq_tok = board_ids[:, 1:65]
    cas_tok = board_ids[:, 66]
    stm_tok = board_ids[:, 67]
    dev = board_ids.device

    def remap(t, table):
        keys = torch.tensor(list(table.keys()), device=dev)
        vals = torch.tensor(list(table.values()), device=dev)
        lut = torch.full((len(token_to_idx),), -1, dtype=torch.long, device=dev)
        lut[keys] = vals
        return lut[t]

    return (remap(sq_tok, SQUARE_TOKID_TO_CLASS),
            remap(stm_tok, STM_TOKID_TO_CLASS),
            remap(cas_tok, CASTLING_TOKID_TO_CLASS))


class BoardEncoder(nn.Module):
    """68 board tokens -> k latent vectors. Bidirectional RoPE transformer +
    Perceiver pooling. No padding (boards are always exactly 68 tokens), so
    attention runs the fused/flash path (mask=None) — validated bit-identical
    in the encoder-mode benchmark."""

    def __init__(self, tok_embedding: nn.Embedding, embed_dim: int,
                 num_heads: int, num_layers: int, num_latents: int,
                 max_seq_len: int = 68, d_ff: int = 1536):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.tok_embedding = tok_embedding          # shared with the decoder stream
        self.num_latents = num_latents
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, head_dim, rope,
                                    max_seq_len, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim=embed_dim)
        self.latent_queries = nn.Parameter(torch.randn(num_latents, embed_dim) * 0.02)
        self.pool = PerceiverPool(embed_dim, num_heads)

    def forward(self, board_ids: torch.Tensor) -> torch.Tensor:
        # board_ids: [N,68] -> latents: [N,k,E]
        n, s = board_ids.shape
        pos = torch.arange(s, device=board_ids.device).unsqueeze(0).expand(n, -1)
        h = self.tok_embedding(board_ids)
        for layer in self.layers:
            h = layer(h, mask=None, input_pos=pos)        # flash (no padding)
        h = self.norm(h)
        latents = self.latent_queries.unsqueeze(0).expand(n, -1, -1)
        return self.pool(latents, h, key_valid_mask=None)  # [N,k,E]


class CausalLatentDecoder(nn.Module):
    """Pure-causal transformer over the mixed (latents | token) embedding
    stream. ``mask=None`` + ``is_causal=True`` -> SDPA flash kernel (this is
    the V2 win: V1's dense prefix mask is gone)."""

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int,
                 max_seq_len: int = 2048, d_ff: int = 1536):
        super().__init__()
        head_dim = embed_dim // num_heads
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = MultiHeadAttention(
                embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=num_heads,
                head_dim=head_dim,
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

    def forward(self, inputs_embeds: torch.Tensor,
                input_pos: torch.Tensor = None) -> torch.Tensor:
        b, s, _ = inputs_embeds.shape
        if input_pos is None:
            input_pos = torch.arange(s, device=inputs_embeds.device).unsqueeze(0).expand(b, -1)
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, mask=None, input_pos=input_pos)   # is_causal -> flash
        return self.norm(h)


class TransitionHead(nn.Module):
    """Parallel absolute world model T(z_t, move) -> board_{t+1}.

    64 square-queries + 1 stm-query + 1 castling-query cross-attend to
    [z_t (k latents), move_emb]; per-query linear classifiers. One forward,
    no causal mask, no autoregression. Chess transitions are deterministic so
    independent per-square heads can represent the single target mode exactly.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.n_q = 64 + 1 + 1                                  # squares + stm + castling
        self.queries = nn.Parameter(torch.randn(self.n_q, embed_dim) * 0.02)
        self.pool = PerceiverPool(embed_dim, num_heads)
        self.sq_head = nn.Linear(embed_dim, N_SQUARE_CLASSES)
        self.stm_head = nn.Linear(embed_dim, N_STM_CLASSES)
        self.cas_head = nn.Linear(embed_dim, N_CASTLING_CLASSES)

    def forward(self, latents: torch.Tensor, move_emb: torch.Tensor):
        # latents: [B,k,E]  move_emb: [B,E]
        b = latents.shape[0]
        ctx = torch.cat([latents, move_emb.unsqueeze(1)], dim=1)   # [B,k+1,E]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)            # [B,66,E]
        q = self.pool(q, ctx, key_valid_mask=None)                 # flash
        sq = self.sq_head(q[:, :64, :])                            # [B,64,13]
        stm = self.stm_head(q[:, 64, :])                           # [B,2]
        cas = self.cas_head(q[:, 65, :])                           # [B,16]
        return {"square": sq, "stm": stm, "castling": cas}


class WDLHead(nn.Module):
    """Encoder-side pure position evaluator (markdowns/12).

    One learned query cross-attends the board's 16 latents (same Perceiver
    pattern as TransitionHead) -> a categorical over the 2-D simplex value
    grid (``value_buckets.N_CELLS``: Q-concentrated, D-uniform). Reads `z_i`
    directly, never the decoder -> structurally leak-free, flash-friendly,
    uniform on every board. Multimodal-capable: can put mass on two distinct
    WDL hypotheses (e.g. "winning or losing, unsure which"). Q and D are read
    off the categorical's mean, so they jointly "see each other".
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.pool = PerceiverPool(embed_dim, num_heads)
        self.head = nn.Linear(embed_dim, N_CELLS)      # 2-D simplex buckets
        self.register_buffer("cell_wdl", CELL_WDL.clone())   # [N_CELLS,3]

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B,k,E] -> bucket logits [B,N_CELLS]
        b = latents.shape[0]
        q = self.query.unsqueeze(0).expand(b, -1, -1)              # [B,1,E]
        q = self.pool(q, latents, key_valid_mask=None)             # [B,1,E]
        return self.head(q[:, 0, :])                               # [B,N_CELLS]

    def mean_wdl(self, logits: torch.Tensor) -> torch.Tensor:
        """bucket logits [...,N_CELLS] -> mean WDL [...,3] (the value to act
        on: E[(W,D,L)] under the predicted categorical)."""
        p = torch.softmax(logits.float(), dim=-1)
        return p @ self.cell_wdl.to(p.dtype)


class ChessDecoderV2(nn.Module):
    """Full V2 model. One nn.Module, one optimizer, joint loss (markdowns/11).

    The encoder is a submodule: it receives gradients from the decoder's
    policy/value loss (through the latents) and from the transition loss,
    exactly like a vision encoder inside a VLM. Vocab / policy_index / value
    buckets are reused unchanged from V1 so eval/ and rl/ stay compatible.
    """

    def __init__(self, vocab_size, embed_dim=1024, num_heads=16,
                 num_encoder_layers=10, num_decoder_layers=12,
                 num_latents=NUM_LATENTS_DEFAULT, board_max_seq_len=68,
                 decoder_max_seq_len=2048, d_ff=1536,
                 num_fourier_freq=128,
                 # accepted-but-unused legacy kwargs (old bucketed value head):
                 n_buckets=100, value_hidden_size=256, wl_sigma=0.4):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_latents = num_latents
        self.embed_dim = embed_dim

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)

        self.board_encoder = BoardEncoder(
            self.tok_embedding, embed_dim, num_heads, num_encoder_layers,
            num_latents, board_max_seq_len, d_ff)
        self.decoder = CausalLatentDecoder(
            embed_dim, num_heads, num_decoder_layers, decoder_max_seq_len, d_ff)
        self.transition_head = TransitionHead(embed_dim, num_heads)
        # Encoder-side joint WDL position evaluator (markdowns/12).
        self.wdl_head = WDLHead(embed_dim, num_heads)

        self.policy_head = nn.Linear(embed_dim, move_vocab_size)
        self.thinking_policy_head = nn.Linear(embed_dim, move_vocab_size)
        self.fourier_encoder = FourierEncoder(embed_dim, num_fourier_freq)

        self.pad_id = token_to_idx["pad"]

    # ---- building blocks -------------------------------------------------

    def encode_boards(self, board_ids: torch.Tensor) -> torch.Tensor:
        """[N,68] -> [N,k,E]. Batch every board of a game in one parallel
        call (no autoregression across boards — the big throughput win)."""
        return self.board_encoder(board_ids)

    def embed_value(self, value: torch.Tensor) -> torch.Tensor:
        """Fourier features for a scalar [N] -> [N,E]."""
        return self.fourier_encoder(value)

    def embed_wdl(self, wdl: torch.Tensor) -> torch.Tensor:
        """Value-token injection (markdowns/12): WDL [N,3] -> [N,E] as
        Fourier(Q) + Fourier(D), with Q = W-L and D = D. One token carrying
        the full joint value, re-injected so later moves are eval-aware."""
        w, d, l = wdl[:, 0], wdl[:, 1], wdl[:, 2]
        return self.fourier_encoder(w - l) + self.fourier_encoder(d)

    @torch.no_grad()
    def predict_wdl(self, board_ids: torch.Tensor) -> torch.Tensor:
        """[N,68] board ids -> mean WDL [N,3] (pure position eval; the value
        to act on, E[(W,D,L)] under the predicted simplex categorical)."""
        return self.wdl_head.mean_wdl(self.wdl_head(self.encode_boards(board_ids)))

    # ---- engine-free rollout (Phase D) -----------------------------------

    def decode_transition(self, out: dict) -> torch.Tensor:
        """Transition-head logits dict -> board_ids [B,68] in the model's own
        tokenization (start_pos, 64 squares, end_pos, castling, stm). Exact
        inverse of board_tokens_to_transition_targets — no chess library, so
        rollouts stay engine-free."""
        dev = out["square"].device
        B = out["square"].shape[0]
        sq = out["square"].argmax(-1)                            # [B,64]
        stm = out["stm"].argmax(-1)                              # [B]
        cas = out["castling"].argmax(-1)                         # [B]
        sq_lut = torch.tensor(SQUARE_CLASS_TO_TOKID, device=dev)
        stm_lut = torch.tensor(STM_CLASS_TO_TOKID, device=dev)
        cas_lut = torch.tensor(CASTLING_CLASS_TO_TOKID, device=dev)
        ids = torch.empty(B, 68, dtype=torch.long, device=dev)
        ids[:, 0] = token_to_idx["start_pos"]
        ids[:, 1:65] = sq_lut[sq]
        ids[:, 65] = token_to_idx["end_pos"]
        ids[:, 66] = cas_lut[cas]
        ids[:, 67] = stm_lut[stm]
        return ids

    @torch.no_grad()
    def rollout_next(self, latents: torch.Tensor, move_emb: torch.Tensor):
        """(z_t [B,k,E], move_emb [B,E]) -> (board_ids [B,68], z_{t+1}
        [B,k,E]) entirely from the model: imagine the next board with the
        parallel transition head, re-tokenize, re-encode. The generation loop
        for thinking/RL appends z_{t+1} to the causal stream — no engine."""
        out = self.transition_head(latents, move_emb)
        board_ids = self.decode_transition(out)
        return board_ids, self.encode_boards(board_ids)

    @staticmethod
    def scheduled_sample_latents(gt_latents: torch.Tensor,
                                 pred_latents: torch.Tensor,
                                 p: float) -> torch.Tensor:
        """Per-board Bernoulli(p) swap of ground-truth board latents for the
        transition head's own predicted latents (exposure-bias fix). p ramps
        0 -> p_max over finetuning so the model trains on the boards it will
        actually roll out at inference. gt/pred: [N,k,E]."""
        if p <= 0.0:
            return gt_latents
        n = gt_latents.shape[0]
        use_pred = (torch.rand(n, device=gt_latents.device) < p)
        m = use_pred.view(n, 1, 1).to(gt_latents.dtype)
        return (1 - m) * gt_latents + m * pred_latents

    # ---- engine-free autoregressive generation (Phase E rollout / F-multi) -

    @torch.no_grad()
    def generate_v2(self, root_board_ids: torch.Tensor, max_plies: int,
                    temperature: float = 1.0):
        """Engine-free game rollout from a root board (no chess library, no
        C++ engine). Loop: decode causal stream -> sample move from
        policy_head (capture log-prob) -> transition_head imagines the next
        board -> re-encode -> append latents -> repeat.

        root_board_ids: [1,68]. Returns a list of per-ply dicts
        {move_sub_id, move_logprob, board_ids[1,68]} — the V2-native rollout
        record GRPO consumes (log-prob recompute re-points to the same
        policy-head positions) and `predict_move_n` reads the last move from.
        Greedy/argmax when temperature==0."""
        self.eval()
        dev = next(self.parameters()).device
        k = self.num_latents
        z = self.encode_boards(root_board_ids)                  # [1,k,E]
        stream = [z]                                            # list of [1,*,E]
        plies = []
        for _ in range(max_plies):
            seq = torch.cat(stream, dim=1)                      # [1,S,E]
            h = self.decoder(seq)
            logits = self.policy_head(h[0, -1, :])              # at last latent
            logp = torch.log_softmax(logits, dim=-1)
            if temperature == 0.0:
                sub = int(torch.argmax(logits))
            else:
                sub = int(torch.multinomial(
                    torch.softmax(logits / temperature, -1), 1))
            move_full = move_idx_to_full_idx[sub]
            move_emb = self.tok_embedding(
                torch.tensor([move_full], device=dev))          # [1,E]
            board_ids, z_next = self.rollout_next(z, move_emb)  # engine-free
            plies.append({"move_sub_id": sub,
                          "move_logprob": float(logp[sub]),
                          "board_ids": board_ids})
            # append move embedding + next-board latents to the causal stream
            stream.append(move_emb.unsqueeze(1))                # [1,1,E]
            stream.append(z_next)                               # [1,k,E]
            z = z_next
        return plies

    @torch.no_grad()
    def predict_move_n(self, root_fen: str, n_history_plies: int = 0,
                       temperature: float = 0.0) -> str:
        """Multi-ply eval (Phase F-multi): engine-free roll forward
        ``n_history_plies`` then return the next predicted move (UCI). With
        n_history_plies==0 this matches ``predict_move``."""
        tokens = fen_to_position_tokens(root_fen)
        bid = torch.tensor([[token_to_idx[t] for t in tokens]],
                           dtype=torch.long,
                           device=next(self.parameters()).device)
        plies = self.generate_v2(bid, max_plies=n_history_plies + 1,
                                 temperature=temperature)
        sub = plies[-1]["move_sub_id"]
        s = idx_to_token[move_idx_to_full_idx[sub]]
        return {'e1h1': 'e1g1', 'e1a1': 'e1c1',
                'e8h8': 'e8g8', 'e8a8': 'e8c8'}.get(s, s)

    # ---- inference: degenerate single-board case (eval compat) -----------

    @torch.no_grad()
    def predict_move(self, fen: str, temperature: float = 1.0,
                     force_legal: bool = True) -> str:
        """Single-position FEN -> best move. Same semantics/signature as
        ChessEncoder.predict_move so PytorchModelAdapter / elo_eval work
        unchanged. Sequence = z_0 (k latents); policy read at last latent."""
        self.eval()
        device = next(self.parameters()).device
        tokens = fen_to_position_tokens(fen)
        board_ids = torch.tensor([[token_to_idx[t] for t in tokens]],
                                 dtype=torch.long, device=device)
        latents = self.encode_boards(board_ids)                 # [1,k,E]
        h = self.decoder(latents)                                # [1,k,E]
        logits = self.policy_head(h[0, -1, :])                   # [move_vocab]

        if force_legal:
            board = chess.Board(fen)
            legal = []
            for move in board.legal_moves:
                uci = move.uci()
                if board.is_castling(move):
                    uci = {'e1g1': 'e1h1', 'e1c1': 'e1a1',
                           'e8g8': 'e8h8', 'e8c8': 'e8a8'}.get(uci, uci)
                if uci in move_token_to_idx:
                    legal.append(move_token_to_idx[uci])
            if legal:
                m = torch.full_like(logits, float('-inf'))
                m[legal] = 0
                logits = logits + m

        if temperature == 0.0:
            sub = torch.argmax(logits).item()
        else:
            sub = torch.multinomial(torch.softmax(logits / temperature, -1), 1).item()
        move_str = idx_to_token[move_idx_to_full_idx[sub]]
        return {'e1h1': 'e1g1', 'e1a1': 'e1c1',
                'e8h8': 'e8g8', 'e8a8': 'e8c8'}.get(move_str, move_str)

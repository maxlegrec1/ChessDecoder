"""
Elo evaluation for thinking ChessDecoder models.

Wraps the thinking inference (start_think -> variations -> end_think -> final_move)
into a predict_move interface compatible with model_vs_stockfish.
"""

import argparse
import chess
import time
import torch
import torch.nn.functional as F

from src.eval.elo_eval import model_vs_stockfish
from src.models.vocab import (
    vocab_size, token_to_idx, idx_to_token,
    board_idx_to_full_idx, move_idx_to_full_idx,
    board_token_to_idx, move_token_to_idx,
)
from src.dataloader.data import fen_to_position_tokens
from scripts.think import load_model, to_standard_uci, sample_token

# Castling: standard UCI -> pseudo (model) UCI
_STANDARD_TO_PSEUDO = {"e1g1": "e1h1", "e1c1": "e1a1", "e8g8": "e8h8", "e8c8": "e8a8"}
_PSEUDO_TO_STANDARD = {v: k for k, v in _STANDARD_TO_PSEUDO.items()}

# Board sub-vocab indices for structural tokens
_BOARD_END_VAR_IDX = board_token_to_idx["end_var"]
_BOARD_END_THINK_IDX = board_token_to_idx["end_think"]

# Move tokens (for validation)
_MOVE_TOKENS = set()
for _t in idx_to_token.values():
    if len(_t) == 4 and _t[0] in "abcdefgh" and _t[1] in "12345678" and _t[2] in "abcdefgh" and _t[3] in "12345678":
        _MOVE_TOKENS.add(_t)
    elif len(_t) == 5 and _t[0] in "abcdefgh" and _t[1] in "12345678" and _t[2] in "abcdefgh" and _t[3] in "12345678" and _t[4] in "qrbn":
        _MOVE_TOKENS.add(_t)


def _sample_move_from_logits(logits, temperature, fen=None, force_legal=True):
    """Sample a move from move-sub-vocab logits, optionally masking to legal moves."""
    if force_legal and fen is not None:
        board = chess.Board(fen)
        legal_indices = []
        for move in board.legal_moves:
            uci = move.uci()
            if board.is_castling(move):
                uci = _STANDARD_TO_PSEUDO.get(uci, uci)
            if uci in move_token_to_idx:
                legal_indices.append(move_token_to_idx[uci])
        if legal_indices:
            mask = torch.full_like(logits, float('-inf'))
            mask[legal_indices] = 0
            logits = logits + mask

    move_sub_idx = sample_token(logits, temperature)
    full_idx = move_idx_to_full_idx[move_sub_idx]
    tok = idx_to_token[full_idx]
    return to_standard_uci(tok)


class ThinkingModelWrapper:
    """Wraps a ChessDecoder model to use thinking inference for predict_move."""

    def __init__(self, model, device, max_seq_len, think_temperature=0.0):
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len
        self.think_temperature = think_temperature
        self.total_tokens = 0
        self.total_time = 0.0

    @torch.no_grad()
    def predict_move(self, fen, temperature=0.0):
        """Run thinking inference and return the final move as standard UCI."""
        t0 = time.time()
        model = self.model
        device = self.device
        max_seq_len = self.max_seq_len
        temp = self.think_temperature

        token_ids = []
        block_ids = []
        wl_entries = []
        d_entries = []
        next_block = [0]
        orphan_ctr = [10000]

        def orphan():
            orphan_ctr[0] += 1
            return orphan_ctr[0]

        def append(tok_id, bid):
            token_ids.append(tok_id)
            block_ids.append(bid)

        def full():
            return len(token_ids) >= max_seq_len

        def done(move):
            self.total_tokens += len(token_ids)
            self.total_time += time.time() - t0
            # Save for inspection
            self.last_token_ids = list(token_ids)
            self.last_wl_entries = list(wl_entries)
            self.last_d_entries = list(d_entries)
            return move

        def prefix_forward():
            S = len(token_ids)
            inp = torch.tensor([token_ids], dtype=torch.long, device=device)
            blk = torch.tensor([block_ids], dtype=torch.long, device=device)
            wl_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
            d_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
            wl_val = torch.zeros(1, S, dtype=torch.float16, device=device)
            d_val = torch.zeros(1, S, dtype=torch.float16, device=device)
            for p, v in wl_entries:
                wl_pos[0, p] = True; wl_val[0, p] = v
            for p, v in d_entries:
                d_pos[0, p] = True; d_val[0, p] = v
            return model(inp, mask_type="prefix", block_id=blk,
                         wl_values=wl_val, d_values=d_val,
                         wl_positions=wl_pos, d_positions=d_pos)

        def causal_forward():
            S = len(token_ids)
            inp = torch.tensor([token_ids], dtype=torch.long, device=device)
            wl_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
            d_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
            wl_val = torch.zeros(1, S, dtype=torch.float16, device=device)
            d_val = torch.zeros(1, S, dtype=torch.float16, device=device)
            for p, v in wl_entries:
                wl_pos[0, p] = True; wl_val[0, p] = v
            for p, v in d_entries:
                d_pos[0, p] = True; d_val[0, p] = v
            return model(inp, mask_type="causal",
                         wl_values=wl_val, d_values=d_val,
                         wl_positions=wl_pos, d_positions=d_pos)

        def predict_wl(move_pos):
            h = prefix_forward()
            logits = model.wl_head(h[0, move_pos:move_pos+1, :])
            wl_idx = torch.argmax(logits, dim=-1)
            return model.wl_bucket_centers[wl_idx].item()

        def predict_d(wl_pos):
            h = prefix_forward()
            logits = model.d_head(h[0, wl_pos:wl_pos+1, :])
            d_idx = torch.argmax(logits, dim=-1)
            return model.d_bucket_centers[d_idx].item()

        def emit_wl_d(move_pos):
            wl = predict_wl(move_pos)
            wl_pos_idx = len(token_ids)
            append(token_to_idx["wl_value"], orphan())
            wl_entries.append((wl_pos_idx, wl))
            d = predict_d(wl_pos_idx)
            d_pos_idx = len(token_ids)
            append(token_to_idx["d_value"], orphan())
            d_entries.append((d_pos_idx, d))

        def emit_board():
            bid = next_block[0]; next_block[0] += 1
            for _ in range(68):
                if full():
                    break
                h = causal_forward()
                logits = model.board_head(h)[0, -1, :]
                board_sub_idx = torch.argmax(logits).item()
                full_idx = board_idx_to_full_idx[board_sub_idx]
                append(full_idx, bid)

        # 1. Root board
        root_tokens = fen_to_position_tokens(fen)
        bid = next_block[0]; next_block[0] += 1
        for t in root_tokens:
            append(token_to_idx[t], bid)
        if full():
            return done(self._fallback_move(fen, temperature))

        # 2. start_think
        append(token_to_idx["start_think"], orphan())

        # 3. Autoregressive thinking loop
        state = "MOVE"
        first_root_move = None

        while not full():
            if state == "MOVE":
                pos = len(token_ids) - 1
                h = prefix_forward()
                logits = model.thinking_policy_head(h)[0, pos, :]
                move_sub_idx = sample_token(logits, temp)
                full_idx = move_idx_to_full_idx[move_sub_idx]
                tok = idx_to_token[full_idx]
                append(full_idx, orphan())
                # Track first root move as fallback
                if first_root_move is None and tok in _MOVE_TOKENS:
                    first_root_move = to_standard_uci(tok)
                state = "WL_D"

            elif state == "WL_D":
                if full():
                    break
                move_pos = len(token_ids) - 1
                emit_wl_d(move_pos)
                state = "BOARD"

            elif state == "BOARD":
                if full():
                    break
                emit_board()
                state = "AFTER_BOARD"

            elif state == "AFTER_BOARD":
                if full():
                    break
                h = causal_forward()
                logits = model.board_head(h)[0, -1, :]
                board_sub_idx = sample_token(logits, temp)
                if board_sub_idx == _BOARD_END_VAR_IDX:
                    full_idx = board_idx_to_full_idx[board_sub_idx]
                    append(full_idx, orphan())
                    state = "AFTER_END_VAR"
                else:
                    state = "MOVE"

            elif state == "AFTER_END_VAR":
                if full():
                    break
                h = causal_forward()
                logits = model.board_head(h)[0, -1, :]
                board_sub_idx = sample_token(logits, temp)
                if board_sub_idx == _BOARD_END_THINK_IDX:
                    full_idx = board_idx_to_full_idx[board_sub_idx]
                    append(full_idx, orphan())
                    state = "FINAL"
                else:
                    state = "MOVE"

            elif state == "FINAL":
                if full():
                    break
                # Sample final move from policy_head with legal-move masking
                pos = len(token_ids) - 1
                h = prefix_forward()
                logits = model.policy_head(h)[0, pos, :]
                return done(_sample_move_from_logits(logits, temperature, fen=fen, force_legal=True))

        # Thinking ran out of space before reaching FINAL
        # Fallback: use first root move if we got one, else direct policy head
        if first_root_move is not None:
            # Validate it's legal
            board = chess.Board(fen)
            try:
                move = chess.Move.from_uci(first_root_move)
                if move in board.legal_moves:
                    return done(first_root_move)
            except ValueError:
                pass
        return done(self._fallback_move(fen, temperature))

    @torch.no_grad()
    def _fallback_move(self, fen, temperature):
        """Direct policy head on root position (no thinking)."""
        return self.model.predict_move(fen, temperature=temperature, force_legal=True)


def main():
    parser = argparse.ArgumentParser(description="Elo evaluation for thinking ChessDecoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--elo", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for the final move selection")
    parser.add_argument("--think-temperature", type=float, default=0.0,
                        help="Temperature for thinking (variations/board generation)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    model, max_seq_len = load_model(args.checkpoint, device)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max seq len: {max_seq_len}")

    wrapper = ThinkingModelWrapper(
        model, device, max_seq_len,
        think_temperature=args.think_temperature,
    )

    model_vs_stockfish(
        model=wrapper,
        model1_name="thinking-decoder",
        num_games=args.num_games,
        temperature=args.temperature,
        elo=args.elo,
    )


if __name__ == "__main__":
    main()

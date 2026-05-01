"""
Thinking inference for ChessDecoder.

Generates a thinking sequence autoregressively, printing tokens as they come.
Stops when max_seq_len is reached.
"""

import argparse
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import (vocab_size, token_to_idx, idx_to_token,
                              board_idx_to_full_idx, move_idx_to_full_idx,
                              board_token_to_idx)
from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.utils.uci import normalize_castling


@dataclass
class ThinkingResult:
    """Captured output of run_thinking().

    Used by the C++/Python parity harness (scripts/verify_decoder_parity.py)
    and by tests/test_engine_parity.py. Token indices are full-vocab.
    final_* fields are None if generation hit max_seq_len before FINAL.

    For temp=0 parity: compare token_ids and wl_bucket_indices/d_bucket_indices
    (argmax indices) for exact match. The wl_entries/d_entries float values
    are subject to FP16 noise.
    """
    token_ids: list = field(default_factory=list)
    block_ids: list = field(default_factory=list)
    wl_entries: list = field(default_factory=list)         # [(position, value)]
    d_entries: list = field(default_factory=list)          # [(position, value)]
    wl_bucket_indices: list = field(default_factory=list)  # [(position, bucket_idx)]
    d_bucket_indices: list = field(default_factory=list)   # [(position, bucket_idx)]
    final_move: str = None                                  # UCI string
    final_wl: float = None
    final_d: float = None
    final_wl_bucket: int = None
    final_d_bucket: int = None
    ended_thinking: bool = False
    truncated: bool = False                                 # hit max_seq_len

_PIECE_SYMBOLS = {
    "white_king": "K", "white_queen": "Q", "white_rook": "R",
    "white_bishop": "B", "white_knight": "N", "white_pawn": "P",
    "black_king": "k", "black_queen": "q", "black_rook": "r",
    "black_bishop": "b", "black_knight": "n", "black_pawn": "p",
}

# Tokens that indicate a move was just generated
_MOVE_TOKENS = set()
for _t in idx_to_token.values():
    if len(_t) == 4 and _t[0] in "abcdefgh" and _t[1] in "12345678" and _t[2] in "abcdefgh" and _t[3] in "12345678":
        _MOVE_TOKENS.add(_t)
    elif len(_t) == 5 and _t[0] in "abcdefgh" and _t[1] in "12345678" and _t[2] in "abcdefgh" and _t[3] in "12345678" and _t[4] in "qrbn":
        _MOVE_TOKENS.add(_t)


def sample_token(logits, temperature):
    """Sample from logits, returns sub-vocab index."""
    if temperature <= 0:
        return torch.argmax(logits).item()
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


# Pre-compute board sub-vocab lookup for signal tokens
_BOARD_END_VAR_IDX = board_token_to_idx["end_var"]
_BOARD_END_THINK_IDX = board_token_to_idx["end_think"]
_BOARD_CONTINUE_VAR_IDX = board_token_to_idx["continue_var"]
_BOARD_NEW_VARIATION_IDX = board_token_to_idx["new_variation"]
_BOARD_GENERIC_MOVE_IDX = board_token_to_idx["generic_move"]


def board_tokens_to_fen(tokens):
    """Reconstruct FEN from 68 token strings."""
    if len(tokens) != 68:
        return "<invalid>"
    squares = tokens[1:65]
    fen_rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            piece = squares[rank * 8 + file]
            sym = _PIECE_SYMBOLS.get(piece)
            if sym:
                if empty > 0:
                    row += str(empty); empty = 0
                row += sym
            else:
                empty += 1
        if empty > 0:
            row += str(empty)
        fen_rows.append(row)
    castling = tokens[66] if tokens[66] != "no_castling_rights" else "-"
    stm = "w" if tokens[67] == "white_to_move" else "b"
    return f"{'/'.join(fen_rows)} {stm} {castling} - 0 1"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    max_seq_len = config["model"]["max_seq_len"]

    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=max_seq_len,
        d_ff=config["model"].get("d_ff"),
        n_buckets=config["model"].get("n_buckets", 100),
        value_hidden_size=config["model"].get("value_hidden_size", 256),
        num_fourier_freq=config["model"].get("num_fourier_freq", 128),
        wl_sigma=config["model"].get("wl_sigma", 0.4),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    # Save bucket centers before .half() — they must stay FP32 for
    # exact match with C++ (which loads them from FP32 files).
    wl_centers = model.wl_bucket_centers.clone()
    d_centers = model.d_bucket_centers.clone()
    model.half()
    model.wl_bucket_centers = wl_centers
    model.d_bucket_centers = d_centers
    model.eval()
    return model, max_seq_len


def reconstruct_wdl(wl, d):
    w = max(0.0, min(1.0, (1 - d + wl) / 2))
    l = max(0.0, min(1.0, (1 - d - wl) / 2))
    d = max(0.0, min(1.0, d))
    return w, d, l


@torch.no_grad()
def run_thinking(model, fen, temperature=0.0, device="cuda", max_seq_len=1024,
                 verbose=True):
    """Run the Python thinking-decoder state machine for a single FEN.

    Args:
        model: ChessDecoder loaded via load_model() (already half + eval).
        fen: starting FEN.
        temperature: sampling temperature; 0.0 = argmax (deterministic).
        device: torch device.
        max_seq_len: hard cap on emitted tokens; truncates mid-state.
        verbose: print state-machine progress to stdout (CLI-style).
                 Set False when calling from parity harnesses or tests.

    Returns:
        ThinkingResult with full-vocab token_ids, wl/d entries, final move,
        and final wl/d (None if generation truncated before FINAL).
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    token_ids = []
    block_ids = []
    wl_entries = []
    d_entries = []
    next_block = [0]
    orphan_ctr = [10000]
    final_move = None
    final_wl = None
    final_d = None
    ended_thinking = False

    def orphan():
        orphan_ctr[0] += 1
        return orphan_ctr[0]

    def append(tok_id, bid):
        token_ids.append(tok_id)
        block_ids.append(bid)

    def full():
        return len(token_ids) >= max_seq_len

    # FourierEncoder weights are FP16 after model.half(); WL/D values must
    # match dtype to avoid the FP32 @ FP16 dtype mismatch.
    val_dtype = next(model.parameters()).dtype

    def prefix_forward():
        S = len(token_ids)
        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
        blk = torch.tensor([block_ids], dtype=torch.long, device=device)
        wl_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
        d_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
        wl_val = torch.zeros(1, S, dtype=val_dtype, device=device)
        d_val = torch.zeros(1, S, dtype=val_dtype, device=device)
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
        wl_val = torch.zeros(1, S, dtype=val_dtype, device=device)
        d_val = torch.zeros(1, S, dtype=val_dtype, device=device)
        for p, v in wl_entries:
            wl_pos[0, p] = True; wl_val[0, p] = v
        for p, v in d_entries:
            d_pos[0, p] = True; d_val[0, p] = v
        return model(inp, mask_type="causal",
                     wl_values=wl_val, d_values=d_val,
                     wl_positions=wl_pos, d_positions=d_pos)

    def sample_move(head, pos):
        """Sample a move from policy/thinking_policy head. Returns full vocab idx."""
        h = prefix_forward()
        logits = head(h)[0, pos, :]  # (move_vocab_size,)
        move_sub_idx = sample_token(logits, temperature)
        return move_idx_to_full_idx[move_sub_idx]  # full vocab idx

    wl_bucket_indices = []
    d_bucket_indices = []

    def predict_wl(move_pos):
        h = prefix_forward()
        logits = model.wl_head(h[0, move_pos:move_pos+1, :])
        wl_idx = torch.argmax(logits, dim=-1).item()
        return wl_idx, model.wl_bucket_centers[wl_idx].item()

    def predict_d(wl_pos):
        h = prefix_forward()
        logits = model.d_head(h[0, wl_pos:wl_pos+1, :])
        d_idx = torch.argmax(logits, dim=-1).item()
        return d_idx, model.d_bucket_centers[d_idx].item()

    def emit_wl_d(move_pos):
        """Predict WL/D, append fourier tokens, print values."""
        wl_idx, wl = predict_wl(move_pos)
        wl_pos_idx = len(token_ids)
        append(token_to_idx["wl_value"], orphan())
        wl_entries.append((wl_pos_idx, wl))
        wl_bucket_indices.append((wl_pos_idx, wl_idx))

        d_idx, d = predict_d(wl_pos_idx)
        d_pos_idx = len(token_ids)
        append(token_to_idx["d_value"], orphan())
        d_entries.append((d_pos_idx, d))
        d_bucket_indices.append((d_pos_idx, d_idx))

        w, dr, l = reconstruct_wdl(wl, d)
        vprint(f"  wl/d -> W={w:.2f} D={dr:.2f} L={l:.2f}")
        return wl_idx, wl, d_idx, d

    def emit_board():
        """Generate 68 board tokens via board_head, print reconstructed FEN."""
        bid = next_block[0]; next_block[0] += 1
        tok_strs = []
        for i in range(68):
            if full():
                break
            h = causal_forward()
            logits = model.board_head(h)[0, -1, :]  # (board_vocab_size,)
            board_sub_idx = torch.argmax(logits).item()
            full_idx = board_idx_to_full_idx[board_sub_idx]
            append(full_idx, bid)
            tok_strs.append(idx_to_token[full_idx])
        if verbose:
            fen_str = board_tokens_to_fen(tok_strs) if len(tok_strs) == 68 else "<truncated>"
            vprint(f"  board: {fen_str}")

    # ===== State machine =====
    #
    # The sequence format the model was trained on:
    #   [board_68] start_think
    #     root_move wl d [board_68] pv_move wl d [board_68] ... end_var
    #     root_move wl d [board_68] ... end_var
    #   end_think final_move wl d
    #
    # States track where we are in this format. At each step we use the
    # appropriate head, append the token(s), and advance. We stop when
    # we hit max_seq_len -- wherever that happens to be.

    # 1. Root board (deterministic from FEN)
    root_tokens = fen_to_position_tokens(fen)
    bid = next_block[0]; next_block[0] += 1
    for t in root_tokens:
        append(token_to_idx[t], bid)
    vprint(f"root: {fen}")
    if full():
        vprint(f"Sequence length: {len(token_ids)} / {max_seq_len}")
        return ThinkingResult(
            token_ids=token_ids, block_ids=block_ids,
            wl_entries=wl_entries, d_entries=d_entries,
            truncated=True,
        )

    # 2. start_think
    append(token_to_idx["start_think"], orphan())
    vprint("start_think")

    # 3. Autoregressive generation
    #    State: what we expect to generate next
    #    MOVE     -> sample a move (root_move or pv_move) from thinking_policy_head
    #    WL_D     -> predict wl/d from value heads
    #    BOARD    -> generate 68 board tokens from board_head
    #    AFTER_BOARD -> decide: pv_move or end_var (use board_head to pick)
    #    END_VAR  -> append end_var, then decide: new variation or end_think
    #    FINAL    -> sample final move from policy_head + wl/d, then done

    state = "MOVE"  # first thing after start_think is a root_move
    var_idx = 0
    final_wl_bucket = None
    final_d_bucket = None

    while not full():
        if state == "MOVE":
            # Sample move from thinking_policy_head at last position
            pos = len(token_ids) - 1
            idx = sample_move(model.thinking_policy_head, pos)
            tok = idx_to_token[idx]
            append(idx, orphan())
            if verbose:
                if tok in _MOVE_TOKENS:
                    # Check if this is a root_move (after start_think or end_var)
                    prev_tok = idx_to_token[token_ids[-2]] if len(token_ids) >= 2 else ""
                    if prev_tok in ("start_think", "end_var"):
                        var_idx += 1
                        vprint(f"  variation {var_idx} root_move: {normalize_castling(tok)}")
                    else:
                        vprint(f"    pv_move: {normalize_castling(tok)}")
                else:
                    vprint(f"  unexpected token from thinking_policy_head: {tok}")
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
            # Use board_head (causal) to decide: end_var or continue_var (PV continuation)
            if full():
                break
            h = causal_forward()
            logits = model.board_head(h)[0, -1, :]  # (board_vocab_size,)
            board_sub_idx = sample_token(logits, temperature)

            if board_sub_idx == _BOARD_END_VAR_IDX:
                full_idx = board_idx_to_full_idx[board_sub_idx]
                append(full_idx, orphan())
                vprint("  end_var")
                state = "AFTER_END_VAR"
            else:
                # continue_var (or any non-end_var): continue PV with thinking_policy_head
                # Don't append continue_var to context (it's just a signal)
                state = "MOVE"

        elif state == "AFTER_END_VAR":
            # Use board_head to decide: end_think or new_variation (start another variation)
            if full():
                break
            h = causal_forward()
            logits = model.board_head(h)[0, -1, :]  # (board_vocab_size,)
            board_sub_idx = sample_token(logits, temperature)

            if board_sub_idx == _BOARD_END_THINK_IDX:
                full_idx = board_idx_to_full_idx[board_sub_idx]
                append(full_idx, orphan())
                vprint("end_think")
                ended_thinking = True
                state = "FINAL"
            else:
                # new_variation (or any non-end_think): start new variation
                # Don't append new_variation to context (it's just a signal)
                state = "MOVE"

        elif state == "FINAL":
            if full():
                break
            pos = len(token_ids) - 1
            idx = sample_move(model.policy_head, pos)
            tok = idx_to_token[idx]
            append(idx, orphan())
            final_move = normalize_castling(tok)
            vprint(f"final_move: {final_move}")

            if not full():
                move_pos = len(token_ids) - 1
                wl_idx, wl, d_idx, d = emit_wl_d(move_pos)
                final_wl = wl
                final_d = d
                final_wl_bucket = wl_idx
                final_d_bucket = d_idx
                if verbose:
                    w, dr, l = reconstruct_wdl(wl, d)
                    vprint(f"\nFinal WDL: W={w:.3f} D={dr:.3f} L={l:.3f}")
            break

    vprint(f"Sequence length: {len(token_ids)} / {max_seq_len}")
    return ThinkingResult(
        token_ids=token_ids, block_ids=block_ids,
        wl_entries=wl_entries, d_entries=d_entries,
        wl_bucket_indices=wl_bucket_indices,
        d_bucket_indices=d_bucket_indices,
        final_move=final_move, final_wl=final_wl, final_d=final_d,
        final_wl_bucket=final_wl_bucket, final_d_bucket=final_d_bucket,
        ended_thinking=ended_thinking,
        truncated=(len(token_ids) >= max_seq_len) and final_move is None,
    )


def main():
    parser = argparse.ArgumentParser(description="Thinking inference for ChessDecoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fen", default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    model, max_seq_len = load_model(args.checkpoint, device)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max seq len: {max_seq_len}")
    print()

    run_thinking(
        model, args.fen,
        temperature=args.temperature,
        device=device,
        max_seq_len=max_seq_len,
    )


if __name__ == "__main__":
    main()

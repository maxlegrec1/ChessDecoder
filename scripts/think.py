"""
Thinking inference for ChessDecoder.

Generates a thinking sequence autoregressively, printing tokens as they come.
Stops when max_seq_len is reached.
"""

import argparse
import torch
import torch.nn.functional as F

from src.models.model import ChessDecoder
from src.models.vocab import vocab_size, token_to_idx, idx_to_token
from src.dataloader.data import fen_to_position_tokens

_PSEUDO_TO_STANDARD = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}

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


def to_standard_uci(model_uci):
    return _PSEUDO_TO_STANDARD.get(model_uci, model_uci)


def sample_token(logits, temperature):
    if temperature <= 0:
        return torch.argmax(logits).item()
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


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
    model.eval()
    return model, max_seq_len


def reconstruct_wdl(wl, d):
    w = max(0.0, min(1.0, (1 - d + wl) / 2))
    l = max(0.0, min(1.0, (1 - d - wl) / 2))
    d = max(0.0, min(1.0, d))
    return w, d, l


@torch.no_grad()
def run_thinking(model, fen, temperature=0.0, device="cuda", max_seq_len=1024):

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

    def prefix_forward():
        S = len(token_ids)
        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
        blk = torch.tensor([block_ids], dtype=torch.long, device=device)
        wl_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
        d_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
        wl_val = torch.zeros(1, S, device=device)
        d_val = torch.zeros(1, S, device=device)
        for p, v in wl_entries:
            wl_pos[0, p] = True; wl_val[0, p] = v
        for p, v in d_entries:
            d_pos[0, p] = True; d_val[0, p] = v
        return model(inp, mask_type="prefix", block_id=blk,
                     wl_values=wl_val, d_values=d_val,
                     wl_positions=wl_pos, d_positions=d_pos)

    def causal_forward():
        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
        return model(inp, mask_type="causal")

    def sample_move(head, pos):
        h = prefix_forward()
        logits = head(h)[0, pos, :]
        return sample_token(logits, temperature)

    def predict_wl(move_pos):
        h = prefix_forward()
        logits = model.wl_head(h[0, move_pos:move_pos+1, :])
        probs = F.softmax(logits.float(), dim=-1)
        return (probs * model.wl_bucket_centers.unsqueeze(0)).sum(-1).item()

    def predict_d(wl_pos):
        h = prefix_forward()
        logits = model.d_head(h[0, wl_pos:wl_pos+1, :])
        probs = F.softmax(logits.float(), dim=-1)
        return (probs * model.d_bucket_centers.unsqueeze(0)).sum(-1).item()

    def emit_wl_d(move_pos):
        """Predict WL/D, append fourier tokens, print values."""
        wl = predict_wl(move_pos)
        wl_pos_idx = len(token_ids)
        append(token_to_idx["wl_value"], orphan())
        wl_entries.append((wl_pos_idx, wl))

        d = predict_d(wl_pos_idx)
        d_pos_idx = len(token_ids)
        append(token_to_idx["d_value"], orphan())
        d_entries.append((d_pos_idx, d))

        w, dr, l = reconstruct_wdl(wl, d)
        print(f"  wl/d -> W={w:.2f} D={dr:.2f} L={l:.2f}")
        return wl, d

    def emit_board():
        """Generate 68 board tokens via board_head, print reconstructed FEN."""
        bid = next_block[0]; next_block[0] += 1
        tok_strs = []
        for i in range(68):
            if full():
                break
            h = causal_forward()
            logits = model.board_head(h)[0, -1, :]
            tok = torch.argmax(logits).item()
            append(tok, bid)
            tok_strs.append(idx_to_token[tok])
        fen_str = board_tokens_to_fen(tok_strs) if len(tok_strs) == 68 else "<truncated>"
        print(f"  board: {fen_str}")

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
    print(f"root: {fen}")
    if full():
        print(f"Sequence length: {len(token_ids)} / {max_seq_len}")
        return

    # 2. start_think
    append(token_to_idx["start_think"], orphan())
    print("start_think")

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
    ended_thinking = False

    while not full():
        if state == "MOVE":
            # Sample move from thinking_policy_head at last position
            pos = len(token_ids) - 1
            idx = sample_move(model.thinking_policy_head, pos)
            tok = idx_to_token[idx]
            append(idx, orphan())
            if tok in _MOVE_TOKENS:
                # Check if this is a root_move (after start_think or end_var)
                prev_tok = idx_to_token[token_ids[-2]] if len(token_ids) >= 2 else ""
                if prev_tok in ("start_think", "end_var"):
                    var_idx += 1
                    print(f"  variation {var_idx} root_move: {to_standard_uci(tok)}")
                else:
                    print(f"    pv_move: {to_standard_uci(tok)}")
            else:
                print(f"  unexpected token from thinking_policy_head: {tok}")
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
            # Use board_head (causal) to let the model decide: next pv_move or end_var
            if full():
                break
            h = causal_forward()
            logits = model.board_head(h)[0, -1, :]
            next_idx = sample_token(logits, temperature)
            next_tok = idx_to_token[next_idx]

            if next_tok == "end_var":
                append(next_idx, orphan())
                print("  end_var")
                state = "AFTER_END_VAR"
            else:
                # Model wants to continue the PV -- sample the actual move
                # from thinking_policy_head instead (board_head isn't trained for moves)
                state = "MOVE"

        elif state == "AFTER_END_VAR":
            # Use board_head to decide: new variation or end_think
            if full():
                break
            h = causal_forward()
            logits = model.board_head(h)[0, -1, :]
            next_idx = sample_token(logits, temperature)
            next_tok = idx_to_token[next_idx]

            if next_tok == "end_think":
                append(next_idx, orphan())
                print("end_think")
                ended_thinking = True
                state = "FINAL"
            else:
                # Start a new variation
                state = "MOVE"

        elif state == "FINAL":
            if full():
                break
            pos = len(token_ids) - 1
            idx = sample_move(model.policy_head, pos)
            tok = idx_to_token[idx]
            append(idx, orphan())
            print(f"final_move: {to_standard_uci(tok)}")

            if not full():
                move_pos = len(token_ids) - 1
                wl, d = emit_wl_d(move_pos)
                w, dr, l = reconstruct_wdl(wl, d)
                print(f"\nFinal WDL: W={w:.3f} D={dr:.3f} L={l:.3f}")
            break

    print(f"Sequence length: {len(token_ids)} / {max_seq_len}")


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

"""
Thinking inference for ChessDecoder.

Given a FEN position, generates a thinking sequence autoregressively:
  [board_68] start_think
    root_move wl d [board_68] pv_move wl d [board_68] ... end_var
    root_move ... end_var
  end_think final_move wl d

Board tokens are generated via board_head (causal masking).
Move tokens are sampled via thinking_policy_head / policy_head (prefix masking).
WL/D values are predicted via wl_head / d_head and injected as fourier encodings.
"""

import argparse
import chess
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


def to_standard_uci(model_uci):
    return _PSEUDO_TO_STANDARD.get(model_uci, model_uci)


def sample_token(logits, temperature):
    if temperature <= 0:
        return torch.argmax(logits).item()
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def reconstruct_wdl(wl, d):
    w = max(0.0, min(1.0, (1 - d + wl) / 2))
    l = max(0.0, min(1.0, (1 - d - wl) / 2))
    d = max(0.0, min(1.0, d))
    return {"win": w, "draw": d, "loss": l}


def board_tokens_to_fen(board_token_ids):
    """Reconstruct a FEN string from 68 generated board tokens."""
    tokens = [idx_to_token[t] for t in board_token_ids]
    # tokens[0] = start_pos, tokens[1..64] = squares (a1..h8), tokens[65] = end_pos,
    # tokens[66] = castling, tokens[67] = stm
    if len(tokens) != 68:
        return "<invalid board>"

    squares = tokens[1:65]

    # chess.SQUARES order: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
    # FEN goes rank 8 (top) to rank 1 (bottom), file a to h
    fen_rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty_count = 0
        for file in range(8):
            sq_idx = rank * 8 + file
            piece = squares[sq_idx]
            sym = _PIECE_SYMBOLS.get(piece)
            if sym:
                if empty_count > 0:
                    row += str(empty_count)
                    empty_count = 0
                row += sym
            else:
                empty_count += 1
        if empty_count > 0:
            row += str(empty_count)
        fen_rows.append(row)

    placement = "/".join(fen_rows)
    castling = tokens[66]
    castling_str = castling if castling != "no_castling_rights" else "-"
    stm_char = "w" if tokens[67] == "white_to_move" else "b"

    return f"{placement} {stm_char} {castling_str} - 0 1"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["max_seq_len"],
        d_ff=config["model"].get("d_ff"),
        n_buckets=config["model"].get("n_buckets", 100),
        value_hidden_size=config["model"].get("value_hidden_size", 256),
        num_fourier_freq=config["model"].get("num_fourier_freq", 128),
        wl_sigma=config["model"].get("wl_sigma", 0.4),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_thinking(model, fen, max_variations=3, max_depth=5, temperature=0.0, device="cuda"):
    """Generate a full thinking sequence for the given FEN."""

    # --- running state ---
    token_ids = []
    block_ids = []
    wl_entries = []   # [(position, value), ...]
    d_entries = []    # [(position, value), ...]
    next_block = [0]
    orphan_ctr = [10000]

    def orphan():
        orphan_ctr[0] += 1
        return orphan_ctr[0]

    def append(tok_id, bid):
        token_ids.append(tok_id)
        block_ids.append(bid)

    # --- forward helpers ---
    def prefix_forward():
        S = len(token_ids)
        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
        blk = torch.tensor([block_ids], dtype=torch.long, device=device)
        wl_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
        d_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
        wl_val = torch.zeros(1, S, device=device)
        d_val = torch.zeros(1, S, device=device)
        for p, v in wl_entries:
            wl_pos[0, p] = True
            wl_val[0, p] = v
        for p, v in d_entries:
            d_pos[0, p] = True
            d_val[0, p] = v
        return model(inp, mask_type="prefix", block_id=blk,
                     wl_values=wl_val, d_values=d_val,
                     wl_positions=wl_pos, d_positions=d_pos)

    def causal_forward():
        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
        return model(inp, mask_type="causal")

    # --- prediction helpers ---
    def predict_wl_at(pos):
        h = prefix_forward()
        logits = model.wl_head(h[0, pos:pos+1, :])
        probs = F.softmax(logits.float(), dim=-1)
        return (probs * model.wl_bucket_centers.unsqueeze(0)).sum(-1).item()

    def predict_d_at(pos):
        h = prefix_forward()
        logits = model.d_head(h[0, pos:pos+1, :])
        probs = F.softmax(logits.float(), dim=-1)
        return (probs * model.d_bucket_centers.unsqueeze(0)).sum(-1).item()

    def sample_move_from(head, pos):
        h = prefix_forward()
        logits = head(h)[0, pos, :]
        idx = sample_token(logits, temperature)
        return idx

    def generate_board():
        """Generate 68 board tokens autoregressively via board_head."""
        bid = next_block[0]
        next_block[0] += 1
        generated = []
        for _ in range(68):
            h = causal_forward()
            logits = model.board_head(h)[0, -1, :]
            tok = torch.argmax(logits).item()  # greedy for board tokens
            append(tok, bid)
            generated.append(tok)
        return generated

    def predict_wl_d(move_pos):
        """Predict WL and D for the move at move_pos. Appends wl/d tokens."""
        # 1. WL from hidden state at move position
        wl_val = predict_wl_at(move_pos)
        wl_pos = len(token_ids)
        append(token_to_idx["wl_value"], orphan())
        wl_entries.append((wl_pos, wl_val))

        # 2. D from hidden state at wl position (WL fourier now injected)
        d_val = predict_d_at(wl_pos)
        d_pos = len(token_ids)
        append(token_to_idx["d_value"], orphan())
        d_entries.append((d_pos, d_val))

        return wl_val, d_val

    # ===== Build the sequence =====

    # 1. Root board (given from FEN)
    root_tokens = fen_to_position_tokens(fen)
    bid = next_block[0]; next_block[0] += 1
    for t in root_tokens:
        append(token_to_idx[t], bid)

    # 2. start_think
    append(token_to_idx["start_think"], orphan())

    # 3. Variation loop
    variations = []
    for var_idx in range(max_variations):
        # Root move: predicted from start_think or previous end_var
        predict_pos = len(token_ids) - 1
        root_move_idx = sample_move_from(model.thinking_policy_head, predict_pos)
        root_move_token = idx_to_token[root_move_idx]
        append(root_move_idx, orphan())

        var_info = {"root_move": to_standard_uci(root_move_token), "nodes": []}

        for depth in range(max_depth):
            # WL/D prediction and token insertion
            move_pos = len(token_ids) - 1
            wl_val, d_val = predict_wl_d(move_pos)
            wdl = reconstruct_wdl(wl_val, d_val)

            # Generate board autoregressively
            board_token_ids = generate_board()
            board_fen = board_tokens_to_fen(board_token_ids)

            node = {"wdl": wdl, "fen": board_fen, "pv_move": None}

            # PV continuation move (only for non-terminal nodes)
            if depth < max_depth - 1:
                stm_pos = len(token_ids) - 1
                pv_idx = sample_move_from(model.thinking_policy_head, stm_pos)
                pv_token = idx_to_token[pv_idx]
                append(pv_idx, orphan())
                node["pv_move"] = to_standard_uci(pv_token)

            var_info["nodes"].append(node)

        # end_var
        append(token_to_idx["end_var"], orphan())
        variations.append(var_info)

    # 4. end_think
    append(token_to_idx["end_think"], orphan())

    # 5. Final move from policy_head
    end_think_pos = len(token_ids) - 1
    final_idx = sample_move_from(model.policy_head, end_think_pos)
    final_token = idx_to_token[final_idx]
    append(final_idx, orphan())

    # 6. Final WL/D
    move_pos = len(token_ids) - 1
    wl_val, d_val = predict_wl_d(move_pos)
    final_wdl = reconstruct_wdl(wl_val, d_val)

    return {
        "fen": fen,
        "variations": variations,
        "final_move": to_standard_uci(final_token),
        "final_wdl": final_wdl,
        "seq_length": len(token_ids),
    }


def print_result(result):
    print(f"Position: {result['fen']}")
    print()
    print("=== Thinking ===")
    print()

    for i, var in enumerate(result["variations"]):
        print(f"Variation {i + 1}: {var['root_move']}")
        for node in var["nodes"]:
            wdl = node["wdl"]
            print(f"  [{wdl['win']:.2f} / {wdl['draw']:.2f} / {wdl['loss']:.2f}]  {node['fen']}")
            if node["pv_move"]:
                print(f"    -> {node['pv_move']}")
        print()

    print("=== Decision ===")
    wdl = result["final_wdl"]
    print(f"Move: {result['final_move']}")
    print(f"WDL:  W={wdl['win']:.3f}  D={wdl['draw']:.3f}  L={wdl['loss']:.3f}")
    print(f"Sequence length: {result['seq_length']} tokens")


def main():
    parser = argparse.ArgumentParser(description="Thinking inference for ChessDecoder")
    parser.add_argument("--checkpoint", required=True, help="Path to finetuned checkpoint")
    parser.add_argument("--fen", default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        help="FEN string")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for moves (0 = greedy)")
    parser.add_argument("--max-variations", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print()

    result = run_thinking(
        model, args.fen,
        max_variations=args.max_variations,
        max_depth=args.max_depth,
        temperature=args.temperature,
        device=device,
    )

    print_result(result)


if __name__ == "__main__":
    main()

"""Head-to-head: V2 + C++ PUCT MCTS  vs  Leela BT4 raw policy.

Both engines play at T=0 (deterministic). To get a non-degenerate sample we
replay each opening from scripts/opening_book.py before letting the engines
take over, alternating colors so each engine gets equal White / Black share.

PGN headers tag whoever was White/Black for each game; aggregate by *engine*,
not by color, so the result is genuinely "wins for V2-MCTS" not "wins as White".

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/h2h_v2mcts_vs_leela.py \\
        [SIMS=800] [CPUCT=1.5] [MAX_PLIES=200]
"""
import sys
import time
from datetime import datetime

import chess
import chess.pgn
import torch

from chessdecoder.models.leela import BT4
from chessdecoder.mcts_v2 import V2MCTS
from scripts.opening_book import OPENING_LINES


SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
CPUCT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
MAX_PLIES = int(sys.argv[3]) if len(sys.argv) > 3 else 200
LEELA_CKPT = "chessdecoder/models/leela/model.pt"

# ---- Load both engines onto the same GPU. They take turns; memory is fine.
print(f"Loading V2-MCTS (sims={SIMS}, cpuct={CPUCT})", flush=True)
v2 = V2MCTS(simulations=SIMS, cpuct=CPUCT, temperature=0.0,
            max_batch_leaves=32)

print(f"Loading Leela BT4 from {LEELA_CKPT}", flush=True)
leela = BT4().to("cuda").eval()
state = torch.load(LEELA_CKPT, map_location="cuda", weights_only=False)
leela.load_state_dict(state)


# ---- Engine adapters: each gets the current FEN + the move history so far,
# returns a UCI move. V2-MCTS ignores history (single-board mode). Leela
# requires the full history to populate its 8 history planes correctly
# (verified in the SF eval — without history Leela degrades by ~900 Elo).


@torch.no_grad()
def v2_move(fen: str, history: list[str]) -> str:
    r = v2.search(fen)
    return r.action


@torch.no_grad()
def leela_move(fen: str, history: list[str]) -> str:
    if history:
        return leela.get_move_from_fen_no_thinking(list(history), T=0.0)
    return leela.get_move_from_fen_no_thinking(fen, T=0.0)


# ---- Game loop
def play_game(white_fn, white_name, black_fn, black_name,
              opening: list[str], pgn_writer=None, round_no: int = 0):
    board = chess.Board()
    history: list[str] = []
    for mv in opening:           # book moves (forced)
        board.push(chess.Move.from_uci(mv))
        history.append(mv)

    plies_played = 0
    while not board.is_game_over(claim_draw=True) and plies_played < MAX_PLIES:
        fn = white_fn if board.turn == chess.WHITE else black_fn
        uci = fn(board.fen(), history)
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            print(f"  illegal-string by "
                  f"{'White' if board.turn == chess.WHITE else 'Black'}: {uci}",
                  flush=True)
            return "0-1" if board.turn == chess.WHITE else "1-0"
        if mv not in board.legal_moves:
            print(f"  illegal-move by "
                  f"{'White' if board.turn == chess.WHITE else 'Black'}: {uci}",
                  flush=True)
            return "0-1" if board.turn == chess.WHITE else "1-0"
        board.push(mv)
        history.append(uci)
        plies_played += 1

    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif (board.is_stalemate() or board.is_insufficient_material()
          or board.can_claim_threefold_repetition()
          or board.can_claim_fifty_moves()):
        result = "1/2-1/2"
    else:
        result = "*"

    if pgn_writer is not None:
        game = chess.pgn.Game()
        game.headers["Event"] = "V2-MCTS vs BT4"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(round_no)
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Result"] = result
        node = game
        b2 = chess.Board()
        for u in history:
            mv = chess.Move.from_uci(u)
            node = node.add_variation(mv)
            b2.push(mv)
        pgn_writer.write(str(game) + "\n\n")
        pgn_writer.flush()

    return result


def main():
    n_open = len(OPENING_LINES)
    print(f"\nOpening book: {n_open} lines  →  {n_open*2} games", flush=True)
    print(f"Time budget per game ≈ {SIMS*0.5/1000*40:.0f}s (MCTS-side only)\n",
          flush=True)

    # Aggregate by *engine* (not by color).
    v2_wins = leela_wins = draws = 0
    games_played = 0

    pgn_path = (f"pgns/h2h_v2mcts{SIMS}_vs_leela_"
                f"{datetime.now():%Y%m%d_%H%M%S}.pgn")
    print(f"PGN: {pgn_path}", flush=True)
    pgn_f = open(pgn_path, "w")

    t_start = time.time()
    for opening_idx, opening in enumerate(OPENING_LINES):
        for color_swap in (False, True):
            if not color_swap:
                white_fn, white_name = v2_move, "V2-MCTS"
                black_fn, black_name = leela_move, "BT4"
            else:
                white_fn, white_name = leela_move, "BT4"
                black_fn, black_name = v2_move, "V2-MCTS"

            t0 = time.time()
            result = play_game(white_fn, white_name, black_fn, black_name,
                               opening, pgn_writer=pgn_f,
                               round_no=games_played + 1)
            dt = time.time() - t0
            games_played += 1

            # Map result to engine outcomes.
            if result == "1/2-1/2":
                draws += 1
                outcome = "draw"
            elif result == "1-0":   # White won
                if white_name == "V2-MCTS":
                    v2_wins += 1; outcome = "V2-MCTS win"
                else:
                    leela_wins += 1; outcome = "BT4 win"
            elif result == "0-1":   # Black won
                if black_name == "V2-MCTS":
                    v2_wins += 1; outcome = "V2-MCTS win"
                else:
                    leela_wins += 1; outcome = "BT4 win"
            else:
                draws += 1
                outcome = "* (no result, treated as draw)"

            elapsed = time.time() - t_start
            print(f"  [{games_played:3d}/{n_open*2}] open#{opening_idx + 1:02d} "
                  f"({len(opening)} book ply, {white_name[:7]:7s}=W vs "
                  f"{black_name[:7]:7s}=B): {result:8s} -> {outcome:18s}  "
                  f"({dt:5.1f}s game, {elapsed/60:.1f}m total, "
                  f"running: V2={v2_wins} BT4={leela_wins} D={draws})",
                  flush=True)

    pgn_f.close()

    n = games_played
    score = v2_wins + 0.5 * draws
    pct = 100 * score / n if n else 0
    print(f"\n===== FINAL: V2-MCTS({SIMS}) vs Leela BT4 raw policy =====")
    print(f"  V2-MCTS wins: {v2_wins}")
    print(f"  BT4 wins:     {leela_wins}")
    print(f"  Draws:        {draws}")
    print(f"  V2-MCTS score: {score:.1f}/{n} = {pct:.1f}%")
    # Bayeselo-style Elo-from-score (no draw modeling — same formula as
    # chessdecoder/eval/stats.py:estimate_elo).
    import math
    if 0.001 < pct/100 < 0.999:
        elo_diff = -400 * math.log10(1 / (score / n) - 1)
        print(f"  Estimated Elo diff: {elo_diff:+.0f} (V2-MCTS over BT4 raw)")
    else:
        print("  Elo diff: saturated (no losses or no wins)")


if __name__ == "__main__":
    main()

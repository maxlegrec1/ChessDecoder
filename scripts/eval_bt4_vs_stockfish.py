#!/usr/bin/env python3
"""Play the REAL lc0 BT4 (ChessRL PyTorch port) vs Stockfish — same settings as
scripts/eval_vs_stockfish.py so the score is directly comparable to our model.

BT4's own code does all the heavy lifting: BT4.get_move_from_fen_no_thinking()
takes a move-history list (full 8-ply history), encodes it (mirroring black to
white-to-move), runs the net, masks to legal moves, and returns a UCI move in
the real board frame. We just feed it the game's move stack and apply the move.

  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_bt4_vs_stockfish.py \
      --games 120 --elo 2500 [--ckpt /workspace/ChessRL/src/models/leela/model.pt]
"""
import argparse, sys

import torch, chess, chess.engine

SF = "bin/stockfish"


def load_bt4(chessrl, ckpt, dev):
    sys.path.insert(0, chessrl)
    from src.models.leela.model import BT4
    m = BT4().to(dev)
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"loaded BT4 {ckpt} (missing {len(miss)}, unexpected {len(unexp)})", flush=True)
    m.eval()
    return m


def _apply(board, bt4_uci):
    """Map BT4's returned UCI to a python-chess legal move (handles lc0 king->rook
    castling notation e1h1 vs standard e1g1, either way)."""
    for mv in board.legal_moves:
        if mv.uci() == bt4_uci:
            return mv
        if board.is_castling(mv):
            frm = chess.square_name(mv.from_square)
            rf = "h" if board.is_kingside_castling(mv) else "a"
            if frm + rf + frm[1] == bt4_uci:
                return mv
    return chess.Move.from_uci(bt4_uci)            # fallback (raises if illegal)


@torch.no_grad()
def bt4_move(m, board):
    hist = [mv.uci() for mv in board.move_stack]   # full history (standard UCIs)
    inp = hist if hist else chess.STARTING_FEN     # empty -> start position
    return _apply(board, m.get_move_from_fen_no_thinking(inp, T=0.0))


def play_game(m, engine, model_white, sf_limit):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < 300:
        if board.turn == (chess.WHITE if model_white else chess.BLACK):
            board.push(bt4_move(m, board))
        else:
            board.push(engine.play(board, sf_limit).move)
    r = board.result(claim_draw=True)
    if r == "*" or r == "1/2-1/2":
        return 0.5
    return 1.0 if ((r == "1-0") == model_white) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ChessRL/src/models/leela/model.pt")
    ap.add_argument("--chessrl", default="/workspace/ChessRL")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--elo", type=int, default=2500)
    ap.add_argument("--sf-time", type=float, default=0.1)
    a = ap.parse_args()
    dev = "cuda"
    m = load_bt4(a.chessrl, a.ckpt, dev)
    engine = chess.engine.SimpleEngine.popen_uci(SF)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": a.elo})
    sf_limit = chess.engine.Limit(time=a.sf_time)
    print(f"BT4 vs Stockfish UCI_Elo={a.elo}, sf_time={a.sf_time}s, {a.games} games, temp=0\n", flush=True)

    w = d = l = 0
    for g in range(a.games):
        s = play_game(m, engine, model_white=(g % 2 == 0), sf_limit=sf_limit)
        if s == 1.0: w += 1
        elif s == 0.0: l += 1
        else: d += 1
        print(f"  game {g+1:>3}/{a.games} ({'W' if g%2==0 else 'B'}): "
              f"{'win ' if s==1 else 'loss' if s==0 else 'draw'}  | running W{w} D{d} L{l}",
              flush=True)
    engine.quit()
    n = w + d + l
    print(f"\n=== BT4 vs SF Elo {a.elo} ===")
    print(f"  W {w}  D {d}  L {l}   (n={n})")
    print(f"  win rate (wins/n):  {w/n:.1%}")
    print(f"  score (W+0.5D)/n:   {(w + 0.5 * d) / n:.1%}")


if __name__ == "__main__":
    main()

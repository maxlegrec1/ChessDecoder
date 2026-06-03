#!/usr/bin/env python3
"""Play the encoder model vs Stockfish (UCI_LimitStrength) and report W/D/L.

The model picks moves by argmax of its policy head over the LEGAL moves
(temperature 0). Plays N games, alternating colours. Reports win/draw/loss
counts and score from the model's perspective.

  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vs_stockfish.py \
      --ckpt PATH --games 40 --elo 2500 [--ffn-type moe --moe-experts 16 \
      --moe-topk 4 --moe-expert-dff 384]
"""
import argparse, sys
import numpy as np
import torch
import chess, chess.engine

from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size, move_token_to_idx, move_vocab_size
from chessdecoder.dataloader.loader import fen_to_ids

SF = "bin/stockfish"


def load_model(args, dev):
    m = ChessEncoder(
        vocab_size=vocab_size, embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, d_ff=args.d_ff, attention_variant="geom",
        policy_head="cross_attn", input_mode="lc0_64", ffn_type=args.ffn_type,
        moe_num_experts=args.moe_experts, moe_top_k=args.moe_topk,
        moe_expert_d_ff=args.moe_expert_dff).to(dev)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): (v.to_local() if hasattr(v, "to_local")
          else getattr(v, "_data", v)) for k, v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt}  (missing {len(miss)}, unexpected {len(unexp)})", flush=True)
    m.eval()
    return m


@torch.no_grad()
def pick_move(model, board, dev):
    out = np.empty(68, dtype=np.int32)
    fen_to_ids(board.fen(), out)
    bid = torch.from_numpy(out.astype(np.int64)).unsqueeze(0).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(bid)["policy"][0].float()              # [1924]
    best, best_lp = None, -1e30
    for mv in board.legal_moves:
        idx = move_token_to_idx.get(mv.uci())
        lp = logits[idx].item() if idx is not None else -1e29  # unknown move -> last resort
        if lp > best_lp:
            best_lp, best = lp, mv
    return best


def play_game(model, engine, dev, model_white, sf_limit):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < 300:
        if board.turn == (chess.WHITE if model_white else chess.BLACK):
            board.push(pick_move(model, board, dev))
        else:
            board.push(engine.play(board, sf_limit).move)
    r = board.result(claim_draw=True)          # "1-0" / "0-1" / "1/2-1/2" / "*"
    if r == "*":
        return 0.5                              # hit ply cap -> draw
    if r == "1/2-1/2":
        return 0.5
    model_won = (r == "1-0") == model_white
    return 1.0 if model_won else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--elo", type=int, default=2500)
    ap.add_argument("--sf-time", type=float, default=0.1)
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--num-heads", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=15)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--ffn-type", default="dense")
    ap.add_argument("--moe-experts", type=int, default=8)
    ap.add_argument("--moe-topk", type=int, default=2)
    ap.add_argument("--moe-expert-dff", type=int, default=768)
    args = ap.parse_args()
    dev = "cuda"
    model = load_model(args, dev)
    engine = chess.engine.SimpleEngine.popen_uci(SF)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})
    sf_limit = chess.engine.Limit(time=args.sf_time)
    print(f"vs Stockfish UCI_Elo={args.elo}, sf_time={args.sf_time}s, {args.games} games, temp=0\n", flush=True)

    w = d = l = 0
    for g in range(args.games):
        s = play_game(model, engine, dev, model_white=(g % 2 == 0), sf_limit=sf_limit)
        if s == 1.0: w += 1
        elif s == 0.0: l += 1
        else: d += 1
        print(f"  game {g+1:>3}/{args.games} ({'W' if g%2==0 else 'B'}): "
              f"{'win ' if s==1 else 'loss' if s==0 else 'draw'}  "
              f"| running W{w} D{d} L{l}", flush=True)
    engine.quit()
    n = w + d + l
    score = (w + 0.5 * d) / n
    print(f"\n=== {args.ckpt} vs SF Elo {args.elo} ===")
    print(f"  W {w}  D {d}  L {l}   (n={n})")
    print(f"  win rate (wins/n):  {w/n:.1%}")
    print(f"  score (W+0.5D)/n:   {score:.1%}")


if __name__ == "__main__":
    main()

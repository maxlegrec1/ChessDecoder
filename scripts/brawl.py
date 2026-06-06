#!/usr/bin/env python3
"""Model-vs-model brawl: two checkpoints play N games (temp 0), report head-to-head.

Each model picks moves by policy argmax over legal moves, with the SAME
history-reconstruction + lc0-castling handling as the Stockfish eval (so a
history-trained model gets its 8-ply input from the game's move stack). Useful
for a sensitive A/B between two of our own models (e.g. history vs no-history),
which is lower-variance than comparing both against Stockfish.

  CUDA_VISIBLE_DEVICES=0 uv run python scripts/brawl.py \
      --ckpt-a HIST.pt   --history-a 8 --label-a history \
      --ckpt-b PLAIN.pt  --history-b 1 --label-b plain \
      --games 200
"""
import argparse, os, sys, random

import torch, chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vs_stockfish import pick_move                      # history + castling aware

from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size


def build(ckpt, dev, ffn="dense", experts=8, topk=2, expert_dff=768, history=1,
          embed=1024, heads=32, layers=15, dff=1536):
    m = ChessEncoder(vocab_size=vocab_size, embed_dim=embed, num_heads=heads,
                     num_layers=layers, d_ff=dff, attention_variant="geom",
                     policy_head="cross_attn", input_mode="lc0_64", ffn_type=ffn,
                     moe_num_experts=experts, moe_top_k=topk,
                     moe_expert_d_ff=expert_dff, history=history).to(dev)
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): (v.to_local() if hasattr(v, "to_local")
          else getattr(v, "_data", v)) for k, v in sd.items()}
    miss, _ = m.load_state_dict(sd, strict=False)
    print(f"loaded {ckpt} (history={history}, missing {len(miss)})", flush=True)
    m.eval()
    return m


def play_game(ma, mb, dev, a_white, opening_plies, rng):
    board = chess.Board()
    # Random opening (same for both models this game) so deterministic temp-0
    # models don't replay one identical game. Different each game -> variety.
    for _ in range(opening_plies):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
    while not board.is_game_over(claim_draw=True) and board.ply() < 300:
        a_to_move = (board.turn == chess.WHITE) == a_white
        board.push(pick_move(ma if a_to_move else mb, board, dev))
    r = board.result(claim_draw=True)
    if r == "*" or r == "1/2-1/2":
        return 0.5
    return 1.0 if ((r == "1-0") == a_white) else 0.0           # from A's view


def main():
    ap = argparse.ArgumentParser()
    for s in ("a", "b"):
        ap.add_argument(f"--ckpt-{s}", required=True)
        ap.add_argument(f"--history-{s}", type=int, default=1)
        ap.add_argument(f"--ffn-{s}", default="dense")
        ap.add_argument(f"--experts-{s}", type=int, default=8)
        ap.add_argument(f"--topk-{s}", type=int, default=2)
        ap.add_argument(f"--expert-dff-{s}", type=int, default=768)
        ap.add_argument(f"--label-{s}", default=s.upper())
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--opening-plies", type=int, default=8,
                    help="random plies to start each game (variety for temp-0 play)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    ma = build(a.ckpt_a, dev, a.ffn_a, a.experts_a, a.topk_a, a.expert_dff_a, a.history_a)
    mb = build(a.ckpt_b, dev, a.ffn_b, a.experts_b, a.topk_b, a.expert_dff_b, a.history_b)
    print(f"\nBRAWL: {a.label_a} (A) vs {a.label_b} (B) — {a.games} games, temp=0\n", flush=True)

    rng = random.Random(a.seed)
    w = d = l = 0
    for g in range(a.games):
        # both models play the SAME random opening in game g (fair); A alternates colour
        s = play_game(ma, mb, dev, a_white=(g % 2 == 0),
                      opening_plies=a.opening_plies, rng=random.Random(a.seed * 100000 + g))
        if s == 1.0: w += 1
        elif s == 0.0: l += 1
        else: d += 1
        print(f"  game {g+1:>3}/{a.games} (A={'W' if g%2==0 else 'B'}): "
              f"{'A win ' if s==1 else 'B win ' if s==0 else 'draw  '}"
              f"| {a.label_a} W{w} D{d} L{l}", flush=True)
    n = w + d + l
    print(f"\n=== {a.label_a} (A) vs {a.label_b} (B), n={n} ===")
    print(f"  {a.label_a}: W {w}  D {d}  L {l}")
    print(f"  {a.label_a} score: {(w + 0.5 * d) / n:.1%}   "
          f"({a.label_b} score: {(l + 0.5 * d) / n:.1%})")


if __name__ == "__main__":
    main()

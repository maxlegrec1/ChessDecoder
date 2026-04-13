"""
Evaluate ELO of a non-thinking (pretrain/finetune) model against Stockfish.

The model is loaded directly from a checkpoint — no TorchScript export needed.
All board inputs are exactly 68 tokens; N positions are forwarded in one GPU op.

Usage:
    # Plain single-board inference (no history)
    uv run python scripts/eval_elo_nonthinker.py \
        --checkpoint checkpoints/checkpoint_616000.pt \
        --num-games 200 --elo 1500

    # Condition each move on the last N (board, move) pairs
    uv run python scripts/eval_elo_nonthinker.py \
        --checkpoint checkpoints/checkpoint_616000.pt \
        --max-n 5 --num-games 200 --elo 1500
"""

import argparse

from chessdecoder.eval.engine import build_nonthinker_engine
from chessdecoder.eval.elo_eval import model_vs_stockfish


def main():
    parser = argparse.ArgumentParser(description="ELO evaluation for non-thinking checkpoints")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a checkpoint_*.pt file.")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--elo", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-n", type=int, default=0,
                        help="History depth: condition each move on the last N (board, move) pairs. "
                             "0 = no history (default).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    engine = build_nonthinker_engine(
        args.checkpoint,
        batch_size=1,   # sequential game play — one position at a time
        device=args.device,
        max_n=args.max_n,
    )
    model_vs_stockfish(
        model=engine,
        model1_name="nonthinker",
        num_games=args.num_games,
        temperature=args.temperature,
        elo=args.elo,
    )


if __name__ == "__main__":
    main()

"""
Evaluate ELO of a thinking model (full COT) against Stockfish.

Usage:
    uv run python scripts/eval_elo_thinking.py --export-dir exports/base --num-games 200 --elo 2000
"""

import argparse

from chessdecoder.eval.engine import build_thinking_single_engine
from chessdecoder.eval.elo_eval import model_vs_stockfish


def main():
    parser = argparse.ArgumentParser(description="ELO evaluation with thinking inference")
    parser.add_argument("--export-dir", default="exports/base")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--elo", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    engine = build_thinking_single_engine(args.export_dir)
    model_vs_stockfish(
        model=engine,
        model1_name="thinking-decoder",
        num_games=args.num_games,
        temperature=args.temperature,
        elo=args.elo,
    )


if __name__ == "__main__":
    main()

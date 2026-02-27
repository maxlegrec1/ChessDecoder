"""
Elo evaluation using the C++ libtorch thinking inference engine.

Usage:
    uv run python scripts/test_evaluate_thinking_cpp.py --num-games 100 --elo 2000
    uv run python scripts/test_evaluate_thinking_cpp.py --export-dir export --num-games 50 --elo 1500
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _decoder_inference_cpp as cpp
from src.eval.elo_eval import model_vs_stockfish


def main():
    parser = argparse.ArgumentParser(description="Elo evaluation with C++ thinking engine")
    parser.add_argument("--export-dir", default="export")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--elo", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    export = Path(args.export_dir)
    engine = cpp.ThinkingInferenceEngine(
        str(export / "backbone.pt"),
        str(export / "weights"),
        str(export / "vocab.json"),
        str(export / "config.json"),
    )

    model_vs_stockfish(
        model=engine,
        model1_name="cpp-decoder",
        num_games=args.num_games,
        temperature=args.temperature,
        elo=args.elo,
    )


if __name__ == "__main__":
    main()

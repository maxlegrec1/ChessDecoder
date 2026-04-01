"""
Evaluate model accuracy on unseen variation positions using C++ thinking engine.

Usage:
    uv run python scripts/eval_unseen_variations.py --export-dir export_eval_mid --parquet single_unseen_parquet/training-run2-test91-20251117-2317.parquet --n 100
"""

import argparse
import random
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _decoder_inference_cpp as cpp


def normalize_castling(move):
    mapping = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
    return mapping.get(move, move)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export = Path(args.export_dir)
    engine = cpp.ThinkingInferenceEngine(
        str(export / "backbone.pt"),
        str(export / "weights"),
        str(export / "vocab.json"),
        str(export / "config.json"),
    )

    df = pd.read_parquet(args.parquet, columns=["fen", "best_move", "mcts_action"])
    # Deduplicate by FEN, sample N
    df = df.drop_duplicates(subset="fen")
    rng = random.Random(args.seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    df = df.iloc[indices[:args.n]].reset_index(drop=True)

    mcts_correct = 0
    best_correct = 0
    completed = 0

    for i, row in df.iterrows():
        fen = row["fen"]
        sp_move = engine.predict_move(fen, 0.0)
        if sp_move:
            sp_move = normalize_castling(sp_move)
            completed += 1
            if sp_move == normalize_castling(row["mcts_action"]):
                mcts_correct += 1
            if sp_move == normalize_castling(row["best_move"]):
                best_correct += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} | mcts_acc={mcts_correct/(completed+1e-8):.3f} best_acc={best_correct/(completed+1e-8):.3f}", flush=True)

    print(f"\n{'='*50}")
    print(f"Export dir: {args.export_dir}")
    print(f"Positions evaluated: {completed}/{len(df)}")
    print(f"MCTS acc:  {mcts_correct}/{completed} = {mcts_correct/(completed+1e-8):.3f}")
    print(f"Best acc:  {best_correct}/{completed} = {best_correct/(completed+1e-8):.3f}")


if __name__ == "__main__":
    main()

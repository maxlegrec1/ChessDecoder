"""
Evaluate pass@k: generate k COTs per FEN, check if any final move matches best_move.

Usage:
    uv run python scripts/pass_at_k.py \
        --export-dir export \
        --num-fens 200 --k 10 \
        --think-temp 0.8 --policy-temp 0.5
"""

import argparse
import os
import random
import time

import pyarrow.parquet as pq

import _decoder_inference_cpp as cpp


def load_fen_bestmove_pairs(n, seed, data_dir):
    """Load (fen, best_move) pairs from pretraining parquet files."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    rng = random.Random(seed)
    fname = rng.choice(files)
    table = pq.read_table(os.path.join(data_dir, fname), columns=["fen", "best_move"])
    total = len(table)
    indices = rng.sample(range(total), min(n * 3, total))
    seen = set()
    pairs = []
    for i in indices:
        fen = table.column("fen")[i].as_py()
        if fen not in seen:
            seen.add(fen)
            pairs.append((fen, table.column("best_move")[i].as_py()))
        if len(pairs) >= n:
            break
    return pairs[:n]


def evaluate_pass_at_k(engine, pairs, k):
    """Returns pass@k accuracy."""
    correct = 0
    for fen, best_move in pairs:
        moves = set()
        for _ in range(k):
            moves.add(engine.predict_move(fen))
        if best_move in moves:
            correct += 1
    return correct / len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k with per-head temperatures")
    parser.add_argument("--export-dir", default="exports")
    parser.add_argument("--data-dir", default="/home/maxime/parquet_files_decoder/")
    parser.add_argument("--num-fens", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--board-temp", type=float, default=0.0)
    parser.add_argument("--think-temp", type=float, default=0.0)
    parser.add_argument("--policy-temp", type=float, default=0.0)
    parser.add_argument("--wl-temp", type=float, default=0.0)
    parser.add_argument("--d-temp", type=float, default=0.0)
    args = parser.parse_args()

    engine = cpp.ThinkingInferenceEngine(
        f"{args.export_dir}/backbone.pt",
        f"{args.export_dir}/weights",
        f"{args.export_dir}/vocab.json",
        f"{args.export_dir}/config.json",
    )

    engine.board_temperature = args.board_temp
    engine.think_temperature = args.think_temp
    engine.policy_temperature = args.policy_temp
    engine.wl_temperature = args.wl_temp
    engine.d_temperature = args.d_temp

    pairs = load_fen_bestmove_pairs(args.num_fens, args.seed, args.data_dir)
    print(f"Loaded {len(pairs)} FEN/best_move pairs")
    print(f"Temps: board={args.board_temp} think={args.think_temp} "
          f"policy={args.policy_temp} wl={args.wl_temp} d={args.d_temp}")
    print(f"Evaluating pass@{args.k} ...")

    t0 = time.time()
    score = evaluate_pass_at_k(engine, pairs, args.k)
    elapsed = time.time() - t0

    print(f"\npass@{args.k}: {score:.1%} ({int(score * len(pairs))}/{len(pairs)})")
    print(f"Time: {elapsed:.0f}s, {engine.total_tokens / elapsed:.0f} tok/s")


if __name__ == "__main__":
    main()

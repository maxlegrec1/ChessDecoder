"""
Evaluate pass@k for both single and batched engines, side by side.

Usage:
    uv run python scripts/pass_at_k_batched.py \
        --num-fens 30 --k 5 --think-temp 1.5 --policy-temp 1.5
"""

import argparse
import os
import random
import time

import pyarrow.parquet as pq

import _decoder_inference_cpp as cpp


def load_fen_bestmove_pairs(n, seed, data_dir):
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


def evaluate_pass_at_k_single(engine, pairs, k):
    correct = 0
    for fen, best_move in pairs:
        moves = set()
        for _ in range(k):
            moves.add(engine.predict_move(fen))
        if best_move in moves:
            correct += 1
    return correct


def evaluate_pass_at_k_batched(engine, pairs, k):
    correct = 0
    for fen, best_move in pairs:
        moves = set()
        for _ in range(k):
            results = engine.predict_moves([fen])
            moves.add(results[0].move)
        if best_move in moves:
            correct += 1
    return correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="exports/base")
    parser.add_argument("--data-dir", default="/home/maxime/parquet_files_decoder/")
    parser.add_argument("--num-fens", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--board-temp", type=float, default=0.0)
    parser.add_argument("--think-temp", type=float, default=1.5)
    parser.add_argument("--policy-temp", type=float, default=1.5)
    parser.add_argument("--wl-temp", type=float, default=0.0)
    parser.add_argument("--d-temp", type=float, default=0.0)
    args = parser.parse_args()

    single = cpp.ThinkingSingleInferenceEngine(
        f"{args.export_dir}/backbone.pt",
        f"{args.export_dir}/weights",
        f"{args.export_dir}/vocab.json",
        f"{args.export_dir}/config.json",
    )
    batched = cpp.ThinkingBatchedInferenceEngine(
        f"{args.export_dir}/backbone.pt",
        f"{args.export_dir}/weights",
        f"{args.export_dir}/vocab.json",
        f"{args.export_dir}/config.json",
        1,
    )

    for eng in [single, batched]:
        eng.board_temperature = args.board_temp
        eng.think_temperature = args.think_temp
        eng.policy_temperature = args.policy_temp
        eng.wl_temperature = args.wl_temp
        eng.d_temperature = args.d_temp

    pairs = load_fen_bestmove_pairs(args.num_fens, args.seed, args.data_dir)
    N = len(pairs)
    print(f"Loaded {N} FEN/best_move pairs")
    print(f"Temps: think={args.think_temp} policy={args.policy_temp}")

    t0 = time.time()
    s_correct = evaluate_pass_at_k_single(single, pairs, args.k)
    t1 = time.time()
    b_correct = evaluate_pass_at_k_batched(batched, pairs, args.k)
    t2 = time.time()

    print(f"\nSingle  pass@{args.k}: {s_correct}/{N} = {s_correct/N:.1%}  ({t1-t0:.0f}s)")
    print(f"Batched pass@{args.k}: {b_correct}/{N} = {b_correct/N:.1%}  ({t2-t1:.0f}s)")


if __name__ == "__main__":
    main()

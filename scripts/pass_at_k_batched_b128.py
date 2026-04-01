"""
Evaluate pass@k using the batched engine with B=128 for throughput.
Generates k rollouts per FEN by batching them together.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="exports/base")
    parser.add_argument("--data-dir", default="/home/maxime/parquet_files_decoder/")
    parser.add_argument("--num-fens", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think-temp", type=float, default=1.5)
    parser.add_argument("--policy-temp", type=float, default=1.5)
    args = parser.parse_args()

    B = args.batch_size
    K = args.k

    engine = cpp.BatchedInferenceEngine(
        f"{args.export_dir}/backbone.pt",
        f"{args.export_dir}/weights",
        f"{args.export_dir}/vocab.json",
        f"{args.export_dir}/config.json",
        B,
    )
    engine.think_temperature = args.think_temp
    engine.policy_temperature = args.policy_temp
    engine.board_temperature = 0.0
    engine.wl_temperature = 0.0
    engine.d_temperature = 0.0

    pairs = load_fen_bestmove_pairs(args.num_fens, args.seed, args.data_dir)
    N = len(pairs)
    print(f"Loaded {N} FEN/best_move pairs")
    print(f"Batch size: {B}, k={K}, temps: think={args.think_temp} policy={args.policy_temp}")

    correct = 0
    total_tokens = 0
    t0 = time.time()

    for idx, (fen, best_move) in enumerate(pairs):
        # Generate K rollouts for this FEN by batching K copies
        # Process in chunks of B if K > B
        moves = set()
        remaining = K
        while remaining > 0:
            chunk = min(remaining, B)
            batch_fens = [fen] * chunk
            results = engine.predict_moves(batch_fens, args.think_temp)
            for r in results[:chunk]:
                moves.add(r.move)
                total_tokens += len(r.token_ids)
            remaining -= chunk

        if best_move in moves:
            correct += 1

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{N}] pass@{K}={correct}/{idx+1} "
                  f"({correct/(idx+1):.1%}), {total_tokens/elapsed:.0f} tok/s")

    elapsed = time.time() - t0
    print(f"\npass@{K}: {correct}/{N} = {correct/N:.1%}")
    print(f"Time: {elapsed:.0f}s, {total_tokens/elapsed:.0f} tok/s")


if __name__ == "__main__":
    main()

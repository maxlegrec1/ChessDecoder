"""
Evaluate pass@k using the batched engine with B=128 for throughput.
Generates k rollouts per FEN by batching them together.
"""

import argparse
import time

import _decoder_inference_cpp as cpp

from chessdecoder.dataloader.sampling import load_pretrain_positions


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

    engine = cpp.ThinkingBatchedInferenceEngine(
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

    records = load_pretrain_positions(args.data_dir, args.num_fens, args.seed)
    pairs = [(r["fen"], r["best_move"]) for r in records]
    N = len(pairs)
    print(f"Loaded {N} FEN/best_move pairs")
    print(f"Batch size: {B}, k={K}, temps: think={args.think_temp} policy={args.policy_temp}")

    correct = 0
    total_tokens = 0
    processed = 0
    t0 = time.time()

    # Pack multiple FENs per batch: each FEN is replicated K times, and we fit
    # floor(B / K) distinct FENs per call. With K=1 this uses the whole batch
    # for distinct FENs; with K=2 B=32 we get 16 FENs × 2 copies, etc.
    if K > B:
        raise ValueError(f"K={K} must be <= batch_size={B}")
    fens_per_batch = B // K

    for start in range(0, N, fens_per_batch):
        chunk_pairs = pairs[start:start + fens_per_batch]
        batch_fens = []
        for fen, _ in chunk_pairs:
            batch_fens.extend([fen] * K)

        results = engine.predict_moves(batch_fens, args.think_temp)

        for i, (fen, best_move) in enumerate(chunk_pairs):
            moves = set()
            for j in range(K):
                r = results[i * K + j]
                moves.add(r.move)
                total_tokens += len(r.token_ids)
            if best_move in moves:
                correct += 1
            processed += 1

        elapsed = time.time() - t0
        print(f"  [{processed}/{N}] pass@{K}={correct}/{processed} "
              f"({correct/processed:.1%}), {total_tokens/elapsed:.0f} tok/s")

    elapsed = time.time() - t0
    print(f"\npass@{K}: {correct}/{N} = {correct/N:.1%}")
    print(f"Time: {elapsed:.0f}s, {total_tokens/elapsed:.0f} tok/s")


if __name__ == "__main__":
    main()

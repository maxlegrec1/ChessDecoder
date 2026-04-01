"""
Search for optimal per-head temperatures maximizing pass@k.

Phase 1: coarse grid over board/think/policy (wl/d fixed at 0).
Results saved to JSON for later analysis.

Usage:
    uv run python scripts/search_temperatures.py \
        --export-dir export --num-fens 200 --k 10
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _decoder_inference_cpp as cpp

from pass_at_k import load_fen_bestmove_pairs, evaluate_pass_at_k


def main():
    parser = argparse.ArgumentParser(description="Grid search per-head temperatures for pass@k")
    parser.add_argument("--export-dir", default="exports/base")
    parser.add_argument("--data-dir", default="/home/maxime/parquet_files_decoder/")
    parser.add_argument("--num-fens", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="temperature_search_results.json")
    args = parser.parse_args()

    engine = cpp.ThinkingInferenceEngine(
        f"{args.export_dir}/backbone.pt",
        f"{args.export_dir}/weights",
        f"{args.export_dir}/vocab.json",
        f"{args.export_dir}/config.json",
    )

    pairs = load_fen_bestmove_pairs(args.num_fens, args.seed, args.data_dir)
    print(f"Loaded {len(pairs)} FEN/best_move pairs")

    # Phase 1: coarse grid (wl/d fixed at 0)
    board_temps = [0.0, 0.3]
    think_temps = [0.0, 0.5, 1.0, 1.5]
    policy_temps = [0.0, 0.5, 1.0, 1.5]

    configs = list(itertools.product(board_temps, think_temps, policy_temps))
    total = len(configs)
    print(f"Grid: {len(board_temps)} x {len(think_temps)} x {len(policy_temps)} = {total} configs")
    print(f"Evaluating pass@{args.k} on {len(pairs)} FENs per config ...\n")

    results = []
    t_start = time.time()

    for i, (bt, tt, pt) in enumerate(configs):
        engine.board_temperature = bt
        engine.think_temperature = tt
        engine.policy_temperature = pt
        engine.wl_temperature = 0.0
        engine.d_temperature = 0.0
        engine.total_tokens = 0

        t0 = time.time()
        score = evaluate_pass_at_k(engine, pairs, args.k)
        elapsed = time.time() - t0
        tok_s = engine.total_tokens / elapsed if elapsed > 0 else 0

        entry = {
            "board_temp": bt,
            "think_temp": tt,
            "policy_temp": pt,
            "wl_temp": 0.0,
            "d_temp": 0.0,
            "score": score,
            "elapsed": round(elapsed, 1),
            "tok_s": round(tok_s),
        }
        results.append(entry)

        eta = (time.time() - t_start) / (i + 1) * (total - i - 1)
        print(f"[{i + 1:3d}/{total}] board={bt:.1f} think={tt:.1f} policy={pt:.1f} "
              f"=> pass@{args.k}={score:.1%}  ({elapsed:.0f}s, {tok_s:.0f} tok/s)  "
              f"ETA {eta / 60:.0f}min")

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"Top 10 configs (pass@{args.k}):")
    print(f"{'=' * 70}")
    for i, r in enumerate(results[:10]):
        print(f"  {i + 1:2d}. {r['score']:.1%}  board={r['board_temp']:.1f} "
              f"think={r['think_temp']:.1f} policy={r['policy_temp']:.1f}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time / 60:.0f}min")

    # Save results
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

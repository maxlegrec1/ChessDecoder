"""
Dump human-readable finetuning sequences from variation parquet files.

Loads variation parquets, calls variation_to_token_ids, and prints a
structured representation of the thinking sequence for each sample.
"""

import argparse
import json
import glob
import os
import random

import numpy as np
import pandas as pd

from src.finetune.data import variation_to_token_ids, _gumbel_reorder, _to_model_uci


def dump_sample(row, sample_idx, max_variations=3, max_depth=5, tau_base=0.3, tau_alpha=1.0):
    """Print a human-readable representation of one variation sample."""
    fen = row["fen"]
    mcts_action = row["mcts_action"]

    # Parse variations
    variations_raw = row["variations"]
    if isinstance(variations_raw, str):
        variations = json.loads(variations_raw)
    else:
        variations = variations_raw

    if not variations:
        print(f"=== Sample {sample_idx} === (no variations, skipped)")
        return

    # Sort and cap (same as variation_to_token_ids)
    variations = sorted(variations, key=lambda v: v.get("visit_count", 0), reverse=True)
    variations = variations[:max_variations]

    # Compute WDL and tau
    W = row["win"] if pd.notna(row["win"]) else 0.0
    D = row["draw"] if pd.notna(row["draw"]) else 0.0
    L = row["loss"] if pd.notna(row["loss"]) else 0.0
    root_wdl = (W, D, L)
    wdl_var = (W + L) - (W - L) ** 2
    tau = tau_base * (1.0 + tau_alpha * wdl_var)

    # Reorder
    variations, ranking = _gumbel_reorder(variations, tau_base, tau_alpha, root_wdl)

    # Also get token IDs for sequence length
    ids, thinking_move_data, final_move_data, value_data, block_boundaries, _, first_is_not_best = \
        variation_to_token_ids(row, max_variations=max_variations, max_depth=max_depth,
                               tau_base=tau_base, tau_alpha=tau_alpha)

    final_move_model = _to_model_uci(mcts_action)

    print(f"=== Sample {sample_idx} ===")
    print(f"FEN: {fen}")
    print(f"Ranking: {ranking}")
    print(f"Root WDL: [{W:.3f}, {D:.3f}, {L:.3f}]")
    print(f"tau: {tau:.3f} (wdl_var={wdl_var:.3f})")
    print("start_think")

    for var_idx, var in enumerate(variations):
        root_move = var["root_move"]
        visit_count = var.get("visit_count", 0)
        visit_fraction = var.get("visit_fraction", 0.0)
        nodes = var.get("nodes", [])
        available_depth = min(max_depth, len(nodes))

        print(f"  {root_move} (visits: {visit_count}, frac: {visit_fraction:.3f})")

        for node_idx, node in enumerate(nodes[:available_depth]):
            node_fen = node["fen"]
            node_wdl = node.get("wdl", [0.0, 0.0, 0.0])
            node_move = node.get("move", "")
            w, d, l = node_wdl
            # Truncate FEN for display
            fen_short = node_fen.split(" ")[0] if len(node_fen) > 40 else node_fen
            print(f"    {fen_short} [{w:.3f}, {d:.3f}, {l:.3f}]")
            if node_move and node_idx < available_depth - 1:
                print(f"      -> {node_move}")

        print("  end_var")

    print("end_think")
    print(f"final_move: {final_move_model} [{W:.3f}, {D:.3f}, {L:.3f}]")
    print(f"first_is_not_best: {first_is_not_best}")
    print(f"seq_length: {len(ids)} tokens")
    print()


def main():
    parser = argparse.ArgumentParser(description="Dump human-readable finetuning sequences")
    parser.add_argument("--parquet-dir", required=True, help="Directory containing variation parquet files")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples to dump")
    parser.add_argument("--max-variations", type=int, default=3, help="Max variations per sample")
    parser.add_argument("--max-depth", type=int, default=5, help="Max PV depth per variation")
    parser.add_argument("--tau-base", type=float, default=0.3, help="Base temperature for Plackett-Luce")
    parser.add_argument("--tau-alpha", type=float, default=1.0, help="WDL variance scaling for adaptive tau")
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {args.parquet_dir}")
        return

    sample_idx = 0
    for file_path in parquet_files:
        if sample_idx >= args.n_samples:
            break

        df = pd.read_parquet(file_path)
        indices = list(range(len(df)))
        random.shuffle(indices)

        for idx in indices:
            if sample_idx >= args.n_samples:
                break

            row = df.iloc[idx]
            variations_raw = row.get("variations", "[]")
            if isinstance(variations_raw, str):
                variations = json.loads(variations_raw)
            else:
                variations = variations_raw

            if not variations:
                continue

            dump_sample(
                row, sample_idx,
                max_variations=args.max_variations,
                max_depth=args.max_depth,
                tau_base=args.tau_base,
                tau_alpha=args.tau_alpha,
            )
            sample_idx += 1


if __name__ == "__main__":
    main()

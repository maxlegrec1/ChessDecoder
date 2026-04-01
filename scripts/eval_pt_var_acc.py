"""
Evaluate pt_best_acc and var_best_acc on 1k samples using the C++ engine.

Usage:
    uv run python scripts/eval_pt_var_acc.py \
        --export-dir export_eval_end \
        --n 1000 --seed 42
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import _decoder_inference_cpp as cpp

_PSEUDO_TO_STANDARD = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}


def _normalize_castling(move):
    return _PSEUDO_TO_STANDARD.get(move, move)


def _sample_one_per_game(df, seed):
    def _pick(g):
        game_seed = hash((seed, g.name)) % (2**31)
        return g.sample(1, random_state=game_seed)
    return df.groupby("game_id", group_keys=False).apply(_pick).reset_index(drop=True)


_STANDARD_START_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _filter_standard_games(df):
    """Remove non-standard games (Chess960, puzzles, etc.) by checking starting FEN."""
    origin = df.loc[df.groupby("game_id")["ply"].idxmin()][["game_id", "fen"]]
    standard_ids = origin[
        origin["fen"].str.split(" ").str[0] == _STANDARD_START_BOARD
    ]["game_id"]
    return df[df["game_id"].isin(standard_ids)]


def load_pretrain_positions(data_dir, n, seed):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    rng = random.Random(seed)
    fname = rng.choice(files)
    print(f"  Pretrain file: {fname}", flush=True)
    df = pd.read_parquet(os.path.join(data_dir, fname),
                         columns=["fen", "best_move", "game_id", "ply"])
    n_before = df["game_id"].nunique()
    df = _filter_standard_games(df)
    n_after = df["game_id"].nunique()
    print(f"  Filtered {n_before - n_after} non-standard games ({n_before} -> {n_after})", flush=True)
    sampled = _sample_one_per_game(df, seed)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    seen = set()
    pairs = []
    for _, row in sampled.iterrows():
        fen = row["fen"]
        if fen not in seen:
            seen.add(fen)
            pairs.append({"fen": fen, "best_move": _normalize_castling(row["best_move"])})
        if len(pairs) >= n:
            break
    print(f"  Loaded {len(pairs)} pretrain positions", flush=True)
    return pairs


def load_variation_positions(data_dir, n, seed):
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    rng = random.Random(seed)
    chosen_files = rng.sample(files, min(3, len(files)))
    print(f"  Variation files: {[os.path.basename(f) for f in chosen_files]}", flush=True)
    dfs = []
    for f in chosen_files:
        df = pd.read_parquet(f, columns=["fen", "best_move", "mcts_action", "game_id"])
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["mcts_action"].notna() & (combined["mcts_action"] != "")]
    sampled = _sample_one_per_game(combined, seed)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    seen = set()
    unique = []
    for _, r in sampled.iterrows():
        if r["fen"] not in seen:
            seen.add(r["fen"])
            unique.append({
                "fen": r["fen"],
                "mcts_action": _normalize_castling(r["mcts_action"]),
                "best_move": _normalize_castling(r["best_move"]),
            })
        if len(unique) >= n:
            break
    print(f"  Loaded {len(unique)} variation positions", flush=True)
    return unique


def evaluate(engine, positions, label, ground_truth_key="best_move"):
    correct = 0
    total = 0
    t0 = time.time()
    for i, pos in enumerate(positions):
        move = engine.predict_move(pos["fen"], 0.0)
        if move:
            move = _normalize_castling(move)
            total += 1
            if move == pos[ground_truth_key]:
                correct += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            acc = correct / total if total else 0
            print(f"  [{label}] {i+1}/{len(positions)}  "
                  f"acc={acc:.3f} ({correct}/{total})  "
                  f"{elapsed:.1f}s", flush=True)
    elapsed = time.time() - t0
    acc = correct / total if total else 0
    return {"correct": correct, "total": total, "acc": acc, "time": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True, help="Exported model directory")
    parser.add_argument("--pretrain-dir", default="/home/maxime/parquet_files_decoder/")
    parser.add_argument("--variation-dir", default="parquets_variations/")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)

    # Load positions
    print("Loading positions...", flush=True)
    pt_positions = load_pretrain_positions(args.pretrain_dir, args.n, args.seed)
    var_positions = load_variation_positions(args.variation_dir, args.n, args.seed)

    # Load engine
    print(f"\nLoading engine from {export_dir}...", flush=True)
    engine = cpp.ThinkingInferenceEngine(
        str(export_dir / "backbone.pt"),
        str(export_dir / "weights"),
        str(export_dir / "vocab.json"),
        str(export_dir / "config.json"),
    )

    # Evaluate
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating var_mcts_acc ({len(var_positions)} positions)...", flush=True)
    var_mcts = evaluate(engine, var_positions, "var_mcts", "mcts_action")

    print(f"\nEvaluating var_best_acc ({len(var_positions)} positions)...", flush=True)
    var_best = evaluate(engine, var_positions, "var_best", "best_move")

    print(f"\nEvaluating pt_best_acc ({len(pt_positions)} positions)...", flush=True)
    pt_best = evaluate(engine, pt_positions, "pt_best", "best_move")

    # Results
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS  ({args.export_dir}, n={args.n}, seed={args.seed})", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  var_mcts_acc:  {var_mcts['acc']:.3f}  ({var_mcts['correct']}/{var_mcts['total']})  [{var_mcts['time']:.1f}s]", flush=True)
    print(f"  var_best_acc:  {var_best['acc']:.3f}  ({var_best['correct']}/{var_best['total']})  [{var_best['time']:.1f}s]", flush=True)
    print(f"  pt_best_acc:   {pt_best['acc']:.3f}  ({pt_best['correct']}/{pt_best['total']})  [{pt_best['time']:.1f}s]", flush=True)
    print(f"\n  var_mcts - pt_best gap: {var_mcts['acc'] - pt_best['acc']:+.3f}", flush=True)
    print(f"  var_best - pt_best gap: {var_best['acc'] - pt_best['acc']:+.3f}", flush=True)


if __name__ == "__main__":
    main()

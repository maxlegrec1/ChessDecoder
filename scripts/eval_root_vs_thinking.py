"""
Compare root policy (no thinking) vs thinking inference accuracy.

Usage:
    uv run python scripts/eval_root_vs_thinking.py \
        --export-dir export_eval_new \
        --n 1000 --seed 42
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import _decoder_inference_cpp as cpp

_PS = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
_STANDARD_START_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def norm(m):
    return _PS.get(m, m) if isinstance(m, str) else m


def _sample_one_per_game(df, seed):
    def _pick(g):
        gs = hash((seed, g.name)) % (2**31)
        return g.sample(1, random_state=gs)
    return df.groupby("game_id", group_keys=False).apply(_pick).reset_index(drop=True)


def _filter_standard_games(df):
    origin = df.loc[df.groupby("game_id")["ply"].idxmin()][["game_id", "fen"]]
    standard_ids = origin[
        origin["fen"].str.split(" ").str[0] == _STANDARD_START_BOARD
    ]["game_id"]
    return df[df["game_id"].isin(standard_ids)]


def load_positions(data_dir, n, seed):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    rng = random.Random(seed)
    fname = rng.choice(files)
    print(f"  File: {fname}", flush=True)
    df = pd.read_parquet(os.path.join(data_dir, fname),
                         columns=["fen", "best_move", "game_id", "ply"])
    n_before = df["game_id"].nunique()
    df = _filter_standard_games(df)
    n_after = df["game_id"].nunique()
    print(f"  Filtered {n_before - n_after} non-standard games", flush=True)
    sampled = _sample_one_per_game(df, seed)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    seen = set()
    pairs = []
    for _, r in sampled.iterrows():
        if r["fen"] not in seen:
            seen.add(r["fen"])
            pairs.append({"fen": r["fen"], "best_move": norm(r["best_move"])})
        if len(pairs) >= n:
            break
    print(f"  Loaded {len(pairs)} positions", flush=True)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--data-dir", default="/home/maxime/parquet_files_decoder/")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)

    print("Loading positions...", flush=True)
    positions = load_positions(args.data_dir, args.n, args.seed)

    print(f"\nLoading engine from {export_dir}...", flush=True)
    engine = cpp.ThinkingInferenceEngine(
        str(export_dir / "backbone.pt"),
        str(export_dir / "weights"),
        str(export_dir / "vocab.json"),
        str(export_dir / "config.json"),
    )

    n = len(positions)

    # Root policy (no thinking)
    print(f"\nEvaluating root policy ({n} positions)...", flush=True)
    root_correct = 0
    t0 = time.time()
    root_moves = []
    for i, p in enumerate(positions):
        move = engine.predict_move_root(p["fen"], 0.0)
        move = norm(move) if move else None
        root_moves.append(move)
        if move and move == p["best_move"]:
            root_correct += 1
        if (i + 1) % 200 == 0:
            print(f"  [root] {i+1}/{n} acc={root_correct/(i+1):.3f}", flush=True)
    root_time = time.time() - t0
    root_acc = root_correct / n

    # Thinking inference
    print(f"\nEvaluating thinking inference ({n} positions)...", flush=True)
    think_correct = 0
    t0 = time.time()
    think_moves = []
    for i, p in enumerate(positions):
        move = engine.predict_move(p["fen"], 0.0)
        move = norm(move) if move else None
        think_moves.append(move)
        if move and move == p["best_move"]:
            think_correct += 1
        if (i + 1) % 200 == 0:
            print(f"  [think] {i+1}/{n} acc={think_correct/(i+1):.3f}", flush=True)
    think_time = time.time() - t0
    think_acc = think_correct / n

    # Per-position comparison
    both_correct = 0
    root_only = 0
    think_only = 0
    both_wrong = 0
    agree = 0
    for i in range(n):
        rc = root_moves[i] == positions[i]["best_move"]
        tc = think_moves[i] == positions[i]["best_move"]
        if rc and tc:
            both_correct += 1
        elif rc and not tc:
            root_only += 1
        elif not rc and tc:
            think_only += 1
        else:
            both_wrong += 1
        if root_moves[i] == think_moves[i]:
            agree += 1

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS  (n={n}, seed={args.seed})", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Root policy (no thinking): {root_correct}/{n} = {root_acc:.1%}  [{root_time:.0f}s]", flush=True)
    print(f"  Thinking inference:        {think_correct}/{n} = {think_acc:.1%}  [{think_time:.0f}s]", flush=True)
    print(f"  Thinking benefit:          {think_acc - root_acc:+.1%}", flush=True)
    print(f"  Speed ratio:               {think_time/root_time:.1f}x slower with thinking", flush=True)
    print(f"\n  Per-position breakdown:", flush=True)
    print(f"    Both correct:              {both_correct:4d} ({both_correct/n:.1%})", flush=True)
    print(f"    Root correct, think wrong: {root_only:4d} ({root_only/n:.1%})", flush=True)
    print(f"    Think correct, root wrong: {think_only:4d} ({think_only/n:.1%})", flush=True)
    print(f"    Both wrong:                {both_wrong:4d} ({both_wrong/n:.1%})", flush=True)
    print(f"    Move agreement:            {agree:4d} ({agree/n:.1%})", flush=True)


if __name__ == "__main__":
    main()

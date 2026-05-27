"""Pre-tokenize every parquet shard into a fast-load .npz cache.

The training loader's per-shard cost (pd.read_parquet -> sort -> tokenize
1.4M FENs through python loops) is ~30-40s per shard, which dominates GPU
idle time at our small-model scale. Doing it once offline and dumping a
flat .npz turns shard load into a ~0.5s SSD read.

Layout: one .npz per parquet, written to ``--cache-dir`` (default
``/mnt/2tb_2/decoder/cached_decoder/``). Each .npz holds:

  board_ids    [R, 68] int16    token ids
  policy_tgt   [R]     int32    move-sub-vocab id or IGNORE_INDEX
  policy_valid [R]     bool
  wdl_mean     [R, 3]  float16  exact (W, D, L) for metrics
  wdl_valid    [R]     bool
  q            [R]     float16  for project_targets at gather time
  d            [R]     float16
  bounds       [G+1]   int32    game-row boundaries (cumulative)

Idempotent: shards whose .npz exists *and* has every expected key are
skipped, so the script is safe to rerun (and to interrupt).
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chessdecoder.dataloader.loader import IGNORE_INDEX, fen_to_ids  # noqa: E402
from chessdecoder.models.vocab import full_idx_to_move_idx, token_to_idx  # noqa: E402

EXPECTED_KEYS = ["board_ids", "policy_tgt", "policy_valid",
                 "wdl_mean", "wdl_valid", "q", "d", "bounds"]


def _is_done(out: Path) -> bool:
    if not out.exists():
        return False
    try:
        with np.load(out) as z:
            return all(k in z.files for k in EXPECTED_KEYS)
    except Exception:
        return False


def convert_one(args):
    parquet_path, cache_dir = args
    name = Path(parquet_path).stem
    out = Path(cache_dir) / f"{name}.npz"
    if _is_done(out):
        return f"skip {name}"

    df = pd.read_parquet(parquet_path)
    df = df[df["played_move"].astype(bool)]
    df = df.sort_values(["game_id", "ply"], kind="stable")
    R = len(df)
    if R == 0:
        return f"empty {name}"

    # board_ids: tokenize FENs in bulk with the fast pure-Python parser.
    fens = df["fen"].to_numpy()
    board_ids = np.empty((R, 68), dtype=np.int16)
    tmp = np.empty(68, dtype=np.int32)
    for i in range(R):
        fen_to_ids(fens[i], tmp)
        board_ids[i] = tmp                                    # int32 -> int16

    # policy_tgt: best_move (full vocab) -> move sub-vocab id.
    best = df["best_move"].to_numpy()
    policy_tgt = np.full(R, IGNORE_INDEX, dtype=np.int32)
    policy_valid = np.zeros(R, dtype=bool)
    for i in range(R):
        m = best[i]
        if m and m in token_to_idx:
            sub = full_idx_to_move_idx.get(token_to_idx[m])
            if sub is not None:
                policy_tgt[i] = sub
                policy_valid[i] = True

    # WDL: orig_q / orig_d as fp16, derived (W,D,L) mean for metrics.
    q_raw = df["orig_q"].to_numpy(dtype=np.float32, na_value=np.nan)
    d_raw = df["orig_d"].to_numpy(dtype=np.float32, na_value=np.nan)
    valid = (~np.isnan(q_raw)) & (~np.isnan(d_raw))
    q = np.clip(q_raw, -1.0, 1.0)
    d = np.clip(d_raw, 0.0, 1.0)
    w = np.clip((1.0 - d + q) * 0.5, 0.0, 1.0)
    l_ = np.clip((1.0 - d - q) * 0.5, 0.0, 1.0)
    wdl_mean = np.stack([w, d, l_], axis=-1)
    s = wdl_mean.sum(-1, keepdims=True)
    np.divide(wdl_mean, np.maximum(s, 1e-8), out=wdl_mean)
    wdl_mean[~valid] = 0.0

    # game boundaries -> int32 (each game is < a few hundred rows; cumulative
    # indices fit in int32 even when concatenated across shards).
    gids = df["game_id"].to_numpy()
    change = np.flatnonzero(gids[1:] != gids[:-1]) + 1
    bounds = np.concatenate([[0], change, [R]]).astype(np.int32)

    tmp_out = out.with_suffix(".tmp.npz")
    np.savez(tmp_out,
             board_ids=board_ids,
             policy_tgt=policy_tgt,
             policy_valid=policy_valid,
             wdl_mean=wdl_mean.astype(np.float16),
             wdl_valid=valid,
             q=q.astype(np.float16),
             d=d.astype(np.float16),
             bounds=bounds)
    os.replace(tmp_out, out)
    sz = out.stat().st_size / 1e6
    return f"done {name}: {R:,} rows, {sz:.0f}MB"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                    default="/mnt/2tb_2/decoder/parquet_files_decoder")
    ap.add_argument("--cache-dir",
                    default="/mnt/2tb_2/decoder/cached_decoder")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel converter processes. Default 2 to share "
                         "disk bandwidth nicely with an in-flight training run.")
    args = ap.parse_args(argv)

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    parquets = sorted(glob.glob(os.path.join(args.source, "*.parquet")))
    print(f"{len(parquets)} parquet shards -> {args.cache_dir}", flush=True)

    jobs = [(p, args.cache_dir) for p in parquets]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(convert_one, jobs)):
            dt = time.time() - t0
            eta = (dt / max(1, i + 1)) * (len(parquets) - i - 1)
            print(f"[{i+1}/{len(parquets)}]  {msg}  "
                  f"(elapsed {dt:.0f}s, eta {eta:.0f}s)", flush=True)
    print(f"all done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

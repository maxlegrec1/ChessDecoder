"""Encoder-only dataloader.

Rows are grouped by ``game_id`` (so we sample positions from across the game
population rather than reading sequentially through a parquet — that would
over-represent neighbouring plies of the same game). For each game we sample
``positions_per_game`` random plies and emit per-game arrays of shape
``[N, ...]``. The training loop reshapes ``[B, N, 68] -> [B*N, 68]`` and the
encoder processes every position independently.

Schema produced per game:
  - board_ids    [N, 68]  : board token ids (encoder input)
  - policy_tgt   [N]      : best-move move-sub-vocab id (IGNORE_INDEX if invalid)
  - policy_valid [N]      : mask for the policy loss
  - wdl_tgt      [N, C]   : soft-categorical target on the 2-D-simplex grid
  - wdl_mean     [N, 3]   : exact target WDL (W,D,L) — for metric only
  - wdl_valid    [N]      : mask for the WDL loss

If a game yields fewer than ``positions_per_game`` valid rows we sample with
replacement (keeps a fixed [N, ...] shape so FP8 + compile sees fixed shapes).
"""
import glob
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader

from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.vocab import (token_to_idx, full_idx_to_move_idx)
from chessdecoder.models.value_buckets import N_CELLS, project_targets

IGNORE_INDEX = -100


def game_to_arrays(game_df, positions_per_game):
    """Returns the per-game arrays described in the module docstring, or
    ``None`` if the game has no rows with a played move (caller skips)."""
    rows = [r for r in game_df.itertuples(index=False)
            if getattr(r, "played_move", None)]
    if not rows:
        return None

    if len(rows) >= positions_per_game:
        idx = random.sample(range(len(rows)), positions_per_game)
    else:
        idx = [random.randrange(len(rows)) for _ in range(positions_per_game)]
    rows = [rows[i] for i in idx]

    N = positions_per_game
    board_ids = torch.zeros(N, 68, dtype=torch.long)
    policy_tgt = torch.full((N,), IGNORE_INDEX, dtype=torch.long)
    policy_valid = torch.zeros(N, dtype=torch.bool)
    wdl_mean = torch.zeros(N, 3, dtype=torch.float32)
    wdl_tgt = torch.zeros(N, N_CELLS, dtype=torch.float32)
    wdl_valid = torch.zeros(N, dtype=torch.bool)

    q_buf, d_buf, q_idx = [], [], []
    for i, row in enumerate(rows):
        toks = fen_to_position_tokens(row.fen)
        board_ids[i] = torch.tensor([token_to_idx[t] for t in toks])

        bm = getattr(row, "best_move", None)
        if bm is not None and bm in token_to_idx:
            policy_tgt[i] = full_idx_to_move_idx[token_to_idx[bm]]
            policy_valid[i] = True

        q, dd = getattr(row, "orig_q", None), getattr(row, "orig_d", None)
        if q is not None and pd.notna(q) and dd is not None and pd.notna(dd):
            q, dd = float(q), float(dd)
            w = (1.0 - dd + q) / 2.0
            l = (1.0 - dd - q) / 2.0
            v = torch.tensor([w, dd, l], dtype=torch.float32).clamp_(0.0, 1.0)
            wdl_mean[i] = v / v.sum().clamp_(min=1e-8)
            q_buf.append(max(-1.0, min(1.0, w - l)))
            d_buf.append(dd)
            q_idx.append(i)
            wdl_valid[i] = True

    if q_idx:
        cat = project_targets(torch.tensor(q_buf), torch.tensor(d_buf))
        wdl_tgt[torch.tensor(q_idx)] = cat

    return {
        "board_ids": board_ids,
        "policy_tgt": policy_tgt, "policy_valid": policy_valid,
        "wdl_tgt": wdl_tgt, "wdl_mean": wdl_mean, "wdl_valid": wdl_valid,
    }


class ChessIterableDataset(IterableDataset):
    """Iterate parquet shards, group rows by ``game_id``, yield per-game arrays."""

    def __init__(self, parquet_dir, positions_per_game=8, shuffle_files=True,
                 shuffle_games=True, seed=42, rank=0, world_size=1):
        self.parquet_dir = parquet_dir
        self.positions_per_game = positions_per_game
        self.shuffle_files = shuffle_files
        self.shuffle_games = shuffle_games
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self.files:
            print(f"Warning: No parquet files found in {parquet_dir}")

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else 0
        es = self.seed + self.epoch * 100003 + self.rank * 7919 + wid * 997
        random.seed(es)
        np.random.seed(es % (2**32))

        rank_files = self.files[self.rank::self.world_size]
        if wi is None:
            files = list(rank_files)
        else:
            per = int(math.ceil(len(rank_files) / float(wi.num_workers)))
            files = rank_files[wid * per: min((wid + 1) * per, len(rank_files))]
        if self.shuffle_files:
            random.shuffle(files)

        for fp in files:
            try:
                df = pd.read_parquet(fp)
                # Sort once by (game_id, ply) so each game's rows are a
                # contiguous slice. ~3s for a 1.4M-row parquet vs minutes of
                # pandas overhead from per-game .get_group() / .indices.
                df = df.sort_values(["game_id", "ply"], kind="stable")
                gids = df["game_id"].to_numpy()
                # Group boundaries: indices where game_id changes + edges.
                change = np.flatnonzero(gids[1:] != gids[:-1]) + 1
                bounds = np.concatenate(
                    [[0], change, [len(gids)]]).astype(np.int64)
                ngames = len(bounds) - 1
                order = np.arange(ngames)
                if self.shuffle_games:
                    np.random.shuffle(order)
                for gi in order:
                    s, e = bounds[gi], bounds[gi + 1]
                    gdf = df.iloc[s:e]                    # already ply-sorted
                    sample = game_to_arrays(gdf, self.positions_per_game)
                    if sample is not None:
                        yield sample
            except Exception as e:
                print(f"Error reading file {fp}: {e}")
                continue


def get_dataloader(parquet_dir, batch_size=16, num_workers=0,
                   positions_per_game=8, seed=42, rank=0, world_size=1):
    ds = ChessIterableDataset(parquet_dir, positions_per_game=positions_per_game,
                              seed=seed, rank=rank, world_size=world_size)
    # spawn context for num_workers>0: workers must not fork after the parent
    # initialized CUDA (FP8 model on the GPU). Forked workers would inherit a
    # broken CUDA context and hang on the first batch. persistent_workers
    # amortizes spawn's ~10-30s startup over the whole training run.
    kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs), ds

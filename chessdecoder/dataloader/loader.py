"""Encoder-only dataloader (pre-tokenized shard cache).

A shard (one parquet file, ~1.4M rows) is loaded once and immediately
tokenized in bulk into NumPy/Tensor arrays kept on the worker. Per-batch work
then collapses to (a) pick game-row indices, (b) gather precomputed rows.

Schema produced per game:
  - board_ids    [N, 68]  : board token ids
  - policy_tgt   [N]      : best-move move-sub-vocab id (-100 if invalid)
  - policy_valid [N]      : mask for the policy loss
  - wdl_tgt      [N, C]   : soft-categorical target on the 2-D simplex grid
  - wdl_mean     [N, 3]   : exact target WDL (W,D,L) — used as a metric only
  - wdl_valid    [N]      : mask for the WDL loss

If a game yields fewer than ``positions_per_game`` valid rows we sample with
replacement (keeps a fixed [N, ...] shape so FP8 + compile sees fixed shapes).
"""
import gc
import glob
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader

from chessdecoder.models.vocab import (token_to_idx, full_idx_to_move_idx,
                                       castling_tokens)
from chessdecoder.models.value_buckets import N_CELLS, project_targets

IGNORE_INDEX = -100


# ---------- fast FEN tokenization (no python-chess) -------------------------

# Map FEN piece chars to their vocab token ids. python-chess names pieces as
# {white,black}_{king,queen,rook,bishop,knight,pawn}.
_PIECE_IDX = {
    "P": token_to_idx["white_pawn"], "N": token_to_idx["white_knight"],
    "B": token_to_idx["white_bishop"], "R": token_to_idx["white_rook"],
    "Q": token_to_idx["white_queen"], "K": token_to_idx["white_king"],
    "p": token_to_idx["black_pawn"], "n": token_to_idx["black_knight"],
    "b": token_to_idx["black_bishop"], "r": token_to_idx["black_rook"],
    "q": token_to_idx["black_queen"], "k": token_to_idx["black_king"],
}
_EMPTY_IDX = token_to_idx["empty"]
_START_IDX = token_to_idx["start_pos"]
_END_IDX = token_to_idx["end_pos"]
_WTM_IDX = token_to_idx["white_to_move"]
_BTM_IDX = token_to_idx["black_to_move"]
_NO_CASTLE_IDX = token_to_idx["no_castling_rights"]
# castling_tokens already enumerates every subset of "KQkq" in canonical order
# (matching the FEN castling field), so a direct lookup works.
_CASTLE_IDX = {c: token_to_idx[c] for c in castling_tokens
               if c != "no_castling_rights"}


def fen_to_ids(fen: str, out: np.ndarray) -> None:
    """Write 68 token ids into ``out[:]`` for a single FEN."""
    out[0] = _START_IDX
    parts = fen.split(" ", 4)
    pieces, side, castling = parts[0], parts[1], parts[2]
    # squares are written in chess.SQUARES order: a1, b1, ..., h8 (rank-major,
    # rank 1 at the bottom). FEN walks rank 8 -> rank 1, file a -> h.
    sq = np.empty(64, dtype=np.int32)
    sq.fill(_EMPTY_IDX)
    for fen_rank, rank_str in enumerate(pieces.split("/")):
        chess_rank = 7 - fen_rank
        f = 0
        base = chess_rank * 8
        for ch in rank_str:
            d = ord(ch) - 48
            if 0 <= d <= 8:
                f += d
            else:
                sq[base + f] = _PIECE_IDX[ch]
                f += 1
    out[1:65] = sq
    out[65] = _END_IDX
    out[66] = _NO_CASTLE_IDX if castling == "-" else _CASTLE_IDX[castling]
    out[67] = _WTM_IDX if side == "w" else _BTM_IDX


def tokenize_shard(df: pd.DataFrame):
    """Tokenize every row of a shard into compact arrays.

    Returns a dict of NumPy/Tensor arrays of length R (the number of rows).
    Rows lacking ``played_move`` are dropped first (callers rely on this so
    every row corresponds to a valid training position).
    """
    df = df[df["played_move"].astype(bool)]
    R = len(df)
    if R == 0:
        return None

    fens = df["fen"].to_numpy()
    board_ids = np.empty((R, 68), dtype=np.int32)
    for i in range(R):
        fen_to_ids(fens[i], board_ids[i])

    # policy target: map best_move -> move sub-vocab id (or IGNORE_INDEX).
    best = df["best_move"].to_numpy()
    policy_tgt = np.full(R, IGNORE_INDEX, dtype=np.int64)
    policy_valid = np.zeros(R, dtype=bool)
    for i in range(R):
        m = best[i]
        if m and m in token_to_idx:
            full = token_to_idx[m]
            sub = full_idx_to_move_idx.get(full)
            if sub is not None:
                policy_tgt[i] = sub
                policy_valid[i] = True

    # WDL: orig_q / orig_d -> (W,D,L). Vectorized across the whole shard.
    # We *don't* expand the [R, N_CELLS] cell-simplex target upfront — at
    # ~1.4M rows x 405 cells x 4 bytes that's 2GB of mostly-cold floats per
    # shard. The cached q/d are tiny (~22MB) and projecting just the picked
    # rows per batch costs <1 ms.
    q_raw = df["orig_q"].to_numpy(dtype=np.float32, na_value=np.nan)
    d_raw = df["orig_d"].to_numpy(dtype=np.float32, na_value=np.nan)
    valid = (~np.isnan(q_raw)) & (~np.isnan(d_raw))
    q = np.clip(q_raw, -1.0, 1.0).astype(np.float32)
    d = np.clip(d_raw, 0.0, 1.0).astype(np.float32)
    w = np.clip((1.0 - d + q) * 0.5, 0.0, 1.0)
    l_ = np.clip((1.0 - d - q) * 0.5, 0.0, 1.0)
    wdl_mean = np.stack([w, d, l_], axis=-1)
    s = wdl_mean.sum(-1, keepdims=True)
    np.divide(wdl_mean, np.maximum(s, 1e-8), out=wdl_mean)
    wdl_mean[~valid] = 0.0

    return {
        # int32 cache (half the RAM of int64); promote to long on gather.
        "board_ids": torch.from_numpy(board_ids),         # int32 [R, 68]
        "policy_tgt": torch.from_numpy(policy_tgt),       # int64 [R]
        "policy_valid": torch.from_numpy(policy_valid),   # bool  [R]
        "wdl_mean": torch.from_numpy(wdl_mean),           # f32   [R, 3]
        "wdl_valid": torch.from_numpy(valid),             # bool  [R]
        "q": torch.from_numpy(q),                         # f32   [R]
        "d": torch.from_numpy(d),                         # f32   [R]
        "_game_id": df["game_id"].to_numpy(),
    }


# ---------- per-game sampler ------------------------------------------------

def _pick_rows(s: int, e: int, n: int) -> np.ndarray:
    """Pick ``n`` row indices uniformly from [s, e). Sampling without
    replacement if the game has enough rows, with replacement otherwise."""
    k = e - s
    if k >= n:
        return np.random.choice(k, n, replace=False).astype(np.int64) + s
    return np.random.randint(s, e, size=n, dtype=np.int64)


def _gather_game(shard, idx_arr):
    """Slice every per-row tensor at ``idx_arr`` into a [N, ...] dict.

    We *do not* call ``project_targets`` here — at B=2048, N=1 that's 2048
    small per-yield torch ops in worker land (CPython overhead dominates).
    Instead the training loop runs ``project_targets`` once per batch on
    GPU, on the concatenated [B*N] q/d, which is essentially free.
    """
    t = torch.as_tensor(idx_arr, dtype=torch.long)
    return {
        "board_ids":    shard["board_ids"][t].to(torch.long),  # [N, 68]
        "policy_tgt":   shard["policy_tgt"][t],                # [N]
        "policy_valid": shard["policy_valid"][t],              # [N]
        "wdl_mean":     shard["wdl_mean"][t],                  # [N, 3]
        "wdl_valid":    shard["wdl_valid"][t],                 # [N]
        "q":            shard["q"][t],                         # [N]  for GPU projection
        "d":            shard["d"][t],                         # [N]
    }


# ---------- cached-shard loader ---------------------------------------------

def _load_cached_shard(fp: str):
    """Load a .npz cache produced by ``scripts/precompute_decoder_cache.py``.

    Reads in ~0.5s on SATA SSD vs ~30-40s for parquet+tokenize. Returns the
    same dict the parquet path produces (minus the now-unneeded ``_game_id``;
    the cache already stores game boundaries directly).
    """
    with np.load(fp) as z:
        return {
            "board_ids": torch.from_numpy(z["board_ids"].astype(np.int32)),
            "policy_tgt": torch.from_numpy(z["policy_tgt"].astype(np.int64)),
            "policy_valid": torch.from_numpy(z["policy_valid"]),
            "wdl_mean": torch.from_numpy(z["wdl_mean"].astype(np.float32)),
            "wdl_valid": torch.from_numpy(z["wdl_valid"]),
            "q": torch.from_numpy(z["q"].astype(np.float32)),
            "d": torch.from_numpy(z["d"].astype(np.float32)),
            "_bounds": z["bounds"].astype(np.int64),
        }


# ---------- iterable dataset ------------------------------------------------

class ChessIterableDataset(IterableDataset):
    """Iterate shards, yield per-game arrays.

    Two backends, chosen by extension at iteration time:
    - ``.npz`` from the offline cache: instant load, just tensor gather.
    - ``.parquet``: original path — pd.read_parquet + sort + tokenize.

    If ``cache_dir`` is set, the .npz cache wins; we fall back to the parquet
    dir for any shard the cache doesn't (yet) have. This keeps a half-built
    cache safe to use while the rest is still being precomputed.
    """

    def __init__(self, parquet_dir, positions_per_game=8, shuffle_files=True,
                 shuffle_games=True, seed=42, rank=0, world_size=1,
                 cache_dir=None):
        self.parquet_dir = parquet_dir
        self.cache_dir = cache_dir
        self.positions_per_game = positions_per_game
        self.shuffle_files = shuffle_files
        self.shuffle_games = shuffle_games
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        # Prefer cached .npz when available, else fall back to parquet.
        parquets = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        files = []
        n_cached = 0
        for p in parquets:
            stem = os.path.splitext(os.path.basename(p))[0]
            cached = (os.path.join(cache_dir, f"{stem}.npz")
                      if cache_dir else None)
            if cached and os.path.exists(cached):
                files.append(cached)
                n_cached += 1
            else:
                files.append(p)
        self.files = files
        if not self.files:
            print(f"Warning: No shards found in {parquet_dir} or {cache_dir}")
        elif cache_dir:
            print(f"Loader: {n_cached}/{len(self.files)} shards from cache, "
                  f"{len(self.files) - n_cached} from parquet")

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

        N = self.positions_per_game
        shard = None
        for fp in files:
            try:
                # Free the previous shard before reading the next — pandas
                # peaks at ~3-4GB during a parquet read+tokenize, and on a
                # 15GB box two workers reloading simultaneously is enough to
                # OOM-kill us (lost an attention-sweep run that way).
                if shard is not None:
                    del shard
                    shard = None
                    gc.collect()
                if fp.endswith(".npz"):
                    shard = _load_cached_shard(fp)
                    bounds = shard.pop("_bounds")
                else:
                    df = pd.read_parquet(fp)
                    df = df.sort_values(["game_id", "ply"], kind="stable")
                    shard = tokenize_shard(df)
                    del df
                    if shard is None:
                        continue
                    gids = shard["_game_id"]
                    change = np.flatnonzero(gids[1:] != gids[:-1]) + 1
                    bounds = np.concatenate(
                        [[0], change, [len(gids)]]).astype(np.int64)
                ngames = len(bounds) - 1
                order = np.arange(ngames)
                if self.shuffle_games:
                    np.random.shuffle(order)
                for gi in order:
                    s, e = int(bounds[gi]), int(bounds[gi + 1])
                    idx = _pick_rows(s, e, N)
                    yield _gather_game(shard, idx)
            except Exception as e:
                print(f"Error reading file {fp}: {e}")
                continue


def get_dataloader(parquet_dir, batch_size=16, num_workers=0,
                   positions_per_game=8, seed=42, rank=0, world_size=1,
                   cache_dir=None):
    ds = ChessIterableDataset(parquet_dir, positions_per_game=positions_per_game,
                              seed=seed, rank=rank, world_size=world_size,
                              cache_dir=cache_dir)
    # pin_memory: copy each batch into page-locked host memory so the
    # subsequent ``.to(device, non_blocking=True)`` in the training loop can
    # overlap H2D with the prior step's GPU compute. Without this the GPU
    # stalls on each transfer.
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True)
    if num_workers > 0:
        # spawn context: workers must not fork after the parent initialized
        # CUDA (FP8 model on the GPU). Forked workers would inherit a broken
        # CUDA context and hang on the first batch. persistent_workers
        # amortizes spawn's ~10-30s startup over the whole training run.
        # prefetch_factor=4: keep a deeper queue of ready batches so a slow
        # shard transition can't stall the GPU.
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(ds, **kwargs), ds

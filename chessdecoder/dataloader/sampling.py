"""Position sampling helpers for training and evaluation.

Shared between finetuning, RL, and eval scripts so they all sample from
pretrain/variation parquets with identical semantics: one position per
game, standard games only (Chess960, puzzles, etc. filtered out).
"""

import glob
import os
import random

import pandas as pd

from chessdecoder.utils.uci import normalize_castling

_STANDARD_START_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def sample_one_per_game(df, seed):
    """Sample 1 row per game with a per-game seed to avoid index bias."""
    def _pick(g):
        game_seed = hash((seed, g.name)) % (2**31)
        return g.sample(1, random_state=game_seed)
    return df.groupby("game_id", group_keys=False).apply(
        _pick, include_groups=False
    ).reset_index(drop=True)


def filter_standard_games(df):
    """Remove non-standard games (Chess960, puzzles, etc.) by checking starting FEN."""
    origin = df.loc[df.groupby("game_id")["ply"].idxmin()][["game_id", "fen"]]
    standard_ids = origin[
        origin["fen"].str.split(" ").str[0] == _STANDARD_START_BOARD
    ]["game_id"]
    return df[df["game_id"].isin(standard_ids)]


def load_pretrain_positions(data_dir, n, seed):
    """Load (fen, best_move) pairs from pretrain parquets.

    Samples 1 position per game, excludes non-standard games (Chess960, etc.).
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    rng = random.Random(seed)
    fname = rng.choice(files)
    df = pd.read_parquet(os.path.join(data_dir, fname),
                         columns=["fen", "best_move", "game_id", "ply"])
    df = filter_standard_games(df)
    sampled = sample_one_per_game(df, seed)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    seen = set()
    pairs = []
    for _, row in sampled.iterrows():
        fen = row["fen"]
        if fen not in seen:
            seen.add(fen)
            pairs.append({"fen": fen, "best_move": normalize_castling(row["best_move"])})
        if len(pairs) >= n:
            break
    return pairs


def load_variation_positions(data_dir, n, seed):
    """Load (fen, best_move, mcts_action) from variation parquets.

    Samples 1 position per game from 3 randomly chosen files.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    rng = random.Random(seed)
    chosen_files = rng.sample(files, min(3, len(files)))
    dfs = []
    for f in chosen_files:
        df = pd.read_parquet(f, columns=["fen", "best_move", "mcts_action", "game_id"])
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["mcts_action"].notna() & (combined["mcts_action"] != "")]
    sampled = sample_one_per_game(combined, seed)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    seen = set()
    unique = []
    for _, r in sampled.iterrows():
        if r["fen"] not in seen:
            seen.add(r["fen"])
            unique.append({
                "fen": r["fen"],
                "mcts_action": normalize_castling(r["mcts_action"]),
                "best_move": normalize_castling(r["best_move"]),
            })
        if len(unique) >= n:
            break
    return unique

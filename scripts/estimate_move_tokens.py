#!/usr/bin/env python3
"""
Estimate the average number of move tokens in the sequences produced by the dataloader.

This reuses the dataloader's slicing/truncation logic so you can report exactly what a
single epoch would expose to the model. The script samples one sequence per game using
the same random start index choice the dataset uses.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from src.dataloader.data import game_to_token_ids
from src.dataloader.loader import ChessIterableDataset


def iter_sequences(
    parquet_dir: Path,
    max_seq_len: int,
    limit: int | None,
    seed: int,
) -> Iterator[int]:
    """
    Yield the number of move tokens kept in each sequence (i.e., length of wdl_data)
    after applying the dataset's random start + truncation.
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset = ChessIterableDataset(str(parquet_dir), max_seq_len=max_seq_len, shuffle_files=False, shuffle_games=False)
    files = list(dataset.files)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    produced = 0
    for file_path in files:
        df = pd.read_parquet(file_path)
        game_ids = df["game_id"].unique()

        for game_id in game_ids:
            if limit is not None and produced >= limit:
                return

            game_df = df[df["game_id"] == game_id].sort_values("ply")
            ids, wdl_data, _block_boundaries, value_data = game_to_token_ids(game_df)

            if not ids:
                produced += 1
                yield 0
                continue

            valid_starts = [0] + [vd[1] + 1 for vd in value_data[:-1]]
            start_idx = random.choice(valid_starts)

            sliced_ids = ids[start_idx:]
            filtered_wdl = [
                (m_idx - start_idx, best, wdl, valid)
                for m_idx, best, wdl, valid in wdl_data
                if m_idx >= start_idx
            ]

            if len(sliced_ids) > max_seq_len:
                sliced_ids = sliced_ids[:max_seq_len]
                filtered_wdl = [d for d in filtered_wdl if d[0] < max_seq_len]

            yield len(filtered_wdl)
            produced += 1


def compute_stats(counts: Iterable[int]) -> tuple[int, float, float]:
    total = 0
    values = []
    for count in counts:
        values.append(count)
        total += 1
    if total == 0:
        return 0, 0.0, 0.0

    mean = sum(values) / total
    variance = sum((value - mean) ** 2 for value in values) / total
    std_dev = math.sqrt(variance)
    return total, mean, std_dev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate average move token count after dataloader slicing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--parquet-dir",
        type=Path,
        default=Path("parquets"),
        help="Directory with the parquet shards consumed by the dataloader.",
    )
    parser.add_argument(
        "-m",
        "--max-seq-len",
        type=int,
        default=2048,
        help="Sequence length used by the dataloader (truncation length).",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Stop after processing this many games.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed used for reproducible start selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = iter_sequences(args.parquet_dir, args.max_seq_len, args.limit, args.seed)
    processed, mean, std_dev = compute_stats(counts)

    if processed == 0:
        print("No games processed.")
        return

    print(f"Processed {processed} games.")
    print(f"Average move tokens per sequence: {mean:.3f}")
    print(f"Standard deviation: {std_dev:.3f}")


if __name__ == "__main__":
    main()


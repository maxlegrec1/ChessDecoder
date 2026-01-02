#!/usr/bin/env python3
"""
Extract `(fen, best_move)` pairs from the parquet files consumed by the dataloader.

This helper reuses the dataloader's file discovery so you can inspect the same inputs
that the training/inference pipeline sees without modifying the dataset itself.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from src.dataloader.loader import ChessIterableDataset


def iter_fen_move_pairs(
    parquet_dir: Path,
    max_pairs: int | None = None,
) -> Iterator[tuple[str, int | None, int | None, str, str]]:
    """
    Yield tuples `(source_file, game_id, ply, fen, best_move)` in the same ordering
    that the dataloader observes when shuffling is disabled.
    """
    dataset = ChessIterableDataset(str(parquet_dir), shuffle_files=False, shuffle_games=False)
    files = list(dataset.files)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    produced = 0
    for file_path in files:
        df = pd.read_parquet(file_path, columns=["game_id", "ply", "fen", "best_move"])
        for row in df.itertuples(index=False):
            best_move = getattr(row, "best_move")
            if not best_move:
                continue

            yield (
                Path(file_path).name,
                getattr(row, "game_id", None),
                getattr(row, "ply", None),
                getattr(row, "fen", ""),
                best_move,
            )

            produced += 1
            if max_pairs is not None and produced >= max_pairs:
                return


def write_csv(
    rows: Iterable[Sequence[str | int | None]],
    path: Path,
) -> None:
    """Stream rows to a CSV file so the extraction can handle large datasets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["source_file", "game_id", "ply", "fen", "best_move"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump (fen, best_move) pairs the dataloader consumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--parquet-dir",
        type=Path,
        default=Path("parquets"),
        help="Directory containing the parquet shards.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Stop after emitting this many pairs (useful for sampling).",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=None,
        help="If provided, writes the pairs to this CSV instead of printing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = iter_fen_move_pairs(args.parquet_dir, max_pairs=args.limit)

    if args.output_csv:
        write_csv(rows, args.output_csv)
    else:
        for source_file, game_id, ply, fen, best_move in rows:
            context = f"{source_file} | game {game_id or '?'} ply {ply or '?'}"
            print(f"{context} -> {best_move}\n{fen}\n")


if __name__ == "__main__":
    main()


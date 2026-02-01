#!/usr/bin/env python3
"""
Generate MCTS variation data for finetuning.

Reads existing parquets (fen, played_move, game_id, ply, ...),
reconstructs move history per game, runs Leela MCTS with variation
extraction on each position, and writes enriched parquets.

Usage:
    uv run python scripts/generate_variations.py \
        --parquet-dir parquets \
        --output-dir parquets_variations \
        --simulations 600 \
        --engine-path model_dynamic_leela.trt \
        --parallel-trees 128
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Ensure the project root is on the path for imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess

from src.mcts import LeelaMCTS


_STANDARD_CASTLING = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}


def _is_standard_game(origin_fen: str, moves: list[str]) -> bool:
    """Check that every move in the game is replayable with standard UCI.

    Returns False for Chess960 games (non-standard castling notation)
    or games with broken move chains.
    """
    board = chess.Board(origin_fen)
    for move_str in moves:
        uci = _STANDARD_CASTLING.get(move_str, move_str)
        try:
            move = board.parse_uci(uci)
        except (chess.InvalidMoveError, chess.IllegalMoveError):
            return False
        if move not in board.legal_moves:
            return False
        board.push(move)
    return True


def _normalize_history(moves: list[str]) -> list[str]:
    """Map the 4 standard-chess pseudo-castling notations to standard UCI."""
    return [_STANDARD_CASTLING.get(m, m) for m in moves]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MCTS variations for finetuning data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("parquets"),
        help="Input parquet directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("parquets_variations"),
        help="Output directory for enriched parquets.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=600,
        help="MCTS simulations per position.",
    )
    parser.add_argument(
        "--max-variations",
        type=int,
        default=5,
        help="Top-K variations to extract.",
    )
    parser.add_argument(
        "--max-variation-depth",
        type=int,
        default=20,
        help="Maximum PV depth per variation.",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default="model_dynamic_leela.trt",
        help="TRT engine file path.",
    )
    parser.add_argument(
        "--positions-per-game",
        type=int,
        default=8,
        help="Number of positions to sample per game (0 = all).",
    )
    parser.add_argument(
        "--games-limit",
        type=int,
        default=None,
        help="Max games per file (for testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already exist in output-dir.",
    )
    parser.add_argument(
        "--cpuct",
        type=float,
        default=1.5,
        help="PUCT exploration constant.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Policy temperature.",
    )
    parser.add_argument(
        "--parallel-trees",
        type=int,
        default=128,
        help="Number of parallel MCTS trees per batch.",
    )
    parser.add_argument(
        "--max-gpu-batch",
        type=int,
        default=256,
        help="Max GPU batch size for dynamic engine.",
    )
    return parser.parse_args()


def process_file(
    parquet_path: Path,
    output_path: Path,
    mcts: LeelaMCTS,
    args: argparse.Namespace,
) -> None:
    """Process a single parquet file using batched parallel MCTS."""
    df = pd.read_parquet(parquet_path)

    game_ids = df["game_id"].unique()
    if args.games_limit is not None:
        game_ids = game_ids[: args.games_limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    n_positions = 0
    n_skipped = 0
    t0 = time.perf_counter()
    pending: list[tuple[dict, str, list[str]]] = []  # (row_dict, origin_fen, history)

    def flush_batch(batch: list[tuple[dict, str, list[str]]]) -> None:
        nonlocal writer, n_positions
        positions = [(fen, hist) for _, fen, hist in batch]
        try:
            results = mcts.run_parallel(
                positions,
                simulations=args.simulations,
                max_variations=args.max_variations,
                max_variation_depth=args.max_variation_depth,
                engine_path=args.engine_path,
                max_batch_size=args.max_gpu_batch,
            )
        except Exception as e:
            print(f"  Batch failed: {e}", file=sys.stderr)
            results = [{"action": None, "value": None, "variations": []}] * len(batch)

        rows = []
        for (row, _, _), res in zip(batch, results):
            enriched = dict(row)
            enriched["variations"] = json.dumps(
                res.get("variations", []), ensure_ascii=False
            )
            enriched["mcts_action"] = res.get("action")
            val = res.get("value")
            enriched["mcts_value"] = (
                json.dumps(list(val), ensure_ascii=False) if val else None
            )
            rows.append(enriched)

        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        n_positions += len(rows)

    try:
        for game_id in tqdm(game_ids, desc=f"  {parquet_path.name}", unit="game"):
            game_df = df[df["game_id"] == game_id].sort_values("ply").reset_index(drop=True)
            all_moves = list(game_df["played_move"])
            origin_fen = game_df.iloc[0]["fen"]

            if not _is_standard_game(origin_fen, all_moves):
                n_skipped += 1
                continue

            n = len(game_df)
            if args.positions_per_game > 0 and args.positions_per_game < n:
                sampled = sorted(random.sample(range(n), args.positions_per_game))
            else:
                sampled = list(range(n))

            for idx in sampled:
                row = game_df.iloc[idx].to_dict()
                history = _normalize_history(all_moves[:idx])
                pending.append((row, origin_fen, history))
                if len(pending) >= args.parallel_trees:
                    flush_batch(pending)
                    pending = []

        if pending:
            flush_batch(pending)
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - t0
    n_games = len(game_ids) - n_skipped
    print(
        f"  Wrote {output_path.name}: {n_games} games, "
        f"{n_positions} positions in {elapsed:.1f}s "
        f"({n_positions / elapsed:.1f} pos/s)"
        f"{f' (skipped {n_skipped} non-standard games)' if n_skipped else ''}"
    )


def main() -> None:
    args = parse_args()

    parquet_files = sorted(args.parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {args.parquet_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet files in {args.parquet_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Simulations: {args.simulations}")
    print(f"Max variations: {args.max_variations}")
    print(f"Max variation depth: {args.max_variation_depth}")
    print(f"Positions per game: {args.positions_per_game or 'all'}")
    print(f"Engine: {args.engine_path}")
    print(f"Parallel trees: {args.parallel_trees}")
    print(f"Max GPU batch: {args.max_gpu_batch}")
    if args.games_limit:
        print(f"Games limit per file: {args.games_limit}")
    print()

    mcts = LeelaMCTS(
        engine_path=args.engine_path,
        simulations=args.simulations,
        cpuct=args.cpuct,
        temperature=args.temperature,
    )

    for parquet_path in parquet_files:
        output_path = args.output_dir / parquet_path.name
        if args.resume and output_path.exists():
            print(f"Skipping {parquet_path.name} (output exists)")
            continue

        print(f"Processing {parquet_path.name}...")
        process_file(parquet_path, output_path, mcts, args)


if __name__ == "__main__":
    main()

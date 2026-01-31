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
        --engine-path leela_minibatch.trt
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

from src.mcts import LeelaMCTS

# Leela stores castling as "king captures rook" (e1h1, e8a8, etc.)
# but the C++ chess library expects standard UCI (e1g1, e8c8, etc.)
_PSEUDO_TO_STANDARD_CASTLING = {
    "e1h1": "e1g1",
    "e1a1": "e1c1",
    "e8h8": "e8g8",
    "e8a8": "e8c8",
}


def _normalize_history(moves: list[str]) -> list[str]:
    """Convert pseudo-castling notation from parquet to standard UCI."""
    return [_PSEUDO_TO_STANDARD_CASTLING.get(m, m) for m in moves]


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
        default="leela_minibatch.trt",
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
    return parser.parse_args()


def process_game(
    game_df: pd.DataFrame,
    mcts: LeelaMCTS,
    simulations: int,
    max_variations: int,
    max_variation_depth: int,
    positions_per_game: int,
) -> list[dict]:
    """Process sampled positions in a single game, returning enriched rows."""
    game_df = game_df.sort_values("ply").reset_index(drop=True)
    all_moves = list(game_df["played_move"])

    # Choose which positions to run MCTS on
    n = len(game_df)
    if positions_per_game > 0 and positions_per_game < n:
        sampled_indices = sorted(random.sample(range(n), positions_per_game))
    else:
        sampled_indices = list(range(n))

    origin_fen = game_df.iloc[0]["fen"]
    rows = []

    for idx in sampled_indices:
        row = game_df.iloc[idx]
        history = _normalize_history(all_moves[:idx])

        try:
            result = mcts.run_with_variations(
                origin_fen,
                history,
                simulations=simulations,
                max_variations=max_variations,
                max_variation_depth=max_variation_depth,
            )
        except Exception as e:
            print(
                f"  Warning: MCTS failed for game={row['game_id']} "
                f"ply={row['ply']}: {e}",
                file=sys.stderr,
            )
            result = {"action": None, "value": None, "variations": []}

        enriched = row.to_dict()
        enriched["variations"] = json.dumps(
            result.get("variations", []), ensure_ascii=False
        )
        enriched["mcts_action"] = result.get("action")
        value = result.get("value")
        enriched["mcts_value"] = (
            json.dumps(list(value), ensure_ascii=False) if value else None
        )
        rows.append(enriched)

    return rows


def process_file(
    parquet_path: Path,
    output_path: Path,
    mcts: LeelaMCTS,
    args: argparse.Namespace,
) -> None:
    """Process a single parquet file."""
    df = pd.read_parquet(parquet_path)

    game_ids = df["game_id"].unique()
    if args.games_limit is not None:
        game_ids = game_ids[: args.games_limit]

    all_rows: list[dict] = []
    t0 = time.perf_counter()

    for game_id in tqdm(game_ids, desc=f"  {parquet_path.name}", unit="game"):
        game_df = df[df["game_id"] == game_id]
        enriched = process_game(
            game_df,
            mcts,
            args.simulations,
            args.max_variations,
            args.max_variation_depth,
            args.positions_per_game,
        )
        all_rows.extend(enriched)

    elapsed = time.perf_counter() - t0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(all_rows)
    pq.write_table(table, output_path)

    n_games = len(game_ids)
    n_positions = len(all_rows)
    print(
        f"  Wrote {output_path.name}: {n_games} games, "
        f"{n_positions} positions in {elapsed:.1f}s "
        f"({n_positions / elapsed:.1f} pos/s)"
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

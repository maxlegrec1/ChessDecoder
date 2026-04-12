"""Lichess-puzzle tactics accuracy, batched C++ engine, with rating/theme buckets.

Usage
-----
    # First run: download the puzzle CSV (~100MB zst, ~800MB csv, 4.9M puzzles)
    bash scripts/download_puzzles.sh

    # Basic run (1000 puzzles, any rating/theme, report overall + rating buckets)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_tactics.py \
        --export-dir exports/export_282k --max-puzzles 1000

    # Compare two models on the same puzzle set
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_tactics.py \
        --exports exports/export_282k=baseline \
                  exports/export_rl_new_step_25=rl_s25 \
        --max-puzzles 1000 \
        --rating-min 1400 --rating-max 2200 \
        --bucket-by rating

    # Focus on specific themes
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_tactics.py \
        --exports exports/export_282k=baseline \
        --max-puzzles 2000 \
        --themes-any fork pin skewer \
        --bucket-by theme --report-themes fork pin skewer
"""

import argparse
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable] + sys.argv)

from pathlib import Path

import _decoder_inference_cpp as cpp

from chessdecoder.eval.tactics import (
    Aggregate,
    PuzzleResult,
    aggregate,
    bucket_by_rating,
    bucket_by_theme,
    evaluate_puzzles,
    load_puzzles,
    theme_counts,
)


DEFAULT_RATING_BUCKETS = [0, 1000, 1400, 1800, 2200, 2600, 3200]
DEFAULT_REPORT_THEMES = [
    "mateIn1", "mateIn2", "mateIn3",
    "fork", "pin", "skewer",
    "sacrifice", "deflection", "decoy", "attraction",
    "discoveredAttack", "doubleCheck",
    "backRankMate", "hangingPiece",
    "endgame", "middlegame", "opening",
    "short", "long",
]


def build_engine(export_dir: str, batch_size: int):
    engine = cpp.BatchedInferenceEngine(
        f"{export_dir}/backbone.pt",
        f"{export_dir}/weights",
        f"{export_dir}/vocab.json",
        f"{export_dir}/config.json",
        batch_size,
    )
    for attr in ("board_temperature", "think_temperature",
                 "policy_temperature", "wl_temperature", "d_temperature"):
        setattr(engine, attr, 0.0)
    return engine


def parse_export_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        path, label = spec.split("=", 1)
    else:
        path, label = spec, Path(spec).name
    return path, label


def print_bucket_report(
    label: str,
    all_results: list[PuzzleResult],
    args,
):
    print(f"\n{'='*80}")
    print(f"  Model: {label}")
    print(f"{'='*80}")
    print(aggregate(all_results).format("OVERALL"))

    if args.bucket_by == "rating":
        print(f"\n  --- rating buckets ---")
        for (lo, hi), group in bucket_by_rating(all_results, args.rating_buckets):
            if not group:
                continue
            print()
            print(aggregate(group).format(f"rating [{lo:>4}, {hi:<4})"))
    elif args.bucket_by == "theme":
        print(f"\n  --- theme buckets ---")
        for theme, group in bucket_by_theme(all_results, args.report_themes):
            if not group:
                continue
            print()
            print(aggregate(group).format(f"theme '{theme}'"))


def print_paired_deltas(
    all_results_by_label: dict[str, list[PuzzleResult]],
):
    """For each pair of labels, show paired deltas on first_move accuracy."""
    labels = list(all_results_by_label.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*80}")
    print(f"  Paired deltas on first_move_accuracy")
    print(f"{'='*80}")
    print(f"  {'pair (A vs B)':40} {'A_wins':>8} {'B_wins':>8} {'ties_both_right':>18} {'Δ':>8}")

    # Align results by puzzle_id — they're evaluated on the same set, same order
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            A, B = labels[i], labels[j]
            ra = all_results_by_label[A]
            rb = all_results_by_label[B]
            # Index by puzzle id to be safe
            by_id_a = {r.puzzle.puzzle_id: r for r in ra}
            by_id_b = {r.puzzle.puzzle_id: r for r in rb}
            common = sorted(set(by_id_a) & set(by_id_b))
            a_wins = sum(1 for pid in common
                         if by_id_a[pid].first_move_correct and not by_id_b[pid].first_move_correct)
            b_wins = sum(1 for pid in common
                         if not by_id_a[pid].first_move_correct and by_id_b[pid].first_move_correct)
            both = sum(1 for pid in common
                       if by_id_a[pid].first_move_correct and by_id_b[pid].first_move_correct)
            delta = (a_wins - b_wins) / len(common) * 100 if common else 0.0
            print(f"  {A:>18} vs {B:<18} {a_wins:>8} {b_wins:>8} {both:>18} {delta:+7.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--puzzles-csv", default="data/lichess_db_puzzle.csv",
                        help="Path to the decompressed Lichess CSV.")
    parser.add_argument("--export-dir",
                        help="Single model export dir. Shorthand for --exports DIR=model.")
    parser.add_argument("--exports", nargs="+", default=[],
                        help="Multiple model specs: path[=label] path[=label] ...")
    parser.add_argument("--max-puzzles", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rating-min", type=int, default=None)
    parser.add_argument("--rating-max", type=int, default=None)
    parser.add_argument("--themes-any", nargs="*", default=None,
                        help="Include puzzles tagged with ANY of these themes.")
    parser.add_argument("--themes-all", nargs="*", default=None,
                        help="Include puzzles tagged with ALL of these themes.")
    parser.add_argument("--themes-none", nargs="*", default=None,
                        help="Exclude puzzles tagged with ANY of these themes.")
    parser.add_argument("--min-popularity", type=int, default=None)
    parser.add_argument("--min-plays", type=int, default=None)

    parser.add_argument("--bucket-by", choices=["rating", "theme", "none"], default="rating")
    parser.add_argument("--rating-buckets", type=int, nargs="+",
                        default=DEFAULT_RATING_BUCKETS,
                        help="Bucket edges (ascending). Buckets are [edge_i, edge_{i+1}).")
    parser.add_argument("--report-themes", nargs="+", default=DEFAULT_REPORT_THEMES,
                        help="Themes to bucket+report when --bucket-by theme.")
    parser.add_argument("--list-themes", action="store_true",
                        help="After loading, print theme counts and exit (no model eval).")

    args = parser.parse_args()

    # Resolve export specs
    export_specs: list[tuple[str, str]] = []
    if args.export_dir:
        export_specs.append((args.export_dir, Path(args.export_dir).name))
    for s in args.exports:
        export_specs.append(parse_export_spec(s))
    if not export_specs and not args.list_themes:
        parser.error("provide --export-dir or --exports")

    # Load puzzles
    rating_range = None
    if args.rating_min is not None or args.rating_max is not None:
        rating_range = (args.rating_min or 0, args.rating_max or 9999)
    themes_any = set(args.themes_any) if args.themes_any else None
    themes_all = set(args.themes_all) if args.themes_all else None
    themes_none = set(args.themes_none) if args.themes_none else None

    print(f"Loading puzzles from {args.puzzles_csv} ...")
    print(f"  rating_range={rating_range} themes_any={themes_any} "
          f"themes_all={themes_all} max={args.max_puzzles} seed={args.seed}")
    puzzles = load_puzzles(
        args.puzzles_csv,
        rating_range=rating_range,
        themes_any=themes_any,
        themes_all=themes_all,
        themes_none=themes_none,
        min_popularity=args.min_popularity,
        min_nb_plays=args.min_plays,
        max_puzzles=args.max_puzzles,
        seed=args.seed,
    )
    print(f"  got {len(puzzles)} puzzles "
          f"(avg rating={sum(p.rating for p in puzzles)/max(len(puzzles),1):.0f}, "
          f"avg plies={sum(p.num_plies for p in puzzles)/max(len(puzzles),1):.1f})")

    if args.list_themes:
        from chessdecoder.eval.tactics import PuzzleResult as _PR
        # Reuse theme_counts on dummy PuzzleResults
        dummies = [_PR(p, [], [], "") for p in puzzles]
        counts = theme_counts(dummies)
        print(f"\n  {'theme':28} count")
        for theme, n in counts.most_common():
            print(f"  {theme:28} {n:>6}")
        return

    # Run each model on the same puzzle set
    all_results: dict[str, list[PuzzleResult]] = {}
    for path, label in export_specs:
        print(f"\n--- Running {label} ({path}) ---")
        engine = build_engine(path, args.batch_size)
        results = evaluate_puzzles(engine, puzzles, batch_size=args.batch_size)
        all_results[label] = results
        del engine  # free GPU before next model

    # Per-model bucketed report
    for label, results in all_results.items():
        print_bucket_report(label, results, args)

    # Paired comparison
    print_paired_deltas(all_results)


if __name__ == "__main__":
    main()

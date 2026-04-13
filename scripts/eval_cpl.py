"""ChessBench action-value CPL eval, batched C++ engine, with paired-bootstrap CIs.

Usage
-----
    # First run: download the action-value test split (~141MB)
    mkdir -p data && curl -L \
        https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag \
        -o data/action_value_test.bag

    # Single model
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_cpl.py \
        --export-dir exports/export_282k --max-positions 1000

    # Compare three models with paired deltas
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_cpl.py \
        --exports exports/export_282k=baseline \
                  exports/export_rl_step_37_fresh=rl_old \
                  exports/export_rl_new_step_25=rl_new \
        --max-positions 1000
"""

import argparse
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable] + sys.argv)

from pathlib import Path

import _decoder_inference_cpp as cpp

from chessdecoder.eval.cpl import (
    PositionResult,
    aggregate,
    bucket_by_best_winprob,
    bucket_by_num_legal,
    evaluate_positions,
    load_positions,
    mcnemar_pvalue,
    paired_delta_winprob_ci,
)


DEFAULT_BEST_WP_BUCKETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
DEFAULT_NUM_LEGAL_BUCKETS = [2, 10, 20, 30, 40, 100]


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
    results: list[PositionResult],
    args,
):
    print(f"\n{'=' * 80}")
    print(f"  Model: {label}")
    print(f"{'=' * 80}")
    print(aggregate(results).format("OVERALL"))

    if args.bucket_by == "best_winprob":
        print(f"\n  --- best_winprob buckets ---")
        for (lo, hi), group in bucket_by_best_winprob(results, args.best_wp_buckets):
            if not group:
                continue
            print()
            print(aggregate(group).format(f"best_wp [{lo:.2f}, {hi:.2f})"))
    elif args.bucket_by == "num_legal":
        print(f"\n  --- num_legal_moves buckets ---")
        for (lo, hi), group in bucket_by_num_legal(results, args.num_legal_buckets):
            if not group:
                continue
            print()
            print(aggregate(group).format(f"num_legal [{lo:>3}, {hi:<3})"))


def print_paired_deltas(all_results: dict[str, list[PositionResult]]):
    """For each pair of labels: paired Δwin% delta, paired McNemar on optimal-rate."""
    labels = list(all_results.keys())
    if len(labels) < 2:
        return

    print(f"\n{'=' * 80}")
    print(f"  Paired comparisons")
    print(f"{'=' * 80}")
    print(f"  Δwin% delta = mean(Δwin%_A) − mean(Δwin%_B)")
    print(f"    positive ⇒ A loses MORE win-prob than B (A is worse).")
    print(f"  McNemar p-value on 'chose optimal move' (lower = more significant).")
    print()
    header = (f"  {'pair (A vs B)':40} "
              f"{'mean ΔΔwin% [95% CI]':>32}  "
              f"{'A_opt':>6} {'B_opt':>6} {'p_McN':>8}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            A, B = labels[i], labels[j]
            ra, rb = all_results[A], all_results[B]
            mean, (lo, hi) = paired_delta_winprob_ci(ra, rb, n_resamples=2000, seed=0)

            by_id_a = {r.position.fen: r for r in ra}
            by_id_b = {r.position.fen: r for r in rb}
            common = sorted(set(by_id_a) & set(by_id_b))
            a_only = sum(1 for fen in common
                         if by_id_a[fen].is_optimal and not by_id_b[fen].is_optimal)
            b_only = sum(1 for fen in common
                         if not by_id_a[fen].is_optimal and by_id_b[fen].is_optimal)
            p_mcn = mcnemar_pvalue(a_only, b_only)
            ci_str = f"{mean * 100:+5.2f}  [{lo * 100:+5.2f}, {hi * 100:+5.2f}]"
            print(f"  {A:>18} vs {B:<18} {ci_str:>32}  "
                  f"{a_only:>6} {b_only:>6} {p_mcn:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag-path", default="data/action_value_test.bag",
                        help="Path to the action-value test .bag file.")
    parser.add_argument("--export-dir",
                        help="Single model export dir. Shorthand for --exports DIR=model.")
    parser.add_argument("--exports", nargs="+", default=[],
                        help="Multiple model specs: path[=label] path[=label] ...")
    parser.add_argument("--max-positions", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-legal-moves", type=int, default=2,
                        help="Filter out positions with fewer legal moves "
                             "(forced moves are trivially correct).")

    parser.add_argument("--bucket-by", choices=["best_winprob", "num_legal", "none"],
                        default="best_winprob")
    parser.add_argument("--best-wp-buckets", type=float, nargs="+",
                        default=DEFAULT_BEST_WP_BUCKETS,
                        help="Bucket edges on best-move win-prob (ascending).")
    parser.add_argument("--num-legal-buckets", type=int, nargs="+",
                        default=DEFAULT_NUM_LEGAL_BUCKETS,
                        help="Bucket edges on num legal moves (ascending).")

    args = parser.parse_args()

    export_specs: list[tuple[str, str]] = []
    if args.export_dir:
        export_specs.append((args.export_dir, Path(args.export_dir).name))
    for s in args.exports:
        export_specs.append(parse_export_spec(s))
    if not export_specs:
        parser.error("provide --export-dir or --exports")

    print(f"Loading action-values from {args.bag_path} ...")
    positions = load_positions(
        args.bag_path,
        max_positions=args.max_positions,
        seed=args.seed,
        min_legal_moves=args.min_legal_moves,
    )
    avg_legal = sum(p.num_legal_moves for p in positions) / max(len(positions), 1)
    avg_best = sum(p.best_winprob for p in positions) / max(len(positions), 1)
    print(f"  got {len(positions)} positions  "
          f"(avg num_legal={avg_legal:.1f}, avg best_winprob={avg_best:.3f})")

    all_results: dict[str, list[PositionResult]] = {}
    for path, label in export_specs:
        print(f"\n--- Running {label} ({path}) ---")
        engine = build_engine(path, args.batch_size)
        results = evaluate_positions(engine, positions, batch_size=args.batch_size)
        all_results[label] = results
        del engine

    for label, results in all_results.items():
        print_bucket_report(label, results, args)

    print_paired_deltas(all_results)


if __name__ == "__main__":
    main()

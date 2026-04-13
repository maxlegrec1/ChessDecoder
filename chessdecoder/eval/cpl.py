"""Centipawn-loss-style eval against the DeepMind ChessBench action-value test split.

For each FEN in the dataset, every legal move has a Stockfish 16 win-probability
`win_prob ∈ [0, 1]` (50ms/state-action, unbounded depth, max skill). For our
model we just need a single forward pass per FEN; the model's chosen move's
win-prob is looked up in the precomputed table — no live engine needed.

We report per-position:
    Δwin% = best_winprob - chosen_winprob

and aggregate into:
    * mean Δwin% (with paired-bootstrap CI when comparing models)
    * optimal-move rate (chosen == argmax SF)
    * top-3 / top-5 rate (chosen ∈ top-K SF moves)
    * blunder rate (Δwin% > 0.20)
    * rank histogram (rank of chosen move in SF ordering)

Buckets:
    * by best-move win-prob (decisive vs balanced positions)
    * by number of legal moves (forced vs many-choice)

The action-value test split has ~62k positions / 1.84M annotations. Decoding
the .bag file + group-by-fen takes ~3s; subsample with `max_positions` for
faster runs (default 1000).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from chessdecoder.eval.bagz import iter_action_value_records
from chessdecoder.eval.stats import bootstrap_ci_mean, wilson_ci
from chessdecoder.utils.uci import normalize_castling


# --------------------------------------------------------------------------
# Data class + loading
# --------------------------------------------------------------------------


@dataclass
class Position:
    fen: str
    # move (UCI) -> Stockfish win_prob in [0, 1] for side-to-move
    move_winprobs: dict[str, float]

    @property
    def num_legal_moves(self) -> int:
        return len(self.move_winprobs)

    @property
    def best_winprob(self) -> float:
        return max(self.move_winprobs.values())

    def rank_of(self, move: str) -> int:
        """1-based rank (1 = best). Returns len+1 if move is not legal/annotated."""
        if move not in self.move_winprobs:
            return len(self.move_winprobs) + 1
        ordered = sorted(self.move_winprobs.values(), reverse=True)
        wp = self.move_winprobs[move]
        # Tie-aware: rank = 1 + number of strictly greater win_probs.
        return 1 + sum(1 for v in ordered if v > wp)


def load_positions(
    bag_path: str | Path,
    *,
    max_positions: int | None = None,
    seed: int = 42,
    min_legal_moves: int = 2,
) -> list[Position]:
    """Stream the action-value bag, group by FEN, then sample.

    `min_legal_moves=2` filters out forced-move positions (no choice → trivially
    correct, nothing to grade).
    """
    by_fen: dict[str, dict[str, float]] = defaultdict(dict)
    for fen, move, wp in iter_action_value_records(bag_path):
        by_fen[fen][move] = wp

    positions = [
        Position(fen=fen, move_winprobs=dict(moves))
        for fen, moves in by_fen.items()
        if len(moves) >= min_legal_moves
    ]
    rng = random.Random(seed)
    rng.shuffle(positions)
    if max_positions is not None:
        positions = positions[:max_positions]
    return positions


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass
class PositionResult:
    position: Position
    model_move: str          # UCI as returned by the engine (after normalize_castling)
    chosen_winprob: float    # SF win-prob of the model's move; 0.0 if illegal/unannotated
    rank: int                # 1-based; len+1 if illegal/unannotated
    legal: bool              # was the model's move present in SF's legal-move list?

    @property
    def delta_winprob(self) -> float:
        """Loss versus the optimal SF move. Always ≥ 0."""
        return self.position.best_winprob - self.chosen_winprob

    @property
    def is_optimal(self) -> bool:
        return self.rank == 1


def evaluate_positions(
    engine,
    positions: list[Position],
    *,
    batch_size: int = 32,
    progress: bool = True,
) -> list[PositionResult]:
    """One model forward pass per position; look up SF win-prob in the table."""
    results: list[PositionResult] = []
    n = len(positions)
    for chunk_start in range(0, n, batch_size):
        chunk = positions[chunk_start:chunk_start + batch_size]
        fens = [p.fen for p in chunk]
        outs = engine.predict_moves(fens, 0.0)

        for p, out in zip(chunk, outs):
            move = normalize_castling(out.move) if out.move else ""
            wp = p.move_winprobs.get(move)
            if wp is None:
                results.append(PositionResult(
                    position=p, model_move=move,
                    chosen_winprob=0.0,                    # treat illegal as worst-case
                    rank=p.num_legal_moves + 1,
                    legal=False,
                ))
            else:
                results.append(PositionResult(
                    position=p, model_move=move,
                    chosen_winprob=wp,
                    rank=p.rank_of(move),
                    legal=True,
                ))

        if progress:
            done = chunk_start + len(chunk)
            print(f"  [cpl] {done}/{n} positions ({100.0 * done / n:.0f}%)",
                  flush=True)

    return results


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


BLUNDER_THRESHOLD = 0.20  # Δwin% > 20 absolute pp counts as a blunder.


@dataclass
class Aggregate:
    n_positions: int
    n_illegal: int
    mean_delta_winprob: float
    delta_ci: tuple[float, float]
    optimal_rate: float
    optimal_ci: tuple[float, float]
    top3_rate: float
    top3_ci: tuple[float, float]
    top5_rate: float
    top5_ci: tuple[float, float]
    blunder_rate: float
    blunder_ci: tuple[float, float]
    # Histogram bins: rank=1, rank=2, rank=3, rank in [4..10], rank>=11 (incl. illegal)
    rank_hist: dict[str, int] = field(default_factory=dict)

    def format(self, label: str = "") -> str:
        head = f"{label}  n={self.n_positions}  illegal={self.n_illegal}"
        lines = [
            head,
            f"  mean Δwin%   : {self.mean_delta_winprob * 100:5.2f}  "
            f"[95% CI {self.delta_ci[0] * 100:5.2f}, {self.delta_ci[1] * 100:5.2f}]",
            f"  optimal rate : {self.optimal_rate * 100:5.2f}%  "
            f"[95% CI {self.optimal_ci[0] * 100:5.2f}, {self.optimal_ci[1] * 100:5.2f}]",
            f"  top-3 rate   : {self.top3_rate * 100:5.2f}%  "
            f"[95% CI {self.top3_ci[0] * 100:5.2f}, {self.top3_ci[1] * 100:5.2f}]",
            f"  top-5 rate   : {self.top5_rate * 100:5.2f}%  "
            f"[95% CI {self.top5_ci[0] * 100:5.2f}, {self.top5_ci[1] * 100:5.2f}]",
            f"  blunder rate : {self.blunder_rate * 100:5.2f}%  "
            f"(Δwin% > {int(BLUNDER_THRESHOLD * 100)}pp) "
            f"[95% CI {self.blunder_ci[0] * 100:5.2f}, {self.blunder_ci[1] * 100:5.2f}]",
            f"  rank hist    : "
            + "  ".join(f"{k}={v}" for k, v in self.rank_hist.items()),
        ]
        return "\n".join(lines)


def _rank_bucket(rank: int, num_legal: int, legal: bool) -> str:
    if not legal:
        return "illegal"
    if rank == 1:
        return "1"
    if rank == 2:
        return "2"
    if rank == 3:
        return "3"
    if rank <= 10:
        return "4-10"
    return "11+"


def aggregate(
    results: list[PositionResult],
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
) -> Aggregate:
    n = len(results)
    if n == 0:
        return Aggregate(0, 0, 0.0, (0.0, 0.0), 0.0, (0.0, 0.0),
                         0.0, (0.0, 0.0), 0.0, (0.0, 0.0),
                         0.0, (0.0, 0.0), {})

    deltas = [r.delta_winprob for r in results]
    n_optimal = sum(1 for r in results if r.is_optimal)
    n_top3 = sum(1 for r in results if r.rank <= 3)
    n_top5 = sum(1 for r in results if r.rank <= 5)
    n_blunder = sum(1 for r in results if r.delta_winprob > BLUNDER_THRESHOLD)
    n_illegal = sum(1 for r in results if not r.legal)

    rank_hist: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4-10": 0, "11+": 0, "illegal": 0}
    for r in results:
        rank_hist[_rank_bucket(r.rank, r.position.num_legal_moves, r.legal)] += 1

    return Aggregate(
        n_positions=n,
        n_illegal=n_illegal,
        mean_delta_winprob=sum(deltas) / n,
        delta_ci=bootstrap_ci_mean(deltas, n_resamples=bootstrap_resamples, seed=bootstrap_seed),
        optimal_rate=n_optimal / n,
        optimal_ci=wilson_ci(n_optimal, n),
        top3_rate=n_top3 / n,
        top3_ci=wilson_ci(n_top3, n),
        top5_rate=n_top5 / n,
        top5_ci=wilson_ci(n_top5, n),
        blunder_rate=n_blunder / n,
        blunder_ci=wilson_ci(n_blunder, n),
        rank_hist=rank_hist,
    )


# --------------------------------------------------------------------------
# Buckets
# --------------------------------------------------------------------------


def bucket_by_best_winprob(
    results: list[PositionResult],
    edges: list[float],
) -> list[tuple[tuple[float, float], list[PositionResult]]]:
    """Group results by best-move win-prob (e.g. 0.0/0.2/0.4/0.6/0.8/1.0).

    Useful axis: 0.0–0.2 = lost positions, 0.4–0.6 = balanced, 0.8–1.0 = winning.
    """
    buckets = [((edges[i], edges[i + 1]), []) for i in range(len(edges) - 1)]
    for r in results:
        wp = r.position.best_winprob
        for (lo, hi), group in buckets:
            if lo <= wp < hi:
                group.append(r)
                break
        else:
            # Include the upper boundary in the last bucket.
            if abs(wp - edges[-1]) < 1e-9 and buckets:
                buckets[-1][1].append(r)
    return buckets


def bucket_by_num_legal(
    results: list[PositionResult],
    edges: list[int],
) -> list[tuple[tuple[int, int], list[PositionResult]]]:
    """Group by number of legal moves in the position."""
    buckets = [((edges[i], edges[i + 1]), []) for i in range(len(edges) - 1)]
    for r in results:
        n = r.position.num_legal_moves
        for (lo, hi), group in buckets:
            if lo <= n < hi:
                group.append(r)
                break
    return buckets


# --------------------------------------------------------------------------
# Paired comparison helpers
# --------------------------------------------------------------------------


def paired_delta_winprob_ci(
    results_a: list[PositionResult],
    results_b: list[PositionResult],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Mean of (Δwin%_A − Δwin%_B) across positions, with paired bootstrap CI.

    Positive = A loses MORE win-prob on average → A is worse.
    Both result lists must be aligned by position (same FEN at same index).
    """
    by_id_a = {r.position.fen: r for r in results_a}
    by_id_b = {r.position.fen: r for r in results_b}
    common = sorted(set(by_id_a) & set(by_id_b))
    diffs = [by_id_a[fen].delta_winprob - by_id_b[fen].delta_winprob
             for fen in common]
    if not diffs:
        return 0.0, (0.0, 0.0)
    mean_diff = sum(diffs) / len(diffs)
    ci = bootstrap_ci_mean(diffs, n_resamples=n_resamples, seed=seed)
    return mean_diff, ci


def mcnemar_pvalue(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (binomial 0.5).

    `a_only` = #positions where A optimal, B not. `b_only` = #B optimal, A not.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    # Two-sided: 2 * sum_{i=0..k} C(n,i) * 0.5^n
    log_half = math.log(0.5)
    log_p = -math.inf
    for i in range(k + 1):
        log_term = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + n * log_half
        # log-sum-exp
        if log_term > log_p:
            log_p, log_term = log_term, log_p
        log_p = log_p + math.log1p(math.exp(log_term - log_p))
    return min(1.0, 2 * math.exp(log_p))

"""Shared statistical primitives for chess model evaluation.

Single source of truth for all metric math used across eval modules:
  * wilson_ci          — used in tactics.py and cpl.py (was duplicated)
  * bootstrap_ci_mean  — used in cpl.py
  * estimate_elo       — used in elo_eval.py (was duplicated across game-play eval scripts)
  * win_rate           — inline formula repeated across game-play eval scripts
"""

from __future__ import annotations

import math
import random
from math import log10


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Z-score for the desired confidence level (default 1.96 → 95%).

    Returns:
        (lower, upper) bounds in [0, 1].
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci_mean(
    values: list[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a flat list.

    Args:
        values: Observed values.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 → 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        (lower, upper) confidence bounds.
    """
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo_idx = int(math.floor(n_resamples * alpha / 2))
    hi_idx = int(math.ceil(n_resamples * (1 - alpha / 2))) - 1
    return means[lo_idx], means[min(hi_idx, n_resamples - 1)]


# --------------------------------------------------------------------------
# ELO estimation
# --------------------------------------------------------------------------


def win_rate(wins: int | float, draws: int | float, total: int) -> float:
    """Score-based win rate: (wins + 0.5 * draws) / total.

    Returns 0.0 if total == 0.
    """
    if total == 0:
        return 0.0
    return (wins + 0.5 * draws) / total


def estimate_elo(wr: float, stockfish_elo: int) -> float:
    """Estimate model ELO from win-rate against a known Stockfish ELO.

    Uses the standard 400-point ELO formula:
        elo = stockfish_elo - 400 * log10((1 - win_rate) / win_rate)

    Edge cases:
        win_rate == 0 → returns 0
        win_rate == 1 → returns stockfish_elo + 800 (capped proxy for dominance)
        win_rate == 0.5 → returns stockfish_elo exactly

    Args:
        wr: Win rate in [0, 1].
        stockfish_elo: The Stockfish ELO setting used during play.

    Returns:
        Estimated model ELO as a float.
    """
    if wr <= 0.0:
        return 0.0
    if wr >= 1.0:
        return float(stockfish_elo + 800)
    return stockfish_elo - 400.0 * log10((1.0 - wr) / wr)

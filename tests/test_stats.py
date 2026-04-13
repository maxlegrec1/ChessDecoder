"""Tests for chessdecoder.eval.stats — all statistical primitives."""

import math

import pytest

from chessdecoder.eval.stats import (
    bootstrap_ci_mean,
    estimate_elo,
    win_rate,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------


def test_wilson_ci_zero_trials():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_all_successes():
    lo, hi = wilson_ci(10, 10)
    assert lo > 0.7
    assert hi == 1.0


def test_wilson_ci_no_successes():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi < 0.3


def test_wilson_ci_fifty_percent():
    lo, hi = wilson_ci(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    # Should be symmetric around 0.5
    assert abs((lo + hi) / 2 - 0.5) < 0.01


def test_wilson_ci_bounds_are_valid_probabilities():
    for k, n in [(0, 1), (1, 1), (3, 7), (99, 100)]:
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# bootstrap_ci_mean
# ---------------------------------------------------------------------------


def test_bootstrap_ci_empty():
    assert bootstrap_ci_mean([]) == (0.0, 0.0)


def test_bootstrap_ci_brackets_true_mean():
    values = [0.0, 0.0, 0.5, 1.0, 1.0]  # mean = 0.5
    lo, hi = bootstrap_ci_mean(values, n_resamples=2000, seed=0)
    assert lo <= 0.5 <= hi
    assert hi - lo > 0.01  # non-trivial interval


def test_bootstrap_ci_constant_values():
    # If all values are the same, CI should be a point.
    lo, hi = bootstrap_ci_mean([0.3] * 20, n_resamples=500, seed=42)
    assert abs(lo - 0.3) < 1e-9
    assert abs(hi - 0.3) < 1e-9


def test_bootstrap_ci_reproducible():
    values = [float(i) / 10 for i in range(10)]
    a = bootstrap_ci_mean(values, n_resamples=500, seed=7)
    b = bootstrap_ci_mean(values, n_resamples=500, seed=7)
    assert a == b


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------


def test_win_rate_zero_games():
    assert win_rate(0, 0, 0) == 0.0


def test_win_rate_all_wins():
    assert win_rate(10, 0, 10) == pytest.approx(1.0)


def test_win_rate_all_draws():
    assert win_rate(0, 10, 10) == pytest.approx(0.5)


def test_win_rate_mixed():
    # 3 wins, 2 draws, 5 total → (3 + 1) / 5 = 0.8
    assert win_rate(3, 2, 5) == pytest.approx(0.8)


def test_win_rate_all_losses():
    assert win_rate(0, 0, 10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# estimate_elo
# ---------------------------------------------------------------------------


def test_estimate_elo_zero_win_rate():
    assert estimate_elo(0.0, 1500) == 0.0


def test_estimate_elo_full_win_rate():
    # Returns stockfish_elo + 800
    assert estimate_elo(1.0, 1500) == pytest.approx(2300.0)


def test_estimate_elo_fifty_percent():
    # Equal performance → same ELO as opponent
    assert estimate_elo(0.5, 1500) == pytest.approx(1500.0)


def test_estimate_elo_above_fifty():
    elo = estimate_elo(0.75, 1500)
    assert elo > 1500


def test_estimate_elo_below_fifty():
    elo = estimate_elo(0.25, 1500)
    assert elo < 1500


def test_estimate_elo_formula_consistency():
    # At 75% wr vs 1500: Δ = -400*log10(0.25/0.75) = 400*log10(3) ≈ 190.85
    expected = 1500 - 400.0 * math.log10(0.25 / 0.75)
    assert estimate_elo(0.75, 1500) == pytest.approx(expected)


def test_estimate_elo_near_zero():
    # Very low (but non-zero) win rate should give a very low ELO.
    elo = estimate_elo(0.01, 1500)
    assert elo < 1000

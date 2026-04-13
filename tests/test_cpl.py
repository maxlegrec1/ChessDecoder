"""Pure-Python tests for the CPL eval module (no GPU, no real .bag download)."""

import os
import struct
import tempfile

import pytest

from chessdecoder.eval.bagz import (
    BagFileReader,
    decode_action_value,
    iter_action_value_records,
)
from chessdecoder.eval.cpl import (
    Aggregate,
    BLUNDER_THRESHOLD,
    Position,
    PositionResult,
    aggregate,
    bucket_by_best_winprob,
    bucket_by_num_legal,
    evaluate_positions,
    load_positions,
    mcnemar_pvalue,
    paired_delta_winprob_ci,
)
from chessdecoder.eval.stats import bootstrap_ci_mean, wilson_ci


# ---------------------------------------------------------------------------
# bagz reader: round-trip synthetic bag files
# ---------------------------------------------------------------------------


def _encode_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _encode_action_value(fen: str, move: str, win_prob: float) -> bytes:
    fb = fen.encode("utf-8")
    mb = move.encode("utf-8")
    return (_encode_varint(len(fb)) + fb
            + _encode_varint(len(mb)) + mb
            + struct.pack(">d", win_prob))


def _write_bag(path: str, records: list[bytes]) -> None:
    """Write a uncompressed-bag file: concatenated records, then int64 limits, then index_start uint64."""
    with open(path, "wb") as f:
        offsets = []
        cur = 0
        for r in records:
            f.write(r)
            cur += len(r)
            offsets.append(cur)
        index_start = cur
        for off in offsets:
            f.write(struct.pack("<q", off))
        # The final 8 bytes ARE the last limit entry (= index_start), per bagz.py.
        # That last limit was already written above, so we just have to verify
        # the math: index_size = file_size - index_start = 8 * len(offsets) ✓.


@pytest.fixture
def bag_path(tmp_path):
    records = [
        _encode_action_value("fen-A", "e2e4", 0.55),
        _encode_action_value("fen-A", "d2d4", 0.50),
        _encode_action_value("fen-B", "g8f6", 0.42),
        _encode_action_value("fen-with-much-longer-content-and-spaces here", "a7a8q", 0.95),
    ]
    p = str(tmp_path / "test.bag")
    _write_bag(p, records)
    return p


def test_bag_reader_roundtrip(bag_path):
    reader = BagFileReader(bag_path)
    assert len(reader) == 4
    decoded = [decode_action_value(reader[i]) for i in range(len(reader))]
    assert decoded[0] == ("fen-A", "e2e4", pytest.approx(0.55))
    assert decoded[1] == ("fen-A", "d2d4", pytest.approx(0.50))
    assert decoded[2] == ("fen-B", "g8f6", pytest.approx(0.42))
    assert decoded[3][0] == "fen-with-much-longer-content-and-spaces here"
    assert decoded[3][1] == "a7a8q"
    assert decoded[3][2] == pytest.approx(0.95)


def test_bag_reader_negative_index(bag_path):
    reader = BagFileReader(bag_path)
    last = decode_action_value(reader[-1])
    assert last[1] == "a7a8q"


def test_bag_reader_out_of_range(bag_path):
    reader = BagFileReader(bag_path)
    with pytest.raises(IndexError):
        _ = reader[10]


def test_iter_action_value_records(bag_path):
    items = list(iter_action_value_records(bag_path))
    assert len(items) == 4
    fens = [f for f, _, _ in items]
    assert fens == ["fen-A", "fen-A", "fen-B",
                    "fen-with-much-longer-content-and-spaces here"]


def test_load_positions_groups_by_fen(bag_path):
    positions = load_positions(bag_path, max_positions=None, min_legal_moves=1)
    assert len(positions) == 3
    by_fen = {p.fen: p for p in positions}
    assert set(by_fen) == {"fen-A", "fen-B",
                           "fen-with-much-longer-content-and-spaces here"}
    assert by_fen["fen-A"].move_winprobs == {"e2e4": pytest.approx(0.55),
                                              "d2d4": pytest.approx(0.50)}
    assert by_fen["fen-A"].best_winprob == pytest.approx(0.55)
    assert by_fen["fen-A"].num_legal_moves == 2


def test_load_positions_min_legal_moves_filter(bag_path):
    positions = load_positions(bag_path, max_positions=None, min_legal_moves=2)
    # Only fen-A has ≥ 2 moves.
    assert [p.fen for p in positions] == ["fen-A"]


def test_load_positions_max_and_seed(bag_path):
    a = load_positions(bag_path, max_positions=2, seed=42, min_legal_moves=1)
    b = load_positions(bag_path, max_positions=2, seed=42, min_legal_moves=1)
    assert [p.fen for p in a] == [p.fen for p in b]
    assert len(a) == 2


# ---------------------------------------------------------------------------
# Position.rank_of
# ---------------------------------------------------------------------------


def test_rank_of_basic():
    p = Position("X", {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1})
    assert p.rank_of("a") == 1
    # Tie: both b and c have rank 2 (no strict greater except a).
    assert p.rank_of("b") == 2
    assert p.rank_of("c") == 2
    assert p.rank_of("d") == 4
    # Unknown move
    assert p.rank_of("z") == 5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _result(wp_map, model_move, best_wp=None):
    p = Position("X", wp_map)
    if model_move in wp_map:
        return PositionResult(p, model_move, wp_map[model_move],
                              p.rank_of(model_move), legal=True)
    return PositionResult(p, model_move, 0.0,
                          p.num_legal_moves + 1, legal=False)


def test_aggregate_basic_metrics():
    # 4 positions, simple bookkeeping. Use values that avoid float drift around 0.20.
    results = [
        _result({"a": 0.9, "b": 0.4, "c": 0.1}, "a"),    # optimal, Δ=0
        _result({"a": 0.9, "b": 0.4, "c": 0.1}, "b"),    # rank 2, Δ=0.5 → blunder
        _result({"a": 0.9, "b": 0.75, "c": 0.1}, "b"),   # rank 2, Δ=0.15 → not blunder
        _result({"a": 0.9, "b": 0.75, "c": 0.1}, ""),    # illegal → blunder (Δ=0.9)
    ]
    agg = aggregate(results, bootstrap_resamples=200)
    assert agg.n_positions == 4
    assert agg.n_illegal == 1
    # mean Δ = (0 + 0.5 + 0.15 + 0.9) / 4 = 0.3875
    assert abs(agg.mean_delta_winprob - 0.3875) < 1e-9
    assert abs(agg.optimal_rate - 0.25) < 1e-9
    assert abs(agg.top3_rate - 0.75) < 1e-9
    # Blunders: position 1 (Δ=0.5) and 3 (Δ=0.9, illegal)
    assert abs(agg.blunder_rate - 0.5) < 1e-9
    assert agg.rank_hist["1"] == 1
    assert agg.rank_hist["2"] == 2
    assert agg.rank_hist["illegal"] == 1


def test_aggregate_empty():
    agg = aggregate([])
    assert agg.n_positions == 0
    assert agg.mean_delta_winprob == 0.0


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = wilson_ci(50, 100)
    # Symmetric around 0.5, narrow band
    assert 0.40 < lo < 0.5 < hi < 0.60


def test_bootstrap_ci_mean_brackets_true_mean():
    values = [0.0, 0.0, 0.5, 1.0, 1.0]   # mean 0.5
    lo, hi = bootstrap_ci_mean(values, n_resamples=2000, seed=0)
    assert lo <= 0.5 <= hi
    # Should be a non-trivial interval.
    assert hi - lo > 0.01


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------


def test_bucket_by_best_winprob():
    rs = [
        _result({"a": 0.05, "b": 0.0}, "a"),    # bucket [0.0, 0.2)
        _result({"a": 0.55, "b": 0.5}, "a"),    # bucket [0.4, 0.6)
        _result({"a": 0.95, "b": 0.5}, "a"),    # bucket [0.8, 1.0]
    ]
    buckets = bucket_by_best_winprob(rs, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert len(buckets[0][1]) == 1  # the 0.05 one
    assert len(buckets[2][1]) == 1  # the 0.55 one
    assert len(buckets[4][1]) == 1  # the 0.95 one (upper-boundary inclusive)


def test_bucket_by_num_legal():
    rs = [
        _result({"a": 0.5, "b": 0.4}, "a"),
        _result({"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2, "e": 0.1, "f": 0.05}, "a"),
    ]
    buckets = bucket_by_num_legal(rs, [1, 5, 10, 100])
    assert len(buckets[0][1]) == 1
    assert len(buckets[1][1]) == 1


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------


def test_paired_delta_zero_when_identical():
    rs = [_result({"a": 0.9, "b": 0.4}, "a")]
    mean, ci = paired_delta_winprob_ci(rs, rs, n_resamples=200)
    assert mean == 0.0


def test_paired_delta_positive_when_a_worse():
    p_map = {"a": 0.9, "b": 0.4}
    rs_a = [PositionResult(Position("X", p_map), "b", 0.4, 2, legal=True)]
    rs_b = [PositionResult(Position("X", p_map), "a", 0.9, 1, legal=True)]
    mean, _ = paired_delta_winprob_ci(rs_a, rs_b, n_resamples=200)
    # A's Δ=0.5, B's Δ=0 → diff=+0.5
    assert abs(mean - 0.5) < 1e-9


def test_mcnemar_zero_discordant():
    assert mcnemar_pvalue(0, 0) == 1.0


def test_mcnemar_extreme_imbalance():
    # 20 A-only wins, 0 B-only → very small p-value.
    p = mcnemar_pvalue(20, 0)
    assert p < 1e-5


# ---------------------------------------------------------------------------
# evaluate_positions with stub engine
# ---------------------------------------------------------------------------


class _StubResult:
    def __init__(self, move): self.move = move


class _StubEngine:
    def __init__(self, fen_to_move):
        self._map = fen_to_move

    def predict_moves(self, fens, _temp):
        return [_StubResult(self._map.get(f, "")) for f in fens]


def test_evaluate_positions_optimal_move():
    p = Position("F1", {"e2e4": 0.6, "d2d4": 0.5})
    engine = _StubEngine({"F1": "e2e4"})
    results = evaluate_positions(engine, [p], batch_size=4, progress=False)
    assert len(results) == 1
    assert results[0].is_optimal
    assert results[0].delta_winprob == pytest.approx(0.0)


def test_evaluate_positions_suboptimal_move():
    p = Position("F1", {"e2e4": 0.6, "d2d4": 0.5})
    engine = _StubEngine({"F1": "d2d4"})
    results = evaluate_positions(engine, [p], batch_size=4, progress=False)
    assert results[0].rank == 2
    assert results[0].delta_winprob == pytest.approx(0.1)


def test_evaluate_positions_illegal_move_treated_as_worst():
    p = Position("F1", {"e2e4": 0.6, "d2d4": 0.5})
    engine = _StubEngine({"F1": "z9z9"})
    results = evaluate_positions(engine, [p], batch_size=4, progress=False)
    assert not results[0].legal
    assert results[0].delta_winprob == pytest.approx(0.6)
    assert results[0].rank == 3

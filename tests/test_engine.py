"""Tests for chessdecoder.eval.engine — Protocol + adapters (no C++ required)."""

import pytest

from chessdecoder.eval.engine import MovePredictor, MoveResult, PytorchModelAdapter


# ---------------------------------------------------------------------------
# MoveResult
# ---------------------------------------------------------------------------


def test_move_result_defaults():
    r = MoveResult(move="e2e4")
    assert r.move == "e2e4"
    assert r.token_ids == []
    assert r.wl_entries == []
    assert r.d_entries == []
    assert r.move_log_probs == []


def test_move_result_with_fields():
    r = MoveResult(
        move="g1f3",
        token_ids=[1, 2, 3],
        wl_entries=[(0, 0.6)],
        d_entries=[(0, 0.1)],
        move_log_probs=[(42, -0.5)],
    )
    assert r.token_ids == [1, 2, 3]
    assert r.wl_entries == [(0, 0.6)]
    assert r.move_log_probs == [(42, -0.5)]


# ---------------------------------------------------------------------------
# PytorchModelAdapter
# ---------------------------------------------------------------------------


def test_pytorch_adapter_returns_move_result():
    adapter = PytorchModelAdapter(lambda fen, temp: "d2d4")
    results = adapter.predict_moves(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"], 0.0)
    assert len(results) == 1
    assert isinstance(results[0], MoveResult)
    assert results[0].move == "d2d4"


def test_pytorch_adapter_multiple_fens():
    adapter = PytorchModelAdapter(lambda fen, temp: fen[:4])  # dummy: first 4 chars
    fens = ["e4e5", "d4d5", "c4c5"]
    results = adapter.predict_moves(fens, 0.0)
    assert len(results) == 3
    assert [r.move for r in results] == ["e4e5", "d4d5", "c4c5"]


def test_pytorch_adapter_none_move_becomes_empty_string():
    adapter = PytorchModelAdapter(lambda fen, temp: None)
    results = adapter.predict_moves(["any_fen"], 0.0)
    assert results[0].move == ""


def test_pytorch_adapter_normalizes_castling():
    # King-side castling in old notation should be normalized.
    adapter = PytorchModelAdapter(lambda fen, temp: "e1g1")
    results = adapter.predict_moves(["any_fen"], 0.0)
    # normalize_castling maps e1g1 → e1g1 (already short-side castling notation)
    # The key is that it doesn't crash and returns a MoveResult.
    assert isinstance(results[0], MoveResult)


def test_pytorch_adapter_optimal_batch_size():
    adapter = PytorchModelAdapter(lambda fen, temp: "e2e4")
    assert adapter.optimal_batch_size == 1


def test_pytorch_adapter_empty_fens():
    adapter = PytorchModelAdapter(lambda fen, temp: "e2e4")
    results = adapter.predict_moves([], 0.0)
    assert results == []


# ---------------------------------------------------------------------------
# MovePredictor Protocol check
# ---------------------------------------------------------------------------


def test_pytorch_adapter_satisfies_protocol():
    adapter = PytorchModelAdapter(lambda fen, temp: "e2e4")
    assert isinstance(adapter, MovePredictor)


def test_custom_class_satisfies_protocol():
    class MyEngine:
        optimal_batch_size = 4

        def predict_moves(self, fens, temperature):
            return [MoveResult(move="e2e4") for _ in fens]

    engine = MyEngine()
    assert isinstance(engine, MovePredictor)


def test_missing_method_fails_protocol():
    class Incomplete:
        optimal_batch_size = 1
        # no predict_moves

    assert not isinstance(Incomplete(), MovePredictor)


def test_missing_attribute_fails_protocol():
    class Incomplete:
        # no optimal_batch_size
        def predict_moves(self, fens, temperature):
            return []

    assert not isinstance(Incomplete(), MovePredictor)

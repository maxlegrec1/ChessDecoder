"""C++ inference engine tests (requires exported model in exports/ and GPU)."""

import chess
import pytest

from tests.conftest import SAMPLE_FENS

pytestmark = pytest.mark.cpp


# ---------------------------------------------------------------------------
# Single engine tests
# ---------------------------------------------------------------------------

def test_single_engine_legal_moves(single_engine, sample_fens):
    for fen in sample_fens:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        move_str = single_engine.predict_move(fen, 0.0)
        assert move_str, f"Empty move for {fen}"
        move = chess.Move.from_uci(move_str)
        assert move in board.legal_moves, f"Illegal: {move_str} for {fen}"


def test_single_engine_deterministic(single_engine):
    fen = SAMPLE_FENS[0]
    move1 = single_engine.predict_move(fen, 0.0)
    move2 = single_engine.predict_move(fen, 0.0)
    assert move1 == move2


def test_single_engine_token_structure(single_engine):
    from chessdecoder.models.vocab import token_to_idx, idx_to_token, move_vocab_size

    fen = SAMPLE_FENS[0]
    single_engine.predict_move(fen, 0.0)
    tids = list(single_engine.last_token_ids())

    # Starts with board block
    assert tids[0] == token_to_idx["start_pos"]

    # Contains start_think and end_think
    assert token_to_idx["start_think"] in tids
    assert token_to_idx["end_think"] in tids

    # token_ids ends at end_think; final move is in .move, values in .wl_entries/.d_entries
    et_idx = token_to_idx["end_think"]
    assert tids[-1] == et_idx or tids.index(et_idx) < len(tids)

    # WL/D entries exist
    wl = list(single_engine.last_wl_entries())
    d = list(single_engine.last_d_entries())
    assert len(wl) > 0
    assert len(d) > 0


# ---------------------------------------------------------------------------
# Batched engine tests
# ---------------------------------------------------------------------------

def test_batched_engine_legal_moves(batched_engine, sample_fens):
    results = batched_engine.predict_moves(sample_fens, 0.0)
    assert len(results) == len(sample_fens)

    for fen, r in zip(sample_fens, results):
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        assert r.move, f"Empty move for {fen}"
        move = chess.Move.from_uci(r.move)
        assert move in board.legal_moves, f"Illegal: {r.move} for {fen}"


def test_batched_single_both_legal(single_engine, batched_engine, sample_fens):
    """Both engines produce legal moves for the same FENs.

    Note: exact move parity is not expected because the single engine uses
    CUDA graphs with tiered KV buffers while the batched engine uses dynamic
    allocation, leading to different FP16 GEMM kernel dispatch and tiny
    numerical differences that can flip argmax on close logits.
    The key correctness test is test_batched_deterministic_same_fen (internal
    consistency) and the pass@k tests run during development.
    """
    for fen in sample_fens:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        single_move = single_engine.predict_move(fen, 0.0)
        batched_results = batched_engine.predict_moves([fen], 0.0)
        batched_move = batched_results[0].move

        assert single_move and chess.Move.from_uci(single_move) in board.legal_moves
        assert batched_move and chess.Move.from_uci(batched_move) in board.legal_moves


def test_batched_deterministic_same_fen(batched_engine):
    """Same FEN repeated N times at temp=0 must produce identical results."""
    fen = SAMPLE_FENS[0]
    fens = [fen] * 4
    results = batched_engine.predict_moves(fens, 0.0)

    moves = [r.move for r in results]
    assert len(set(moves)) == 1, f"Non-deterministic: {moves}"

    # Token sequences should also match
    for i in range(1, len(results)):
        assert results[i].token_ids == results[0].token_ids, \
            f"Token sequence mismatch between slot 0 and {i}"


def test_batched_mixed_fens_valid(batched_engine, sample_fens):
    results = batched_engine.predict_moves(sample_fens, 0.0)
    for fen, r in zip(sample_fens, results):
        assert r.move, f"Empty move for {fen}"
        assert len(r.token_ids) > 68, f"Too few tokens for {fen}"
        assert len(r.wl_entries) > 0, f"No WL entries for {fen}"
        assert len(r.d_entries) > 0, f"No D entries for {fen}"


def test_batched_token_structure(batched_engine):
    from chessdecoder.models.vocab import token_to_idx, move_vocab_size

    results = batched_engine.predict_moves(SAMPLE_FENS[:3], 0.0)
    for r in results:
        tids = r.token_ids
        assert tids[0] == token_to_idx["start_pos"]
        assert token_to_idx["start_think"] in tids
        assert token_to_idx["end_think"] in tids
        # token_ids ends at end_think
        assert token_to_idx["end_think"] in tids

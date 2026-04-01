"""RL reward function tests (pure Python, no GPU)."""

import pytest

from src.models.vocab import token_to_idx, move_vocab_size
from src.rl.rewards import (
    move_quality_reward, format_reward, coherence_reward,
    CompositeReward,
)


# Token IDs for building synthetic sequences
_START_POS = token_to_idx["start_pos"]
_END_POS = token_to_idx["end_pos"]
_EMPTY = token_to_idx["empty"]
_NO_CASTLE = token_to_idx["no_castling_rights"]
_WTM = token_to_idx["white_to_move"]
_START_THINK = token_to_idx["start_think"]
_END_THINK = token_to_idx["end_think"]
_END_VAR = token_to_idx["end_var"]
_WL = token_to_idx["wl_value"]
_D = token_to_idx["d_value"]
_E2E4 = token_to_idx["e2e4"]
_D2D4 = token_to_idx["d2d4"]


def _make_board_block():
    """Minimal 68-token board block."""
    return ([_START_POS] + [_EMPTY] * 64 + [_END_POS, _NO_CASTLE, _WTM])


def _make_valid_thinking_seq(root_move=_E2E4, final_move=_E2E4):
    """Build a well-formed thinking token sequence."""
    board = _make_board_block()
    return (
        board
        + [_START_THINK, root_move, _WL, _D]
        + board
        + [_END_VAR]
        + [_END_THINK, final_move, _WL, _D]
    )


# --- move_quality_reward ---

def test_move_quality_exact_match():
    assert move_quality_reward("e2e4", [], {"best_move": "e2e4"}) == 1.0


def test_move_quality_mismatch():
    assert move_quality_reward("d2d4", [], {"best_move": "e2e4"}) == 0.0


def test_move_quality_castling_normalization():
    # Both "e1h1" (pseudo) and "e1g1" (standard) should normalize the same
    assert move_quality_reward("e1g1", [], {"best_move": "e1g1"}) == 1.0


# --- format_reward ---

def test_format_valid_structure():
    token_ids = _make_valid_thinking_seq()
    assert format_reward("e2e4", token_ids, {}) == 1.0


def test_format_no_start_think():
    token_ids = [_EMPTY] * 10
    assert format_reward("e2e4", token_ids, {}) == 0.0


def test_format_no_end_think():
    board = _make_board_block()
    token_ids = board + [_START_THINK, _E2E4, _WL, _D] + board + [_END_VAR]
    # No end_think -> truncated
    assert format_reward("e2e4", token_ids, {}) == -0.5


def test_format_no_end_var():
    board = _make_board_block()
    # Has start_think and end_think but no end_var between them
    token_ids = board + [_START_THINK, _E2E4, _WL, _D, _END_THINK, _E2E4]
    assert format_reward("e2e4", token_ids, {}) == 0.0


def test_format_broken_board_block():
    # Board block that's too short (only 67 tokens instead of 68)
    short_board = [_START_POS] + [_EMPTY] * 63 + [_END_POS, _NO_CASTLE]  # 67 tokens
    token_ids = (
        _make_board_block()
        + [_START_THINK, _E2E4, _WL, _D]
        + short_board  # incomplete
        + [_END_VAR, _END_THINK, _E2E4]
    )
    assert format_reward("e2e4", token_ids, {}) == 0.0


# --- coherence_reward ---

def test_coherence_move_explored():
    token_ids = _make_valid_thinking_seq(root_move=_E2E4, final_move=_E2E4)
    assert coherence_reward("e2e4", token_ids, {}) == 1.0


def test_coherence_move_not_explored():
    token_ids = _make_valid_thinking_seq(root_move=_E2E4, final_move=_D2D4)
    # Final move d2d4 not explored as root move (root was e2e4)
    assert coherence_reward("d2d4", token_ids, {}) == 0.0


def test_coherence_no_thinking():
    assert coherence_reward("e2e4", [_EMPTY] * 10, {}) == 0.0


# --- CompositeReward ---

def test_composite_weighted_sum():
    token_ids = _make_valid_thinking_seq()
    cr = CompositeReward({"move_quality": 0.5, "format": 0.3})
    total, components = cr("e2e4", token_ids, {"best_move": "e2e4"})
    expected = 0.5 * 1.0 + 0.3 * 1.0
    assert abs(total - expected) < 1e-6
    assert components["move_quality"] == 1.0
    assert components["format"] == 1.0


def test_composite_unknown_reward_raises():
    with pytest.raises(ValueError, match="Unknown reward"):
        CompositeReward({"nonexistent": 1.0})

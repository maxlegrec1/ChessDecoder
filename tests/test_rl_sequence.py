"""Tests for RL sequence parsing (pure Python, no GPU)."""

import torch

from src.models.vocab import token_to_idx, move_vocab_size, POSITION_TOKEN_LENGTH
from src.rl.sequence import parse_rollout, collate_rollouts


# Token IDs
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
_PAD = token_to_idx["pad"]
_E2E4 = token_to_idx["e2e4"]
_D2D4 = token_to_idx["d2d4"]


def _board_block():
    return [_START_POS] + [_EMPTY] * 64 + [_END_POS, _NO_CASTLE, _WTM]


def _make_rollout():
    """Build a minimal valid rollout with 1 variation."""
    board = _board_block()
    token_ids = (
        board                               # root board (68)
        + [_START_THINK]                     # pos 68
        + [_E2E4, _WL, _D]                  # root move + values (69, 70, 71)
        + board                              # board after move (72-139)
        + [_END_VAR]                         # 140
        + [_END_THINK]                       # 141
        + [_D2D4, _WL, _D]                  # final move + values (142, 143, 144)
    )
    wl_entries = [(70, 0.1), (143, 0.2)]
    d_entries = [(71, 0.3), (144, 0.4)]
    return token_ids, wl_entries, d_entries


def test_parse_rollout_shapes():
    token_ids, wl_entries, d_entries = _make_rollout()
    max_seq_len = 256
    result = parse_rollout(token_ids, wl_entries, d_entries, max_seq_len)

    for key in ["input_ids", "block_id", "wl_positions", "d_positions",
                "wl_values", "d_values", "thinking_move_mask",
                "final_move_mask", "move_token_ids"]:
        assert result[key].shape == (max_seq_len,), f"{key} shape mismatch"
    assert isinstance(result["seq_len"], int)


def test_parse_rollout_block_ids():
    token_ids, wl, d = _make_rollout()
    result = parse_rollout(token_ids, wl, d, 256)

    # Two board blocks: positions [0,68) and [72,140)
    block_id = result["block_id"]
    assert block_id[0].item() == 0      # first block
    assert block_id[67].item() == 0     # still first block
    assert block_id[72].item() == 1     # second block
    assert block_id[139].item() == 1    # still second block

    # Orphan tokens have unique IDs >= n_blocks=2
    assert block_id[68].item() >= 2     # start_think
    assert block_id[69].item() >= 2     # move


def test_parse_rollout_masks_exclusive():
    token_ids, wl, d = _make_rollout()
    result = parse_rollout(token_ids, wl, d, 256)
    overlap = result["thinking_move_mask"] & result["final_move_mask"]
    assert not overlap.any()


def test_parse_rollout_thinking_move():
    token_ids, wl, d = _make_rollout()
    result = parse_rollout(token_ids, wl, d, 256)

    # e2e4 at position 69 — mask should be at prediction position 68 (start_think)
    assert result["thinking_move_mask"][68].item() is True
    assert result["thinking_move_mask"][69].item() is False
    # move_token_id at prediction position should be valid
    assert 0 <= result["move_token_ids"][68].item() < move_vocab_size


def test_parse_rollout_final_move():
    token_ids, wl, d = _make_rollout()
    result = parse_rollout(token_ids, wl, d, 256)

    # d2d4 at position 142 — mask should be at prediction position 141 (end_think)
    assert result["final_move_mask"][141].item() is True
    assert result["final_move_mask"][142].item() is False
    assert 0 <= result["move_token_ids"][141].item() < move_vocab_size


def test_parse_rollout_wl_d_positions():
    token_ids, wl_entries, d_entries = _make_rollout()
    result = parse_rollout(token_ids, wl_entries, d_entries, 256)

    for pos, val in wl_entries:
        assert result["wl_positions"][pos].item() is True
        assert abs(result["wl_values"][pos].item() - val) < 1e-6
    for pos, val in d_entries:
        assert result["d_positions"][pos].item() is True
        assert abs(result["d_values"][pos].item() - val) < 1e-6


def test_parse_rollout_padding():
    token_ids, wl, d = _make_rollout()
    max_seq_len = 256
    result = parse_rollout(token_ids, wl, d, max_seq_len)
    seq_len = result["seq_len"]

    # Beyond seq_len should be pad tokens
    for i in range(seq_len, max_seq_len):
        assert result["input_ids"][i].item() == _PAD


def test_collate_rollouts_batch():
    token_ids, wl, d = _make_rollout()
    parsed = [parse_rollout(token_ids, wl, d, 256) for _ in range(3)]
    batch = collate_rollouts(parsed, torch.device("cpu"))

    for key in ["input_ids", "block_id", "thinking_move_mask", "final_move_mask"]:
        assert batch[key].shape == (3, 256), f"{key} batch shape mismatch"
    assert batch["seq_lens"].shape == (3,)

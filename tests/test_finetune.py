"""Finetune variation sequence and tensor building tests (pure Python, no GPU)."""

import json
import pytest
import pyarrow.parquet as pq

from src.models.vocab import (
    token_to_idx, idx_to_token, board_token_to_idx,
    move_vocab_size, POSITION_TOKEN_LENGTH,
)
from src.finetune.data import variation_to_token_ids
from src.finetune.loader import FinetuneIterableDataset


@pytest.fixture(scope="module")
def sample_variation_row():
    """Load first valid row from variation parquets."""
    import glob
    files = sorted(glob.glob("parquets_variations/*.parquet"))
    if not files:
        pytest.skip("No variation parquet files found")
    table = pq.read_table(files[0])
    df = table.to_pandas()
    for _, row in df.iterrows():
        try:
            variations = row["variations"]
            if isinstance(variations, str):
                variations = json.loads(variations)
            if len(variations) >= 2:
                return row
        except Exception:
            continue
    pytest.skip("No valid variation row found")


@pytest.fixture(scope="module")
def variation_result(sample_variation_row):
    """Parsed variation sequence from the sample row."""
    return variation_to_token_ids(sample_variation_row, max_variations=3, max_depth=5)


def test_variation_starts_with_board_and_start_think(variation_result):
    ids = variation_result[0]
    assert ids[0] == token_to_idx["start_pos"]
    assert len(ids) > POSITION_TOKEN_LENGTH
    assert ids[POSITION_TOKEN_LENGTH] == token_to_idx["start_think"]


def test_variation_ends_with_final_move_wl_d(variation_result):
    ids = variation_result[0]
    # Last 3 tokens should be: final_move, wl_value, d_value
    assert ids[-2] == token_to_idx["wl_value"]
    assert ids[-1] == token_to_idx["d_value"]
    # The token before wl_value should be a move token
    final_move_id = ids[-3]
    assert 0 <= final_move_id < move_vocab_size


def test_variation_contains_end_think(variation_result):
    ids = variation_result[0]
    assert token_to_idx["end_think"] in ids


def test_variation_thinking_move_positions(variation_result):
    ids = variation_result[0]
    thinking_move_data = variation_result[1]
    assert len(thinking_move_data) > 0

    for pos, move_str in thinking_move_data:
        assert 0 <= pos < len(ids), f"Position {pos} out of bounds"
        assert move_str in token_to_idx, f"Move '{move_str}' not in vocab"
        # The token AT the prediction position should be a predecessor
        # (start_think, end_var, or board STM)


def test_variation_final_move_data(variation_result):
    ids = variation_result[0]
    final_move_data = variation_result[2]
    assert final_move_data is not None
    pos, move_str = final_move_data
    assert ids[pos] == token_to_idx["end_think"]
    assert move_str in token_to_idx


def test_variation_block_boundaries(variation_result):
    ids = variation_result[0]
    block_boundaries = variation_result[4]
    assert len(block_boundaries) >= 1  # at least root board

    prev_end = -1
    for start, end in block_boundaries:
        assert start >= 0
        assert end <= len(ids)
        assert end - start == POSITION_TOKEN_LENGTH
        assert ids[start] == token_to_idx["start_pos"]
        assert start > prev_end, "Overlapping blocks"
        prev_end = end


def test_variation_value_data(variation_result):
    ids = variation_result[0]
    value_data = variation_result[3]
    assert len(value_data) > 0

    for wl_pos, d_pos, wl, d, is_valid in value_data:
        assert ids[wl_pos] == token_to_idx["wl_value"]
        assert ids[d_pos] == token_to_idx["d_value"]
        assert d_pos == wl_pos + 1


# --- Tensor building tests ---

@pytest.fixture(scope="module")
def dataset():
    """Minimal FinetuneIterableDataset (no actual data loading, just methods)."""
    return FinetuneIterableDataset(
        pretrain_parquet_dir=None,
        variation_parquet_dir=None,
        max_seq_len=1024,
    )


def test_build_variation_tensors_masks_exclusive(dataset, variation_result):
    ids, thinking_move_data, final_move_data, value_data, block_boundaries = (
        variation_result[0], variation_result[1], variation_result[2],
        variation_result[3], variation_result[4],
    )
    first_is_not_best = variation_result[6]
    max_depth_end_var = variation_result[7]
    max_var_end_think = variation_result[8]

    tensors = dataset._build_variation_tensors(
        ids, thinking_move_data, final_move_data, value_data,
        block_boundaries, first_is_not_best,
        max_depth_end_var, max_var_end_think,
    )
    overlap = tensors["move_mask"] & tensors["thinking_move_mask"]
    assert not overlap.any(), "move_mask and thinking_move_mask overlap"


def test_build_variation_tensors_board_target_at_move_positions(dataset, variation_result):
    ids, thinking_move_data, final_move_data, value_data, block_boundaries = (
        variation_result[0], variation_result[1], variation_result[2],
        variation_result[3], variation_result[4],
    )
    tensors = dataset._build_variation_tensors(
        ids, thinking_move_data, final_move_data, value_data,
        block_boundaries, variation_result[6],
        variation_result[7], variation_result[8],
    )
    board_targets = tensors["board_target_ids"]
    move_mask = tensors["move_mask"]
    think_mask = tensors["thinking_move_mask"]
    any_move = move_mask | think_mask

    IGNORE = -100
    generic = board_token_to_idx["generic_move"]
    continue_var = board_token_to_idx["continue_var"]
    new_var = board_token_to_idx["new_variation"]

    # At positions where a move is predicted, board target should NOT be IGNORE
    for i in range(any_move.shape[0]):
        if any_move[i]:
            bt = board_targets[i].item()
            assert bt != IGNORE, f"Board target is IGNORE at move position {i}"
            assert bt in (generic, continue_var, new_var), \
                f"Unexpected board target {bt} at move position {i}"

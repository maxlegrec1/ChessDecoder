"""Parse C++ engine rollout output into model-ready training tensors.

This is the RL equivalent of _build_variation_tensors in chessdecoder/finetune/loader.py,
but operates on generated sequences from the C++ inference engine rather than
ground-truth data from parquet files.

Note: uses ``wl_values``/``d_values`` (not ``wl_targets``/``d_targets`` as in finetune)
because these are inference outputs, not ground-truth labels.
"""

import torch

from chessdecoder.models.vocab import (
    token_to_idx, full_idx_to_move_idx, move_vocab_size,
    POSITION_TOKEN_LENGTH,
)

_START_THINK = token_to_idx["start_think"]
_END_THINK = token_to_idx["end_think"]
_END_VAR = token_to_idx["end_var"]
_START_POS = token_to_idx["start_pos"]
_WL_VALUE = token_to_idx["wl_value"]
_D_VALUE = token_to_idx["d_value"]
_PAD = token_to_idx["pad"]


def _is_move_token(tok_id: int) -> bool:
    return 0 <= tok_id < move_vocab_size


def parse_rollout(
    token_ids: list[int],
    wl_entries: list[tuple[int, float]],
    d_entries: list[tuple[int, float]],
    max_seq_len: int,
) -> dict:
    """Parse a single rollout's token sequence into model-ready tensors.

    Walks the token_ids using the same state machine as the C++ engine
    to identify board blocks, move positions, and value positions.

    Returns dict with tensors sized [max_seq_len].
    """
    seq_len = min(len(token_ids), max_seq_len)

    input_ids = torch.full((max_seq_len,), _PAD, dtype=torch.long)
    input_ids[:seq_len] = torch.tensor(token_ids[:seq_len], dtype=torch.long)

    thinking_move_mask = torch.zeros(max_seq_len, dtype=torch.bool)
    final_move_mask = torch.zeros(max_seq_len, dtype=torch.bool)
    move_token_ids = torch.full((max_seq_len,), -1, dtype=torch.long)

    wl_positions = torch.zeros(max_seq_len, dtype=torch.bool)
    d_positions = torch.zeros(max_seq_len, dtype=torch.bool)
    wl_values = torch.zeros(max_seq_len, dtype=torch.float32)
    d_values = torch.zeros(max_seq_len, dtype=torch.float32)

    # Build WL/D lookup from entries
    wl_lookup = {pos: val for pos, val in wl_entries if pos < max_seq_len}
    d_lookup = {pos: val for pos, val in d_entries if pos < max_seq_len}

    # Fill WL/D positions and values
    for pos, val in wl_lookup.items():
        wl_positions[pos] = True
        wl_values[pos] = val
    for pos, val in d_lookup.items():
        d_positions[pos] = True
        d_values[pos] = val

    # Walk tokens to identify blocks and move positions
    block_boundaries: list[tuple[int, int]] = []
    i = 0
    in_thinking = False

    while i < seq_len:
        tok = token_ids[i]

        # Board block detection
        if tok == _START_POS and i + POSITION_TOKEN_LENGTH <= seq_len:
            block_boundaries.append((i, i + POSITION_TOKEN_LENGTH))
            i += POSITION_TOKEN_LENGTH
            continue

        if tok == _START_THINK:
            in_thinking = True
            i += 1
            continue

        if tok == _END_THINK:
            in_thinking = False
            # Final move is predicted from end_think position (i)
            j = i + 1
            while j < seq_len:
                if _is_move_token(token_ids[j]):
                    final_move_mask[i] = True
                    move_token_ids[i] = full_idx_to_move_idx[token_ids[j]]
                    break
                j += 1
            i += 1
            continue

        if tok == _END_VAR:
            i += 1
            continue

        # Move token inside thinking region — mark the prediction position (i-1)
        if in_thinking and _is_move_token(tok):
            thinking_move_mask[i - 1] = True
            move_token_ids[i - 1] = full_idx_to_move_idx[tok]
            i += 1
            continue

        i += 1

    # Build block_id tensor: board blocks get sequential IDs, orphans get unique IDs
    n_blocks = len(block_boundaries)
    block_id = torch.arange(max_seq_len, dtype=torch.long) + n_blocks
    for block_num, (b_start, b_end) in enumerate(block_boundaries):
        block_id[b_start:b_end] = block_num

    return {
        "input_ids": input_ids,
        "block_id": block_id,
        "wl_positions": wl_positions,
        "d_positions": d_positions,
        "wl_values": wl_values,
        "d_values": d_values,
        "thinking_move_mask": thinking_move_mask,
        "final_move_mask": final_move_mask,
        "move_token_ids": move_token_ids,
        "seq_len": seq_len,
    }


def collate_rollouts(parsed_list: list[dict], device: torch.device) -> dict:
    """Stack a list of parsed rollouts into a batched dict of tensors.

    Args:
        parsed_list: list of dicts from parse_rollout().
        device: target device.

    Returns:
        Batched dict with tensors of shape [B, max_seq_len].
    """
    keys = ["input_ids", "block_id", "wl_positions", "d_positions",
            "wl_values", "d_values", "thinking_move_mask", "final_move_mask",
            "move_token_ids"]
    batch = {}
    for k in keys:
        batch[k] = torch.stack([p[k] for p in parsed_list]).to(device)
    batch["seq_lens"] = torch.tensor([p["seq_len"] for p in parsed_list],
                                     dtype=torch.long, device=device)
    return batch

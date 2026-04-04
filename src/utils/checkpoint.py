"""Shared checkpoint-handling utilities.

Currently exposes :func:`migrate_state_dict`, the logic for migrating older
checkpoints to the current sub-vocab head layout. Previously this was
duplicated in ``src/train/train.py`` and ``src/finetune/train.py``.
"""

import torch

from src.models.vocab import (
    vocab_size,
    board_vocab_size,
    move_vocab_size,
    board_idx_to_full_idx,
    move_idx_to_full_idx,
)


def migrate_state_dict(state_dict):
    """Migrate old checkpoint: sub-vocab heads, expand embedding, clone thinking_policy_head."""
    # Expand tok_embedding if needed
    if "tok_embedding.weight" in state_dict:
        t = state_dict["tok_embedding.weight"]
        if t.shape[0] < vocab_size:
            pad = torch.zeros(vocab_size - t.shape[0], *t.shape[1:], dtype=t.dtype, device=t.device)
            state_dict["tok_embedding.weight"] = torch.cat([t, pad], dim=0)

    # board_head: extract rows for board sub-vocab from old full-vocab weights
    if "board_head.weight" in state_dict:
        old_w = state_dict["board_head.weight"]
        old_b = state_dict["board_head.bias"]
        if old_w.shape[0] > board_vocab_size:
            # Old checkpoint has full-vocab head — extract sub-vocab rows.
            # Some board sub-vocab tokens may have indices >= old vocab size (e.g. generic_move)
            old_vocab_sz = old_w.shape[0]
            new_w = torch.zeros(board_vocab_size, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
            new_b = torch.zeros(board_vocab_size, dtype=old_b.dtype, device=old_b.device)
            for i, full_idx in enumerate(board_idx_to_full_idx):
                if full_idx < old_vocab_sz:
                    new_w[i] = old_w[full_idx]
                    new_b[i] = old_b[full_idx]
            state_dict["board_head.weight"] = new_w
            state_dict["board_head.bias"] = new_b

    # policy_head / thinking_policy_head: extract rows for move sub-vocab
    for head in ["policy_head", "thinking_policy_head"]:
        if f"{head}.weight" not in state_dict:
            continue
        old_w = state_dict[f"{head}.weight"]
        old_b = state_dict[f"{head}.bias"]
        if old_w.shape[0] > move_vocab_size:
            old_vocab_sz = old_w.shape[0]
            new_w = torch.zeros(move_vocab_size, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
            new_b = torch.zeros(move_vocab_size, dtype=old_b.dtype, device=old_b.device)
            for i, full_idx in enumerate(move_idx_to_full_idx):
                if full_idx < old_vocab_sz:
                    new_w[i] = old_w[full_idx]
                    new_b[i] = old_b[full_idx]
            state_dict[f"{head}.weight"] = new_w
            state_dict[f"{head}.bias"] = new_b

    # Clone policy_head -> thinking_policy_head (if not already present)
    if "thinking_policy_head.weight" not in state_dict and "policy_head.weight" in state_dict:
        state_dict["thinking_policy_head.weight"] = state_dict["policy_head.weight"].clone()
        state_dict["thinking_policy_head.bias"] = state_dict["policy_head.bias"].clone()

    return state_dict

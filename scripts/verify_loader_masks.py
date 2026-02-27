"""
Verify that the finetuning dataloader produces correct masks and targets.

Tests both variation and pretrain sequences from the loader, checking:
1. WL/D targets are set at all move prediction positions (move_mask | thinking_move_mask)
2. move_mask and thinking_move_mask are mutually exclusive
3. wl_positions and d_positions have matching counts
4. Value targets at prediction positions match the values at wl_value/d_value token positions
5. Board target overrides are correct at move prediction positions
6. Sequence structure is valid (tokens in expected sub-vocabularies)
"""

import sys
import json
import torch
import numpy as np
from src.finetune.loader import FinetuneIterableDataset
from src.finetune.data import variation_to_token_ids
from src.models.vocab import (
    token_to_idx, idx_to_token, board_token_to_idx,
    full_idx_to_board_idx, full_idx_to_move_idx,
    board_idx_to_full_idx, move_idx_to_full_idx,
)

IGNORE_INDEX = -100

# --- Helpers ---

def token_name(token_id):
    return idx_to_token.get(token_id, f"<unk:{token_id}>")


def check_sample(sample, label, verbose=False):
    """Run all checks on a single sample dict. Returns (passed, errors)."""
    errors = []
    input_ids = sample["input_ids"]
    board_target_ids = sample["board_target_ids"]
    move_target_ids = sample["move_target_ids"]
    move_mask = sample["move_mask"]
    thinking_move_mask = sample["thinking_move_mask"]
    wl_positions = sample["wl_positions"]
    d_positions = sample["d_positions"]
    wl_targets = sample["wl_targets"]
    d_targets = sample["d_targets"]
    wdl_valid = sample["wdl_valid"]
    block_id = sample["block_id"]

    pad_id = token_to_idx["pad"]
    seq_len = (input_ids != pad_id).sum().item()

    # 1. move_mask and thinking_move_mask are mutually exclusive
    overlap = (move_mask & thinking_move_mask).sum().item()
    if overlap > 0:
        errors.append(f"move_mask and thinking_move_mask overlap at {overlap} positions")

    # 2. All move_mask/thinking_move_mask positions are within seq_len
    any_move = move_mask | thinking_move_mask
    any_move_positions = any_move.nonzero(as_tuple=True)[0]
    if any_move_positions.numel() > 0 and any_move_positions.max().item() >= seq_len:
        errors.append(f"Move mask positions extend beyond seq_len={seq_len}")

    # 3. wl_positions and d_positions have matching counts
    n_wl = wl_positions.sum().item()
    n_d = d_positions.sum().item()
    if n_wl != n_d:
        errors.append(f"wl_positions count ({n_wl}) != d_positions count ({n_d})")

    # 4. All move prediction positions have valid wl_targets and wdl_valid
    for pos in any_move_positions.tolist():
        if not wdl_valid[pos]:
            errors.append(f"  pos {pos} ({token_name(input_ids[pos].item())}): wdl_valid=False")
        if wl_targets[pos].item() == 0.0 and d_targets[pos].item() == 0.0:
            # Could be legitimate (drawn position), just flag for review
            pass

    # 5. For each move prediction position, check that wl_pos = pos+2 exists in wl_positions
    #    and that wl_targets[pos] == wl_targets[wl_pos]
    value_mismatch = 0
    for pos in any_move_positions.tolist():
        wl_tok_pos = pos + 2
        if wl_tok_pos < seq_len:
            if not wl_positions[wl_tok_pos]:
                errors.append(f"  pos {pos}: expected wl_positions[{wl_tok_pos}]=True, got False "
                              f"(token at +2: {token_name(input_ids[wl_tok_pos].item())})")
            else:
                wl_at_pred = wl_targets[pos].item()
                wl_at_tok = wl_targets[wl_tok_pos].item()
                if abs(wl_at_pred - wl_at_tok) > 1e-6:
                    errors.append(f"  pos {pos}: wl_targets[{pos}]={wl_at_pred:.4f} != wl_targets[{wl_tok_pos}]={wl_at_tok:.4f}")
                    value_mismatch += 1

    # 6. Board target overrides at move prediction positions
    generic_move_idx = board_token_to_idx["generic_move"]
    new_variation_idx = board_token_to_idx["new_variation"]
    continue_var_idx = board_token_to_idx["continue_var"]
    start_think_id = token_to_idx["start_think"]
    end_var_id = token_to_idx["end_var"]
    end_think_id = token_to_idx["end_think"]

    for pos in any_move_positions.tolist():
        tok_id = input_ids[pos].item()
        bt = board_target_ids[pos].item()
        if tok_id == start_think_id:
            if bt != generic_move_idx:
                errors.append(f"  pos {pos} (start_think): board_target should be generic_move ({generic_move_idx}), got {bt}")
        elif tok_id == end_var_id:
            if bt != new_variation_idx:
                errors.append(f"  pos {pos} (end_var): board_target should be new_variation ({new_variation_idx}), got {bt}")
        elif tok_id == end_think_id:
            if bt != generic_move_idx:
                errors.append(f"  pos {pos} (end_think): board_target should be generic_move ({generic_move_idx}), got {bt}")
        else:
            # Should be a stm position (continue_var for thinking, generic_move for final)
            if thinking_move_mask[pos]:
                if bt != continue_var_idx:
                    errors.append(f"  pos {pos} ({token_name(tok_id)}): thinking_move board_target should be continue_var ({continue_var_idx}), got {bt}")

    # 7. move_target_ids are set (not IGNORE_INDEX) at all move prediction positions
    for pos in any_move_positions.tolist():
        mt = move_target_ids[pos].item()
        if mt == IGNORE_INDEX:
            errors.append(f"  pos {pos}: move_target_ids is IGNORE_INDEX at move prediction position")

    # 8. Check token at pos+1 (the actual move) is a valid move token
    for pos in any_move_positions.tolist():
        move_pos = pos + 1
        if move_pos < seq_len:
            move_tok_id = input_ids[move_pos].item()
            if move_tok_id not in full_idx_to_move_idx:
                errors.append(f"  pos {pos}: token at pos+1 ({token_name(move_tok_id)}) is not a move token")

    # 9. Verify wl_value/d_value tokens are at the right positions
    wl_value_id = token_to_idx["wl_value"]
    d_value_id = token_to_idx["d_value"]
    for pos in wl_positions.nonzero(as_tuple=True)[0].tolist():
        if input_ids[pos].item() != wl_value_id:
            errors.append(f"  wl_positions[{pos}]=True but token is {token_name(input_ids[pos].item())}, expected wl_value")
    for pos in d_positions.nonzero(as_tuple=True)[0].tolist():
        if input_ids[pos].item() != d_value_id:
            errors.append(f"  d_positions[{pos}]=True but token is {token_name(input_ids[pos].item())}, expected d_value")

    # Summary
    n_final = move_mask.sum().item()
    n_thinking = thinking_move_mask.sum().item()

    if verbose or errors:
        print(f"\n{'='*60}")
        print(f"{label}: seq_len={seq_len}, final_moves={n_final}, thinking_moves={n_thinking}, "
              f"wl_positions={n_wl}, d_positions={n_d}")
        if errors:
            print(f"  ERRORS ({len(errors)}):")
            for e in errors:
                print(f"    {e}")
        else:
            print(f"  ALL CHECKS PASSED")

    return len(errors) == 0, errors, {"seq_len": seq_len, "n_final": n_final, "n_thinking": n_thinking, "n_wl": n_wl}


def dump_sequence(sample, max_tokens=200):
    """Print the token sequence for visual inspection."""
    input_ids = sample["input_ids"]
    move_mask = sample["move_mask"]
    thinking_move_mask = sample["thinking_move_mask"]
    wl_positions = sample["wl_positions"]
    d_positions = sample["d_positions"]
    wl_targets = sample["wl_targets"]
    d_targets = sample["d_targets"]
    wdl_valid = sample["wdl_valid"]
    board_target_ids = sample["board_target_ids"]

    pad_id = token_to_idx["pad"]
    seq_len = (input_ids != pad_id).sum().item()

    print(f"\n--- Sequence dump (first {min(max_tokens, seq_len)} of {seq_len} tokens) ---")
    for i in range(min(max_tokens, seq_len)):
        tok = token_name(input_ids[i].item())
        flags = []
        if move_mask[i]:
            flags.append("MOVE")
        if thinking_move_mask[i]:
            flags.append("THINK")
        if wl_positions[i]:
            flags.append(f"WL_POS(wl={wl_targets[i]:.3f})")
        if d_positions[i]:
            flags.append(f"D_POS(d={d_targets[i]:.3f})")
        if wdl_valid[i] and (move_mask[i] or thinking_move_mask[i]):
            flags.append(f"tgt_wl={wl_targets[i]:.3f},d={d_targets[i]:.3f}")

        bt = board_target_ids[i].item()
        if bt != IGNORE_INDEX:
            full_id = board_idx_to_full_idx[bt] if 0 <= bt < len(board_idx_to_full_idx) else -1
            bt_name = idx_to_token.get(full_id, f"board_{bt}")
            flags.append(f"bt={bt_name}")

        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {i:4d}: {tok:20s}{flag_str}")


def main():
    import glob
    import os

    variation_dir = "/datadrive/variations/"
    pretrain_dir = "/home/maxime/parquet_files_decoder/"

    # Check data exists
    var_files = sorted(glob.glob(os.path.join(variation_dir, "*.parquet")))
    pt_files = sorted(glob.glob(os.path.join(pretrain_dir, "*.parquet")))
    print(f"Variation files: {len(var_files)}")
    print(f"Pretrain files: {len(pt_files)}")

    if not var_files:
        print("No variation files found, trying local path...")
        variation_dir = "parquets_variations/"
        var_files = sorted(glob.glob(os.path.join(variation_dir, "*.parquet")))
        print(f"Variation files (local): {len(var_files)}")

    # Create dataset with variation_ratio=1 to get only variation samples
    dataset = FinetuneIterableDataset(
        variation_parquet_dir=variation_dir,
        pretrain_parquet_dir=pretrain_dir,
        max_seq_len=4096,
        variation_ratio=1.0,
        max_variations=3,
        max_depth=5,
        tau_base=0.3,
        tau_alpha=1.0,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("CHECKING VARIATION SAMPLES")
    print("=" * 60)

    n_checked = 0
    n_passed = 0
    total_final = 0
    total_thinking = 0
    total_wl = 0

    for i, sample in enumerate(dataset):
        if i >= 100:
            break
        passed, errs, stats = check_sample(sample, f"Variation #{i}", verbose=(i < 3))
        if i < 2:
            dump_sequence(sample, max_tokens=120)
        n_checked += 1
        if passed:
            n_passed += 1
        total_final += stats["n_final"]
        total_thinking += stats["n_thinking"]
        total_wl += stats["n_wl"]

    print(f"\n{'='*60}")
    print(f"VARIATION RESULTS: {n_passed}/{n_checked} samples passed all checks")
    print(f"  Avg final moves/sample: {total_final/max(n_checked,1):.1f}")
    print(f"  Avg thinking moves/sample: {total_thinking/max(n_checked,1):.1f}")
    print(f"  Avg wl_positions/sample: {total_wl/max(n_checked,1):.1f}")
    print(f"  WL head now trains on: {(total_final+total_thinking)/max(n_checked,1):.1f} positions/sample (was {total_final/max(n_checked,1):.1f})")

    # Also check pretrain samples
    print("\n" + "=" * 60)
    print("CHECKING PRETRAIN SAMPLES")
    print("=" * 60)

    dataset_pt = FinetuneIterableDataset(
        variation_parquet_dir=variation_dir,
        pretrain_parquet_dir=pretrain_dir,
        max_seq_len=256,
        variation_ratio=0.0,  # Only pretrain
        max_variations=3,
        max_depth=5,
        seed=42,
    )

    n_checked_pt = 0
    n_passed_pt = 0
    total_final_pt = 0

    for i, sample in enumerate(dataset_pt):
        if i >= 50:
            break
        passed, errs, stats = check_sample(sample, f"Pretrain #{i}", verbose=(i < 1))
        n_checked_pt += 1
        if passed:
            n_passed_pt += 1
        total_final_pt += stats["n_final"]

    print(f"\n{'='*60}")
    print(f"PRETRAIN RESULTS: {n_passed_pt}/{n_checked_pt} samples passed all checks")
    print(f"  Avg move predictions/sample: {total_final_pt/max(n_checked_pt,1):.1f}")

    # Final verdict
    print(f"\n{'='*60}")
    all_passed = (n_passed == n_checked) and (n_passed_pt == n_checked_pt)
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILURES: {n_checked - n_passed} variation, {n_checked_pt - n_passed_pt} pretrain")
        sys.exit(1)


if __name__ == "__main__":
    main()

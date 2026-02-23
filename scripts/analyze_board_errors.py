"""
Analyze board prediction failures in finetuned.pt on variation data.

Loads variation parquets, runs inference with the finetuned model, and
identifies patterns in positions where board_total_acc == 0 (entire board wrong).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import chess
from collections import Counter, defaultdict
from tqdm import tqdm

from src.models.model import ChessDecoder
from src.models.vocab import (
    vocab_size, token_to_idx, idx_to_token,
    board_vocab_size, move_vocab_size,
    board_idx_to_full_idx, move_idx_to_full_idx,
    full_idx_to_board_idx, full_idx_to_move_idx,
    board_token_to_idx, board_vocab,
)
from src.finetune.loader import FinetuneIterableDataset
from src.finetune.data import variation_to_token_ids
from src.finetune.train import migrate_state_dict
from torch.utils.data import DataLoader


def load_model(checkpoint_path, device):
    """Load finetuned model."""
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=1024,
        num_heads=16,
        num_layers=12,
        max_seq_len=4096,
        d_ff=1536,
        n_buckets=100,
        value_hidden_size=256,
        num_fourier_freq=128,
        wl_sigma=0.4,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = migrate_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def analyze_batch(model, batch, device):
    """
    Run inference on a batch and return per-block error analysis.

    Returns list of dicts with info about each board block.
    """
    IGNORE_INDEX = -100

    input_ids = batch["input_ids"].to(device)
    board_target_ids = batch["board_target_ids"].to(device)
    move_target_ids = batch["move_target_ids"].to(device)
    move_mask = batch["move_mask"].to(device)
    thinking_move_mask = batch["thinking_move_mask"].to(device)
    wl_positions = batch["wl_positions"].to(device)
    d_positions = batch["d_positions"].to(device)
    wl_targets = batch["wl_targets"].to(device)
    d_targets = batch["d_targets"].to(device)
    block_id = batch["block_id"].to(device)

    B, S = input_ids.shape

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        # Fourier inputs
        wl_fourier_input = torch.zeros_like(wl_targets)
        d_fourier_input = torch.zeros_like(d_targets)
        if wl_positions.any():
            wl_fourier_input[wl_positions] = model.discretize_to_bucket(
                wl_targets[wl_positions], model.wl_bucket_centers)
        if d_positions.any():
            d_fourier_input[d_positions] = model.discretize_to_bucket(
                d_targets[d_positions], model.d_bucket_centers)

        # Causal pass
        h_causal = model(
            input_ids, mask_type="causal",
            wl_values=wl_fourier_input, d_values=d_fourier_input,
            wl_positions=wl_positions, d_positions=d_positions,
        )
        board_logits = model.board_head(h_causal)

    # Board mask (same as training)
    board_mask = board_target_ids != IGNORE_INDEX
    any_move = move_mask | thinking_move_mask
    first_move_idx = any_move.int().argmax(dim=1)
    has_moves = any_move.any(dim=1)
    first_move_idx[~has_moves] = any_move.size(1)
    indices = torch.arange(any_move.size(1), device=device).unsqueeze(0)
    pre_first_move_mask = indices < first_move_idx.unsqueeze(1)
    board_mask = board_mask & (~pre_first_move_mask)

    preds_board = torch.argmax(board_logits, dim=-1)
    board_correct = (preds_board == board_target_ids) & board_mask

    # Now analyze per-block
    results = []

    for b in range(B):
        # Get unique block IDs for this sample (blocks are board blocks)
        sample_block_ids = block_id[b]
        sample_mask = board_mask[b]
        sample_correct = board_correct[b]
        sample_targets = board_target_ids[b]
        sample_preds = preds_board[b]
        sample_input = input_ids[b]

        # Find unique block IDs that have masked positions
        active_positions = sample_mask.nonzero(as_tuple=True)[0]
        if active_positions.numel() == 0:
            continue

        active_block_ids = sample_block_ids[active_positions]
        unique_blocks = active_block_ids.unique()

        for blk in unique_blocks:
            blk_positions = (sample_block_ids == blk) & sample_mask
            blk_pos_indices = blk_positions.nonzero(as_tuple=True)[0]

            if blk_pos_indices.numel() == 0:
                continue

            blk_correct = sample_correct[blk_pos_indices]
            blk_targets = sample_targets[blk_pos_indices]
            blk_preds = sample_preds[blk_pos_indices]
            blk_inputs = sample_input[blk_pos_indices]

            n_correct = blk_correct.sum().item()
            n_total = blk_correct.numel()
            all_correct = (n_correct == n_total)

            # Get full block range (including unmasked positions)
            all_blk_positions = (sample_block_ids == blk).nonzero(as_tuple=True)[0]
            blk_start = all_blk_positions[0].item()
            blk_end = all_blk_positions[-1].item() + 1

            # Figure out what kind of block this is
            # Check what token precedes the block
            prev_token_id = sample_input[blk_start - 1].item() if blk_start > 0 else -1
            prev_token = idx_to_token.get(prev_token_id, "N/A")

            # Check what token follows the block
            next_token_id = sample_input[blk_end].item() if blk_end < S else -1
            next_token = idx_to_token.get(next_token_id, "N/A")

            # Determine context: root board, variation board, etc.
            if blk.item() == 0:
                context = "root_board"
            else:
                context = "variation_board"

            # Collect per-position errors within this block
            errors = []
            for i, pos in enumerate(blk_pos_indices):
                if not blk_correct[i].item():
                    target_idx = blk_targets[i].item()
                    pred_idx = blk_preds[i].item()
                    input_idx = blk_inputs[i].item()

                    # Intra-block offset
                    intra_offset = pos.item() - blk_start

                    target_token = board_vocab[target_idx] if 0 <= target_idx < len(board_vocab) else f"idx_{target_idx}"
                    pred_token = board_vocab[pred_idx] if 0 <= pred_idx < len(board_vocab) else f"idx_{pred_idx}"
                    input_token = idx_to_token.get(input_idx, f"idx_{input_idx}")

                    errors.append({
                        "intra_offset": intra_offset,
                        "target": target_token,
                        "pred": pred_token,
                        "input": input_token,
                    })

            results.append({
                "batch_idx": b,
                "block_id": blk.item(),
                "context": context,
                "n_correct": n_correct,
                "n_total": n_total,
                "all_correct": all_correct,
                "prev_token": prev_token,
                "next_token": next_token,
                "blk_start": blk_start,
                "blk_end": blk_end,
                "errors": errors,
            })

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model("finetuned.pt", device)

    # Create dataset (variation only, no pretrain)
    variation_dir = "parquets_variations"
    import glob
    variation_files = sorted(glob.glob(os.path.join(variation_dir, "*.parquet")))
    print(f"Found {len(variation_files)} variation files")

    dataset = FinetuneIterableDataset(
        variation_files=variation_files,
        max_seq_len=4096,
        variation_ratio=1.0,  # 100% variation data
        max_variations=3,
        max_depth=5,
        shuffle_files=False,
        shuffle_games=False,
        tau_base=0.3,
        tau_alpha=1.0,
        seed=42,
    )

    dataloader = DataLoader(dataset, batch_size=16, num_workers=0)

    # Collect stats
    total_blocks = 0
    correct_blocks = 0
    wrong_blocks = 0

    # Aggregate error patterns
    error_by_context = Counter()         # context -> count of wrong blocks
    total_by_context = Counter()         # context -> count of all blocks
    error_by_intra_offset = Counter()    # intra_offset -> count of errors
    error_by_target_token = Counter()    # target_token -> count of errors
    error_by_pred_token = Counter()      # pred_token -> count of errors
    error_confusion = Counter()          # (target, pred) -> count
    error_by_prev_token = Counter()      # prev_token -> count of wrong blocks
    total_by_prev_token = Counter()      # prev_token -> count of all blocks

    # Errors per position in block
    errors_at_offset = defaultdict(list)  # offset -> list of (target, pred) tuples

    # Track how many errors per wrong block
    error_count_distribution = Counter()  # n_errors -> count of blocks

    # Track blocks by n_total
    error_by_block_size = Counter()      # block_size -> count of wrong blocks
    total_by_block_size = Counter()      # block_size -> count of all blocks

    n_batches = 0
    max_batches = 200  # Process enough data

    print(f"Processing {max_batches} batches...")
    for batch in tqdm(dataloader, total=max_batches):
        if n_batches >= max_batches:
            break

        block_results = analyze_batch(model, batch, device)

        for blk in block_results:
            total_blocks += 1
            total_by_context[blk["context"]] += 1
            total_by_prev_token[blk["prev_token"]] += 1
            total_by_block_size[blk["n_total"]] += 1

            if blk["all_correct"]:
                correct_blocks += 1
            else:
                wrong_blocks += 1
                n_errors = len(blk["errors"])
                error_count_distribution[n_errors] += 1
                error_by_context[blk["context"]] += 1
                error_by_prev_token[blk["prev_token"]] += 1
                error_by_block_size[blk["n_total"]] += 1

                for err in blk["errors"]:
                    error_by_intra_offset[err["intra_offset"]] += 1
                    error_by_target_token[err["target"]] += 1
                    error_by_pred_token[err["pred"]] += 1
                    error_confusion[(err["target"], err["pred"])] += 1
                    errors_at_offset[err["intra_offset"]].append((err["target"], err["pred"]))

        n_batches += 1

    print(f"\n{'='*80}")
    print(f"ANALYSIS RESULTS ({n_batches} batches, {total_blocks} blocks)")
    print(f"{'='*80}")

    print(f"\n--- Overall ---")
    print(f"Total blocks:   {total_blocks}")
    print(f"Correct blocks: {correct_blocks} ({100*correct_blocks/total_blocks:.1f}%)")
    print(f"Wrong blocks:   {wrong_blocks} ({100*wrong_blocks/total_blocks:.1f}%)")

    print(f"\n--- Error rate by context ---")
    for ctx in sorted(total_by_context.keys()):
        tot = total_by_context[ctx]
        err = error_by_context.get(ctx, 0)
        print(f"  {ctx:20s}: {err}/{tot} wrong ({100*err/tot:.1f}%)")

    print(f"\n--- Error rate by previous token ---")
    for tok, tot in sorted(total_by_prev_token.items(), key=lambda x: -x[1])[:20]:
        err = error_by_prev_token.get(tok, 0)
        print(f"  prev='{tok:20s}': {err}/{tot} wrong ({100*err/tot:.1f}%)")

    print(f"\n--- Errors per wrong block (distribution) ---")
    for n_err, cnt in sorted(error_count_distribution.items()):
        print(f"  {n_err} errors: {cnt} blocks")

    print(f"\n--- Error rate by block size (n_total masked positions) ---")
    for sz, tot in sorted(total_by_block_size.items()):
        err = error_by_block_size.get(sz, 0)
        print(f"  size={sz:3d}: {err}/{tot} wrong ({100*err/tot:.1f}%)")

    print(f"\n--- Most common error positions (intra-block offset) ---")
    for offset, cnt in sorted(error_by_intra_offset.items(), key=lambda x: -x[1])[:20]:
        # Interpret offset: 0=start_pos predicts first square, 1-64=squares, 65=end_pos->castling, 66=castling->stm
        if offset == 0:
            desc = "start_pos→square[0]"
        elif 1 <= offset <= 64:
            desc = f"square[{offset-1}]→square[{offset}]"
        elif offset == 65:
            desc = "end_pos→castling"
        elif offset == 66:
            desc = "castling→stm"
        else:
            desc = f"offset_{offset}"
        print(f"  offset={offset:3d} ({desc:25s}): {cnt} errors")

    print(f"\n--- Most common target tokens in errors ---")
    for tok, cnt in error_by_target_token.most_common(20):
        print(f"  target='{tok}': {cnt} errors")

    print(f"\n--- Most common predicted tokens in errors ---")
    for tok, cnt in error_by_pred_token.most_common(20):
        print(f"  pred='{tok}': {cnt} errors")

    print(f"\n--- Top 30 confusion pairs (target → pred) ---")
    for (tgt, pred), cnt in error_confusion.most_common(30):
        print(f"  '{tgt}' → '{pred}': {cnt}")

    # Detailed analysis of structural tokens
    print(f"\n--- Structural token errors ---")
    structural = ["end_var", "continue_var", "new_variation", "end_think", "generic_move", "start_think"]
    for tok in structural:
        target_cnt = error_by_target_token.get(tok, 0)
        pred_cnt = error_by_pred_token.get(tok, 0)
        print(f"  {tok:20s}: target_errors={target_cnt}, pred_as={pred_cnt}")

    # Analyze what's happening at castling position (offset 65)
    print(f"\n--- Errors at castling position (offset 65) ---")
    if 65 in errors_at_offset:
        castle_errors = errors_at_offset[65]
        castle_confusion = Counter(castle_errors)
        for (tgt, pred), cnt in castle_confusion.most_common(10):
            print(f"  target='{tgt}' → pred='{pred}': {cnt}")
    else:
        print("  No errors at offset 65")

    # Analyze what's happening at stm position (offset 66)
    print(f"\n--- Errors at STM position (offset 66) ---")
    if 66 in errors_at_offset:
        stm_errors = errors_at_offset[66]
        stm_confusion = Counter(stm_errors)
        for (tgt, pred), cnt in stm_confusion.most_common(10):
            print(f"  target='{tgt}' → pred='{pred}': {cnt}")
    else:
        print("  No errors at offset 66")

    # Analyze errors at offset > 66 (these would be structural tokens like continue_var, end_var, etc.)
    print(f"\n--- Errors at offset > 66 (post-board structural) ---")
    for offset in sorted(errors_at_offset.keys()):
        if offset > 66:
            confusion = Counter(errors_at_offset[offset])
            print(f"  offset {offset}:")
            for (tgt, pred), cnt in confusion.most_common(5):
                print(f"    target='{tgt}' → pred='{pred}': {cnt}")


if __name__ == "__main__":
    main()

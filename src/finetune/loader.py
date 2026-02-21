"""
Finetuning dataloader that mixes normal pretraining data with thinking variation data.

Uses configurable variation_ratio to control the mix (e.g. 0.2 = 20% thinking, 80% normal).
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
import math
import random
import json

from src.dataloader.data import game_to_token_ids
from src.models.vocab import (token_to_idx, full_idx_to_board_idx, full_idx_to_move_idx,
                              board_token_to_idx)
from src.finetune.data import variation_to_token_ids


class FinetuneIterableDataset(IterableDataset):
    def __init__(
        self,
        pretrain_parquet_dir=None,
        variation_parquet_dir=None,
        max_seq_len=1024,
        variation_ratio=0.2,
        max_variations=3,
        max_depth=5,
        shuffle_files=True,
        shuffle_games=True,
        skip_board_prob=0.0,
        tau_base=0.3,
        tau_alpha=1.0,
        pretrain_files=None,
        variation_files=None,
    ):
        self.max_seq_len = max_seq_len
        self.variation_ratio = variation_ratio
        self.max_variations = max_variations
        self.max_depth = max_depth
        self.shuffle_files = shuffle_files
        self.shuffle_games = shuffle_games
        self.skip_board_prob = skip_board_prob
        self.tau_base = tau_base
        self.tau_alpha = tau_alpha
        self.pad_id = token_to_idx["pad"]

        if pretrain_files is not None:
            self.pretrain_files = pretrain_files
        elif pretrain_parquet_dir is not None:
            self.pretrain_files = sorted(glob.glob(os.path.join(pretrain_parquet_dir, "*.parquet")))
        else:
            self.pretrain_files = []

        if variation_files is not None:
            self.variation_files = variation_files
        elif variation_parquet_dir is not None:
            self.variation_files = sorted(glob.glob(os.path.join(variation_parquet_dir, "*.parquet")))
        else:
            self.variation_files = []

        if not self.pretrain_files:
            print(f"Warning: No pretrain parquet files")
        if not self.variation_files:
            print(f"Warning: No variation parquet files")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Split files among workers
        if worker_info is None:
            pretrain_files = list(self.pretrain_files)
            variation_files = list(self.variation_files)
        else:
            nw = worker_info.num_workers
            wid = worker_info.id
            # Shard pretrain files across workers (large dataset, one pass per epoch)
            per_w = int(math.ceil(len(self.pretrain_files) / nw))
            pretrain_files = self.pretrain_files[wid * per_w: min((wid + 1) * per_w, len(self.pretrain_files))]
            # Don't shard variation files — each worker gets all of them.
            # Variation data is small and recycled many times per epoch anyway.
            # Sharding with fewer files than workers leaves some workers with 0 files,
            # causing their variation_epoch counter to spike meaninglessly.
            variation_files = list(self.variation_files)

        pretrain_iter = self._pretrain_iter(pretrain_files)
        variation_iter = self._variation_iter(variation_files)
        pretrain_epoch = 0

        while True:
            if random.random() < self.variation_ratio:
                sample = next(variation_iter, None)
                if sample is None:
                    # Variation data exhausted — epoch is over
                    return
                sample["pretrain_epoch"] = torch.tensor(pretrain_epoch, dtype=torch.long)
            else:
                sample = next(pretrain_iter, None)
                if sample is None:
                    # Restart pretrain iterator
                    pretrain_epoch += 1
                    pretrain_iter = self._pretrain_iter(pretrain_files)
                    sample = next(pretrain_iter, None)
                    if sample is None:
                        # No pretrain data at all, fall through to variation
                        sample = next(variation_iter, None)
                        if sample is None:
                            return
                sample["pretrain_epoch"] = torch.tensor(pretrain_epoch, dtype=torch.long)

            yield sample

    def _pretrain_iter(self, files):
        """Iterate over pretraining data (normal sequences, no thinking)."""
        files = list(files)
        if self.shuffle_files:
            random.shuffle(files)

        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                grouped = df.groupby('game_id', sort=False)
                game_ids = list(grouped.groups.keys())

                if self.shuffle_games:
                    np.random.shuffle(game_ids)

                for game_id in game_ids:
                    game_df = grouped.get_group(game_id).sort_values('ply')

                    ids, wdl_data, block_boundaries, value_data = game_to_token_ids(
                        game_df, skip_board_prob=self.skip_board_prob
                    )

                    # Random start (same as ChessIterableDataset)
                    valid_starts = [0] + [vd[1] + 1 for vd in value_data[:-1]]
                    start_idx = random.choice(valid_starts)

                    ids = ids[start_idx:]
                    wdl_data = [(m_idx - start_idx, best, wdl, valid)
                                for m_idx, best, wdl, valid in wdl_data
                                if m_idx >= start_idx]
                    value_data = [(wl_pos - start_idx, d_pos - start_idx, wl, d, valid)
                                  for wl_pos, d_pos, wl, d, valid in value_data
                                  if d_pos >= start_idx]
                    adjusted_boundaries = []
                    for (b_start, b_end) in block_boundaries:
                        adj_start = b_start - start_idx
                        adj_end = b_end - start_idx
                        if adj_end > 0 and adj_start < len(ids):
                            adjusted_boundaries.append((max(0, adj_start), min(len(ids), adj_end)))

                    # Truncate
                    if len(ids) > self.max_seq_len:
                        ids = ids[:self.max_seq_len]
                        wdl_data = [d for d in wdl_data if d[0] < self.max_seq_len]
                        value_data = [vd for vd in value_data if vd[1] < self.max_seq_len]
                        adjusted_boundaries = [(s, min(e, self.max_seq_len))
                                               for (s, e) in adjusted_boundaries
                                               if s < self.max_seq_len]

                    value_data = [vd for vd in value_data
                                  if vd[0] < self.max_seq_len and vd[1] < self.max_seq_len]
                    valid_move_indices = set()
                    for vd in value_data:
                        valid_move_indices.add(vd[0] - 1)
                    wdl_data = [d for d in wdl_data if d[0] in valid_move_indices]

                    yield self._build_pretrain_tensors(ids, wdl_data, value_data, adjusted_boundaries)

            except Exception as e:
                print(f"Error reading pretrain file {file_path}: {e}")
                continue

    def _variation_iter(self, files):
        """Iterate over variation data (thinking sequences)."""
        files = list(files)
        if self.shuffle_files:
            random.shuffle(files)

        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                indices = list(range(len(df)))
                if self.shuffle_games:
                    random.shuffle(indices)

                for idx in indices:
                    row = df.iloc[idx]

                    # Skip rows without variations
                    variations_raw = row.get("variations", "[]")
                    if isinstance(variations_raw, str):
                        variations = json.loads(variations_raw)
                    else:
                        variations = variations_raw
                    if not variations:
                        continue

                    try:
                        ids, thinking_move_data, final_move_data, value_data, block_boundaries, ranking, first_is_not_best, max_depth_end_var_positions, max_var_end_think_position = \
                            variation_to_token_ids(
                                row,
                                max_variations=self.max_variations,
                                max_depth=self.max_depth,
                                tau_base=self.tau_base,
                                tau_alpha=self.tau_alpha,
                            )
                    except Exception as e:
                        print(f"Error converting variation row {idx}: {e}")
                        continue

                    # Skip if sequence too long
                    if len(ids) > self.max_seq_len:
                        # Try with fewer variations/depth
                        for reduced_vars in range(self.max_variations - 1, 0, -1):
                            for reduced_depth in range(self.max_depth, 0, -1):
                                ids, thinking_move_data, final_move_data, value_data, block_boundaries, ranking, first_is_not_best, max_depth_end_var_positions, max_var_end_think_position = \
                                    variation_to_token_ids(
                                        row, max_variations=reduced_vars, max_depth=reduced_depth,
                                        tau_base=self.tau_base, tau_alpha=self.tau_alpha,
                                    )
                                if len(ids) <= self.max_seq_len:
                                    break
                            if len(ids) <= self.max_seq_len:
                                break

                    if len(ids) > self.max_seq_len:
                        continue

                    if final_move_data is None:
                        continue

                    yield self._build_variation_tensors(
                        ids, thinking_move_data, final_move_data, value_data, block_boundaries,
                        first_is_not_best=first_is_not_best,
                        max_depth_end_var_positions=max_depth_end_var_positions,
                        max_var_end_think_position=max_var_end_think_position,
                    )

            except Exception as e:
                print(f"Error reading variation file {file_path}: {e}")
                continue

    def _build_pretrain_tensors(self, ids, wdl_data, value_data, block_boundaries):
        """Build tensor dict for a pretrain sample (identical to ChessIterableDataset)."""
        seq_len = len(ids)
        IGNORE_INDEX = -100

        input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        board_target_ids = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        move_target_ids = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        thinking_move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        wl_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        d_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        wl_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
        d_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
        wdl_valid = torch.zeros((self.max_seq_len,), dtype=torch.bool)

        input_ids[:seq_len] = torch.tensor(ids, dtype=torch.long)

        # Shifted board targets (mapped to board sub-vocab indices)
        if seq_len > 1:
            for i in range(seq_len - 1):
                full_idx = input_ids[i + 1].item()
                board_target_ids[i] = full_idx_to_board_idx.get(full_idx, IGNORE_INDEX)

        # Move targets: override stm positions with generic_move + move sub-vocab target
        generic_move_board_idx = board_token_to_idx["generic_move"]
        for move_idx, best_move, wdl, is_valid_wdl in wdl_data:
            stm_pos = move_idx - 1
            if 0 <= stm_pos < self.max_seq_len:
                board_target_ids[stm_pos] = generic_move_board_idx
                move_target_ids[stm_pos] = full_idx_to_move_idx[token_to_idx[best_move]]
                move_mask[stm_pos] = True

        for wl_pos, d_pos, wl, d, is_valid in value_data:
            if wl_pos < self.max_seq_len:
                wl_positions[wl_pos] = True
                wl_targets[wl_pos] = wl
                wdl_valid[wl_pos] = is_valid
            if d_pos < self.max_seq_len:
                d_positions[d_pos] = True
                d_targets[d_pos] = d
                wdl_valid[d_pos] = is_valid
            stm_pos = wl_pos - 2
            if 0 <= stm_pos < self.max_seq_len:
                wl_targets[stm_pos] = wl
                d_targets[stm_pos] = d
                wdl_valid[stm_pos] = is_valid

        max_block_num = len(block_boundaries)
        block_id = torch.arange(self.max_seq_len, dtype=torch.long) + max_block_num
        for block_num, (b_start, b_end) in enumerate(block_boundaries):
            block_id[b_start:b_end] = block_num

        return {
            "input_ids": input_ids,
            "board_target_ids": board_target_ids,
            "move_target_ids": move_target_ids,
            "move_mask": move_mask,
            "thinking_move_mask": thinking_move_mask,
            "wl_positions": wl_positions,
            "d_positions": d_positions,
            "wl_targets": wl_targets,
            "d_targets": d_targets,
            "wdl_valid": wdl_valid,
            "block_id": block_id,
            "first_is_not_best": torch.tensor(False, dtype=torch.bool),
            "continue_var_mask": torch.zeros((self.max_seq_len,), dtype=torch.bool),
            "new_variation_mask": torch.zeros((self.max_seq_len,), dtype=torch.bool),
            "end_var_max_depth_mask": torch.zeros((self.max_seq_len,), dtype=torch.bool),
            "end_think_max_var_mask": torch.zeros((self.max_seq_len,), dtype=torch.bool),
        }

    def _build_variation_tensors(self, ids, thinking_move_data, final_move_data, value_data, block_boundaries, first_is_not_best=False, max_depth_end_var_positions=None, max_var_end_think_position=None):
        """Build tensor dict for a thinking variation sample."""
        seq_len = len(ids)
        IGNORE_INDEX = -100

        input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        board_target_ids = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        move_target_ids = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        thinking_move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        wl_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        d_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        wl_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
        d_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
        wdl_valid = torch.zeros((self.max_seq_len,), dtype=torch.bool)

        input_ids[:seq_len] = torch.tensor(ids, dtype=torch.long)

        # Shifted board targets (mapped to board sub-vocab indices)
        if seq_len > 1:
            for i in range(seq_len - 1):
                full_idx = input_ids[i + 1].item()
                board_target_ids[i] = full_idx_to_board_idx.get(full_idx, IGNORE_INDEX)

        # Thinking move targets: set move_target_ids + override board_target_ids
        start_think_id = token_to_idx["start_think"]
        end_var_id = token_to_idx["end_var"]
        generic_move_board_idx = board_token_to_idx["generic_move"]

        # Continuation masks for metrics only
        continue_var_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        new_variation_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)

        for predict_from_pos, move_token in thinking_move_data:
            if not (0 <= predict_from_pos < self.max_seq_len):
                continue
            # Move target in move sub-vocab
            move_target_ids[predict_from_pos] = full_idx_to_move_idx[token_to_idx[move_token]]
            thinking_move_mask[predict_from_pos] = True

            # Board target override based on input token type
            tok_id = ids[predict_from_pos]
            if tok_id == start_think_id:
                board_target_ids[predict_from_pos] = generic_move_board_idx
            elif tok_id == end_var_id:
                board_target_ids[predict_from_pos] = board_token_to_idx["new_variation"]
                new_variation_mask[predict_from_pos] = True
            else:
                # Board stm position - PV continuation
                board_target_ids[predict_from_pos] = board_token_to_idx["continue_var"]
                continue_var_mask[predict_from_pos] = True

        # Final move target (predicted from end_think via policy_head)
        if final_move_data is not None:
            end_think_pos, final_move_token = final_move_data
            if 0 <= end_think_pos < self.max_seq_len:
                move_target_ids[end_think_pos] = full_idx_to_move_idx[token_to_idx[final_move_token]]
                move_mask[end_think_pos] = True
                board_target_ids[end_think_pos] = generic_move_board_idx

        # Value targets
        for wl_pos, d_pos, wl, d, is_valid in value_data:
            if wl_pos < self.max_seq_len:
                wl_positions[wl_pos] = True
                wl_targets[wl_pos] = wl
                wdl_valid[wl_pos] = is_valid
            if d_pos < self.max_seq_len:
                d_positions[d_pos] = True
                d_targets[d_pos] = d
                wdl_valid[d_pos] = is_valid

        # Store WL/D targets at the final move position for convenience
        if final_move_data is not None:
            end_think_pos, _ = final_move_data
            if value_data:
                last_wl_pos, last_d_pos, last_wl, last_d, last_valid = value_data[-1]
                if 0 <= end_think_pos < self.max_seq_len:
                    wl_targets[end_think_pos] = last_wl
                    d_targets[end_think_pos] = last_d
                    wdl_valid[end_think_pos] = last_valid

        # Block IDs
        max_block_num = len(block_boundaries)
        block_id = torch.arange(self.max_seq_len, dtype=torch.long) + max_block_num
        for block_num, (b_start, b_end) in enumerate(block_boundaries):
            block_id[b_start:b_end] = block_num

        # Max depth / max variation masks
        end_var_max_depth_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        end_think_max_var_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)

        if max_depth_end_var_positions:
            for pos in max_depth_end_var_positions:
                if 0 <= pos < self.max_seq_len:
                    end_var_max_depth_mask[pos] = True

        if max_var_end_think_position is not None and 0 <= max_var_end_think_position < self.max_seq_len:
            end_think_max_var_mask[max_var_end_think_position] = True

        return {
            "input_ids": input_ids,
            "board_target_ids": board_target_ids,
            "move_target_ids": move_target_ids,
            "move_mask": move_mask,
            "thinking_move_mask": thinking_move_mask,
            "wl_positions": wl_positions,
            "d_positions": d_positions,
            "wl_targets": wl_targets,
            "d_targets": d_targets,
            "wdl_valid": wdl_valid,
            "block_id": block_id,
            "first_is_not_best": torch.tensor(first_is_not_best, dtype=torch.bool),
            "continue_var_mask": continue_var_mask,
            "new_variation_mask": new_variation_mask,
            "end_var_max_depth_mask": end_var_max_depth_mask,
            "end_think_max_var_mask": end_think_max_var_mask,
        }


def get_finetune_dataloader(
    pretrain_parquet_dir,
    variation_parquet_dir,
    batch_size=16,
    num_workers=0,
    max_seq_len=1024,
    variation_ratio=0.2,
    max_variations=3,
    max_depth=5,
    skip_board_prob=0.0,
    tau_base=0.3,
    tau_alpha=1.0,
):
    dataset = FinetuneIterableDataset(
        pretrain_parquet_dir=pretrain_parquet_dir,
        variation_parquet_dir=variation_parquet_dir,
        max_seq_len=max_seq_len,
        variation_ratio=variation_ratio,
        max_variations=max_variations,
        max_depth=max_depth,
        shuffle_files=True,
        shuffle_games=True,
        skip_board_prob=skip_board_prob,
        tau_base=tau_base,
        tau_alpha=tau_alpha,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def get_finetune_train_val_dataloaders(
    pretrain_parquet_dir,
    variation_parquet_dir,
    train_split=0.8,
    batch_size=16,
    num_workers=0,
    max_seq_len=1024,
    variation_ratio=0.2,
    max_variations=3,
    max_depth=5,
    skip_board_prob=0.0,
    tau_base=0.3,
    tau_alpha=1.0,
):
    """Create train and validation dataloaders by splitting files."""
    all_pretrain = sorted(glob.glob(os.path.join(pretrain_parquet_dir, "*.parquet")))
    all_variation = sorted(glob.glob(os.path.join(variation_parquet_dir, "*.parquet")))

    # Split by file count
    n_pt = int(len(all_pretrain) * train_split)
    n_vt = int(len(all_variation) * train_split)
    # Ensure at least 1 file in val if possible
    if n_pt == len(all_pretrain) and len(all_pretrain) > 1:
        n_pt -= 1
    if n_vt == len(all_variation) and len(all_variation) > 1:
        n_vt -= 1

    train_pretrain = all_pretrain[:n_pt]
    val_pretrain = all_pretrain[n_pt:]
    train_variation = all_variation[:n_vt]
    val_variation = all_variation[n_vt:]

    print(f"Train/val split: pretrain {len(train_pretrain)}/{len(val_pretrain)} files, "
          f"variation {len(train_variation)}/{len(val_variation)} files")

    common_kwargs = dict(
        max_seq_len=max_seq_len,
        variation_ratio=variation_ratio,
        max_variations=max_variations,
        max_depth=max_depth,
        skip_board_prob=skip_board_prob,
        tau_base=tau_base,
        tau_alpha=tau_alpha,
    )

    train_dataset = FinetuneIterableDataset(
        pretrain_files=train_pretrain,
        variation_files=train_variation,
        shuffle_files=True,
        shuffle_games=True,
        **common_kwargs,
    )
    val_dataset = FinetuneIterableDataset(
        pretrain_files=val_pretrain,
        variation_files=val_variation,
        shuffle_files=False,
        shuffle_games=False,
        **common_kwargs,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=max(1, num_workers // 2))

    return train_loader, val_loader

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
from src.models.vocab import token_to_idx
from finetune.data import variation_to_token_ids


class FinetuneIterableDataset(IterableDataset):
    def __init__(
        self,
        pretrain_parquet_dir,
        variation_parquet_dir,
        max_seq_len=1024,
        variation_ratio=0.2,
        max_variations=3,
        max_depth=5,
        shuffle_files=True,
        shuffle_games=True,
        skip_board_prob=0.0,
        tau_base=0.3,
        tau_alpha=1.0,
    ):
        self.pretrain_parquet_dir = pretrain_parquet_dir
        self.variation_parquet_dir = variation_parquet_dir
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

        self.pretrain_files = sorted(glob.glob(os.path.join(pretrain_parquet_dir, "*.parquet")))
        self.variation_files = sorted(glob.glob(os.path.join(variation_parquet_dir, "*.parquet")))

        if not self.pretrain_files:
            print(f"Warning: No pretrain parquet files found in {pretrain_parquet_dir}")
        if not self.variation_files:
            print(f"Warning: No variation parquet files found in {variation_parquet_dir}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Split files among workers
        if worker_info is None:
            pretrain_files = list(self.pretrain_files)
            variation_files = list(self.variation_files)
        else:
            nw = worker_info.num_workers
            wid = worker_info.id
            per_w = int(math.ceil(len(self.pretrain_files) / nw))
            pretrain_files = self.pretrain_files[wid * per_w: min((wid + 1) * per_w, len(self.pretrain_files))]
            per_w_v = int(math.ceil(len(self.variation_files) / nw))
            variation_files = self.variation_files[wid * per_w_v: min((wid + 1) * per_w_v, len(self.variation_files))]

        pretrain_iter = self._pretrain_iter(pretrain_files)
        variation_iter = self._variation_iter(variation_files)
        variation_epoch = 0

        while True:
            if random.random() < self.variation_ratio:
                sample = next(variation_iter, None)
                if sample is None:
                    # Restart variation iterator
                    variation_epoch += 1
                    variation_iter = self._variation_iter(variation_files)
                    sample = next(variation_iter, None)
                    if sample is None:
                        # No variation data at all, fall through to pretrain
                        sample = next(pretrain_iter, None)
                        if sample is None:
                            return
                sample["variation_epoch"] = torch.tensor(variation_epoch, dtype=torch.long)
            else:
                sample = next(pretrain_iter, None)
                if sample is None:
                    return
                sample["variation_epoch"] = torch.tensor(variation_epoch, dtype=torch.long)

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
                        ids, thinking_move_data, final_move_data, value_data, block_boundaries, ranking, first_is_not_best = \
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
                                ids, thinking_move_data, final_move_data, value_data, block_boundaries, ranking, first_is_not_best = \
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
                    )

            except Exception as e:
                print(f"Error reading variation file {file_path}: {e}")
                continue

    def _build_pretrain_tensors(self, ids, wdl_data, value_data, block_boundaries):
        """Build tensor dict for a pretrain sample (identical to ChessIterableDataset)."""
        seq_len = len(ids)

        input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        target_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        thinking_move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        wl_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        d_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        wl_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
        d_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
        wdl_valid = torch.zeros((self.max_seq_len,), dtype=torch.bool)

        input_ids[:seq_len] = torch.tensor(ids, dtype=torch.long)
        if seq_len > 1:
            target_ids[:seq_len - 1] = input_ids[1:seq_len]

        for move_idx, best_move, wdl, is_valid_wdl in wdl_data:
            stm_pos = move_idx - 1
            if 0 <= stm_pos < self.max_seq_len:
                target_ids[stm_pos] = token_to_idx[best_move]
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
            move_pos = wl_pos - 1
            for pos in [move_pos, wl_pos, d_pos]:
                if 0 <= pos < self.max_seq_len:
                    target_ids[pos] = self.pad_id

        max_block_num = len(block_boundaries)
        block_id = torch.arange(self.max_seq_len, dtype=torch.long) + max_block_num
        for block_num, (b_start, b_end) in enumerate(block_boundaries):
            block_id[b_start:b_end] = block_num

        # Pre-board mask: positions just before each board block
        # that should predict start_pos via the board_head
        pre_board_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        start_pos_id = token_to_idx["start_pos"]
        for (b_start, b_end) in block_boundaries:
            pre_board_pos = b_start - 1
            if 0 <= pre_board_pos < self.max_seq_len:
                target_ids[pre_board_pos] = start_pos_id
                pre_board_mask[pre_board_pos] = True

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "move_mask": move_mask,
            "thinking_move_mask": thinking_move_mask,
            "wl_positions": wl_positions,
            "d_positions": d_positions,
            "wl_targets": wl_targets,
            "d_targets": d_targets,
            "wdl_valid": wdl_valid,
            "block_id": block_id,
            "pre_board_mask": pre_board_mask,
            "first_is_not_best": torch.tensor(False, dtype=torch.bool),
        }

    def _build_variation_tensors(self, ids, thinking_move_data, final_move_data, value_data, block_boundaries, first_is_not_best=False):
        """Build tensor dict for a thinking variation sample."""
        seq_len = len(ids)

        input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        target_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        thinking_move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        wl_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        d_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        wl_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
        d_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
        wdl_valid = torch.zeros((self.max_seq_len,), dtype=torch.bool)

        input_ids[:seq_len] = torch.tensor(ids, dtype=torch.long)

        # Default next-token targets (shifted by 1) for board generation
        if seq_len > 1:
            target_ids[:seq_len - 1] = input_ids[1:seq_len]

        # Thinking move targets (variation root moves + PV continuation moves)
        for predict_from_pos, move_token in thinking_move_data:
            if 0 <= predict_from_pos < self.max_seq_len:
                target_ids[predict_from_pos] = token_to_idx[move_token]
                thinking_move_mask[predict_from_pos] = True

        # Final move target (predicted from end_think via policy_head)
        if final_move_data is not None:
            end_think_pos, final_move_token = final_move_data
            if 0 <= end_think_pos < self.max_seq_len:
                target_ids[end_think_pos] = token_to_idx[final_move_token]
                move_mask[end_think_pos] = True

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

        # Exclude move, wl, d positions from board generation targets
        # For thinking moves: the token after predict_from_pos is the move token
        for predict_from_pos, _ in thinking_move_data:
            move_token_pos = predict_from_pos + 1
            if 0 <= move_token_pos < self.max_seq_len:
                target_ids[move_token_pos] = self.pad_id

        # For final move: token after end_think
        if final_move_data is not None:
            end_think_pos, _ = final_move_data
            final_move_pos = end_think_pos + 1
            if 0 <= final_move_pos < self.max_seq_len:
                target_ids[final_move_pos] = self.pad_id

        # Exclude wl and d token positions from board generation
        for wl_pos, d_pos, _, _, _ in value_data:
            for pos in [wl_pos, d_pos]:
                if 0 <= pos < self.max_seq_len:
                    target_ids[pos] = self.pad_id

        # Store WL/D targets at the position that predicts WL (move token before wl_pos)
        # For the final move: stm_pos = wl_pos - 2 pattern doesn't apply
        # For variation values: wl is predicted from the move token before it
        # We store at the move_mask/thinking_move_mask positions for convenience
        if final_move_data is not None:
            end_think_pos, _ = final_move_data
            # The final wl/d are the last entry in value_data
            if value_data:
                last_wl_pos, last_d_pos, last_wl, last_d, last_valid = value_data[-1]
                if 0 <= end_think_pos < self.max_seq_len:
                    wl_targets[end_think_pos] = last_wl
                    d_targets[end_think_pos] = last_d
                    wdl_valid[end_think_pos] = last_valid

        # Block IDs: board groups get shared IDs, everything else gets unique orphan IDs
        max_block_num = len(block_boundaries)
        block_id = torch.arange(self.max_seq_len, dtype=torch.long) + max_block_num
        for block_num, (b_start, b_end) in enumerate(block_boundaries):
            block_id[b_start:b_end] = block_num

        # Pre-board mask: positions just before each board block
        # that should predict start_pos via the board_head
        pre_board_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        start_pos_id = token_to_idx["start_pos"]
        for (b_start, b_end) in block_boundaries:
            pre_board_pos = b_start - 1
            if 0 <= pre_board_pos < self.max_seq_len:
                target_ids[pre_board_pos] = start_pos_id
                pre_board_mask[pre_board_pos] = True

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "move_mask": move_mask,
            "thinking_move_mask": thinking_move_mask,
            "wl_positions": wl_positions,
            "d_positions": d_positions,
            "wl_targets": wl_targets,
            "d_targets": d_targets,
            "wdl_valid": wdl_valid,
            "block_id": block_id,
            "pre_board_mask": pre_board_mask,
            "first_is_not_best": torch.tensor(first_is_not_best, dtype=torch.bool),
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

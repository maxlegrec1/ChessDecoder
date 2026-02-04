import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
import math
import random
from src.dataloader.data import game_to_token_ids
from src.models.vocab import token_to_idx

class ChessIterableDataset(IterableDataset):
    def __init__(self, parquet_dir, max_seq_len=2048, shuffle_files=True, shuffle_games=True, skip_board_prob=0.0):
        self.parquet_dir = parquet_dir
        self.max_seq_len = max_seq_len
        self.shuffle_files = shuffle_files
        self.shuffle_games = shuffle_games
        self.skip_board_prob = skip_board_prob
        self.pad_id = token_to_idx["pad"]

        # Find all parquet files
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self.files:
            print(f"Warning: No parquet files found in {parquet_dir}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Determine which files this worker should read
        if worker_info is None:  # Single-process data loading
            files_to_read = list(self.files)
        else:
            # Split files among workers
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            files_to_read = self.files[iter_start:iter_end]

        if self.shuffle_files:
            random.shuffle(files_to_read)

        for file_path in files_to_read:
            try:
                df = pd.read_parquet(file_path)

                # Use groupby instead of repeated filtering
                grouped = df.groupby('game_id', sort=False)
                game_ids = list(grouped.groups.keys())

                if self.shuffle_games:
                    np.random.shuffle(game_ids)

                for game_id in game_ids:
                    game_df = grouped.get_group(game_id).sort_values('ply')

                    ids, wdl_data, block_boundaries, value_data = game_to_token_ids(
                        game_df, skip_board_prob=self.skip_board_prob
                    )

                    # Valid start indices: 0 (start of game) and d_pos + 1 for each position except last
                    valid_starts = [0] + [vd[1] + 1 for vd in value_data[:-1]]

                    start_idx = random.choice(valid_starts)

                    # Slice the sequence
                    ids = ids[start_idx:]

                    # Adjust wdl_data (move targets) - keep original_move_num unchanged
                    wdl_data = [(m_idx - start_idx, best, wdl, valid, orig_move_num)
                                for m_idx, best, wdl, valid, orig_move_num in wdl_data
                                if m_idx >= start_idx]

                    # Adjust value_data
                    value_data = [(wl_pos - start_idx, d_pos - start_idx, wl, d, valid)
                                  for wl_pos, d_pos, wl, d, valid in value_data
                                  if d_pos >= start_idx]  # d_pos is always after wl_pos

                    # Adjust block_boundaries
                    adjusted_boundaries = []
                    for (b_start, b_end) in block_boundaries:
                        adj_start = b_start - start_idx
                        adj_end = b_end - start_idx
                        if adj_end > 0 and adj_start < len(ids):
                            adjusted_boundaries.append((max(0, adj_start), min(len(ids), adj_end)))

                    # Truncate if necessary
                    if len(ids) > self.max_seq_len:
                        ids = ids[:self.max_seq_len]
                        wdl_data = [d for d in wdl_data if d[0] < self.max_seq_len]
                        value_data = [vd for vd in value_data if vd[1] < self.max_seq_len]
                        adjusted_boundaries = [(b_start, min(b_end, self.max_seq_len))
                                               for (b_start, b_end) in adjusted_boundaries
                                               if b_start < self.max_seq_len]

                    # Ensure positions are fully included: if wl_pos is in range but d_pos is cut,
                    # exclude that position entirely
                    value_data = [vd for vd in value_data
                                  if vd[0] < self.max_seq_len and vd[1] < self.max_seq_len]

                    # Also ensure move positions are valid: if move_idx is in range but
                    # its corresponding wl/d are cut, remove the move from wdl_data too
                    valid_move_indices = set()
                    for vd in value_data:
                        # wl_pos = move_idx + 1, so move_idx = wl_pos - 1
                        valid_move_indices.add(vd[0] - 1)
                    wdl_data = [d for d in wdl_data if d[0] in valid_move_indices]

                    seq_len = len(ids)

                    input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
                    target_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
                    move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
                    move_num = torch.full((self.max_seq_len,), -1, dtype=torch.long)  # Original game move number
                    wl_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
                    d_positions = torch.zeros((self.max_seq_len,), dtype=torch.bool)
                    wl_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
                    d_targets = torch.zeros((self.max_seq_len,), dtype=torch.float32)
                    wdl_valid = torch.zeros((self.max_seq_len,), dtype=torch.bool)

                    input_ids[:seq_len] = torch.tensor(ids, dtype=torch.long)

                    # Default next-token target (shifted by 1)
                    if seq_len > 1:
                        target_ids[:seq_len-1] = input_ids[1:seq_len]

                    # Process move targets and value positions
                    for move_idx, best_move, wdl, is_valid_wdl, orig_move_num in wdl_data:
                        # Move target: at stm position (move_idx - 1), predict the move
                        stm_pos = move_idx - 1
                        if 0 <= stm_pos < self.max_seq_len:
                            target_ids[stm_pos] = token_to_idx[best_move]
                            move_mask[stm_pos] = True
                            move_num[stm_pos] = orig_move_num

                    for wl_pos, d_pos, wl, d, is_valid in value_data:
                        # Mark WL and D placeholder positions
                        if wl_pos < self.max_seq_len:
                            wl_positions[wl_pos] = True
                            wl_targets[wl_pos] = wl
                            wdl_valid[wl_pos] = is_valid
                        if d_pos < self.max_seq_len:
                            d_positions[d_pos] = True
                            d_targets[d_pos] = d
                            wdl_valid[d_pos] = is_valid

                        # Also store targets at stm position for convenience
                        # (stm_pos = wl_pos - 2 since sequence is [...stm, move, wl, d...])
                        stm_pos = wl_pos - 2
                        if 0 <= stm_pos < self.max_seq_len:
                            wl_targets[stm_pos] = wl
                            d_targets[stm_pos] = d
                            wdl_valid[stm_pos] = is_valid

                        # At move, WL, and D positions: set target to pad so they're
                        # excluded from board CE loss
                        move_pos = wl_pos - 1  # move token position
                        for pos in [move_pos, wl_pos, d_pos]:
                            if 0 <= pos < self.max_seq_len:
                                target_ids[pos] = self.pad_id

                    # Build block_id tensor
                    max_block_num = len(adjusted_boundaries)
                    block_id = torch.arange(self.max_seq_len, dtype=torch.long) + max_block_num
                    for block_num, (b_start, b_end) in enumerate(adjusted_boundaries):
                        block_id[b_start:b_end] = block_num

                    # Pre-board mask: positions just before each board block
                    # that should predict start_pos via the board_head
                    pre_board_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
                    start_pos_id = token_to_idx["start_pos"]
                    for (b_start, b_end) in adjusted_boundaries:
                        pre_board_pos = b_start - 1
                        if 0 <= pre_board_pos < self.max_seq_len:
                            target_ids[pre_board_pos] = start_pos_id
                            pre_board_mask[pre_board_pos] = True

                    yield {
                        "input_ids": input_ids,
                        "target_ids": target_ids,
                        "move_mask": move_mask,
                        "move_num": move_num,  # Original game move number (0-indexed)
                        "wl_positions": wl_positions,
                        "d_positions": d_positions,
                        "wl_targets": wl_targets,
                        "d_targets": d_targets,
                        "wdl_valid": wdl_valid,
                        "block_id": block_id,
                        "pre_board_mask": pre_board_mask,
                    }
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

def get_dataloader(parquet_dir, batch_size=16, num_workers=0, max_seq_len=2048, skip_board_prob=0.0):
    dataset = ChessIterableDataset(
        parquet_dir,
        shuffle_files=True,
        shuffle_games=True,
        max_seq_len=max_seq_len,
        skip_board_prob=skip_board_prob
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

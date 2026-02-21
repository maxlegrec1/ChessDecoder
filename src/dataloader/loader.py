import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
import math
import random
from src.dataloader.data import game_to_token_ids
from src.models.vocab import (token_to_idx, full_idx_to_board_idx, full_idx_to_move_idx,
                              board_token_to_idx)

class ChessIterableDataset(IterableDataset):
    def __init__(self, parquet_dir, max_seq_len=2048, shuffle_files=True, shuffle_games=True, skip_board_prob=0.0, seed=42):
        self.parquet_dir = parquet_dir
        self.max_seq_len = max_seq_len
        self.shuffle_files = shuffle_files
        self.shuffle_games = shuffle_games
        self.skip_board_prob = skip_board_prob
        self.pad_id = token_to_idx["pad"]
        self.seed = seed
        self.epoch = 0  # set externally before each epoch for deterministic resumption

        # Find all parquet files
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self.files:
            print(f"Warning: No parquet files found in {parquet_dir}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        # Deterministic seeding: same seed+epoch+worker → same shuffle order
        epoch_seed = self.seed + self.epoch * 100003 + worker_id * 997
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2**32))

        # Determine which files this worker should read
        if worker_info is None:  # Single-process data loading
            files_to_read = list(self.files)
        else:
            # Split files among workers
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
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

                    # Adjust wdl_data (move targets)
                    wdl_data = [(m_idx - start_idx, best, wdl, valid)
                                for m_idx, best, wdl, valid in wdl_data
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
                    IGNORE_INDEX = -100

                    input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
                    board_target_ids = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
                    move_target_ids = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
                    move_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
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

                    # Process move targets: override stm positions with generic_move + move sub-vocab target
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

                    # Build block_id tensor
                    max_block_num = len(adjusted_boundaries)
                    block_id = torch.arange(self.max_seq_len, dtype=torch.long) + max_block_num
                    for block_num, (b_start, b_end) in enumerate(adjusted_boundaries):
                        block_id[b_start:b_end] = block_num

                    yield {
                        "input_ids": input_ids,
                        "board_target_ids": board_target_ids,
                        "move_target_ids": move_target_ids,
                        "move_mask": move_mask,
                        "wl_positions": wl_positions,
                        "d_positions": d_positions,
                        "wl_targets": wl_targets,
                        "d_targets": d_targets,
                        "wdl_valid": wdl_valid,
                        "block_id": block_id,
                    }
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

def get_dataloader(parquet_dir, batch_size=16, num_workers=0, max_seq_len=2048, skip_board_prob=0.0, seed=42):
    dataset = ChessIterableDataset(
        parquet_dir,
        shuffle_files=True,
        shuffle_games=True,
        max_seq_len=max_seq_len,
        skip_board_prob=skip_board_prob,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

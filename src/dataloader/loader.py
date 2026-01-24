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
                game_ids = df['game_id'].unique()
                
                if self.shuffle_games:
                    np.random.shuffle(game_ids)
                    
                for game_id in game_ids:
                    game_df = df[df['game_id'] == game_id].sort_values('ply')

                    ids, wdl_data, block_boundaries = game_to_token_ids(game_df, skip_board_prob=self.skip_board_prob)

                    # Randomly select a start position
                    # Valid start indices are 0 (start of game) and immediately after each move
                    # wdl_data contains (move_idx, ...), so next position starts at move_idx + 1
                    valid_starts = [0] + [d[0] + 1 for d in wdl_data[:-1]]

                    # We might want to avoid starting too close to the end if we want meaningful sequences
                    # But for now, uniform choice is fine
                    start_idx = random.choice(valid_starts)

                    # Slice the sequence
                    ids = ids[start_idx:]

                    # Adjust wdl_data
                    # Filter out moves before start_idx and shift indices
                    wdl_data = [(m_idx - start_idx, best, wdl, valid)
                                for m_idx, best, wdl, valid in wdl_data
                                if m_idx >= start_idx]

                    # Adjust block_boundaries for the sliced sequence
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
                        adjusted_boundaries = [(b_start, min(b_end, self.max_seq_len))
                                               for (b_start, b_end) in adjusted_boundaries
                                               if b_start < self.max_seq_len]

                    input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
                    target_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
                    wdl_targets = torch.zeros((self.max_seq_len, 3), dtype=torch.float32)
                    wdl_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)

                    seq_len = len(ids)
                    input_ids[:seq_len] = torch.tensor(ids, dtype=torch.long)

                    # Default target is offset by 1
                    if seq_len > 1:
                        target_ids[:seq_len-1] = input_ids[1:seq_len]

                    for move_idx, best_move, wdl, is_valid_wdl in wdl_data:
                        target_idx = move_idx - 1
                        if target_idx >= 0 and target_idx < self.max_seq_len:
                            target_ids[target_idx] = token_to_idx[best_move]
                            wdl_targets[target_idx] = torch.tensor(wdl, dtype=torch.float32)
                            if is_valid_wdl:
                                wdl_mask[target_idx] = True

                    # Build block_id tensor
                    # Default: each position gets a unique ID (orphan/padding)
                    max_block_num = len(adjusted_boundaries)
                    block_id = torch.arange(self.max_seq_len, dtype=torch.long) + max_block_num
                    # Assign block IDs to positions within blocks
                    for block_num, (b_start, b_end) in enumerate(adjusted_boundaries):
                        block_id[b_start:b_end] = block_num

                    yield {
                        "input_ids": input_ids,
                        "target_ids": target_ids,
                        "wdl_targets": wdl_targets,
                        "wdl_mask": wdl_mask,
                        "block_id": block_id,
                    }
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

def get_dataloader(parquet_dir, batch_size=16, num_workers=0, max_seq_len=2048, skip_board_prob=0.0):
    # shuffle=True is not supported for IterableDataset in DataLoader
    # We handle shuffling internally
    dataset = ChessIterableDataset(
        parquet_dir, 
        shuffle_files=True, 
        shuffle_games=True, 
        max_seq_len=max_seq_len,
        skip_board_prob=skip_board_prob
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

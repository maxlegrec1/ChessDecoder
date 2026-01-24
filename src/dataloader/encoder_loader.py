import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
import math
import random
from src.dataloader.data import fen_to_position_tokens
from src.models.vocab import token_to_idx, policy_index, policy_to_idx


class ChessEncoderDataset(IterableDataset):
    """
    Dataset for encoder model: extracts (FEN, best_move) pairs.
    
    Each sample is a single position tokenized, with the target being 
    the best move index in the policy vocabulary.
    """
    
    def __init__(
        self, 
        parquet_dir: str, 
        max_seq_len: int = 128, 
        shuffle_files: bool = True,
        shuffle_positions: bool = True
    ):
        self.parquet_dir = parquet_dir
        self.max_seq_len = max_seq_len
        self.shuffle_files = shuffle_files
        self.shuffle_positions = shuffle_positions
        self.pad_id = token_to_idx["pad"]
        
        # Find all parquet files
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self.files:
            print(f"Warning: No parquet files found in {parquet_dir}")
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Determine which files this worker should read
        if worker_info is None:
            files_to_read = list(self.files)
        else:
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
                
                # Get indices and optionally shuffle
                indices = df.index.tolist()
                if self.shuffle_positions:
                    random.shuffle(indices)
                
                for idx in indices:
                    row = df.loc[idx]
                    
                    fen = row['fen']
                    best_move = row['best_move']
                    
                    # Skip if best_move is not in policy vocabulary
                    if best_move not in policy_to_idx:
                        continue
                    
                    # Convert FEN to tokens
                    tokens = fen_to_position_tokens(fen)
                    token_ids = [token_to_idx[t] for t in tokens]
                    
                    # Skip if sequence is too long
                    seq_len = len(token_ids)
                    if seq_len > self.max_seq_len:
                        continue
                    
                    # Pad sequence
                    input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
                    input_ids[:seq_len] = torch.tensor(token_ids, dtype=torch.long)
                    
                    # Create attention mask (True for real tokens, False for padding)
                    attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
                    attention_mask[:seq_len] = True
                    
                    # Target is the best move index in policy vocabulary
                    target = policy_to_idx[best_move]
                    
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "target": torch.tensor(target, dtype=torch.long),
                        "seq_len": seq_len
                    }
                    
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue


def get_encoder_dataloader(
    parquet_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    max_seq_len: int = 128
) -> DataLoader:
    """Create dataloader for encoder training."""
    dataset = ChessEncoderDataset(
        parquet_dir,
        max_seq_len=max_seq_len,
        shuffle_files=True,
        shuffle_positions=True
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


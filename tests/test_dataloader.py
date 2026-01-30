import time
import torch
from src.dataloader.loader import get_dataloader
from src.models.vocab import idx_to_token
import pandas as pd

def test_dataloader():
    parquet_dir = 'parquets/'
    batch_size = 1
    loader = get_dataloader(parquet_dir, batch_size=batch_size)

    print(f"Testing dataloader with batch size {batch_size}...")

    # Check multiple samples to see random starts
    print("\nChecking first 5 samples for random starts:")
    for i, batch in enumerate(loader):
        if i >= 5:
            break

        input_ids = batch["input_ids"]
        first_tokens = [idx_to_token[id.item()] for id in input_ids[0, :100]]
        print(f"Sample {i+1} start: {first_tokens}")

        # Check if it starts with start_pos
        if first_tokens[0] == "start_pos":
            print("  -> Starts at beginning of game")
        else:
            print("  -> Starts in middle of game")

        # Check move_mask
        move_mask = batch["move_mask"]
        target_ids = batch["target_ids"]

        # Get indices where mask is True
        masked_indices = torch.nonzero(move_mask[0]).squeeze()

        if masked_indices.numel() > 0:
            indices_to_check = masked_indices[:5] if masked_indices.numel() > 5 else masked_indices

            print("  Checking move_mask targets:")
            for idx in indices_to_check:
                token_id = target_ids[0, idx].item()
                token = idx_to_token[token_id]
                is_move = len(token) in [4, 5] and token[0] in "abcdefgh" and token[1] in "12345678"
                print(f"    Index {idx}: {token} (Is move? {is_move})")
                if not is_move:
                    print(f"    WARNING: Token at masked index {idx} is NOT a move: {token}")

        # Check WL/D positions
        wl_positions = batch["wl_positions"]
        d_positions = batch["d_positions"]
        print(f"  WL positions: {wl_positions[0].sum().item()}, D positions: {d_positions[0].sum().item()}")

    # Speed test
    print("\nRunning speed test (loading 100 batches)...")
    num_batches = 100
    start_time = time.time()
    for i, _ in enumerate(loader):
        if i >= num_batches - 1:
            break
    end_time = time.time()
    avg_time = (end_time - start_time) / num_batches
    print(f"Average time per batch: {avg_time:.4f}s")
    print(f"Throughput: {batch_size / avg_time:.2f} games/s")

if __name__ == "__main__":
    test_dataloader()

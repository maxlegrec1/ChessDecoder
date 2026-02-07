import torch
from src.dataloader.loader import get_dataloader
from src.models.vocab import token_to_idx
from tqdm import tqdm
import numpy as np

def check_padding():
    parquet_dir = 'parquets/'
    batch_size = 32 # Use a larger batch size for faster statistics
    loader = get_dataloader(parquet_dir, batch_size=batch_size)
    
    pad_id = token_to_idx["pad"]
    max_seq_len = 2048
    
    print(f"Checking padding statistics with batch size {batch_size}...")
    
    padding_lengths = []
    truncated_count = 0
    total_sequences = 0
    
    # Check a fixed number of batches
    num_batches = 50
    
    for i, batch in tqdm(enumerate(loader), total=num_batches):
        if i >= num_batches:
            break
            
        input_ids = batch["input_ids"]
        # input_ids shape: (B, T)
        
        # Count padding tokens for each sequence in the batch
        # pad_id is usually 0 or similar, but let's compare explicitly
        is_pad = (input_ids == pad_id)
        num_pads = is_pad.sum(dim=1).tolist()
        
        padding_lengths.extend(num_pads)
        
        # Check for truncation (0 padding)
        # Note: It's possible a game is exactly 2048 tokens, but unlikely. 
        # Usually 0 padding means it hit the max_seq_len limit.
        for p in num_pads:
            if p == 0:
                truncated_count += 1
                
        total_sequences += len(num_pads)
        
    padding_lengths = np.array(padding_lengths)
    
    avg_padding = np.mean(padding_lengths)
    median_padding = np.median(padding_lengths)
    std_padding = np.std(padding_lengths)
    min_padding = np.min(padding_lengths)
    max_padding = np.max(padding_lengths)
    
    truncation_rate = (truncated_count / total_sequences) * 100
    
    print("\nPadding Statistics:")
    print(f"Total Sequences Analyzed: {total_sequences}")
    print(f"Max Sequence Length: {max_seq_len}")
    print(f"Average Padding Length: {avg_padding:.2f} tokens")
    print(f"Median Padding Length: {median_padding:.2f} tokens")
    print(f"Std Dev Padding: {std_padding:.2f}")
    print(f"Min Padding: {min_padding}")
    print(f"Max Padding: {max_padding}")
    print(f"\nTruncation Rate (0 padding): {truncation_rate:.2f}%")
    print(f"Percentage of sequences with padding: {100 - truncation_rate:.2f}%")
    
    # Histogram buckets
    print("\nPadding Distribution:")
    buckets = [0, 100, 500, 1000, 1500, 2000, 2048]
    hist, _ = np.histogram(padding_lengths, bins=buckets)
    
    for i in range(len(buckets)-1):
        count = hist[i]
        percentage = (count / total_sequences) * 100
        print(f"  {buckets[i]}-{buckets[i+1]}: {count} ({percentage:.2f}%)")
        
    # Specifically for very high padding (short sequences)
    high_pad = (padding_lengths > 1800).sum()
    print(f"  >1800 padding (seq len < 248): {high_pad} ({(high_pad/total_sequences)*100:.2f}%)")

if __name__ == "__main__":
    check_padding()

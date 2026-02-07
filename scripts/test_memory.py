import torch
import time
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size

def test_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ChessDecoder(vocab_size=vocab_size).to(device)
    
    batch_size = 16
    seq_len = 2048
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 1024**2
        
    start_time = time.time()
    with torch.no_grad():
        h = model(x)
    end_time = time.time()

    print(f"Forward pass took: {end_time - start_time:.4f}s")
    print(f"Output shape: Hidden {h.shape}")
    
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU Memory Usage: {peak_mem:.2f} MB")
        print(f"Memory increase: {peak_mem - start_mem:.2f} MB")
    else:
        print("Memory usage tracking only available for CUDA")

if __name__ == "__main__":
    test_memory()

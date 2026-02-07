import torch
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size

def show_size():
    model = ChessDecoder(vocab_size=vocab_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    print("\nSize Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,} parameters")
        
    # Breakdown of a single layer
    layer = model.decoder.layers[0]
    print("\nSingle Layer Breakdown:")
    for name, module in layer.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} parameters")

if __name__ == "__main__":
    show_size()

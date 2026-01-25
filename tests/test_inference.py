import torch
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size

def test_inference():
    # Initialize model
    model = ChessDecoder(vocab_size=vocab_size)
    model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_6.pt"))
    model.eval()
    
    # show number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


    # Test FEN (starting position)
    fen = "5rk1/6p1/7p/p3Np2/8/8/5PPP/2r1R1K1 b - - 1 26"
    
    print(f"Testing inference on FEN: {fen}")
    
    # Predict move with temperature 0.0 (argmax)
    move_greedy, value = model.predict_move_and_value(fen, temperature=0.0)
    print(f"Predicted move (temp=0.0): {move_greedy}")
    print(f"Value: {value}")

if __name__ == "__main__":
    test_inference()

import torch
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size

def test_inference():
    # Initialize model
    model = ChessDecoder(vocab_size=vocab_size)
    model.load_state_dict(torch.load("src/train/checkpoints/checkpoint_epoch_1.pt"))
    model.eval()
    
    # Test FEN (starting position)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print(f"Testing inference on FEN: {fen}")
    
    # Predict move with temperature 1.0
    move = model.predict_move(fen, temperature=1.0)
    print(f"Predicted move (temp=1.0): {move}")
    
    # Predict move with temperature 0.0 (argmax)
    move_greedy = model.predict_move(fen, temperature=0.0)
    print(f"Predicted move (temp=0.0): {move_greedy}")

if __name__ == "__main__":
    test_inference()

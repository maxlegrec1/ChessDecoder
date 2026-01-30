import torch
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size

def test_inference():
    # Initialize model
    model = ChessDecoder(vocab_size=vocab_size, num_layers=12, num_heads=16, embed_dim=1024, d_ff=1536, max_seq_len=256)
    model.load_state_dict(torch.load("checkpoints/run-1_20260128_231050/checkpoint_epoch_8.pt")["model_state_dict"])
    model.eval()
    
    # show number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


    # Test FEN (starting position)
    # fen = "5rk1/6p1/7p/p3Np2/8/8/5PPP/2r1R1K1 b - - 1 26"
    fen = "rnb1k1nr/pppp1ppp/4pq2/8/5P2/P4N2/P1PPP1PP/R1BQKB1R b KQkq - 2 4"
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN   R w KQkq - 0 1"
    fen = "rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq - 0 1"
    fen = "rnbqkbnr/pppppp1p/8/6p1/6P1/8/PPPPPP1P/RNBQKBNR w KQkq - 0 2"
    fen = "3k1q2/ppp3r1/3p4/3Pb3/2P1N3/7Q/PP4PP/4R1K1 b - - 1 28"
    fen = "r1bqr1k1/ppp2ppp/1n1p1b2/3P3Q/2P2P2/2NBB3/PP4PP/R4RK1 b - - 2 14"
    fen = "r1bqr1k1/ppp2p1p/1n1p1bp1/3P4/2P2P2/2NBBQ2/PP4PP/R4RK1 b - - 1 15"
    fen = "r1bqr1k1/ppp2p2/1n1p1bp1/3P3p/2P2P2/2NBBQ2/PP4PP/R4RK1 w - - 0 16"
    print(f"Testing inference on FEN: {fen}")
    
    # Predict move with temperature 0.0 (argmax)
    move_greedy, value = model.predict_move_and_value(fen, temperature=0.0)
    print(f"Predicted move (temp=0.0): {move_greedy}")
    print(f"Value: {value}")

if __name__ == "__main__":
    test_inference()

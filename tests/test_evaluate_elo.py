from src.eval.elo_eval import model_vs_stockfish
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size
import torch


model = ChessDecoder(vocab_size=vocab_size, embed_dim=1024, num_heads=16, num_layers=12, max_seq_len=256, d_ff=1536)
model.load_state_dict(torch.load("checkpoints/run-1_20260128_231050/checkpoint_epoch_9.pt")["model_state_dict"])
# model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_4.pt"))
model.eval()

model_vs_stockfish(model, model1_name="decoder", num_games=10, temperature=0.0, elo=2000)
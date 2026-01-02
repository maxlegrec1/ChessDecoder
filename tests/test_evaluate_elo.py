from src.eval.elo_eval import model_vs_stockfish
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size
import torch


model = ChessDecoder(vocab_size=vocab_size)
model.load_state_dict(torch.load("src/train/checkpoints/checkpoint_epoch_3.pt"))
# model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_4.pt"))
model.eval()

model_vs_stockfish(model, model1_name="decoder", num_games=10, temperature=0.0, elo=1500)
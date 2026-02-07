from src.eval.elo_eval import model_vs_stockfish
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size
import torch


def migrate_state_dict(state_dict, new_vocab_size):
    """Migrate old checkpoint: expand vocab dim from old to new size, clone policy_head to thinking_policy_head."""
    for key in ["tok_embedding.weight", "board_head.weight", "board_head.bias",
                "policy_head.weight", "policy_head.bias"]:
        t = state_dict[key]
        old_size = t.shape[0]
        if old_size < new_vocab_size:
            pad = torch.zeros(new_vocab_size - old_size, *t.shape[1:], dtype=t.dtype, device=t.device)
            state_dict[key] = torch.cat([t, pad], dim=0)

    state_dict["thinking_policy_head.weight"] = state_dict["policy_head.weight"].clone()
    state_dict["thinking_policy_head.bias"] = state_dict["policy_head.bias"].clone()
    return state_dict


model = ChessDecoder(vocab_size=vocab_size, embed_dim=1024, num_heads=16, num_layers=12, max_seq_len=256, d_ff=1536)
# model.load_state_dict(migrate_state_dict(
#     torch.load("checkpoints/run-1_20260128_231050/checkpoint_epoch_10.pt")["model_state_dict"],
#     vocab_size
# ))
model.load_state_dict(torch.load("checkpoints/run-1_20260206_233243/checkpoint_epoch_2.pt")["model_state_dict"])
model.eval()

model_vs_stockfish(model, model1_name="decoder", num_games=100, temperature=0.0, elo=1500)
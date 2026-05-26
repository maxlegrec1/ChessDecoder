"""Single train-step shape + grad smoke (CPU, tiny config)."""
import pandas as pd
import torch
import torch.nn as nn

from chessdecoder.dataloader.loader import game_to_arrays, IGNORE_INDEX
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import move_vocab_size, vocab_size


def _gdf(n_rows=20):
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * n_rows
    return pd.DataFrame([{"fen": f, "played_move": "e2e4", "best_move": "e2e4",
                          "orig_q": 0.1, "orig_d": 0.5, "ply": i, "game_id": "g"}
                         for i, f in enumerate(fens)])


def test_one_train_step():
    """B=2 games × N=4 positions through the encoder, backward through both
    losses, optimizer step — verifies the full path the real loop runs."""
    B, N = 2, 4
    s1 = game_to_arrays(_gdf(20), positions_per_game=N)
    s2 = game_to_arrays(_gdf(20), positions_per_game=N)
    batch = {k: torch.stack([s1[k], s2[k]]) for k in s1}                # [B, N, ...]

    m = ChessEncoder(vocab_size=vocab_size, embed_dim=32, num_heads=4,
                     num_layers=2, seq_len=68, d_ff=64)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    bid = batch["board_ids"].reshape(B * N, 68)
    out = m(bid)
    pol_logits = out["policy"].reshape(B, N, -1)
    wdl_logits = out["wdl"].reshape(B, N, -1)

    pol_tgt = batch["policy_tgt"]
    pol_val = batch["policy_valid"]
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    pol_loss = ce(pol_logits.reshape(-1, move_vocab_size),
                  torch.where(pol_val, pol_tgt,
                              torch.full_like(pol_tgt, IGNORE_INDEX)).reshape(-1))

    wdl_tgt = batch["wdl_tgt"]
    wdl_val = batch["wdl_valid"]
    logp = torch.log_softmax(wdl_logits.float(), -1)
    wdl_ce = -(wdl_tgt * logp).sum(-1)
    wdl_loss = (wdl_ce * wdl_val).sum() / (wdl_val.sum() + 1e-8)

    total = pol_loss + wdl_loss
    assert torch.isfinite(total)
    total.backward()
    opt.step()
    assert torch.isfinite(next(m.parameters())).all()

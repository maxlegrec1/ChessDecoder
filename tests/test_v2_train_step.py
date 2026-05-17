"""Phase C: one full V2 training step (CPU) with the reshaped sequence +
encoder-side joint WDL head (markdowns/12). Validates the joint forward/loss
wiring and that gradients flow back into the board encoder."""
import chess
import pandas as pd
import torch
import torch.nn as nn

from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.dataloader.loader_v2 import (
    game_to_v2_arrays, assemble_decoder_inputs)
from chessdecoder.models.v2.model_v2 import ChessDecoderV2, N_SQUARE_CLASSES

IGNORE_INDEX = -100


def _batch(P_max=8):
    b = chess.Board()
    rows = []
    for ply, uci in enumerate(["e2e4", "e7e5", "g1f3", "b8c6"]):
        rows.append({"game_id": 1, "ply": ply, "fen": b.fen(),
                     "played_move": uci, "best_move": uci,
                     "orig_q": 0.05 * (ply + 1), "orig_d": 0.3})
        b.push_uci(uci)
    a = game_to_v2_arrays(pd.DataFrame(rows), P_max)
    return {k: v.unsqueeze(0) for k, v in a.items()}      # batch size 1


def test_v2_training_step_and_encoder_gradient():
    torch.manual_seed(0)
    m = ChessDecoderV2(vocab_size=vocab_size, embed_dim=32, num_heads=4,
                        num_encoder_layers=1, num_decoder_layers=1,
                        num_latents=4, d_ff=64, num_fourier_freq=8)
    m.train()
    batch = _batch()
    B, P, _ = batch["board_ids"].shape
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    latents = m.encode_boards(batch["board_ids"].reshape(B * P, 68)).reshape(B, P, -1, 32)

    # --- encoder-side joint WDL (leak-free), soft cross-entropy ---
    wdl_logits = m.wdl_head(latents.reshape(B * P, -1, 32)).reshape(B, P, 3)
    vmask = batch["wdl_valid"] & batch["ply_mask"]
    logp = torch.log_softmax(wdl_logits.float(), -1)
    wdl_loss = (-(batch["wdl_tgt"] * logp).sum(-1) * vmask).sum() / (vmask.sum() + 1e-8)

    value_emb = m.embed_wdl(batch["wdl_tgt"].reshape(-1, 3)).reshape(B, P, 32)
    move_emb = m.tok_embedding(batch["move_full"])
    seq, pos = assemble_decoder_inputs(latents, move_emb, value_emb)
    h = m.decoder(seq)

    pmask = batch["policy_valid"] & batch["ply_mask"]
    pol = m.policy_head(h[:, pos["policy_pos"], :])             # at value slot
    policy_loss = ce(pol.reshape(-1, move_vocab_size),
                     torch.where(pmask, batch["policy_tgt"],
                                 torch.full_like(batch["policy_tgt"], IGNORE_INDEX)).reshape(-1))

    out = m.transition_head(latents.reshape(B * P, -1, 32),
                            move_emb.reshape(B * P, 32))
    trm = batch["trans_valid"].reshape(-1)
    sq_t = torch.where(trm.unsqueeze(1), batch["trans_sq"].reshape(B * P, 64),
                       torch.full_like(batch["trans_sq"].reshape(B * P, 64), IGNORE_INDEX))
    trans_loss = ce(out["square"].reshape(-1, N_SQUARE_CLASSES), sq_t.reshape(-1))

    total = 5.0 * policy_loss + trans_loss + wdl_loss
    assert torch.isfinite(total)
    total.backward()

    # gradient must reach the encoder (trained only via decoder policy +
    # encoder-side WDL + transition loss).
    g = m.board_encoder.latent_queries.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert m.wdl_head.query.grad is not None and m.wdl_head.query.grad.abs().sum() > 0
    assert m.tok_embedding.weight.grad is not None

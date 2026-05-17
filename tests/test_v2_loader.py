"""Phase B tests: V2 sequence builder (CPU-only, fast)."""
import random

import chess
import pandas as pd
import torch

from chessdecoder.models.vocab import vocab_size, token_to_idx, full_idx_to_move_idx
from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.dataloader.loader_v2 import (
    game_to_v2_arrays, assemble_decoder_inputs)
from chessdecoder.models.v2.model_v2 import (
    ChessDecoderV2, board_tokens_to_transition_targets)


def _game_df():
    b = chess.Board()
    rows = []
    for ply, uci in enumerate(["e2e4", "e7e5", "g1f3"]):
        rows.append({"game_id": 1, "ply": ply, "fen": b.fen(),
                     "played_move": uci, "best_move": uci,
                     "orig_q": 0.2, "orig_d": 0.5})       # WDL target source
        b.push_uci(uci)
    return pd.DataFrame(rows)


def test_arrays_shapes_and_validity():
    random.seed(1)                       # -> random start = 0 (full 3-ply game)
    a = game_to_v2_arrays(_game_df(), max_plies=8)
    assert a["board_ids"].shape == (8, 68)
    assert a["ply_mask"].tolist() == [True] * 3 + [False] * 5
    assert a["policy_valid"][:3].all() and not a["policy_valid"][3:].any()
    assert a["wdl_valid"][:3].all()
    # transition valid for plies 0,1 (ply 2 is the last -> no next board)
    assert a["trans_valid"].tolist() == [True, True, False] + [False] * 5
    assert a["policy_tgt"][0].item() == full_idx_to_move_idx[token_to_idx["e2e4"]]
    # exact mean WDL from orig_q=0.2, orig_d=0.5:
    # W=(1-0.5+0.2)/2=0.35, D=0.5, L=(1-0.5-0.2)/2=0.15
    assert torch.allclose(a["wdl_mean"][1],
                          torch.tensor([0.35, 0.5, 0.15]), atol=1e-5)
    # soft 2-D-simplex categorical: rows sum to 1 and are unbiased
    # (expectation under the bucket grid recovers the mean WDL)
    from chessdecoder.models.v2.value_buckets import CELL_WDL, N_CELLS
    assert a["wdl_tgt"].shape == (8, N_CELLS)
    assert abs(a["wdl_tgt"][:3].sum(-1).sub(1.0).abs().max().item()) < 1e-5
    recon = a["wdl_tgt"][:3] @ CELL_WDL
    assert torch.allclose(recon, a["wdl_mean"][:3], atol=2e-2)


def test_transition_target_is_next_ply_board():
    random.seed(1)                       # full game from ply 0
    df = _game_df()
    a = game_to_v2_arrays(df, max_plies=8)
    # ply 0 transition target must equal the class encoding of ply 1's board.
    b = chess.Board(); b.push_uci("e2e4")
    nxt = torch.tensor([[token_to_idx[t] for t in fen_to_position_tokens(b.fen())]])
    sq, stm, cas = board_tokens_to_transition_targets(nxt)
    assert torch.equal(a["trans_sq"][0], sq[0])
    assert a["trans_stm"][0].item() == stm[0].item()
    assert a["trans_cas"][0].item() == cas[0].item()


def test_random_start_crops_from_midgame():
    """seed 5 -> start=2 on the 3-ply game: only the last ply survives,
    so P=1, no transition target, and the surviving board is g1f3's."""
    random.seed(5)
    a = game_to_v2_arrays(_game_df(), max_plies=8)
    assert a["ply_mask"].tolist() == [True] + [False] * 7
    assert not a["trans_valid"].any()                 # single ply -> no next board
    assert a["policy_tgt"][0].item() == full_idx_to_move_idx[token_to_idx["g1f3"]]


def test_max_plies_crops_tail():
    """Long game, start 0: window is capped at max_plies (never exceeds ctx)."""
    random.seed(1)
    b = chess.Board()
    rows = []
    for ply, uci in enumerate(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]):
        rows.append({"game_id": 1, "ply": ply, "fen": b.fen(),
                     "played_move": uci, "best_move": uci,
                     "played_q": 0.1, "played_d": 0.2})
        b.push_uci(uci)
    a = game_to_v2_arrays(pd.DataFrame(rows), max_plies=4)
    assert a["ply_mask"].sum().item() == 4              # cropped to max_plies
    assert a["board_ids"].shape == (4, 68)


def test_assemble_layout_and_fourier_injection():
    random.seed(1)                       # start 0 so all 3 plies present
    m = ChessDecoderV2(vocab_size=vocab_size, embed_dim=16, num_heads=2,
                        num_encoder_layers=1, num_decoder_layers=1,
                        num_latents=4, d_ff=32, value_hidden_size=8,
                        num_fourier_freq=4)
    a = game_to_v2_arrays(_game_df(), max_plies=3)
    B, P, k = 1, 3, 4
    latents = m.encode_boards(a["board_ids"]).unsqueeze(0)        # [1,3,4,E]
    move_emb = m.tok_embedding(a["move_full"]).unsqueeze(0)        # [1,3,E]
    value_emb = m.embed_wdl(a["wdl_mean"]).unsqueeze(0)          # [1,3,E]

    seq, pos = assemble_decoder_inputs(latents, move_emb, value_emb)
    L = k + 2                                                     # [z|value|move]
    assert seq.shape == (1, P * L, 16)
    assert pos["ply_len"] == L
    # policy is read at the value slot (eval-aware move choice)
    assert pos["policy_pos"].tolist() == [k, L + k, 2 * L + k]
    assert pos["value_pos"].tolist() == [k, L + k, 2 * L + k]
    assert pos["move_pos"].tolist() == [k + 1, L + k + 1, 2 * L + k + 1]

    # latent slots, value slot (Fourier WDL), move slot are exact.
    assert torch.equal(seq[0, :k], latents[0, 0])
    assert torch.allclose(seq[0, pos["value_pos"][2]],
                          m.embed_wdl(a["wdl_mean"][2:3])[0].to(seq.dtype))
    assert torch.equal(seq[0, pos["move_pos"][1]], move_emb[0, 1])


def test_dataloader_smoke(tmp_path):
    _game_df().to_parquet(tmp_path / "g.parquet")
    from chessdecoder.dataloader.loader_v2 import get_v2_dataloader
    dl, ds = get_v2_dataloader(str(tmp_path), batch_size=2, max_plies=8)
    batch = next(iter(dl))
    assert batch["board_ids"].shape == (1, 8, 68)   # one game in the file
    assert batch["trans_valid"].shape == (1, 8)

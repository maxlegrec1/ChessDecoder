"""Phase D core: general mixed-sequence (segment-splice) assembler.

Reuses finetune/data.py's well-tested variation parser verbatim and only
changes the representation (68-tok board block -> one board Seg -> k latents;
wl/d placeholders -> continuous-value Segs). Validates the §12.0 splice.
"""
import json

import chess
import torch

from chessdecoder.models.vocab import vocab_size, token_to_idx
from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.finetune.data import variation_to_token_ids
from chessdecoder.dataloader.sequence_v2 import (
    Seg, build_mixed_sequence, variation_plan_from_token_ids)
from chessdecoder.models.v2.model_v2 import ChessDecoderV2

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _tiny():
    return ChessDecoderV2(vocab_size=vocab_size, embed_dim=16, num_heads=2,
                          num_encoder_layers=1, num_decoder_layers=1,
                          num_latents=4, d_ff=32, value_hidden_size=8,
                          num_fourier_freq=4)


def _variation_row():
    b = chess.Board(START)
    b.push_uci("e2e4")
    f1 = b.fen()
    b.push_uci("e7e5")
    f2 = b.fen()
    var = {"root_move": "e2e4", "visit_count": 50, "visit_fraction": 0.7,
           "prior": 0.5,
           "nodes": [{"fen": f1, "move": "e7e5", "wdl": [0.5, 0.3, 0.2],
                      "visit_count": 40},
                     {"fen": f2, "move": None, "wdl": [0.4, 0.4, 0.2],
                      "visit_count": 20}]}
    return {"fen": START, "variations": json.dumps([var]),
            "mcts_action": "e2e4", "win": 0.5, "draw": 0.3, "loss": 0.2,
            "played_move": "e2e4", "best_move": "e2e4"}


def test_segment_splice_roundtrip_and_shapes():
    out = variation_to_token_ids(_variation_row(), max_variations=1, max_depth=5)
    ids, _, _, value_data, block_boundaries = out[0], out[1], out[2], out[3], out[4]
    vpos = {}
    for wl_p, d_p, wl, d, _ in value_data:
        vpos[wl_p] = wl
        vpos[d_p] = d

    plan = variation_plan_from_token_ids(ids, block_boundaries, vpos)
    m = _tiny()
    k = m.num_latents
    seq, pos = build_mixed_sequence(m, plan, device="cpu")

    n_board = sum(1 for s in plan if s.kind == "board")
    n_other = sum(1 for s in plan if s.kind != "board")
    assert n_board == len(block_boundaries)
    assert seq.shape == (1, n_board * k + n_other, 16)
    assert pos["S"] == seq.shape[1]

    # spans are contiguous & ordered; board spans are exactly k wide.
    prev = 0
    for i, (a, b) in enumerate(pos["spans"]):
        assert a == prev
        assert b - a == (k if plan[i].kind == "board" else 1)
        assert pos["last"][i] == b - 1
        prev = b

    # token segs carry the right embedding; wl/d carry Fourier(value).
    for i, s in enumerate(plan):
        a, _ = pos["spans"][i]
        if s.kind == "token":
            assert torch.equal(
                seq[0, a], m.tok_embedding(torch.tensor([s.token_id]))[0])
        elif s.kind in ("wl", "d"):
            assert torch.allclose(
                seq[0, a], m.embed_value(torch.tensor([float(s.value)]))[0])
        else:
            assert i in pos["board_latents"]
            assert torch.equal(seq[0, a:a + k], pos["board_latents"][i][0])

    # the non-board token order must match the original flat id order.
    flat_non_board = [t for j, t in enumerate(ids)
                      if not any(a <= j < b for a, b in block_boundaries)]
    plan_tokens = [s.token_id if s.kind == "token" else token_to_idx[
        "wl_value" if s.kind == "wl" else "d_value"]
        for s in plan if s.kind != "board"]
    assert plan_tokens == flat_non_board


def test_decoder_runs_on_spliced_sequence():
    m = _tiny()
    plan = [Seg("board", board_ids=[token_to_idx[t]
                                    for t in fen_to_position_tokens(START)]),
            Seg("token", token_id=token_to_idx["start_think"]),
            Seg("token", token_id=token_to_idx["e2e4"]),
            Seg("wl", value=0.3), Seg("d", value=0.4)]
    seq, pos = build_mixed_sequence(m, plan, device="cpu")
    h = m.decoder(seq)
    assert h.shape == seq.shape
    # thinking_policy_head readable at start_think's position (predicts root move)
    st_idx = pos["last"][1]
    logits = m.thinking_policy_head(h[0, st_idx])
    assert logits.shape == (m.policy_head.out_features,)

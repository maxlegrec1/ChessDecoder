"""Phase D: V2 thinking-finetune loss path (CPU, synthetic variation row).

No variation parquets here (MCTS-generated/gitignored) so this exercises the
full loss wiring + scheduled sampling + encoder grad-flow on a hand-built
multi-variation, multi-depth row.
"""
import json

import chess
import torch

from chessdecoder.models.vocab import vocab_size
from chessdecoder.models.v2.model_v2 import ChessDecoderV2
from chessdecoder.finetune.loader_v2 import variation_to_v2_sample
from chessdecoder.finetune.train_v2 import compute_finetune_v2_loss

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _row():
    def pv(moves):
        b = chess.Board(START)
        nodes = []
        for i, mv in enumerate(moves):
            b.push_uci(mv)
            nodes.append({"fen": b.fen(),
                          "move": moves[i + 1] if i + 1 < len(moves) else None,
                          "wdl": [0.5, 0.3, 0.2], "visit_count": 30 - i})
        return nodes
    vars_ = [
        {"root_move": "e2e4", "visit_count": 80, "visit_fraction": 0.6,
         "prior": 0.5, "nodes": pv(["e2e4", "e7e5", "g1f3"])},
        {"root_move": "d2d4", "visit_count": 40, "visit_fraction": 0.3,
         "prior": 0.3, "nodes": pv(["d2d4", "d7d5"])},
    ]
    return {"fen": START, "variations": json.dumps(vars_),
            "mcts_action": "e2e4", "win": 0.5, "draw": 0.3, "loss": 0.2,
            "played_move": "e2e4", "best_move": "e2e4"}


def _tiny():
    return ChessDecoderV2(vocab_size=vocab_size, embed_dim=16, num_heads=2,
                          num_encoder_layers=1, num_decoder_layers=1,
                          num_latents=4, d_ff=32, value_hidden_size=8,
                          num_fourier_freq=4)


def test_finetune_v2_loss_and_gradients():
    torch.manual_seed(0)
    m = _tiny()
    m.train()
    plan, sup = variation_to_v2_sample(_row(), max_variations=3, max_depth=5)

    assert sum(1 for s in plan if s.kind == "board") >= 4   # root + PV boards
    assert len(sup["thinking"]) >= 3                         # root moves + PV moves
    assert sup["final"] is not None
    assert len(sup["transition_triples"]) >= 2

    losses = compute_finetune_v2_loss(m, plan, sup, ss_p=0.0, device="cpu")
    for k, v in losses.items():
        assert torch.isfinite(v), k
    assert losses["thinking"].item() > 0 and losses["transition"].item() > 0

    total = sum(losses.values())
    total.backward()
    g = m.board_encoder.latent_queries.grad
    assert g is not None and g.abs().sum() > 0                # encoder learns
    assert m.thinking_policy_head.weight.grad is not None


def test_finetune_v2_scheduled_sampling_runs():
    torch.manual_seed(0)
    m = _tiny()
    plan, sup = variation_to_v2_sample(_row(), max_variations=3, max_depth=5)
    losses = compute_finetune_v2_loss(m, plan, sup, ss_p=0.5, device="cpu")
    assert torch.isfinite(losses["transition"])
    losses["transition"].backward()                          # ss path differentiable

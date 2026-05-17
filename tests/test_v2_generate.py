"""Phase E/F-multi: engine-free autoregressive generator (CPU).

generate_v2 is the V2-native rollout primitive: it never calls a chess
library or the C++ engine — the next board comes from the transition head.
GRPO log-prob recompute re-points to the same policy-head positions; the
multi-ply eval (predict_move_n) reads the last move.
"""
import chess
import torch

from chessdecoder.models.vocab import vocab_size, token_to_idx
from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.v2.model_v2 import ChessDecoderV2

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _tiny():
    return ChessDecoderV2(vocab_size=vocab_size, embed_dim=16, num_heads=2,
                          num_encoder_layers=1, num_decoder_layers=1,
                          num_latents=4, d_ff=32, value_hidden_size=8,
                          num_fourier_freq=4)


def _bid(fen):
    return torch.tensor([[token_to_idx[t] for t in fen_to_position_tokens(fen)]],
                        dtype=torch.long)


def test_generate_v2_structure_and_logprobs():
    m = _tiny()
    plies = m.generate_v2(_bid(START), max_plies=6, temperature=0.0)
    assert len(plies) == 6
    for p in plies:
        assert 0 <= p["move_sub_id"] < m.policy_head.out_features
        assert p["move_logprob"] <= 1e-5                      # log-prob <= 0
        assert p["board_ids"].shape == (1, 68)
        # engine-free board is structurally valid in the model's tokenization
        assert p["board_ids"][0, 0].item() == token_to_idx["start_pos"]
        assert p["board_ids"][0, 65].item() == token_to_idx["end_pos"]


def test_generate_v2_greedy_is_deterministic():
    m = _tiny()
    a = m.generate_v2(_bid(START), 5, temperature=0.0)
    b = m.generate_v2(_bid(START), 5, temperature=0.0)
    assert [x["move_sub_id"] for x in a] == [x["move_sub_id"] for x in b]


def test_predict_move_n_returns_uci():
    m = _tiny()
    mv0 = m.predict_move_n(START, n_history_plies=0, temperature=0.0)
    mv3 = m.predict_move_n(START, n_history_plies=3, temperature=0.0)
    # valid UCI strings (legality not enforced — engine-free rollout explores)
    for mv in (mv0, mv3):
        assert isinstance(mv, str)
        chess.Move.from_uci(mv)

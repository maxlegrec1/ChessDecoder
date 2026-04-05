"""Model forward pass and prediction tests (requires GPU)."""

import chess
import pytest
import torch

from chessdecoder.models.vocab import (
    token_to_idx, vocab_size, board_vocab_size, move_vocab_size,
    POSITION_TOKEN_LENGTH,
)
from chessdecoder.dataloader.data import fen_to_position_tokens

pytestmark = pytest.mark.gpu


def test_forward_shape_causal(tiny_model):
    ids = torch.randint(0, vocab_size, (1, POSITION_TOKEN_LENGTH), device="cuda")
    h = tiny_model(ids, mask_type="causal")
    assert h.shape == (1, POSITION_TOKEN_LENGTH, 64)  # embed_dim=64


def test_forward_shape_prefix(tiny_model):
    S = POSITION_TOKEN_LENGTH
    ids = torch.randint(0, vocab_size, (1, S), device="cuda")
    block_id = torch.zeros(1, S, dtype=torch.long, device="cuda")
    h = tiny_model(ids, mask_type="prefix", block_id=block_id)
    assert h.shape == (1, S, 64)


def test_forward_batch(tiny_model):
    ids = torch.randint(0, vocab_size, (4, POSITION_TOKEN_LENGTH), device="cuda")
    h = tiny_model(ids, mask_type="causal")
    assert h.shape == (4, POSITION_TOKEN_LENGTH, 64)


def test_head_output_shapes(tiny_model):
    ids = torch.randint(0, vocab_size, (1, POSITION_TOKEN_LENGTH), device="cuda")
    h = tiny_model(ids, mask_type="causal")

    board_logits = tiny_model.board_head(h)
    assert board_logits.shape == (1, POSITION_TOKEN_LENGTH, board_vocab_size)

    policy_logits = tiny_model.policy_head(h)
    assert policy_logits.shape == (1, POSITION_TOKEN_LENGTH, move_vocab_size)

    think_logits = tiny_model.thinking_policy_head(h)
    assert think_logits.shape == (1, POSITION_TOKEN_LENGTH, move_vocab_size)


def test_causal_no_future_leakage(tiny_model):
    """Token at position 0 must not be affected by changes at position 1."""
    ids = torch.randint(0, vocab_size, (1, 10), device="cuda")

    h1 = tiny_model(ids, mask_type="causal")
    hidden_pos0_v1 = h1[0, 0].clone()

    # Modify token at position 1
    ids[0, 1] = (ids[0, 1] + 1) % vocab_size
    h2 = tiny_model(ids, mask_type="causal")
    hidden_pos0_v2 = h2[0, 0]

    assert torch.allclose(hidden_pos0_v1, hidden_pos0_v2, atol=1e-5)


def test_prefix_bidirectional_within_block(tiny_model):
    """Within a block, changing token 0 should affect token 67."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tokens = fen_to_position_tokens(fen)
    ids = torch.tensor([[token_to_idx[t] for t in tokens]], device="cuda")
    block_id = torch.zeros(1, POSITION_TOKEN_LENGTH, dtype=torch.long, device="cuda")

    h1 = tiny_model(ids, mask_type="prefix", block_id=block_id)
    hidden_last_v1 = h1[0, -1].clone()

    # Modify token at position 1 (a1 square)
    ids[0, 1] = token_to_idx["empty"]
    h2 = tiny_model(ids, mask_type="prefix", block_id=block_id)
    hidden_last_v2 = h2[0, -1]

    # With prefix masking, last position sees all positions in same block
    assert not torch.allclose(hidden_last_v1, hidden_last_v2, atol=1e-5)


def test_predict_move_legal(tiny_model, sample_fens):
    for fen in sample_fens:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        move_str = tiny_model.predict_move(fen, temperature=0.0, force_legal=True)
        move = chess.Move.from_uci(move_str)
        assert move in board.legal_moves, f"Illegal move {move_str} for {fen}"


def test_predict_move_and_value_ranges(tiny_model):
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move_str, wdl = tiny_model.predict_move_and_value(fen, temperature=0.0)
    assert 0.0 <= wdl["win"] <= 1.0
    assert 0.0 <= wdl["draw"] <= 1.0
    assert 0.0 <= wdl["loss"] <= 1.0


def test_bucket_centers(tiny_model):
    wl = tiny_model.wl_bucket_centers
    assert wl.shape == (100,)
    assert wl.min().item() >= -1.0
    assert wl.max().item() <= 1.0

    d = tiny_model.d_bucket_centers
    assert d.shape == (100,)
    assert d.min().item() >= 0.0
    assert d.max().item() <= 1.0


def test_five_heads_exist(tiny_model):
    for head_name in ["board_head", "policy_head", "thinking_policy_head",
                      "wl_head", "d_head"]:
        assert hasattr(tiny_model, head_name)

"""Phase D: engine-free rollout primitive (CPU).

The critical correctness gate: decode_transition must be the EXACT inverse of
board_tokens_to_transition_targets, so a perfectly-predicted transition
reconstructs the next board's 68 tokens bit-for-bit with no chess library.
"""
import chess
import torch
import torch.nn.functional as F

from chessdecoder.models.vocab import vocab_size, token_to_idx
from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.v2.model_v2 import (
    ChessDecoderV2, board_tokens_to_transition_targets,
    N_SQUARE_CLASSES, N_STM_CLASSES, N_CASTLING_CLASSES)


def _bid(fen):
    return torch.tensor([[token_to_idx[t] for t in fen_to_position_tokens(fen)]],
                        dtype=torch.long)


def _tiny():
    return ChessDecoderV2(vocab_size=vocab_size, embed_dim=16, num_heads=2,
                          num_encoder_layers=1, num_decoder_layers=1,
                          num_latents=4, d_ff=32, value_hidden_size=8,
                          num_fourier_freq=4)


def test_decode_transition_is_exact_inverse():
    m = _tiny()
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/8/8/4k3/8/8/4K3/8 b - - 0 1",                       # no castling
    ]
    for fen in fens:
        bid = _bid(fen)
        sq, stm, cas = board_tokens_to_transition_targets(bid)
        out = {                                                  # perfect head
            "square": F.one_hot(sq, N_SQUARE_CLASSES).float(),
            "stm": F.one_hot(stm, N_STM_CLASSES).float(),
            "castling": F.one_hot(cas, N_CASTLING_CLASSES).float(),
        }
        recon = m.decode_transition(out)
        assert torch.equal(recon, bid), fen                      # bit-for-bit


def test_rollout_next_shapes_and_revalid():
    m = _tiny()
    z = m.encode_boards(_bid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    move_emb = m.tok_embedding(torch.tensor([token_to_idx["e2e4"]]))
    board_ids, z_next = m.rollout_next(z, move_emb)
    assert board_ids.shape == (1, 68)
    assert z_next.shape == (1, m.num_latents, m.embed_dim)
    # decoded board ids must be valid vocab ids in the right structural slots
    assert board_ids[0, 0].item() == token_to_idx["start_pos"]
    assert board_ids[0, 65].item() == token_to_idx["end_pos"]
    assert board_ids[0, 67].item() in (token_to_idx["white_to_move"],
                                       token_to_idx["black_to_move"])


def test_scheduled_sample_latents_endpoints_and_mix():
    gt = torch.zeros(50, 4, 8)
    pred = torch.ones(50, 4, 8)
    assert torch.equal(ChessDecoderV2.scheduled_sample_latents(gt, pred, 0.0), gt)
    assert torch.equal(ChessDecoderV2.scheduled_sample_latents(gt, pred, 1.0), pred)
    torch.manual_seed(0)
    mix = ChessDecoderV2.scheduled_sample_latents(gt, pred, 0.5)
    per_board = mix.view(50, -1)[:, 0]
    assert set(per_board.unique().tolist()) <= {0.0, 1.0}        # each board all-GT or all-pred
    assert 0 < per_board.sum() < 50                              # genuinely mixed

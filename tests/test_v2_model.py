"""Phase A unit tests for the full V2 model (CPU-only, fast).

Gates from markdowns/11 §12.3:
  - latent shapes through encoder / decoder / predict_move
  - Fourier WL/D injection parity vs V1's FourierEncoder
  - transition head can represent a real board_{t+1} target exactly
  - the causal decoder uses the flash path (mask=None, is_causal=True)
"""
import chess
import torch

from chessdecoder.models.vocab import vocab_size, token_to_idx
from chessdecoder.models.v2.layers import FourierEncoder
from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.v2.model_v2 import (
    ChessDecoderV2, board_tokens_to_transition_targets,
    N_SQUARE_CLASSES, N_STM_CLASSES, N_CASTLING_CLASSES,
)

torch.manual_seed(0)
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _tiny():
    return ChessDecoderV2(
        vocab_size=vocab_size, embed_dim=32, num_heads=4,
        num_encoder_layers=1, num_decoder_layers=1, num_latents=4,
        d_ff=64, value_hidden_size=16, num_fourier_freq=8)


def _board_ids(fen):
    toks = fen_to_position_tokens(fen)
    return torch.tensor([[token_to_idx[t] for t in toks]], dtype=torch.long)


def test_latent_and_decoder_shapes():
    m = _tiny()
    boards = torch.randint(0, vocab_size, (5, 68))
    z = m.encode_boards(boards)
    assert z.shape == (5, 4, 32)                       # [N, k, E]
    h = m.decoder(z.reshape(1, 5 * 4, 32))
    assert h.shape == (1, 20, 32)                      # decoder preserves shape


def test_predict_move_legal():
    m = _tiny()
    mv = m.predict_move(START, temperature=0.0)
    assert chess.Move.from_uci(mv) in set(chess.Board(START).legal_moves)


def test_fourier_injection_matches_encoder():
    """embed_value must be exactly FourierEncoder(value) with shared weights
    (the WL/D injection wiring is the model's fourier_encoder, nothing else)."""
    m = _tiny()
    ref = FourierEncoder(embed_dim=32, num_frequencies=8)
    ref.load_state_dict(m.fourier_encoder.state_dict())
    x = torch.tensor([-0.73, 0.0, 0.41, 0.99])
    assert torch.equal(m.embed_value(x), ref(x))


def test_transition_target_extraction():
    """Targets must be the exact inverse of fen_to_position_tokens."""
    board_after = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    bid = _board_ids(board_after)
    sq, stm, cas = board_tokens_to_transition_targets(bid)
    assert sq.shape == (1, 64) and (sq >= 0).all() and (sq < N_SQUARE_CLASSES).all()
    assert stm.shape == (1,) and stm.item() == 1            # black to move
    assert cas.shape == (1,) and 0 <= cas.item() < N_CASTLING_CLASSES
    # e4 holds a white pawn after 1.e4 -> not the "empty" class (idx 0).
    assert sq[0, chess.E4].item() != 0


def test_transition_head_can_fit_one_transition_exactly():
    """Chess transitions are deterministic -> independent per-square heads
    can drive the loss on a single (z_t, move) -> board_{t+1} to ~0."""
    m = _tiny()
    z = m.encode_boards(_board_ids(START)).detach()         # [1,k,E]
    move_emb = m.tok_embedding(
        torch.tensor([token_to_idx["e2e4"]])).detach()       # [1,E]
    after = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    tsq, tstm, tcas = board_tokens_to_transition_targets(_board_ids(after))

    opt = torch.optim.Adam(m.transition_head.parameters(), lr=5e-2)
    for _ in range(400):
        opt.zero_grad()
        out = m.transition_head(z, move_emb)
        loss = (torch.nn.functional.cross_entropy(out["square"][0], tsq[0])
                + torch.nn.functional.cross_entropy(out["stm"], tstm)
                + torch.nn.functional.cross_entropy(out["castling"], tcas))
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = m.transition_head(z, move_emb)
    assert out["square"][0].argmax(-1).eq(tsq[0]).all()
    assert out["stm"].argmax(-1).eq(tstm).all()
    assert out["castling"].argmax(-1).eq(tcas).all()
    assert loss.item() < 0.05


def test_decoder_uses_causal_flash_path():
    """The flash win is structural: every decoder attn is is_causal and the
    forward never materializes a dense [B,S,S] mask (which kills flash)."""
    m = _tiny()
    for layer in m.decoder.layers:
        assert layer.attn.is_causal is True

    seen = []
    orig = type(m.decoder.layers[0]).forward

    def spy(self, x, mask=None, input_pos=None):
        seen.append(mask)
        return orig(self, x, mask=mask, input_pos=input_pos)

    try:
        type(m.decoder.layers[0]).forward = spy
        m.decoder(torch.randn(1, 12, 32))
    finally:
        type(m.decoder.layers[0]).forward = orig
    assert seen and all(msk is None for msk in seen)        # no dense mask

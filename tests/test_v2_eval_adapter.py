"""Phase F: V2 drops into the existing eval harness unchanged.

ChessDecoderV2.predict_move matches ChessEncoder.predict_move's
``(fen, temperature) -> uci`` contract, so PytorchModelAdapter (and thus
elo_eval's Stockfish games / tactics / CPL) work with no eval-side changes.
"""
import chess

from chessdecoder.models.vocab import vocab_size
from chessdecoder.models.v2.model_v2 import ChessDecoderV2
from chessdecoder.eval.engine import PytorchModelAdapter

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
]


def test_v2_predict_moves_via_adapter():
    m = ChessDecoderV2(vocab_size=vocab_size, embed_dim=32, num_heads=4,
                        num_encoder_layers=1, num_decoder_layers=1,
                        num_latents=4, d_ff=64, value_hidden_size=16,
                        num_fourier_freq=8)
    adapter = PytorchModelAdapter(
        lambda fen, temp: m.predict_move(fen, temperature=temp, force_legal=True))
    assert adapter.optimal_batch_size == 1
    results = adapter.predict_moves(FENS, temperature=0.0)
    assert len(results) == 2
    for fen, r in zip(FENS, results):
        assert chess.Move.from_uci(r.move) in set(chess.Board(fen).legal_moves)

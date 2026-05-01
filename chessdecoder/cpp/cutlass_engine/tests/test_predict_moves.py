"""End-to-end test: ThinkingEngine.predict_moves on real FENs.

Compares against the Python ChessDecoder.predict_move (no-thinking, argmax).
Both should agree token-by-token on a small model at temp=0.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size
import _cutlass_decoder_cpp as ce
from export_for_cutlass import export_for_cutlass


def test_predict_moves_argmax_matches_python():
    torch.manual_seed(0)
    mc = {
        "embed_dim": 64, "num_heads": 2, "num_layers": 2, "max_seq_len": 128,
        "d_ff": 128, "n_buckets": 100, "value_hidden_size": 64,
        "num_fourier_freq": 32, "wl_sigma": 0.4,
    }
    m = ChessDecoder(vocab_size=vocab_size, **mc).cuda().half()
    m.eval()
    B = 4

    # A few realistic FENs (start position + a few play positions).
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkb1r/pppppppp/5n2/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 2 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    ]
    assert len(fens) == B

    with tempfile.TemporaryDirectory(prefix="cutlass_pred_") as td:
        export_dir = Path(td)
        export_for_cutlass(m, {"model": mc}, export_dir)

        engine = ce.ThinkingEngine(
            backbone_pt="", weights_dir=str(export_dir / "weights"),
            vocab_json=str(export_dir / "vocab.json"),
            config_json=str(export_dir / "config.json"),
            batch_size=B,
        )
        engine.policy_temperature = 0.0

        results = engine.predict_moves(fens, 0.0)
        engine_moves = [r.move for r in results]
        print(f"engine: {engine_moves}")

        # Reference: Python predict_move at temp=0.
        py_moves = [m.predict_move(fen, temperature=0.0) for fen in fens]
        print(f"python: {py_moves}")

        # All B moves should match.
        mismatch = sum(1 for a, b in zip(engine_moves, py_moves) if a != b)
        assert mismatch == 0, f"{mismatch}/{B} mismatch: engine={engine_moves} vs python={py_moves}"

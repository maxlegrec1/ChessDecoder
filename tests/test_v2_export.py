"""Phase G step 1: V2 TorchScript export parity (CPU).

The two fixed-shape modules (BoardEncoder, TransitionHead) must trace to
TorchScript and reproduce eager outputs — the contract the C++ engine
consumes. The causal decoder export (KV-cache rebuild) is the remaining
Phase-G work and is intentionally out of scope here.
"""
import json
import os

import torch

from chessdecoder.models.vocab import vocab_size
from chessdecoder.models.v2.model_v2 import ChessDecoderV2
from chessdecoder.export.export_v2 import export_v2_modules


def test_v2_export_traces_and_parity(tmp_path):
    torch.manual_seed(0)
    m = ChessDecoderV2(vocab_size=vocab_size, embed_dim=16, num_heads=2,
                       num_encoder_layers=1, num_decoder_layers=1,
                       num_latents=4, d_ff=32, value_hidden_size=8,
                       num_fourier_freq=4)
    err = export_v2_modules(m, str(tmp_path), example_batch=2)
    assert err < 1e-4                                       # eager == scripted

    assert os.path.exists(tmp_path / "board_encoder.ts")
    assert os.path.exists(tmp_path / "transition_head.ts")
    cfg = json.load(open(tmp_path / "config.json"))
    assert cfg["num_latents"] == 4 and cfg["board_tokens"] == 68
    assert "NOT EXPORTED" in cfg["decoder"]                 # honest contract

    # reloaded modules run and match eager on a fresh batch
    enc = torch.jit.load(str(tmp_path / "board_encoder.ts"))
    tr = torch.jit.load(str(tmp_path / "transition_head.ts"))
    bids = torch.randint(0, vocab_size, (3, 68))
    with torch.no_grad():
        z = enc(bids)
        assert torch.allclose(z, m.board_encoder(bids), atol=1e-4)
        o = tr(z, torch.randn(3, 16))
        assert set(o.keys()) == {"square", "stm", "castling"}
        assert o["square"].shape == (3, 64, 13)

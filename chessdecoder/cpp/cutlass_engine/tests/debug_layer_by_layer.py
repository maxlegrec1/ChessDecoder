"""Compare engine intermediate state vs Python ChessDecoder layer by layer.

Sub-stage encoding for stop_after_layer:
  -1            : just embedding
   2*L          : after attention sub-block of layer L
   2*L + 1      : after MLP sub-block of layer L
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

torch.manual_seed(0)
mc = {
    "embed_dim": 64, "num_heads": 2, "num_layers": 2, "max_seq_len": 32,
    "d_ff": 128, "n_buckets": 100, "value_hidden_size": 64,
    "num_fourier_freq": 32, "wl_sigma": 0.4,
}
m = ChessDecoder(vocab_size=vocab_size, **mc).cuda().half()
m.eval()
E = mc["embed_dim"]
B = 2

with tempfile.TemporaryDirectory(prefix="cutlass_debug_") as td:
    export_dir = Path(td)
    export_for_cutlass(m, {"model": mc}, export_dir)

    engine = ce.ThinkingEngine(
        backbone_pt="", weights_dir=str(export_dir / "weights"),
        vocab_json=str(export_dir / "vocab.json"),
        config_json=str(export_dir / "config.json"),
        batch_size=B,
    )

    ids = torch.tensor([42, 137], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    active = torch.ones(B, dtype=torch.int32, device="cuda")

    def run_engine(stop):
        past_len = torch.zeros(B, dtype=torch.int32, device="cuda")
        h_eng = torch.zeros(B, E, dtype=torch.float16, device="cuda")
        res_eng = torch.zeros(B, E, dtype=torch.float16, device="cuda")
        engine.forward_decode_partial(ids.data_ptr(), pos.data_ptr(),
                                      active.data_ptr(), past_len.data_ptr(),
                                      stop,
                                      h_eng.data_ptr(), res_eng.data_ptr())
        return h_eng.float() + res_eng.float()  # full residual stream

    # Reference: build the python residual stream stage by stage.
    with torch.no_grad():
        h_py = m.tok_embedding(ids.long().unsqueeze(1))   # [B, 1, E]
        ipos = pos.long().unsqueeze(1)

        # Stage 0: just embedding.
        eng_full = run_engine(-1)
        ref = h_py.squeeze(1).float()
        err = (eng_full - ref).abs().max().item()
        print(f"[stage  -1: embedding]              err={err:.4e}")

        for L in range(mc["num_layers"]):
            # After attention sub-block of layer L.
            attn_in = m.layers[L].sa_norm(h_py)
            # MultiHeadAttention requires either kv_cache or y= for self-attn.
            # Pass y=attn_in to do plain self-attention.
            attn_out = m.layers[L].attn(attn_in, attn_in, mask=None, input_pos=ipos)
            h_after_attn = h_py + attn_out  # python residual

            eng_full = run_engine(2 * L)
            ref = h_after_attn.squeeze(1).float()
            err = (eng_full - ref).abs().max().item()
            print(f"[stage {2*L:>3}: layer {L} after attn]    err={err:.4e}")
            if err > 1e-2:
                print(f"   eng[0,:6]={eng_full[0,:6].tolist()}")
                print(f"   ref[0,:6]={ref[0,:6].tolist()}")

            # After MLP sub-block of layer L.
            mlp_in = m.layers[L].mlp_norm(h_after_attn)
            mlp_out = m.layers[L].mlp(mlp_in)
            h_py = h_after_attn + mlp_out

            eng_full = run_engine(2 * L + 1)
            ref = h_py.squeeze(1).float()
            err = (eng_full - ref).abs().max().item()
            print(f"[stage {2*L+1:>3}: layer {L} after mlp]     err={err:.4e}")
            if err > 1e-2:
                print(f"   eng[0,:6]={eng_full[0,:6].tolist()}")
                print(f"   ref[0,:6]={ref[0,:6].tolist()}")

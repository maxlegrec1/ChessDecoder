"""Export the V2 BoardForward TorchScript from a pretrain checkpoint.

Writes to exports/v2/{board_forward.ts, vocab.json, board_forward_config.json}
— the contract the cpp/v2/ engine consumes.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/export_v2_board_forward.py \\
        [/path/to/checkpoint.pt] [output_dir]
"""
import sys

import torch

from chessdecoder.models.vocab import vocab_size
from chessdecoder.models.v2.model_v2 import ChessDecoderV2
from chessdecoder.export.export_v2 import export_v2_board_forward

DEFAULT_CKPT = ("/home/maxime/ChessDecoder/checkpoints/v2_pretrain_muon1e3/"
                "v2-pretrain-muon1e3_20260518_075615/checkpoint_436000.pt")

CKPT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
OUT = sys.argv[2] if len(sys.argv) > 2 else "exports/v2"

print(f"Loading {CKPT}", flush=True)
ck = torch.load(CKPT, map_location="cuda", weights_only=False)
mc, dc = ck["config"]["model"], ck["config"]["data"]
model = ChessDecoderV2(
    vocab_size=vocab_size,
    embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
    num_encoder_layers=mc["num_encoder_layers"],
    num_decoder_layers=mc["num_decoder_layers"],
    num_latents=mc["num_latents"],
    decoder_max_seq_len=dc["max_plies"] * (mc["num_latents"] + 3),
    d_ff=mc["d_ff"], n_buckets=mc.get("n_buckets", 100),
    value_hidden_size=mc.get("value_hidden_size", 256),
    num_fourier_freq=mc.get("num_fourier_freq", 128),
    wl_sigma=mc.get("wl_sigma", 0.4),
).to("cuda").eval()
model.load_state_dict({k.replace("_orig_mod.", ""): v
                       for k, v in ck["model_state_dict"].items()})
print(f"Loaded model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params",
      flush=True)

max_err = export_v2_board_forward(model, OUT)
print(f"Exported to {OUT}/board_forward.ts + vocab.json + config.json",
      flush=True)
print(f"Eager-vs-scripted parity (max abs err): {max_err:.2e}", flush=True)

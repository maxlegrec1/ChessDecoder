"""Verify the RL adapter exposes a libtorch-compatible interface and that
rollout.py-style usage works end-to-end with the cutlass engine.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/src")
sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/python")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size
from rl_adapter import build_engine_for_rl
from export_for_cutlass import export_for_cutlass


CHECKPOINT_PATH = "/workspace/ChessDecoder/checkpoints/finetune-thinking-v1_20260422_135004/checkpoint_step_326000.pt"

SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "8/8/8/4k3/8/4K3/4Q3/8 w - - 0 1",
]


def main():
    print("=== RL adapter smoke test ===")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]
    model = ChessDecoder(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_layers=mc["num_layers"], max_seq_len=mc["max_seq_len"],
        d_ff=mc.get("d_ff"), n_buckets=mc.get("n_buckets", 100),
        value_hidden_size=mc.get("value_hidden_size", 256),
        num_fourier_freq=mc.get("num_fourier_freq", 128),
        wl_sigma=mc.get("wl_sigma", 0.4))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cuda().half().eval()

    # Mock GRPOConfig
    config = SimpleNamespace(
        think_temperature=0.0,
        policy_temperature=0.0,
        board_temperature=0.0,
        wl_temperature=0.0,
        d_temperature=0.0,
    )

    fens = SAMPLE_FENS
    B = len(fens)

    with tempfile.TemporaryDirectory(prefix="cutlass_rl_") as td:
        export_dir = Path(td)
        export_for_cutlass(model, cfg, export_dir)
        # rl_adapter expects backbone.pt to exist (or to be ignored). Touch it
        # so the path-based existence check (if any) doesn't fail.
        (export_dir / "backbone.pt").touch()

        engine = build_engine_for_rl(str(export_dir), B, config)
        results = engine.predict_moves(fens, config.think_temperature)

        assert len(results) == B, f"Expected {B} results, got {len(results)}"

        for i, r in enumerate(results):
            print(f"\nFEN[{i}]: move={r.move!r} tokens={len(r.token_ids)}")
            # Verify all fields exist and have the expected shape
            for fname in ("move_log_probs", "wl_entries", "d_entries",
                         "wl_bucket_indices", "d_bucket_indices",
                         "wl_log_probs", "d_log_probs"):
                v = getattr(r, fname)
                assert isinstance(v, list), f"{fname} should be list"
                if v:
                    assert isinstance(v[0], tuple), f"{fname}[0] should be tuple"
                    assert len(v[0]) == 2, f"{fname}[0] should be a 2-tuple"
                print(f"  {fname}: len={len(v)} sample={v[:2] if v else '[]'}")

        print("\nPASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

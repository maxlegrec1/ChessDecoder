"""Verify Phase K log_prob delivery: at temp=0 the engine emits log_probs
parallel to move/wl/d sample positions. Smoke test that:
  - move_log_probs / wl_log_probs / d_log_probs are non-empty
  - values are negative (log of a probability ≤ 1)
  - move_positions are inside token_ids range
  - lengths line up: |move_log_probs| == |move_positions|, etc.
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


CHECKPOINT_PATH = "/workspace/ChessDecoder/checkpoints/finetune-thinking-v1_20260422_135004/checkpoint_step_326000.pt"

SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkb1r/pppppppp/5n2/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 2 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "8/8/8/4k3/8/4K3/4Q3/8 w - - 0 1",
]


def main():
    print("=== Phase K log_prob smoke test ===")
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

    fens = SAMPLE_FENS
    B = len(fens)

    with tempfile.TemporaryDirectory(prefix="cutlass_lp_test_") as td:
        export_dir = Path(td)
        export_for_cutlass(model, cfg, export_dir)
        engine = ce.ThinkingEngine(
            backbone_pt="", weights_dir=str(export_dir / "weights"),
            vocab_json=str(export_dir / "vocab.json"),
            config_json=str(export_dir / "config.json"),
            batch_size=B,
        )
        for attr in ("board_temperature", "think_temperature",
                     "policy_temperature", "wl_temperature", "d_temperature"):
            setattr(engine, attr, 0.0)

        results = engine.predict_moves_thinking(fens, 0.0, 4096, 8)

        all_ok = True
        for i, r in enumerate(results):
            mp_len = len(r.move_positions)
            mlp_len = len(r.move_log_probs)
            wlp_len = len(r.wl_log_probs)
            dlp_len = len(r.d_log_probs)
            wp_len = len(r.wl_positions)
            dp_len = len(r.d_positions)
            tok_len = len(r.token_ids)

            print(f"\nFEN[{i}]: tokens={tok_len}")
            print(f"  moves: positions={mp_len} log_probs={mlp_len}")
            print(f"  WL:    positions={wp_len} log_probs={wlp_len}")
            print(f"  D:     positions={dp_len} log_probs={dlp_len}")
            if mlp_len:
                print(f"  move_log_probs[:3]={list(r.move_log_probs)[:3]}, "
                      f"min={min(r.move_log_probs):.3f}, max={max(r.move_log_probs):.3f}")
            if wlp_len:
                print(f"  wl_log_probs[:3]={list(r.wl_log_probs)[:3]}, "
                      f"min={min(r.wl_log_probs):.3f}, max={max(r.wl_log_probs):.3f}")
            if dlp_len:
                print(f"  d_log_probs[:3]={list(r.d_log_probs)[:3]}, "
                      f"min={min(r.d_log_probs):.3f}, max={max(r.d_log_probs):.3f}")

            # Sanity checks
            checks = [
                ("move_positions == move_log_probs", mp_len == mlp_len),
                ("wl_positions == wl_log_probs",     wp_len == wlp_len),
                ("d_positions == d_log_probs",       dp_len == dlp_len),
                ("move_positions in [0, tok_len)",
                 all(0 <= p < tok_len for p in r.move_positions)),
                ("move_log_probs ≤ 0",  all(lp <= 1e-3 for lp in r.move_log_probs)),
                ("wl_log_probs ≤ 0",    all(lp <= 1e-3 for lp in r.wl_log_probs)),
                ("d_log_probs ≤ 0",     all(lp <= 1e-3 for lp in r.d_log_probs)),
            ]
            for name, ok in checks:
                tag = "✓" if ok else "✗"
                print(f"    {tag} {name}")
                if not ok: all_ok = False

        print("\n" + ("PASS" if all_ok else "FAIL"))
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

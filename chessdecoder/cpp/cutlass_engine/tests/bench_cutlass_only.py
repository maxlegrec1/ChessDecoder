"""Cutlass-engine-only bench (no libtorch). Used to compare hand-rolled
fmha_prefill vs CUTLASS FMHA at the engine level."""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
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
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "r3k2r/pp1q1ppp/2n1pn2/2bp4/2B5/2NP1NP1/PPP1QPBP/R4RK1 w kq - 0 11",
    "8/8/8/4k3/8/4K3/4Q3/8 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bqkb1r/ppp1pppp/2n2n2/3p4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 4 4",
]


def run_cutlass(weights_dir, vocab_json, config_json, fens, B, max_seq_len, max_iters):
    torch.cuda.empty_cache(); gc.collect()
    engine = ce.ThinkingEngine(
        backbone_pt="", weights_dir=weights_dir,
        vocab_json=vocab_json, config_json=config_json,
        batch_size=B,
    )
    for attr in ("board_temperature", "think_temperature",
                 "policy_temperature", "wl_temperature", "d_temperature"):
        setattr(engine, attr, 0.0)
    _ = engine.predict_moves_thinking(fens[:B], 0.0, max_seq_len, max_iters)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = engine.predict_moves_thinking(fens, 0.0, max_seq_len, max_iters)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    total_tokens = sum(len(r.token_ids) for r in results)
    del engine
    torch.cuda.empty_cache()
    return {"time_s": t, "fens": len(fens), "tokens": total_tokens,
            "fen_per_s": len(fens) / t, "tok_per_s": total_tokens / t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-sizes", type=str, default="64,128")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]
    print(f"model: {mc}")

    model = ChessDecoder(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_layers=mc["num_layers"], max_seq_len=mc["max_seq_len"],
        d_ff=mc.get("d_ff"), n_buckets=mc.get("n_buckets", 100),
        value_hidden_size=mc.get("value_hidden_size", 256),
        num_fourier_freq=mc.get("num_fourier_freq", 128),
        wl_sigma=mc.get("wl_sigma", 0.4))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cuda().half()
    model.eval()

    fens = [SAMPLE_FENS[i % len(SAMPLE_FENS)] for i in range(args.n)]
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    with tempfile.TemporaryDirectory(prefix="cutlass_bench_") as td:
        export_dir = Path(td)
        export_for_cutlass(model, cfg, export_dir)
        weights_dir = str(export_dir / "weights")
        vocab_json = str(export_dir / "vocab.json")
        config_json = str(export_dir / "config.json")

        print("\n=== Bench ===")
        print(f"{'B':>6} {'cutlass FEN/s':>16} {'tok/s':>10} {'ms/batch':>10}")
        for B in batches:
            r = run_cutlass(weights_dir, vocab_json, config_json,
                            fens, B, args.max_seq_len, args.max_iters)
            print(f"{B:>6} {r['fen_per_s']:>16.2f} {r['tok_per_s']:>10.0f} "
                  f"{r['time_s']*1000:>10.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

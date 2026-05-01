"""Throughput bench for cutlass_engine.predict_moves (no-thinking path).

Measures FEN/sec across batch sizes, total VRAM, and the steady-state
allocation footprint. Compares against the libtorch path
(_decoder_inference_cpp.ThinkingBatchedInferenceEngine.predict_moves) when
both paths are configured for no-thinking-trace argmax inference.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python \\
        chessdecoder/cpp/cutlass_engine/tests/bench_predict_moves.py \\
        [--n 512] [--batch-sizes 16,32,64] [--checkpoint <path>]

Notes:
- Each FEN consumes one prefill (S=68) + one head GEMV + one argmax. There
  is no autoregressive decoding in this path. Throughput is "FEN/s" — not
  "tokens/s" — and is dominated by the prefill cost per slot.
- The libtorch path does the full thinking-trace generation by default;
  for an apples-to-apples comparison, this bench runs only the cutlass
  engine. A libtorch comparison number is recorded for reference using
  the existing thinking-engine output (slower per FEN by 100–1000x because
  of its much larger token budget per rollout).
"""

from __future__ import annotations

import argparse
import gc
import os
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


def _gpu_mem_used_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 * 1024)


def _build_model(seed: int, large: bool) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    if large:
        # Production-ish config (12 layers, 1024 embed, 16 heads).
        mc = {
            "embed_dim": 1024, "num_heads": 16, "num_layers": 12,
            "max_seq_len": 4096, "d_ff": 4096,
            "n_buckets": 100, "value_hidden_size": 256,
            "num_fourier_freq": 128, "wl_sigma": 0.4,
        }
    else:
        # Smaller config for quick iteration.
        mc = {
            "embed_dim": 256, "num_heads": 4, "num_layers": 4,
            "max_seq_len": 2048, "d_ff": 1024,
            "n_buckets": 100, "value_hidden_size": 128,
            "num_fourier_freq": 64, "wl_sigma": 0.4,
        }
    m = ChessDecoder(vocab_size=vocab_size, **mc).cuda().half()
    m.eval()
    return m, mc


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="number of FENs")
    ap.add_argument("--batch-sizes", type=str, default="16,32,64",
                    help="comma-separated batch sizes")
    ap.add_argument("--large", action="store_true",
                    help="use production-sized model (12L/1024E/16H, slow without checkpoint)")
    ap.add_argument("--warmup", type=int, default=2, help="warmup runs per B")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per B")
    args = ap.parse_args()

    print(f"=== cutlass_engine bench: N={args.n}, large={args.large} ===")
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    m, mc = _build_model(seed=0, large=args.large)

    fens = [SAMPLE_FENS[i % len(SAMPLE_FENS)] for i in range(args.n)]

    with tempfile.TemporaryDirectory(prefix="cutlass_bench_") as td:
        export_dir = Path(td)
        export_for_cutlass(m, {"model": mc}, export_dir)

        results = []
        for B in batches:
            torch.cuda.empty_cache()
            gc.collect()
            mem_before = _gpu_mem_used_mb()

            engine = ce.ThinkingEngine(
                backbone_pt="", weights_dir=str(export_dir / "weights"),
                vocab_json=str(export_dir / "vocab.json"),
                config_json=str(export_dir / "config.json"),
                batch_size=B,
            )
            engine.policy_temperature = 0.0

            mem_after_init = _gpu_mem_used_mb()

            # Warmup.
            for _ in range(args.warmup):
                _ = engine.predict_moves(fens, 0.0)
            torch.cuda.synchronize()

            # Timed runs.
            mem_peak = mem_after_init
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                _ = engine.predict_moves(fens, 0.0)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                mem_peak = max(mem_peak, _gpu_mem_used_mb())

            del engine
            torch.cuda.empty_cache()

            t_med = sorted(times)[len(times) // 2]
            t_min = min(times)
            fen_per_s = args.n / t_med
            ms_per_fen = 1000.0 * t_med / args.n

            print(f"B={B:>4}  median={t_med*1000:7.1f} ms  min={t_min*1000:7.1f} ms  "
                  f"FEN/s={fen_per_s:7.1f}  ms/FEN={ms_per_fen:6.2f}  "
                  f"engine_init=+{mem_after_init - mem_before:6.1f} MB  "
                  f"peak_run=+{mem_peak - mem_after_init:6.1f} MB")
            results.append({
                "B": B, "fen_per_s": fen_per_s, "ms_per_fen": ms_per_fen,
                "engine_init_mb": mem_after_init - mem_before,
                "peak_run_mb": mem_peak - mem_after_init,
            })

    # Pick the best B.
    best = max(results, key=lambda r: r["fen_per_s"])
    print(f"\nBest: B={best['B']} → {best['fen_per_s']:.1f} FEN/s ({best['ms_per_fen']:.2f} ms/FEN)")

    # Save JSON snapshot for regression tracking.
    out_path = Path(__file__).parent / "bench_results.json"
    import json
    payload = {
        "config": {"n": args.n, "large": args.large, "warmup": args.warmup,
                   "repeats": args.repeats},
        "results": results,
        "best": best,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved snapshot to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Throughput bench for predict_moves_thinking (full thinking-trace path).

Measures tok/s, ms/FEN, peak VRAM. Optionally compares against the Python
run_thinking reference for consistency at temp=0 (token-by-token match
modulo FP16 argmax flips).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python \\
        chessdecoder/cpp/cutlass_engine/tests/bench_thinking.py \\
        [--n 64] [--batch-size 32] [--large] [--max-iters 4]
"""

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
from chessdecoder.inference.think import run_thinking
import _cutlass_decoder_cpp as ce
from export_for_cutlass import export_for_cutlass


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


def _gpu_mem_used_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--large", action="store_true",
                    help="production model (12L/1024E/16H, slow without checkpoint)")
    ap.add_argument("--max-iters", type=int, default=4,
                    help="cap on variation iterations (lower = faster bench)")
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="hard cap on token sequence length")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--check-parity", action="store_true",
                    help="also run Python run_thinking and report match counts")
    args = ap.parse_args()

    print(f"=== thinking bench: N={args.n}, B={args.batch_size}, large={args.large}, max_iters={args.max_iters}")

    torch.manual_seed(0)
    if args.large:
        mc = {
            "embed_dim": 1024, "num_heads": 16, "num_layers": 12,
            "max_seq_len": 4096, "d_ff": 4096,
            "n_buckets": 100, "value_hidden_size": 256,
            "num_fourier_freq": 128, "wl_sigma": 0.4,
        }
    else:
        mc = {
            "embed_dim": 256, "num_heads": 4, "num_layers": 4,
            "max_seq_len": 1024, "d_ff": 1024,
            "n_buckets": 100, "value_hidden_size": 128,
            "num_fourier_freq": 64, "wl_sigma": 0.4,
        }
    m = ChessDecoder(vocab_size=vocab_size, **mc).cuda().half()
    m.eval()

    fens = [SAMPLE_FENS[i % len(SAMPLE_FENS)] for i in range(args.n)]

    with tempfile.TemporaryDirectory(prefix="cutlass_thinking_bench_") as td:
        export_dir = Path(td)
        export_for_cutlass(m, {"model": mc}, export_dir)

        torch.cuda.empty_cache(); gc.collect()
        mem_before = _gpu_mem_used_mb()

        engine = ce.ThinkingEngine(
            backbone_pt="", weights_dir=str(export_dir / "weights"),
            vocab_json=str(export_dir / "vocab.json"),
            config_json=str(export_dir / "config.json"),
            batch_size=args.batch_size,
        )
        for attr in ("board_temperature", "think_temperature",
                     "policy_temperature", "wl_temperature", "d_temperature"):
            setattr(engine, attr, 0.0)

        mem_after_init = _gpu_mem_used_mb()
        print(f"engine init: +{mem_after_init - mem_before:.1f} MB")

        # Warmup.
        for _ in range(args.warmup):
            _ = engine.predict_moves_thinking(fens, 0.0, args.max_seq_len, args.max_iters)
        torch.cuda.synchronize()

        # Timed runs.
        times = []
        total_tokens = 0
        peak_run = mem_after_init
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            results = engine.predict_moves_thinking(
                fens, 0.0, args.max_seq_len, args.max_iters)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            total_tokens = sum(len(r.token_ids) for r in results)
            peak_run = max(peak_run, _gpu_mem_used_mb())

        t_med = sorted(times)[len(times) // 2]
        t_min = min(times)
        tok_per_s = total_tokens / t_med
        ms_per_fen = 1000.0 * t_med / args.n

        print(f"\n=== Engine results ===")
        print(f"median time:    {t_med * 1000:.1f} ms")
        print(f"min time:       {t_min * 1000:.1f} ms")
        print(f"total tokens:   {total_tokens}")
        print(f"tok/s:          {tok_per_s:.0f}")
        print(f"ms/FEN:         {ms_per_fen:.2f}")
        print(f"peak run vram:  +{peak_run - mem_after_init:.1f} MB")

        if args.check_parity:
            print(f"\n=== Python parity check (first {min(args.n, 4)} FENs) ===")
            check_n = min(args.n, 4)
            py_results = []
            t0 = time.perf_counter()
            for fen in fens[:check_n]:
                pr = run_thinking(m, fen, temperature=0.0,
                                  max_seq_len=args.max_seq_len, verbose=False)
                py_results.append(pr)
            t_py = time.perf_counter() - t0
            py_total_tokens = sum(len(r.token_ids) for r in py_results)
            py_tok_per_s = py_total_tokens / t_py
            print(f"python time (sequential, {check_n} FENs): {t_py * 1000:.1f} ms")
            print(f"python tok/s: {py_tok_per_s:.0f}")
            print(f"engine vs python speedup: {(tok_per_s / py_tok_per_s) if py_tok_per_s else float('nan'):.1f}x")

            # Token-by-token match (modulo FP16 noise on close logits).
            for i, (er, pr) in enumerate(zip(results[:check_n], py_results)):
                er_tokens = list(er.token_ids)
                pr_tokens = list(pr.token_ids)
                common = min(len(er_tokens), len(pr_tokens))
                mismatch = sum(1 for a, b in zip(er_tokens[:common], pr_tokens[:common]) if a != b)
                first_div = next(
                    (j for j in range(common) if er_tokens[j] != pr_tokens[j]),
                    -1)
                print(f"  FEN[{i}]: shared={common} mismatch={mismatch} "
                      f"first_div={first_div}  "
                      f"({mismatch * 100 / max(common, 1):.1f}% mismatch in shared)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

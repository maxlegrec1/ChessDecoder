"""Head-to-head bench: cutlass_engine vs libtorch ThinkingBatchedInferenceEngine.

Measures FEN/sec on the real production checkpoint at varying batch sizes,
and a CB test (small B with large N).

Plots / prints two curves:
  curve A: cutlass_engine FEN/s vs B
  curve B: libtorch FEN/s vs B
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size
from chessdecoder.rl.rollout import export_model
import _cutlass_decoder_cpp as ce
import _decoder_inference_cpp as cpp_libtorch
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
    # Warmup.
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


def run_libtorch(backbone_pt, weights_dir, vocab_json, config_json, fens, B):
    torch.cuda.empty_cache(); gc.collect()
    engine = cpp_libtorch.ThinkingBatchedInferenceEngine(
        backbone_pt, weights_dir, vocab_json, config_json, B)
    for attr in ("board_temperature", "think_temperature",
                 "policy_temperature", "wl_temperature", "d_temperature"):
        setattr(engine, attr, 0.0)
    # Warmup.
    _ = engine.predict_moves(fens[:B], 0.0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    results = engine.predict_moves(fens, 0.0)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    total_tokens = sum(len(r.token_ids) for r in results)
    del engine
    torch.cuda.empty_cache()
    return {"time_s": t, "fens": len(fens), "tokens": total_tokens,
            "fen_per_s": len(fens) / t, "tok_per_s": total_tokens / t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch-sizes", type=str, default="16,32,64,128")
    ap.add_argument("--max-iters", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--cb-test", action="store_true",
                    help="run CB-vs-no-CB test (small B with large N)")
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

    with tempfile.TemporaryDirectory(prefix="cutlass_vs_libtorch_") as td:
        export_dir = Path(td)
        # cutlass-format export.
        export_for_cutlass(model, cfg, export_dir)
        # libtorch-format export (TorchScript .pt).  Need a separate dir.
        export_dir_lt = Path(tempfile.mkdtemp(prefix="lt_export_"))
        export_model(model, cfg, export_dir_lt)

        weights_dir_cu = str(export_dir / "weights")
        vocab_json_cu = str(export_dir / "vocab.json")
        config_json_cu = str(export_dir / "config.json")
        backbone_pt_lt = str(export_dir_lt / "backbone.pt")
        weights_dir_lt = str(export_dir_lt / "weights")
        vocab_json_lt = str(export_dir_lt / "vocab.json")
        config_json_lt = str(export_dir_lt / "config.json")

        results_cu = []
        results_lt = []
        for B in batches:
            print(f"\n--- B={B} ---")
            r_cu = run_cutlass(weights_dir_cu, vocab_json_cu, config_json_cu,
                               fens, B, args.max_seq_len, args.max_iters)
            print(f"  cutlass : {r_cu['fen_per_s']:.2f} FEN/s   "
                  f"({r_cu['tok_per_s']:.0f} tok/s, {r_cu['time_s']*1000:.0f} ms)")
            r_lt = run_libtorch(backbone_pt_lt, weights_dir_lt, vocab_json_lt,
                                config_json_lt, fens, B)
            print(f"  libtorch: {r_lt['fen_per_s']:.2f} FEN/s   "
                  f"({r_lt['tok_per_s']:.0f} tok/s, {r_lt['time_s']*1000:.0f} ms)")
            print(f"  speedup : {r_cu['fen_per_s'] / r_lt['fen_per_s']:.2f}x")
            results_cu.append({"B": B, **r_cu})
            results_lt.append({"B": B, **r_lt})

        cb_results = None
        if args.cb_test:
            print("\n=== CB test ===")
            print(f"  Compares: B=N (no CB) vs B<N (mid-flight refill)")
            N_cb = max(batches) * 2  # e.g., if max B=128, N=256
            fens_cb = [SAMPLE_FENS[i % len(SAMPLE_FENS)] for i in range(N_cb)]
            print(f"\n  N={N_cb}, B={max(batches)}, no CB needed (N == 2B):")
            r_no_cb = run_cutlass(weights_dir_cu, vocab_json_cu, config_json_cu,
                                  fens_cb, max(batches), args.max_seq_len, args.max_iters)
            print(f"    cutlass: {r_no_cb['fen_per_s']:.2f} FEN/s")

            print(f"\n  N={N_cb}, B={max(batches)//4} (CB refills {N_cb//(max(batches)//4)}x):")
            r_cb = run_cutlass(weights_dir_cu, vocab_json_cu, config_json_cu,
                               fens_cb, max(batches)//4, args.max_seq_len, args.max_iters)
            print(f"    cutlass: {r_cb['fen_per_s']:.2f} FEN/s")
            cb_results = {"no_cb": r_no_cb, "cb": r_cb}

    print("\n=== Summary ===")
    print(f"{'B':>6} {'cutlass FEN/s':>16} {'libtorch FEN/s':>16} {'speedup':>10}")
    for rc, rl in zip(results_cu, results_lt):
        print(f"{rc['B']:>6} {rc['fen_per_s']:>16.2f} {rl['fen_per_s']:>16.2f} "
              f"{rc['fen_per_s'] / rl['fen_per_s']:>9.2f}x")

    out = Path(__file__).parent / "bench_cutlass_vs_libtorch.json"
    out.write_text(json.dumps({
        "config": vars(args),
        "cutlass": results_cu,
        "libtorch": results_lt,
        "cb_test": cb_results,
    }, indent=2))
    print(f"\nSnapshot saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Head-to-head bench mimicking the RL rollout loop.

  N FENs, group_size G → effective N*G rollouts processed in chunks of
  inference_batch_size B. Wall-clock includes engine construction +
  exports — same as the RL loop pays per outer step.

Run with no args for the standard 320 FENs × 10 groups × B=128.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/src")

from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size
from chessdecoder.rl.rollout import _build_engine, export_model
from chessdecoder.rl.config import GRPOConfig

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


def _bench_run(export_dir: str, all_fens: list[str], B: int, label: str,
               think_t: float = 0.0):
    """Run the chunked rollout loop just like rollout.generate_rollouts does."""
    config = GRPOConfig(
        inference_batch_size=B,
        think_temperature=think_t,
        policy_temperature=think_t,
        board_temperature=think_t,
        wl_temperature=0.0,
        d_temperature=0.0,
    )
    engine = _build_engine(export_dir, config, B)

    t0 = time.perf_counter()
    total_results = []
    for start in range(0, len(all_fens), B):
        chunk = all_fens[start:start + B]
        if len(chunk) < B:
            # Pad to B (RL chunking pads or accepts uneven last chunk).
            chunk = chunk + [chunk[-1]] * (B - len(chunk))
        raw = engine.predict_moves(chunk, think_t)
        total_results.extend(raw[:len(all_fens) - start])
        print(f"  [{label}] {min(start + B, len(all_fens))}/{len(all_fens)} done "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    total_tokens = sum(len(r.token_ids) for r in total_results[:len(all_fens)])
    return {
        "label": label,
        "n_fens": len(all_fens),
        "elapsed_s": elapsed,
        "fen_per_s": len(all_fens) / elapsed,
        "total_tokens": total_tokens,
        "tok_per_s": total_tokens / elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320, help="distinct FENs")
    ap.add_argument("--g", type=int, default=10, help="group size (rollouts per FEN)")
    ap.add_argument("--b", type=int, default=128, help="inference batch size")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]
    print(f"Model: {mc}")

    model = ChessDecoder(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_layers=mc["num_layers"], max_seq_len=mc["max_seq_len"],
        d_ff=mc.get("d_ff"), n_buckets=mc.get("n_buckets", 100),
        value_hidden_size=mc.get("value_hidden_size", 256),
        num_fourier_freq=mc.get("num_fourier_freq", 128),
        wl_sigma=mc.get("wl_sigma", 0.4))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cuda().half().eval()

    fens = [SAMPLE_FENS[i % len(SAMPLE_FENS)] for i in range(args.n)]
    all_fens = [fen for fen in fens for _ in range(args.g)]
    effective_N = len(all_fens)
    print(f"\nBench config: distinct={args.n}, group={args.g}, "
          f"effective={effective_N}, batch_size={args.b}, temp={args.temp}")

    with tempfile.TemporaryDirectory(prefix="rl_bench_") as td:
        export_dir = Path(td)
        export_model(model, cfg, export_dir)
        # Also export weights for cutlass (different layout)
        from export_for_cutlass import export_for_cutlass
        export_for_cutlass(model, cfg, export_dir)

        backend = os.environ.get("RL_ENGINE", "libtorch").lower()
        label = f"{backend.upper()}{' + CUTLASS_FMHA' if os.environ.get('USE_CUTLASS_FMHA') else ''}"
        result = _bench_run(str(export_dir), all_fens, args.b, label,
                            think_t=args.temp)

    print(f"\n=== {result['label']} ===")
    print(f"  N             : {result['n_fens']}")
    print(f"  Wall clock    : {result['elapsed_s']:.1f}s "
          f"({result['elapsed_s']/60:.1f} min)")
    print(f"  FEN/s         : {result['fen_per_s']:.2f}")
    print(f"  total tokens  : {result['total_tokens']}")
    print(f"  tok/s         : {result['tok_per_s']:.0f}")


if __name__ == "__main__":
    sys.exit(main() or 0)

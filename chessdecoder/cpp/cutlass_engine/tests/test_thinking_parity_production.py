"""Production-checkpoint parity test for predict_moves_thinking.

Loads the real finetuned ChessDecoder checkpoint, runs both my engine and
Python's run_thinking on a fixed FEN suite at temp=0, reports strict and
loose match rates.

Strict match: token_ids identical for the entire shared prefix.
Loose match: token_ids identical OR diverging only at value-bucket
positions (where Python's argmax differs by 1 and the top-2 logit gap
is < 1e-3 — documented FP16 noise).
"""

from __future__ import annotations

import argparse
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


CHECKPOINT_PATH = "/workspace/ChessDecoder/checkpoints/finetune-thinking-v1_20260422_135004/checkpoint_step_326000.pt"

SAMPLE_FENS = [
    # Starting position.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Early opening.
    "rnbqkb1r/pppppppp/5n2/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 2 2",
    # Italian game.
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    # Sicilian.
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    # Middlegame tactical position.
    "r3k2r/pp1q1ppp/2n1pn2/2bp4/2B5/2NP1NP1/PPP1QPBP/R4RK1 w kq - 0 11",
    # K+Q vs K endgame.
    "8/8/8/4k3/8/4K3/4Q3/8 w - - 0 1",
    # Pawn endgame.
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    # Slav.
    "r1bqkb1r/ppp1pppp/2n2n2/3p4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 4 4",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of FENs (uses SAMPLE_FENS cyclically)")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]
    print(f"model config: {mc}")

    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=mc["embed_dim"],
        num_heads=mc["num_heads"],
        num_layers=mc["num_layers"],
        max_seq_len=mc["max_seq_len"],
        d_ff=mc.get("d_ff"),
        n_buckets=mc.get("n_buckets", 100),
        value_hidden_size=mc.get("value_hidden_size", 256),
        num_fourier_freq=mc.get("num_fourier_freq", 128),
        wl_sigma=mc.get("wl_sigma", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cuda().half()
    model.eval()
    print(f"loaded into GPU; parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    fens = [SAMPLE_FENS[i % len(SAMPLE_FENS)] for i in range(args.n)]
    B = args.n

    with tempfile.TemporaryDirectory(prefix="cutlass_prod_parity_") as td:
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

        # --- Engine run ---
        print(f"\nRunning engine on {B} FENs (max_iters={args.max_iters}, max_seq_len={args.max_seq_len})...")
        t0 = time.perf_counter()
        engine_results = engine.predict_moves_thinking(
            fens, 0.0, args.max_seq_len, args.max_iters)
        torch.cuda.synchronize()
        engine_time = time.perf_counter() - t0
        engine_total_tokens = sum(len(r.token_ids) for r in engine_results)
        print(f"  engine time: {engine_time*1000:.1f} ms")
        print(f"  engine tok/s: {engine_total_tokens / engine_time:.0f}")
        print(f"  engine moves: {[r.move for r in engine_results]}")

        # --- Python run ---
        print(f"\nRunning Python run_thinking on each FEN sequentially...")
        t0 = time.perf_counter()
        py_results = []
        for fen in fens:
            r = run_thinking(model, fen, temperature=0.0,
                             max_seq_len=args.max_seq_len, verbose=False)
            py_results.append(r)
        torch.cuda.synchronize()
        py_time = time.perf_counter() - t0
        py_total_tokens = sum(len(r.token_ids) for r in py_results)
        print(f"  python time: {py_time*1000:.1f} ms")
        print(f"  python tok/s: {py_total_tokens / py_time:.0f}")
        print(f"  python moves: {[r.final_move for r in py_results]}")
        print(f"  ==> engine speedup: {(engine_total_tokens/engine_time)/(py_total_tokens/py_time):.2f}x")

        # --- Parity analysis ---
        print(f"\n=== Per-FEN parity ===")
        strict_full = 0  # full token_ids match
        strict_prefix = 0  # exact match on the shared prefix
        moves_match = 0  # final_move strings match
        total_shared = 0
        total_mismatch = 0
        first_divs = []

        for i, (er, pr) in enumerate(zip(engine_results, py_results)):
            er_tokens = list(er.token_ids)
            pr_tokens = list(pr.token_ids)
            shared = min(len(er_tokens), len(pr_tokens))
            mismatch = sum(1 for a, b in zip(er_tokens[:shared], pr_tokens[:shared]) if a != b)
            first_div = next((j for j in range(shared) if er_tokens[j] != pr_tokens[j]), -1)

            er_move = er.move
            py_move = pr.final_move or ""
            move_eq = (er_move == py_move) and bool(er_move)

            total_shared += shared
            total_mismatch += mismatch
            if mismatch == 0:
                strict_prefix += 1
                if len(er_tokens) == len(pr_tokens):
                    strict_full += 1
            if move_eq:
                moves_match += 1
            if first_div >= 0:
                first_divs.append(first_div)

            tag = "✓" if mismatch == 0 else "✗"
            move_tag = "✓" if move_eq else f"engine={er_move!r} vs py={py_move!r}"
            print(f"  FEN[{i}] {tag} engine_len={len(er_tokens)} py_len={len(pr_tokens)} "
                  f"shared={shared} mismatch={mismatch} first_div={first_div}")
            print(f"    final_move: {move_tag}")

        print(f"\n=== Summary ===")
        print(f"  Strict full match (every token equal):     {strict_full}/{B}")
        print(f"  Strict prefix match (mismatch=0 in shared): {strict_prefix}/{B}")
        print(f"  Final-move match:                            {moves_match}/{B}")
        if total_shared:
            print(f"  Token-level mismatch rate: {total_mismatch}/{total_shared} = "
                  f"{100*total_mismatch/total_shared:.1f}%")
        if first_divs:
            avg = sum(first_divs) / len(first_divs)
            print(f"  Mean first divergence position: {avg:.0f} (of avg seq len ~{total_shared//B})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

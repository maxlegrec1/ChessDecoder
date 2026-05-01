"""Parity gate: Python state machine vs C++ batched engine, temp=0.

Runs `chessdecoder.inference.think.run_thinking` (the canonical Python reference)
and the C++ `ThinkingBatchedInferenceEngine` on the same FENs, then asserts:

  - token_ids match exactly (full-vocab, post-root-board)
  - wl_bucket_indices match exactly (argmax indices)
  - d_bucket_indices match exactly (argmax indices)
  - final_move matches exactly

A `--strict` flag fails on any mismatch. `--allow-fp16-noise` accepts up to
N divergence positions where Python's argmax and C++'s argmax differ by 1
(documented FP16 numerical noise from the post-cBLAS log_softmax cast in
batched_engine — see `chessdecoder/cpp/decoder/batched_engine.cpp:163,176`).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/verify_decoder_parity.py \\
        --checkpoint checkpoints/finetune-thinking-v1_20260422_135004/checkpoint_step_326000.pt \\
        --num-positions 16 --temperature 0.0 --strict

Note
----
Python reference takes ~10s per FEN on B200 (no batching, single-position
autoregressive). For quick iteration use --num-positions 8 or 16. The C++
side runs all N positions in one batched call and finishes in seconds.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Determinism: load_pretrain_positions uses Python's randomized hash().
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable] + sys.argv)

import torch

from chessdecoder.dataloader.sampling import load_pretrain_positions
from chessdecoder.inference.think import run_thinking, load_model as load_python_model
from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size
from chessdecoder.rl.config import GRPOConfig
from chessdecoder.rl.rollout import export_model
from chessdecoder.utils.training import load_pretrained_checkpoint
from chessdecoder.utils.uci import normalize_castling


def _diff_summary(name, py_seq, cpp_seq, max_show=5):
    """Return (n_match, n_total, mismatch_positions[:max_show])."""
    n = min(len(py_seq), len(cpp_seq))
    diffs = []
    matches = 0
    for i in range(n):
        if py_seq[i] == cpp_seq[i]:
            matches += 1
        else:
            if len(diffs) < max_show:
                diffs.append((i, py_seq[i], cpp_seq[i]))
    return matches, n, diffs


def compare(py_result, cpp_result, allow_fp16_noise=0):
    """Compare a single FEN's Python vs C++ result. Returns (passed, report)."""
    report = []
    fail = False

    # Token ID parity (full-vocab, includes root board which is deterministic
    # from the FEN; comparing from index 0 is fine).
    py_toks = py_result.token_ids
    cpp_toks = list(cpp_result.token_ids)
    n_match, n_total, diffs = _diff_summary("tokens", py_toks, cpp_toks)
    report.append(f"  tokens: {n_match}/{n_total} match (py_len={len(py_toks)}, cpp_len={len(cpp_toks)})")
    if n_match != n_total or len(py_toks) != len(cpp_toks):
        fail = True
        for i, p, c in diffs:
            report.append(f"    pos {i}: py={p} cpp={c}")

    # WL bucket index parity. Position keys may differ in order; compare as
    # ordered list of bucket indices (positions should match if tokens match).
    py_wl = [b for _, b in py_result.wl_bucket_indices]
    cpp_wl = [b for _, b in cpp_result.wl_bucket_indices]
    n_match, n_total, diffs = _diff_summary("wl", py_wl, cpp_wl)
    fp16_close = sum(1 for _, p, c in diffs if abs(p - c) <= 1)
    extra_diffs = [d for d in diffs if abs(d[1] - d[2]) > 1]
    report.append(f"  wl_buckets: {n_match}/{n_total} match (py_n={len(py_wl)}, cpp_n={len(cpp_wl)}, |Δ|≤1: {fp16_close})")
    if (n_match != n_total or len(py_wl) != len(cpp_wl)) and (
        not allow_fp16_noise or extra_diffs or
        (len(py_wl) - n_match) > allow_fp16_noise
    ):
        fail = True
        for i, p, c in diffs[:5]:
            report.append(f"    wl pos {i}: py_bucket={p} cpp_bucket={c}")

    # D bucket index parity
    py_d = [b for _, b in py_result.d_bucket_indices]
    cpp_d = [b for _, b in cpp_result.d_bucket_indices]
    n_match, n_total, diffs = _diff_summary("d", py_d, cpp_d)
    fp16_close = sum(1 for _, p, c in diffs if abs(p - c) <= 1)
    extra_diffs = [d for d in diffs if abs(d[1] - d[2]) > 1]
    report.append(f"  d_buckets:  {n_match}/{n_total} match (py_n={len(py_d)}, cpp_n={len(cpp_d)}, |Δ|≤1: {fp16_close})")
    if (n_match != n_total or len(py_d) != len(cpp_d)) and (
        not allow_fp16_noise or extra_diffs or
        (len(py_d) - n_match) > allow_fp16_noise
    ):
        fail = True
        for i, p, c in diffs[:5]:
            report.append(f"    d pos {i}: py_bucket={p} cpp_bucket={c}")

    # Final move
    py_move = py_result.final_move
    cpp_move = normalize_castling(cpp_result.move) if cpp_result.move else None
    if py_move == cpp_move:
        report.append(f"  final_move: MATCH ({py_move})")
    else:
        report.append(f"  final_move: MISMATCH py={py_move} cpp={cpp_move}")
        fail = True

    return not fail, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="chessdecoder/rl/config.yaml")
    p.add_argument("--num-positions", type=int, default=16,
                   help="N FENs to test (Python ref is ~10s/FEN, scale accordingly)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42,
                   help="FEN sampling seed (independent of cfg.eval_seed)")
    p.add_argument("--strict", action="store_true",
                   help="Exit nonzero on any mismatch (default: report-only)")
    p.add_argument("--allow-fp16-noise", type=int, default=0,
                   help="Accept up to N bucket-index diffs of magnitude 1 per FEN")
    p.add_argument("--output", default=None,
                   help="Save JSON summary to this path")
    args = p.parse_args()

    if args.temperature != 0.0:
        print("WARNING: parity at temp>0 is distributional only; this script tests bit-exact match.",
              file=sys.stderr)

    cfg = GRPOConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Sample FENs (independent of cfg.eval_seed so we don't collide with
    # the RL eval positions baked into rl/config.yaml) ────────────────────────
    all_pt = sorted(glob.glob(os.path.join(cfg.pretrain_parquet_dir, "*.parquet")))
    if not all_pt:
        raise SystemExit(f"No parquets in {cfg.pretrain_parquet_dir}")
    positions = load_pretrain_positions(
        cfg.pretrain_parquet_dir, args.num_positions, args.seed,
        files=all_pt[:1],  # one parquet is plenty for N up to ~hundreds
    )
    fens = [pos["fen"] for pos in positions]
    print(f"Sampled {len(fens)} FENs (seed={args.seed})\n")

    # ── Python reference ─────────────────────────────────────────────────────
    print("Loading Python reference model ...")
    py_model, max_seq_len = load_python_model(args.checkpoint, device)

    py_results = []
    t0 = time.time()
    for i, fen in enumerate(fens):
        t1 = time.time()
        r = run_thinking(py_model, fen, temperature=args.temperature,
                         device=device, max_seq_len=max_seq_len, verbose=False)
        py_results.append(r)
        print(f"  py [{i+1}/{len(fens)}] tokens={len(r.token_ids)} "
              f"final={r.final_move} ({time.time()-t1:.1f}s)")
    py_total = time.time() - t0
    print(f"Python reference: {py_total:.1f}s total\n")

    # ── C++ batched engine ───────────────────────────────────────────────────
    # Build a fresh export of the same checkpoint (eval-mode FP16) so that
    # both engines see identical weights.
    print("Building C++ engine: exporting checkpoint ...")
    cpp_model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=cfg.model["embed_dim"], num_heads=cfg.model["num_heads"],
        num_layers=cfg.model["num_layers"], max_seq_len=cfg.model["max_seq_len"],
        d_ff=cfg.model.get("d_ff"), n_buckets=cfg.model.get("n_buckets", 100),
        value_hidden_size=cfg.model.get("value_hidden_size", 256),
        num_fourier_freq=cfg.model.get("num_fourier_freq", 128),
        wl_sigma=cfg.model.get("wl_sigma", 0.4),
    ).to(device)
    load_pretrained_checkpoint(cpp_model, args.checkpoint, device)

    export_dir = Path(tempfile.mkdtemp(prefix="parity_"))
    try:
        export_model(cpp_model, {"model": cfg.model}, export_dir)
        del cpp_model
        torch.cuda.empty_cache()

        import _decoder_inference_cpp as cpp
        engine = cpp.ThinkingBatchedInferenceEngine(
            str(export_dir / "backbone.pt"),
            str(export_dir / "weights"),
            str(export_dir / "vocab.json"),
            str(export_dir / "config.json"),
            len(fens),
        )
        for attr in ("board_temperature", "think_temperature",
                     "policy_temperature", "wl_temperature", "d_temperature"):
            setattr(engine, attr, args.temperature)

        t0 = time.time()
        cpp_results = engine.predict_moves(fens, args.temperature)
        cpp_total = time.time() - t0
        print(f"C++ batched: {cpp_total:.1f}s total ({len(fens)/cpp_total:.1f} FEN/s)\n")
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)

    # ── Compare ──────────────────────────────────────────────────────────────
    n_pass = 0
    fen_reports = []
    for i, (fen, py_r, cpp_r) in enumerate(zip(fens, py_results, cpp_results)):
        passed, report = compare(py_r, cpp_r, allow_fp16_noise=args.allow_fp16_noise)
        if passed:
            n_pass += 1
        fen_reports.append({"fen": fen, "passed": passed, "report": report})
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] FEN #{i}: {fen}")
        for line in report:
            print(line)

    summary = {
        "total": len(fens),
        "passed": n_pass,
        "failed": len(fens) - n_pass,
        "py_total_s": py_total,
        "cpp_total_s": cpp_total,
        "cpp_throughput_fen_per_s": len(fens) / cpp_total,
        "temperature": args.temperature,
        "fen_reports": fen_reports,
    }
    print(f"\n=== Parity: {n_pass}/{len(fens)} PASS ===")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.output}")

    if args.strict and n_pass != len(fens):
        sys.exit(1)


if __name__ == "__main__":
    main()

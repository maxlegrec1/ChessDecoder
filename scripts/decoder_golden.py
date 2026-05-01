"""C++ batched engine regression gate.

Two modes:

  --lock GOLDEN.json
      Run the engine on N FENs at temp=0, save token_ids + bucket indices +
      final move per FEN to GOLDEN.json. Use to capture the pre-change baseline.

  --check GOLDEN.json
      Run the engine on the same N FENs (same seed), compare to GOLDEN.json,
      fail on any divergence. Use as a regression gate after Phase 0/2/etc.
      changes that should NOT alter the engine's output.

Why C++ vs C++ rather than Python vs C++:
  Python and C++ already diverge under temp=0 due to FP16 noise in the cuBLAS
  log_softmax cast (chessdecoder/cpp/decoder/batched_engine.cpp:163,176).
  That divergence is independent of our refactor and can't be the gate.
  C++ vs its own pre-change output is the right invariant for refactors that
  preserve numerics (Phase 0: KV-cache layout, Phase 2: sync removal).

Usage
-----
    # Lock baseline before Phase 0
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/decoder_golden.py \\
        --lock tests/golden/cpp_baseline_temp0.json \\
        --checkpoint checkpoints/finetune-thinking-v1_20260422_135004/checkpoint_step_326000.pt \\
        --num-positions 64

    # After Phase 0 changes
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/decoder_golden.py \\
        --check tests/golden/cpp_baseline_temp0.json \\
        --checkpoint <same checkpoint> \\
        --num-positions 64
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

# load_pretrain_positions uses Python's hash() with a per-game seed, which is
# randomized unless PYTHONHASHSEED is set. Re-exec with seed=0 if needed so
# --lock and --check produce the same FEN list.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable] + sys.argv)

import torch

from chessdecoder.dataloader.sampling import load_pretrain_positions
from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size
from chessdecoder.rl.config import GRPOConfig
from chessdecoder.rl.rollout import export_model
from chessdecoder.utils.training import load_pretrained_checkpoint
from chessdecoder.utils.uci import normalize_castling


def run_engine(checkpoint, cfg, fens, temperature, batch_size):
    """Build engine, run on fens, return list of dicts."""
    device = torch.device("cuda")
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=cfg.model["embed_dim"], num_heads=cfg.model["num_heads"],
        num_layers=cfg.model["num_layers"], max_seq_len=cfg.model["max_seq_len"],
        d_ff=cfg.model.get("d_ff"), n_buckets=cfg.model.get("n_buckets", 100),
        value_hidden_size=cfg.model.get("value_hidden_size", 256),
        num_fourier_freq=cfg.model.get("num_fourier_freq", 128),
        wl_sigma=cfg.model.get("wl_sigma", 0.4),
    ).to(device)
    load_pretrained_checkpoint(model, checkpoint, device)

    export_dir = Path(tempfile.mkdtemp(prefix="golden_"))
    try:
        export_model(model, {"model": cfg.model}, export_dir)
        del model
        torch.cuda.empty_cache()

        import _decoder_inference_cpp as cpp
        engine = cpp.ThinkingBatchedInferenceEngine(
            str(export_dir / "backbone.pt"),
            str(export_dir / "weights"),
            str(export_dir / "vocab.json"),
            str(export_dir / "config.json"),
            batch_size,
        )
        for attr in ("board_temperature", "think_temperature",
                     "policy_temperature", "wl_temperature", "d_temperature"):
            setattr(engine, attr, temperature)

        t0 = time.time()
        N = len(fens)
        results = []
        for start in range(0, N, batch_size):
            chunk = fens[start:start + batch_size]
            cpp_results = engine.predict_moves(chunk, temperature)
            for r in cpp_results:
                results.append({
                    "move": normalize_castling(r.move) if r.move else None,
                    "token_ids": list(r.token_ids),
                    "wl_bucket_indices": [list(t) for t in r.wl_bucket_indices],
                    "d_bucket_indices": [list(t) for t in r.d_bucket_indices],
                })
        elapsed = time.time() - t0
        return results, elapsed
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)


def sample_fens(cfg, num_positions, seed):
    all_pt = sorted(glob.glob(os.path.join(cfg.pretrain_parquet_dir, "*.parquet")))
    if not all_pt:
        raise SystemExit(f"No parquets in {cfg.pretrain_parquet_dir}")
    positions = load_pretrain_positions(
        cfg.pretrain_parquet_dir, num_positions, seed,
        files=all_pt[:1],
    )
    return [pos["fen"] for pos in positions]


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lock", metavar="PATH", help="Save engine output to PATH")
    mode.add_argument("--check", metavar="PATH", help="Compare engine output to PATH")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="chessdecoder/rl/config.yaml")
    p.add_argument("--num-positions", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    cfg = GRPOConfig.from_yaml(args.config)
    fens = sample_fens(cfg, args.num_positions, args.seed)
    print(f"Sampled {len(fens)} FENs (seed={args.seed})")

    print(f"Running C++ engine ...")
    results, elapsed = run_engine(args.checkpoint, cfg, fens, args.temperature, args.batch_size)
    print(f"  done in {elapsed:.1f}s ({len(fens)/elapsed:.1f} FEN/s)\n")

    payload = {
        "checkpoint": str(args.checkpoint),
        "temperature": args.temperature,
        "seed": args.seed,
        "num_positions": args.num_positions,
        "fens": fens,
        "results": results,
    }

    if args.lock:
        out = Path(args.lock)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"LOCKED → {out}")
        return

    # --check mode
    with open(args.check) as f:
        golden = json.load(f)

    if golden["fens"] != fens:
        print(f"FAIL: FEN list mismatch (seed/num/parquet differ from golden)")
        sys.exit(1)
    if golden["temperature"] != args.temperature:
        print(f"FAIL: temperature mismatch golden={golden['temperature']} this={args.temperature}")
        sys.exit(1)

    n_pass = 0
    n_fail = 0
    for i, (fen, gr, cr) in enumerate(zip(fens, golden["results"], results)):
        diffs = []
        if gr["move"] != cr["move"]:
            diffs.append(f"move: golden={gr['move']} got={cr['move']}")
        if gr["token_ids"] != cr["token_ids"]:
            n_match = sum(1 for a, b in zip(gr["token_ids"], cr["token_ids"]) if a == b)
            diffs.append(f"tokens: {n_match}/{min(len(gr['token_ids']), len(cr['token_ids']))} "
                         f"(golden_len={len(gr['token_ids'])} got_len={len(cr['token_ids'])})")
        if gr["wl_bucket_indices"] != cr["wl_bucket_indices"]:
            diffs.append(f"wl_buckets differ")
        if gr["d_bucket_indices"] != cr["d_bucket_indices"]:
            diffs.append(f"d_buckets differ")
        if diffs:
            n_fail += 1
            print(f"[FAIL] FEN #{i}: {fen}")
            for d in diffs:
                print(f"  {d}")
        else:
            n_pass += 1

    print(f"\n=== Regression: {n_pass}/{len(fens)} match golden ===")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

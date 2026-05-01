"""Throughput + VRAM-stability bench for the C++ batched inference engine.

Sweeps inference_batch_size, runs N FENs, reports tok/s, FEN/s, latency
percentiles, peak VRAM, and VRAM jitter. Emits a JSON snapshot for
regression comparison across phases of the engine overhaul.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_decoder_engine.py \\
        --checkpoint checkpoints/finetune-thinking-v1_20260422_135004/checkpoint_step_326000.pt \\
        --batch-sizes 16,32,64,128 --num-fens 256 \\
        --output tests/golden/bench_baseline.json

Comparison
----------
    uv run python scripts/bench_decoder_engine.py \\
        --compare tests/golden/bench_baseline.json --output bench_phase{N}.json
"""

import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Determinism: same PYTHONHASHSEED dance as decoder_golden.py.
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


def _nvidia_smi_used_mib():
    """Read GPU memory.used (MiB) via nvidia-smi. Returns None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=2,
        ).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return None


class VRAMSampler:
    """Polls nvidia-smi memory.used in a background thread."""
    def __init__(self, interval_s=0.1):
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            v = _nvidia_smi_used_mib()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.samples


def sample_fens(cfg, num_fens, seed):
    all_pt = sorted(glob.glob(os.path.join(cfg.pretrain_parquet_dir, "*.parquet")))
    if not all_pt:
        raise SystemExit(f"No parquets in {cfg.pretrain_parquet_dir}")
    positions = load_pretrain_positions(
        cfg.pretrain_parquet_dir, num_fens, seed,
        files=all_pt[:1],
    )
    return [pos["fen"] for pos in positions]


def bench_one(engine_factory, fens, batch_size, temperature):
    """Run engine over fens at given batch_size; return per-call timings + tokens."""
    sampler = VRAMSampler(interval_s=0.1)

    # Warmup: one batch to populate CUDA-graph caches in the engine.
    engine = engine_factory(batch_size)
    warmup_fens = fens[:batch_size] if len(fens) >= batch_size else fens + fens[:batch_size - len(fens)]
    _ = engine.predict_moves(warmup_fens, temperature)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    sampler.start()
    per_call_times = []
    per_call_tokens = []
    n_done = 0
    t_total = time.time()
    n_full_batches = len(fens) // batch_size
    n_fens = n_full_batches * batch_size if n_full_batches else len(fens)
    for start in range(0, n_fens, batch_size):
        chunk = fens[start:start + batch_size]
        t0 = time.time()
        results = engine.predict_moves(chunk, temperature)
        torch.cuda.synchronize()
        per_call_times.append(time.time() - t0)
        per_call_tokens.append(sum(len(r.token_ids) for r in results))
        n_done += len(chunk)
    elapsed = time.time() - t_total
    samples = sampler.stop()

    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    nvsmi_peak = max(samples) if samples else None
    nvsmi_jitter = (max(samples) - min(samples)) if samples else None

    total_tok = sum(per_call_tokens)
    return {
        "batch_size": batch_size,
        "n_fens": n_done,
        "n_batches": len(per_call_times),
        "elapsed_s": elapsed,
        "tok_per_s": total_tok / elapsed,
        "fen_per_s": n_done / elapsed,
        "total_tokens": total_tok,
        "avg_tok_per_fen": total_tok / max(n_done, 1),
        "ms_per_fen_p50": 1000 * statistics.median(t / max(batch_size, 1) for t in per_call_times),
        "ms_per_fen_p95": 1000 * (statistics.quantiles([t / max(batch_size, 1) for t in per_call_times], n=20)[18]
                                  if len(per_call_times) >= 20 else max(per_call_times) / max(batch_size, 1)),
        "torch_peak_mb": peak,
        "nvsmi_peak_mb": nvsmi_peak,
        "nvsmi_jitter_mb": nvsmi_jitter,
        "nvsmi_n_samples": len(samples),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="chessdecoder/rl/config.yaml")
    p.add_argument("--batch-sizes", default="16,32,64,128",
                   help="Comma-separated list of inference batch sizes to sweep")
    p.add_argument("--num-fens", type=int, default=256,
                   help="Total FENs to feed at each batch size")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temp matching RL config (think + policy)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--compare", default=None,
                   help="Path to a previous bench JSON; print delta")
    args = p.parse_args()

    cfg = GRPOConfig.from_yaml(args.config)
    fens = sample_fens(cfg, args.num_fens, args.seed)
    print(f"Loaded {len(fens)} FENs (seed={args.seed})")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB\n")

    # Build + load model and export once; engine factories rebuild the C++
    # engine for each batch size (CUDA-graph buffers are pre-allocated to
    # batch_size at construction).
    print("Building model + exporting backbone ...")
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=cfg.model["embed_dim"], num_heads=cfg.model["num_heads"],
        num_layers=cfg.model["num_layers"], max_seq_len=cfg.model["max_seq_len"],
        d_ff=cfg.model.get("d_ff"), n_buckets=cfg.model.get("n_buckets", 100),
        value_hidden_size=cfg.model.get("value_hidden_size", 256),
        num_fourier_freq=cfg.model.get("num_fourier_freq", 128),
        wl_sigma=cfg.model.get("wl_sigma", 0.4),
    ).to(device)
    load_pretrained_checkpoint(model, args.checkpoint, device)

    export_dir = Path(tempfile.mkdtemp(prefix="bench_"))
    try:
        export_model(model, {"model": cfg.model}, export_dir)
        del model
        torch.cuda.empty_cache()

        import _decoder_inference_cpp as cpp
        def make_engine(B):
            engine = cpp.ThinkingBatchedInferenceEngine(
                str(export_dir / "backbone.pt"),
                str(export_dir / "weights"),
                str(export_dir / "vocab.json"),
                str(export_dir / "config.json"),
                B,
            )
            engine.think_temperature = args.temperature
            engine.policy_temperature = args.temperature
            engine.board_temperature = cfg.board_temperature
            engine.wl_temperature = cfg.wl_temperature
            engine.d_temperature = cfg.d_temperature
            return engine

        results = []
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        print(f"{'B':>5}  {'fens':>5}  {'tok/s':>8}  {'fen/s':>7}  "
              f"{'ms/fen p50':>11}  {'p95':>6}  "
              f"{'torch_peak_mb':>13}  {'nvsmi_peak':>10}  {'jitter':>7}")
        print(f"{'-'*5}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*11}  {'-'*6}  "
              f"{'-'*13}  {'-'*10}  {'-'*7}")
        for B in batch_sizes:
            r = bench_one(make_engine, fens, B, args.temperature)
            results.append(r)
            jitter = f"{r['nvsmi_jitter_mb']:.0f}" if r['nvsmi_jitter_mb'] is not None else "n/a"
            nvpeak = f"{r['nvsmi_peak_mb']:.0f}" if r['nvsmi_peak_mb'] is not None else "n/a"
            print(f"{B:>5}  {r['n_fens']:>5}  {r['tok_per_s']:>8.0f}  "
                  f"{r['fen_per_s']:>7.2f}  {r['ms_per_fen_p50']:>10.1f}  "
                  f"{r['ms_per_fen_p95']:>5.1f}  "
                  f"{r['torch_peak_mb']:>12.0f}  {nvpeak:>10}  {jitter:>7}")
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)

    snapshot = {
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": args.checkpoint,
        "temperature": args.temperature,
        "num_fens": args.num_fens,
        "results": results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"\nWrote {args.output}")

    if args.compare:
        with open(args.compare) as f:
            prev = json.load(f)
        prev_by_b = {r["batch_size"]: r for r in prev["results"]}
        print(f"\nDelta vs {args.compare}:")
        print(f"{'B':>5}  {'Δtok/s':>10}  {'Δfen/s':>10}  {'Δpeak_mb':>10}  {'Δjitter':>10}")
        for r in results:
            B = r["batch_size"]
            if B not in prev_by_b:
                continue
            p = prev_by_b[B]
            d_tok = r["tok_per_s"] - p["tok_per_s"]
            d_fen = r["fen_per_s"] - p["fen_per_s"]
            d_peak = r["torch_peak_mb"] - p["torch_peak_mb"]
            d_jit = ((r["nvsmi_jitter_mb"] or 0) - (p["nvsmi_jitter_mb"] or 0))
            print(f"{B:>5}  {d_tok:>+10.0f}  {d_fen:>+10.2f}  {d_peak:>+10.0f}  {d_jit:>+10.0f}")


if __name__ == "__main__":
    main()

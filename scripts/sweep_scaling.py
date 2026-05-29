"""Scaling-law sweep: fit ``L(N, wall_clock)`` and ``L(N, FLOPs)`` for chess.

Trains 5 model sizes log-spaced from ~8M to ~120M params, each at its
memory-efficient per-step batch size (with grad-accumulation only when the
24GB 4090 forces it), targeting roughly equal wall-clock budgets so the
loss curves are directly comparable.

The wandb logs include both ``_runtime`` (wall-clock seconds, auto-set
by wandb at init) and ``train/cumulative_tflops`` (per-micro-batch FLOPs
times step count), so you can fit either the hardware-specific wall-clock
law or the transferable FLOPs law from the same dataset.

After the sweep you have data points across (size, time) and (size, FLOPs)
that, combined with our prior runs at 2.3M / 15M / 27M, span a good 50x
parameter range — enough to fit a Kaplan/Chinchilla-style law and predict
the optimal model size at a 7-day budget without running 7 days per size.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path


# (name, embed_dim, num_heads, num_layers, d_ff, batch_size, grad_accum, max_steps)
# Per-size batch / accum are chosen for memory + FP8 friendliness:
#   - d_ff divisible by 16 so MLP linears stay on the FP8 path
#   - batch_size as large as fits at 24GB; grad_accum makes effective batch 2048
#   - max_steps tuned so each run takes ~5-7h on the 4090
SIZES = [
    # name,  embed, heads, layers, d_ff, batch, accum, max_steps
    ("8M",     320,     5,      4,  1280,  2048,     1,    150000),
    ("15M",    384,     6,      6,  1536,  2048,     1,     80000),
    ("30M",    512,     8,      6,  2048,  2048,     1,     70000),
    ("60M",    640,    10,      8,  2560,  1024,     2,    100000),
    ("120M",   768,    12,     12,  3072,   512,     4,     80000),
]
SEED = 42

COMMON_OVERRIDES = {
    "model.attention_variant": "geom",
    "model.policy_head": "cross_attn",
    "model.input_mode": "lc0_64",
    "data.positions_per_game": 1,
    "data.num_workers": 3,
    "training.optimizer": "muon",
    "training.learning_rate": 3e-3,
    "training.weight_decay": 0.01,
    "training.grad_clip": 1.0,
    "training.num_epochs": 1000,
    "training.log_every_n_steps": 200,
    "training.save_every_n_steps": 1000000,
    "training.use_fp8": True,
    "training.fp8_compile": True,
    "training.seed": SEED,
    "training.resume_from": None,
}

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "sweep_out"
LOG_DIR.mkdir(exist_ok=True)

WANDB_GROUP = os.environ.get(
    "WANDB_RUN_GROUP",
    f"scaling-law-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")


def run_name(size: str) -> str:
    return f"scaling_{size}"


_SUCCESS_MARKER = "Reached max_steps="


def _log_is_complete(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        with log_path.open("rb") as f:
            f.seek(max(0, log_path.stat().st_size - 4096))
            return _SUCCESS_MARKER.encode() in f.read()
    except OSError:
        return False


def launch(spec) -> Path:
    name, embed, heads, layers, dff, B, accum, max_steps = spec
    rname = run_name(name)
    log_path = LOG_DIR / f"{rname}.log"
    if _log_is_complete(log_path):
        print(f"[skip]  {rname}: completed previously ({log_path}).")
        return log_path
    if log_path.exists():
        print(f"[redo]  {rname}: prior log incomplete, rerunning.")
        log_path.unlink()

    overrides = {
        **COMMON_OVERRIDES,
        "model.embed_dim": embed,
        "model.num_heads": heads,
        "model.num_layers": layers,
        "model.d_ff": dff,
        "data.batch_size": B,
        "training.gradient_accumulation_steps": accum,
        "run_name": rname,
        "training.checkpoint_dir": f"checkpoints/sweep/{rname}/",
    }
    set_args = []
    for k, v in overrides.items():
        set_args += ["--set", f"{k}={v if v is not None else 'null'}"]

    cmd = [sys.executable, "-u", "chessdecoder/train/train.py",
           "chessdecoder/train/config.yaml",
           "--max-steps", str(max_steps), *set_args]
    print(f"[run]   {rname}: {max_steps} micro-steps, B={B} accum={accum} "
          f"(effective B={B * accum}) -> {log_path}")
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": "0",
           "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
           "WANDB_RUN_GROUP": WANDB_GROUP,
           "WANDB_TAGS": "sweep,scaling-law",
           "PYTHONPATH": str(ROOT)}
    with log_path.open("w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                             env=env, cwd=str(ROOT))
    print(f"[{'done' if rc == 0 else 'fail'}]  {rname}"
          f"{'' if rc == 0 else f': exit {rc}'}")
    return log_path


_STEP_RE = re.compile(
    r"Step (\d+): loss ([\d.]+) \(pol ([\d.]+) wdl ([\d.]+)\) "
    r"move_acc=([\d.]+) wdl_acc=([\d.]+) q_mae=([\d.]+)")


def parse_log(log_path: Path) -> dict:
    last = {}
    try:
        text = log_path.read_text(errors="ignore")
    except FileNotFoundError:
        return {}
    for m in _STEP_RE.finditer(text):
        last = {"step": int(m[1]), "loss": float(m[2]),
                "pol": float(m[3]), "wdl": float(m[4]),
                "move_acc": float(m[5]), "wdl_acc": float(m[6]),
                "q_mae": float(m[7])}
    return last


def summarize() -> None:
    rows = []
    for spec in SIZES:
        name = spec[0]
        r = parse_log(LOG_DIR / f"{run_name(name)}.log")
        rows.append((name, r))
    rows.sort(key=lambda r: r[1].get("loss", float("inf")))
    print()
    print(f"{'size':<8}  {'step':>7}  {'loss':>7}  {'pol':>6}  {'wdl':>6}  "
          f"{'move_acc':>9}  {'wdl_acc':>8}  {'q_mae':>6}")
    print("-" * 75)
    for name, r in rows:
        if not r:
            print(f"{name:<8}  (no parseable log)")
            continue
        print(f"{name:<8}  {r['step']:>7}  {r['loss']:>7.3f}  "
              f"{r['pol']:>6.3f}  {r['wdl']:>6.3f}  "
              f"{r['move_acc']:>9.3f}  {r['wdl_acc']:>8.3f}  {r['q_mae']:>6.3f}")


def main() -> int:
    for spec in SIZES:
        launch(spec)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

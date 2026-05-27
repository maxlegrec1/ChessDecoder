"""Attention / positional-embedding sweep on a small encoder + big batch.

Runs five variants of the encoder's positional/attention machinery at the
already-best optimizer/LR (muon @ 3e-3), holding everything else fixed:

  baseline   learned absolute pos embedding + plain bidirectional MHA
  rope1d     1D RoPE (rotate Q/K by position 0..67), no token pos embed
  rope2d     2D RoPE — head_dim split, rotated by rank and file
  relpos2d   learned (Δrank, Δfile) attention-logit bias (T5/ALiBi-style)
  geom       MLP over chess geometric features (knight, diag, cheby, ...)
             producing the attention bias

Same launch shape as ``scripts/sweep_lr_opt.py``: subprocesses with ``--set``
overrides, one log file per variant, final move_acc summary table.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path

VARIANTS = ["baseline", "rope1d", "rope2d", "relpos2d", "geom"]
MAX_STEPS = 1500
SEED = 42

COMMON_OVERRIDES = {
    "model.embed_dim": 512,
    "model.num_heads": 8,
    "model.num_layers": 6,
    "model.d_ff": 2048,
    "data.batch_size": 2048,
    "data.positions_per_game": 1,
    # 1 worker not 2 — we sit close to the 15GB RAM ceiling during shard
    # transitions and two workers reloading at once is enough to trigger OOM.
    "data.num_workers": 1,
    "training.optimizer": "muon",
    "training.learning_rate": 3e-3,
    "training.weight_decay": 0.01,
    "training.grad_clip": 1.0,
    "training.gradient_accumulation_steps": 1,
    "training.num_epochs": 1000,
    "training.log_every_n_steps": 25,
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
    f"attn-sweep-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")


def run_name(variant: str) -> str:
    return f"sweep_attn_{variant}"


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


def launch(variant: str) -> Path:
    name = run_name(variant)
    log_path = LOG_DIR / f"{name}.log"
    if _log_is_complete(log_path):
        print(f"[skip]  {name}: completed previously ({log_path}).")
        return log_path
    if log_path.exists():
        print(f"[redo]  {name}: prior log incomplete, rerunning.")
        log_path.unlink()

    overrides = {**COMMON_OVERRIDES,
                 "model.attention_variant": variant,
                 "run_name": name,
                 "training.checkpoint_dir": f"checkpoints/sweep/{name}/"}
    set_args = []
    for k, v in overrides.items():
        set_args += ["--set", f"{k}={v if v is not None else 'null'}"]

    cmd = [sys.executable, "-u", "chessdecoder/train/train.py",
           "chessdecoder/train/config.yaml",
           "--max-steps", str(MAX_STEPS), *set_args]
    print(f"[run]   {name}: {MAX_STEPS} steps -> {log_path}")
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": "0",
           "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
           "WANDB_RUN_GROUP": WANDB_GROUP,
           "WANDB_TAGS": "sweep,attention",
           "PYTHONPATH": str(ROOT)}
    with log_path.open("w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                             env=env, cwd=str(ROOT))
    print(f"[{'done' if rc == 0 else 'fail'}]  {name}"
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
    for v in VARIANTS:
        r = parse_log(LOG_DIR / f"{run_name(v)}.log")
        rows.append((v, r))
    rows.sort(key=lambda r: -(r[1].get("move_acc", -1.0)))
    print()
    print(f"{'variant':<10}  {'step':>6}  {'loss':>7}  {'pol':>6}  {'wdl':>6}  "
          f"{'move_acc':>9}  {'wdl_acc':>8}  {'q_mae':>6}")
    print("-" * 70)
    for v, r in rows:
        if not r:
            print(f"{v:<10}  (no parseable log)")
            continue
        print(f"{v:<10}  {r['step']:>6}  {r['loss']:>7.3f}  "
              f"{r['pol']:>6.3f}  {r['wdl']:>6.3f}  "
              f"{r['move_acc']:>9.3f}  {r['wdl_acc']:>8.3f}  {r['q_mae']:>6.3f}")


def main() -> int:
    for v in VARIANTS:
        launch(v)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

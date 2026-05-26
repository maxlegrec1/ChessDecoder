"""LR x optimizer sweep on a small encoder + big batch.

Launches ``chessdecoder/train/train.py`` once per (optimizer, lr) combo, using
``--set`` overrides — no per-run yaml files. Each run is capped at
``MAX_STEPS`` gradient steps; logs land in ``sweep_out/<run_name>.log``.

After all runs finish, the script tails each log, parses the last "Step N:"
line, and prints a final table sorted by move_acc so the winning config is
obvious. We deliberately don't try to be clever — sequential runs on the
single 4090, fixed seed, fixed data shuffle, same step budget.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path

# --- sweep grid -------------------------------------------------------------
OPTIMIZERS = ["muon", "adamw"]
LRS = [1e-4, 3e-4, 1e-3, 3e-3]
MAX_STEPS = 1500
SEED = 42

# Small model + big batch (A/B regime: cheap to run, ranking still informative).
COMMON_OVERRIDES = {
    "model.embed_dim": 512,
    "model.num_heads": 8,
    "model.num_layers": 6,
    "model.d_ff": 2048,
    # Big batch, max diversity: 2048 games x 1 position each = 2048 positions/step.
    "data.batch_size": 2048,
    "data.positions_per_game": 1,
    # 2 workers + spawn + persistent: enough to hide ~30s parquet loads behind
    # GPU compute, without contending too much on the single SSD.
    "data.num_workers": 2,
    "training.weight_decay": 0.01,
    "training.grad_clip": 1.0,
    "training.gradient_accumulation_steps": 1,
    "training.num_epochs": 1000,    # large; max_steps stops the loop first
    "training.log_every_n_steps": 25,
    "training.save_every_n_steps": 1000000,   # don't save during sweep
    "training.use_fp8": True,
    "training.fp8_compile": True,
    "training.seed": SEED,
    "training.resume_from": None,
}

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "sweep_out"
LOG_DIR.mkdir(exist_ok=True)

# One group per sweep invocation so the 8 runs cluster in the wandb UI; can
# be overridden externally if you want to extend an existing group.
WANDB_GROUP = os.environ.get(
    "WANDB_RUN_GROUP",
    f"lr-opt-sweep-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")


def run_name(opt: str, lr: float) -> str:
    return f"sweep_{opt}_lr{lr:g}".replace("+", "")


def launch(opt: str, lr: float) -> Path:
    name = run_name(opt, lr)
    log_path = LOG_DIR / f"{name}.log"
    if log_path.exists():
        print(f"[skip]  {name}: log already exists ({log_path}).")
        return log_path

    overrides = {**COMMON_OVERRIDES,
                 "training.optimizer": opt,
                 "training.learning_rate": lr,
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
           # wandb on by default so the sweep is fully observable; export
           # ``WANDB_MODE=disabled`` if you want to skip logging.
           "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
           "WANDB_RUN_GROUP": WANDB_GROUP,
           "WANDB_TAGS": "sweep,lr-opt",
           "PYTHONPATH": str(ROOT)}
    with log_path.open("w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                             env=env, cwd=str(ROOT))
    if rc != 0:
        print(f"[fail]  {name}: exit {rc}")
    else:
        print(f"[done]  {name}")
    return log_path


_STEP_RE = re.compile(
    r"Step (\d+): loss ([\d.]+) \(pol ([\d.]+) wdl ([\d.]+)\) "
    r"move_acc=([\d.]+) wdl_acc=([\d.]+) q_mae=([\d.]+) pos/s=(\d+)")


def parse_log(log_path: Path) -> dict:
    """Return the last well-formed Step-line as a dict (or {} if none)."""
    last = {}
    try:
        text = log_path.read_text(errors="ignore")
    except FileNotFoundError:
        return {}
    for m in _STEP_RE.finditer(text):
        last = {"step": int(m[1]), "loss": float(m[2]),
                "pol": float(m[3]), "wdl": float(m[4]),
                "move_acc": float(m[5]), "wdl_acc": float(m[6]),
                "q_mae": float(m[7]), "pos_per_s": int(m[8])}
    return last


def summarize() -> None:
    rows = []
    for opt in OPTIMIZERS:
        for lr in LRS:
            log = LOG_DIR / f"{run_name(opt, lr)}.log"
            r = parse_log(log)
            rows.append((opt, lr, r))
    rows.sort(key=lambda r: -(r[2].get("move_acc", -1.0)))
    print()
    print(f"{'optimizer':<10} {'lr':>8}  {'step':>6}  {'loss':>7}  "
          f"{'pol':>6}  {'wdl':>6}  {'move_acc':>9}  {'wdl_acc':>8}  {'q_mae':>6}  {'pos/s':>6}")
    print("-" * 88)
    for opt, lr, r in rows:
        if not r:
            print(f"{opt:<10} {lr:>8g}  (no parseable log)")
            continue
        print(f"{opt:<10} {lr:>8g}  {r['step']:>6}  {r['loss']:>7.3f}  "
              f"{r['pol']:>6.3f}  {r['wdl']:>6.3f}  {r['move_acc']:>9.3f}  "
              f"{r['wdl_acc']:>8.3f}  {r['q_mae']:>6.3f}  {r['pos_per_s']:>6}")


def main() -> int:
    for opt in OPTIMIZERS:
        for lr in LRS:
            launch(opt, lr)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

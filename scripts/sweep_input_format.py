"""Sweep the input-token format with the *current* best backbone + policy:
geom attention + cross_attn policy. Four modes:

  default          68 tokens: CLS | 64 squares | end_pos | castling | stm
  no_end           67 tokens: drop end_pos (suspected vestigial)
  no_cls_no_end    66 tokens: drop CLS + end_pos (value via mean-pool)
  lc0_64           64 tokens: squares only; stm/castling as broadcast embeds,
                   value via mean-pool. This is the LC0 layout.

Holds everything else fixed: geom backbone, cross_attn policy, muon @ 3e-3,
B=2048 N=1, 7500 steps. Wandb group ``input-format-sweep-<ts>``, tag
``sweep,input-format``.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path

INPUT_MODES = ["lc0_64", "cls_65"]
MAX_STEPS = 7500
SEED = 42

COMMON_OVERRIDES = {
    "model.embed_dim": 512,
    "model.num_heads": 8,
    "model.num_layers": 6,
    "model.d_ff": 2048,
    "model.attention_variant": "geom",
    "model.policy_head": "cross_attn",
    "data.batch_size": 2048,
    "data.positions_per_game": 1,
    "data.num_workers": 3,
    "training.optimizer": "muon",
    "training.learning_rate": 3e-3,
    "training.weight_decay": 0.01,
    "training.grad_clip": 1.0,
    "training.gradient_accumulation_steps": 1,
    "training.num_epochs": 1000,
    "training.log_every_n_steps": 100,
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
    f"input-format-sweep-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")


def run_name(input_mode: str) -> str:
    return f"sweep_inputfmt_{input_mode}"


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


def launch(input_mode: str) -> Path:
    name = run_name(input_mode)
    log_path = LOG_DIR / f"{name}.log"
    if _log_is_complete(log_path):
        print(f"[skip]  {name}: completed previously ({log_path}).")
        return log_path
    if log_path.exists():
        print(f"[redo]  {name}: prior log incomplete, rerunning.")
        log_path.unlink()

    overrides = {**COMMON_OVERRIDES,
                 "model.input_mode": input_mode,
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
           "WANDB_TAGS": "sweep,input-format",
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
    for im in INPUT_MODES:
        r = parse_log(LOG_DIR / f"{run_name(im)}.log")
        rows.append((im, r))
    rows.sort(key=lambda r: -(r[1].get("move_acc", -1.0)))
    print()
    print(f"{'input_mode':<18}  {'step':>6}  {'loss':>7}  {'pol':>6}  {'wdl':>6}  "
          f"{'move_acc':>9}  {'wdl_acc':>8}  {'q_mae':>6}")
    print("-" * 80)
    for im, r in rows:
        if not r:
            print(f"{im:<18}  (no parseable log)")
            continue
        print(f"{im:<18}  {r['step']:>6}  {r['loss']:>7.3f}  "
              f"{r['pol']:>6.3f}  {r['wdl']:>6.3f}  "
              f"{r['move_acc']:>9.3f}  {r['wdl_acc']:>8.3f}  {r['q_mae']:>6.3f}")


def main() -> int:
    for im in INPUT_MODES:
        launch(im)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

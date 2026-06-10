"""Auto-run the king-mobility probe on each new pretrain-v2 checkpoint.

Polls the run dir; for every unseen agent_*.pt runs the frozen probe protocol
(last-token readout) and logs (step, acc, mae) to wandb run 'king-probe' +
appends to agent_data/probe_results.csv.
"""
import csv
import glob
import os
import subprocess
import time

import wandb

RUN_GLOB = "checkpoints/agent_pretrain_v2/*/agent_*.pt"
CSV = "agent_data/probe_results.csv"
POLL_S = 600

os.makedirs("agent_data", exist_ok=True)
seen = set()
if os.path.exists(CSV):
    with open(CSV) as f:
        seen = {r["ckpt"] for r in csv.DictReader(f)}
else:
    with open(CSV, "w") as f:
        csv.writer(f).writerow(["ckpt", "step", "acc", "mae"])

run = wandb.init(project="search-agent", name="king-probe",
                 id="king-probe-v2", resume="allow")

while True:
    cks = [c for c in glob.glob(RUN_GLOB) if not c.endswith("latest_full.pt")]
    cks.sort(key=lambda p: int(p.split("_")[-1].split(".")[0]))
    for ck in cks:
        if ck in seen:
            continue
        import sys
        out = subprocess.run(
            [sys.executable, "scripts/probe_king_moves.py", ck],
            capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}).stdout
        for line in out.splitlines():
            if line.startswith("PROBE step="):
                step = int(line.split("step=")[1].split(":")[0])
                acc = float(line.split("acc=")[1].split()[0])
                mae = float(line.split("mae=")[1].split()[0])
                wandb.log({"probe/king_moves_acc": acc,
                           "probe/king_moves_mae": mae}, step=step)
                with open(CSV, "a") as f:
                    csv.writer(f).writerow([ck, step, acc, mae])
                print(f"probed {ck}: acc={acc} mae={mae}", flush=True)
        seen.add(ck)
    time.sleep(POLL_S)

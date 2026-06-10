"""Stage-A pretraining loop for the search agent.

Streams of packed task examples (tasks.py) -> AgentDecoder -> CE on loss
positions only (head projected on selected positions — never the full
[B,S,31k] logits). Per-task losses/accuracies + exact-board metrics to wandb
(project: search-agent).

Run:  CUDA_VISIBLE_DEVICES=0 uv run python -m chessdecoder.agent.pretrain \
          [chessdecoder/agent/config_stageA.yaml]
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from chessdecoder.agent.model import AgentDecoder
from chessdecoder.agent.tasks import (AgentTaskDataset, build_val_streams,
                                      STREAM_LEN, TASK_MIX)
from chessdecoder.utils.training import load_config
from chessdecoder.utils.muon import build_optimizer

TASK_NAMES = {1: "t1_copy", 2: "t2_apply", 3: "t3_line", 4: "t4_agg",
              5: "t5_distill", 6: "t6_distance"}


def _per_task_metrics(pred: torch.Tensor, tgt: torch.Tensor,
                      task: torch.Tensor, eid: torch.Tensor,
                      losses: torch.Tensor) -> dict:
    """pred/tgt/task/eid/losses are flat tensors over loss positions."""
    out = {}
    correct = (pred == tgt)
    for tid, name in TASK_NAMES.items():
        m = task == tid
        n = int(m.sum())
        if n == 0:
            continue
        out[f"loss/{name}"] = losses[m].mean().item()
        out[f"acc/{name}"] = correct[m].float().mean().item()
    # exact-board: group board-answer positions by example id
    bm = eid >= 0
    if bm.any():
        ids = eid[bm]
        uniq, inv = torch.unique(ids, return_inverse=True)
        ok = torch.ones(len(uniq), dtype=torch.long, device=pred.device)
        ok.scatter_reduce_(0, inv, correct[bm].long(), reduce="amin")
        # split exact-board by task (board answers exist for T1/T2/T3/T6)
        first_of = torch.full((len(uniq),), -1, dtype=torch.long, device=pred.device)
        first_of.scatter_reduce_(0, inv, task[bm].long(), reduce="amax")
        for tid in (1, 2, 3, 6):
            tm = first_of == tid
            if tm.any():
                out[f"board_exact/{TASK_NAMES[tid]}"] = (ok[tm] == 1).float().mean().item()
    return out


def evaluate(model, val_batches, device) -> dict:
    model.eval()
    agg, counts = {}, {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ids, loss_m, task, eid, pos in val_batches:
            ids, pos = ids.to(device), pos.to(device)
            loss_m, task, eid = (loss_m.to(device), task.to(device),
                                 eid.to(device))
            h = model(ids.unsqueeze(0) if ids.dim() == 1 else ids,
                      pos.unsqueeze(0) if pos.dim() == 1 else pos)
            # next-token: hidden at p predicts token at p+1
            tgt_mask = loss_m.clone()
            tgt_mask[..., 0] = False
            sel = tgt_mask.reshape(-1).nonzero(as_tuple=True)[0]
            hf = h.reshape(-1, h.shape[-1])[sel - 1]
            logits = model.logits_at(hf.float())
            tgt = ids.reshape(-1)[sel]
            ce = nn.functional.cross_entropy(logits, tgt, reduction="none")
            m = _per_task_metrics(logits.argmax(-1), tgt,
                                  task.reshape(-1)[sel],
                                  eid.reshape(-1)[sel], ce)
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    model.train()
    return {f"val/{k}": v / counts[k] for k, v in agg.items()}


def train():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "chessdecoder/agent/config_stageA.yaml"
    config = load_config(cfg_path)
    tc, dc, mc = config["training"], config["data"], config["model"]
    device = "cuda"
    torch.manual_seed(tc.get("seed", 42))

    model = AgentDecoder(embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
                         num_layers=mc["num_layers"], d_ff=mc["d_ff"],
                         max_seq_len=mc["max_seq_len"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AgentDecoder: {n_params/1e6:.1f}M params", flush=True)

    ds = AgentTaskDataset(dc["parquet_dir"], dc["label_glob"],
                          seed=tc.get("seed", 42))
    loader = DataLoader(ds, batch_size=dc["batch_size"],
                        num_workers=dc["num_workers"], pin_memory=True,
                        prefetch_factor=2 if dc["num_workers"] else None)

    print("building fixed val streams...", flush=True)
    val_streams = build_val_streams(dc["parquet_dir"], dc["val_labels"],
                                    dc["val_streams"])
    vb = dc["batch_size"]
    val_batches = [tuple(torch.stack([s[j] for s in val_streams[i:i + vb]])
                         for j in range(5))
                   for i in range(0, len(val_streams), vb)]
    print(f"val: {len(val_streams)} streams in {len(val_batches)} batches", flush=True)

    if tc.get("compile", True):
        model = torch.compile(model, dynamic=False)

    optimizer = build_optimizer(model, tc.get("optimizer", "muon"),
                                tc["learning_rate"], tc["weight_decay"])
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    warmup = tc.get("warmup_steps", 500)

    run_dir = os.path.join(tc["checkpoint_dir"],
                           f"{config['run_name']}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    wandb.init(project=config["project_name"], name=config["run_name"],
               config=config)

    step = 0
    resume_from = tc.get("resume_from")
    if resume_from:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        raw = getattr(model, "_orig_mod", model)
        raw.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        else:
            print("resume: no optimizer state in ckpt — momentum rebuilds "
                  "over the first ~100 steps", flush=True)
        step = ck["step"]
        print(f"resumed from {resume_from} at step {step}", flush=True)

    t_win, tok_win = time.time(), 0
    max_steps = tc.get("max_steps", 200_000)
    log_every, val_every = tc["log_every_n_steps"], tc["val_every_n_steps"]
    save_every = tc["save_every_n_steps"]

    for ids, loss_m, task, eid, pos in loader:
        if step >= max_steps:
            break
        ids, pos = ids.to(device, non_blocking=True), pos.to(device, non_blocking=True)
        loss_m = loss_m.to(device, non_blocking=True)
        task, eid = task.to(device, non_blocking=True), eid.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(ids, pos)
        # next-token prediction: hidden at p-1 predicts the token at p.
        tgt_mask = loss_m.clone()
        tgt_mask[:, 0] = False
        sel = tgt_mask.reshape(-1).nonzero(as_tuple=True)[0]
        hf = h.reshape(-1, h.shape[-1])[sel - 1]
        logits = getattr(model, "_orig_mod", model).logits_at(hf.float())
        tgt = ids.reshape(-1)[sel]
        ce = nn.functional.cross_entropy(logits, tgt, reduction="none")
        loss = ce.mean()

        loss.backward()
        # lr warmup
        scale = min(1.0, (step + 1) / warmup)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tok_win += ids.numel()
        if step % log_every == 0:
            with torch.no_grad():
                m = _per_task_metrics(logits.argmax(-1), tgt,
                                      task.reshape(-1)[sel],
                                      eid.reshape(-1)[sel], ce.detach())
            dt = max(time.time() - t_win, 1e-6)
            m.update({"train/loss": loss.item(),
                      "train/tokens_per_s": tok_win / dt,
                      "train/lr_scale": scale, "train/step": step})
            t_win, tok_win = time.time(), 0
            wandb.log(m, step=step)
            print(f"step {step}: loss {loss.item():.4f} "
                  + " ".join(f"{k.split('/')[1]}={v:.3f}"
                             for k, v in m.items() if k.startswith("acc/")),
                  flush=True)

        if step > 0 and step % val_every == 0:
            vm = evaluate(model, val_batches, device)
            wandb.log(vm, step=step)
            print("  VAL " + " ".join(f"{k.split('/', 1)[1]}={v:.3f}"
                                      for k, v in sorted(vm.items())
                                      if "acc" in k or "exact" in k), flush=True)

        if step > 0 and step % save_every == 0:
            raw = getattr(model, "_orig_mod", model)
            torch.save({"model_state_dict": raw.state_dict(), "step": step,
                        "config": config},
                       os.path.join(run_dir, f"agent_{step}.pt"))
            # full state (incl. optimizer) in one rolling file for clean resume
            torch.save({"model_state_dict": raw.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step, "config": config},
                       os.path.join(run_dir, "agent_latest_full.pt"))
            print(f"  saved agent_{step}.pt", flush=True)
        step += 1

    raw = getattr(model, "_orig_mod", model)
    torch.save({"model_state_dict": raw.state_dict(), "step": step,
                "config": config}, os.path.join(run_dir, "agent_final.pt"))
    wandb.finish()


if __name__ == "__main__":
    train()

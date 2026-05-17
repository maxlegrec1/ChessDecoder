"""V2 (ChessEncoderV2) hyperparameter sweep — ONE config per process.

Strict, statistically-relevant HP analysis on ChessFENS at *effective batch
2048*. Primary metric: cross-entropy + top1 on a FIXED held-out shard
(chessfens-00000), identical and paired across every config, so differences
are not confounded by data or eval noise. Stockfish is intentionally NOT used
here (too slow / too high variance for a grid) — it's a downstream check for
the final winner only.

Everything except (lr, weight_decay, seed) is held constant: architecture,
effective batch (2048 = micro_batch x accum), warmup+cosine schedule over the
fixed step budget, grad clip, AMP. Seeds vary init + data shuffling so we can
estimate variance and make a significance claim.

Usage (one config):
  uv run python scripts/v2_hp_sweep.py --lr 1e-4 --wd 0.1 --seed 0 \
      --steps 1500 --val-every 150 --micro-batch 512 --accum 4
"""
import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from chessdecoder.models.v2.encoder_mode import ChessEncoderV2
from chessdecoder.models.vocab import vocab_size, policy_index
from chessdecoder.dataloader.encoder_loader import (
    ChessEncoderDataset, get_encoder_dataloader,
)
from chessdecoder.eval.elo_eval import model_vs_stockfish
from chessdecoder.eval.engine import PytorchModelAdapter


# ---------------------------------------------------------------------------
# Muon optimizer (Keller Jordan): Newton-Schulz-orthogonalized momentum update
# for 2-D hidden weight matrices; AdamW for everything else (embeddings, the
# policy head, latent queries, norms, biases). One Optimizer object with two
# param-group kinds so the existing LambdaLR scales both.
# ---------------------------------------------------------------------------
def _newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class MuonWithAdam(torch.optim.Optimizer):
    def __init__(self, muon_params, adam_params, lr=1e-3, weight_decay=0.0,
                 momentum=0.95, betas=(0.9, 0.95), eps=1e-8):
        groups = [
            dict(params=list(muon_params), kind="muon", lr=lr,
                 weight_decay=weight_decay, momentum=momentum),
            dict(params=list(adam_params), kind="adam", lr=lr,
                 weight_decay=weight_decay, betas=betas, eps=eps),
        ]
        super().__init__(groups, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            if g["kind"] == "muon":
                mom = g["momentum"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p)
                    buf = st["buf"]
                    buf.mul_(mom).add_(p.grad)
                    upd = p.grad.add(buf, alpha=mom)            # Nesterov
                    o = _newtonschulz5(upd)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(o, alpha=-lr * scale)
            else:  # AdamW
                b1, b2 = g["betas"]; eps = g["eps"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    st["step"] += 1
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    bc1 = 1 - b1 ** st["step"]
                    bc2 = 1 - b2 ** st["step"]
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(eps), value=-lr)
        return loss


def build_optimizer(model, name, lr, wd):
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "muon":
        muon_p, adam_p = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_matrix = p.ndim == 2 and not any(
                k in n for k in ("tok_embedding", "policy_head",
                                 "latent_queries"))
            (muon_p if is_matrix else adam_p).append(p)
        return MuonWithAdam(muon_p, adam_p, lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")

OUT_DIR = "/workspace/ChessDecoder/sweep_out"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_val(val_dir, n, device):
    """Fixed, deterministic held-out set (no shuffle): first n valid samples.
    Cached to disk after the first build — identical for every config, so
    every run after the first just loads it (sequential sweep speedup)."""
    cache = os.path.join(OUT_DIR, f"_valcache_n{n}.pt")
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu")
        return (d["ids"].to(device), d["masks"].to(device),
                d["tgts"].to(device))
    ds = ChessEncoderDataset(val_dir, max_seq_len=68, shuffle_files=False,
                             shuffle_positions=False)
    ids, masks, tgts = [], [], []
    for s in ds:
        ids.append(s["input_ids"]); masks.append(s["attention_mask"])
        tgts.append(s["target"])
        if len(ids) >= n:
            break
    ids, masks, tgts = torch.stack(ids), torch.stack(masks), torch.stack(tgts)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"ids": ids, "masks": masks, "tgts": tgts}, cache)
    return ids.to(device), masks.to(device), tgts.to(device)


@torch.no_grad()
def evaluate(model, val, device, vb=2000):
    """Fixed eval batch (vb) so the compiled graph is reused; drops a final
    sub-vb remainder (<=1999 of 200k, deterministic & identical across every
    config, so the paired comparison is unaffected)."""
    model.eval()
    ids, masks, tgts = val
    n_full = (ids.shape[0] // vb) * vb
    tot_loss, tot_n, tot_c1, tot_c3 = 0.0, 0, 0, 0
    lf = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, n_full, vb):
        x, m, t = ids[i:i+vb], masks[i:i+vb], tgts[i:i+vb]
        logits = model(x, attention_mask=m, padded=False).float()
        tot_loss += lf(logits, t).item()
        top3 = logits.topk(3, dim=-1).indices
        tot_c1 += (top3[:, 0] == t).sum().item()
        tot_c3 += (top3 == t.unsqueeze(-1)).any(-1).sum().item()
        tot_n += t.numel()
    model.train()
    return tot_loss / tot_n, tot_c1 / tot_n, tot_c3 / tot_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--val-every", type=int, default=150)
    ap.add_argument("--micro-batch", type=int, default=512)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--train-dir", default="/workspace/ChessDecoder/_cf_train")
    ap.add_argument("--val-dir", default="/workspace/ChessDecoder/_cf_val")
    ap.add_argument("--val-samples", type=int, default=200000)
    ap.add_argument("--sf-every", type=int, default=0)   # 0 = disabled
    ap.add_argument("--sf-games", type=int, default=60)
    ap.add_argument("--group", default="v2-hpsweep")
    args = ap.parse_args()

    eff_bs = args.micro_batch * args.accum
    tag = f"lr{args.lr:g}_{args.optimizer}_wd{args.wd:g}_gc{args.grad_clip:g}"
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, tag + ".csv")
    device = torch.device("cuda")
    set_seed(args.seed)

    raw_model = ChessEncoderV2(
        vocab_size=vocab_size, num_policy_tokens=len(policy_index),
        embed_dim=1024, num_heads=16, num_encoder_layers=10,
        num_decoder_layers=2, num_latents=16, max_seq_len=68, d_ff=1536,
    ).to(device)
    # Static shapes (fixed batch + seq 68) make this ideal for torch.compile.
    # Keep raw_model for Stockfish predict_move (batch-1, avoids compile churn).
    model = torch.compile(raw_model)

    sf_csv = os.path.join(OUT_DIR, tag + "_sf.csv")
    last_elo = 1320.0  # adaptive Stockfish: next match at max(1320,int(last))

    def stockfish_match(stp):
        nonlocal last_elo
        raw_model.eval()
        try:
            mel = max(1320, int(last_elo))
            adapter = PytorchModelAdapter(
                lambda fen, temp: raw_model.predict_move(
                    fen, temperature=temp, force_legal=True))
            wr, est = model_vs_stockfish(
                model=adapter, model1_name=f"{tag}_s{stp}",
                num_games=args.sf_games, temperature=0.0, elo=mel,
                pgn_dir=os.path.join(OUT_DIR, "pgns", tag, f"s{stp}"))
        finally:
            raw_model.train()
        last_elo = est
        wandb.log({"eval/stockfish_winrate": wr,
                   "eval/stockfish_estimated_elo": est,
                   "eval/stockfish_match_elo": mel, "step": stp})
        with open(sf_csv, "a", newline="") as f:
            csv.writer(f).writerow([stp, f"{wr:.4f}", f"{est:.1f}", mel])
        print(f"[{tag}] SF step {stp}: wr={wr:.3f} est_elo={est:.0f} "
              f"(played {mel})", flush=True)

    val = build_val(args.val_dir, args.val_samples, device)
    # Hard guarantee for the unpadded fast-attention path: ChessFENS positions
    # must all be full-length (no padding). Fail loudly if ever violated.
    assert bool(val[1].all()), "val set has padding — unpadded path invalid"
    loader = get_encoder_dataloader(
        args.train_dir, batch_size=args.micro_batch, num_workers=8,
        max_seq_len=68, match_decoder_sampling=False,
    )

    opt = build_optimizer(raw_model, args.optimizer, args.lr, args.wd)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loss_fn = nn.CrossEntropyLoss()

    wandb.init(project="chessencoder-debug", group=args.group, name=tag,
               config=dict(lr=args.lr, wd=args.wd, optimizer=args.optimizer,
                           grad_clip=args.grad_clip, seed=args.seed,
                           eff_bs=eff_bs, steps=args.steps, arch="encoder_v2"),
               reinit=True)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["step", "train_loss", "val_loss", "val_top1", "lr", "elapsed_s"])
    with open(sf_csv, "w", newline="") as f:
        csv.writer(f).writerow(["step", "winrate", "estimated_elo", "match_elo"])

    model.train()
    it = iter(loader)
    t0 = time.time()
    step = 0
    last_train_loss = float("nan")
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(args.accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader); b = next(it)
            # Keep batch shape static for the compiled graph: skip the rare
            # short batch at an iterator/file boundary.
            while b["input_ids"].shape[0] != args.micro_batch:
                try:
                    b = next(it)
                except StopIteration:
                    it = iter(loader); b = next(it)
            x = b["input_ids"].to(device, non_blocking=True)
            m = b["attention_mask"].to(device, non_blocking=True)
            t = b["target"].to(device, non_blocking=True)
            if step == 0:  # one-time guard: training data must be unpadded too
                assert bool(m.all()), "train batch has padding — unpadded path invalid"
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, attention_mask=m, padded=False)
                loss = loss_fn(logits, t) / args.accum
            scaler.scale(loss).backward()
            acc_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        last_train_loss = acc_loss

        if step % args.val_every == 0 or step == args.steps:
            with torch.no_grad():
                lf = logits.float()
                tt3 = lf.topk(3, dim=-1).indices
                tr_top1 = (tt3[:, 0] == t).float().mean().item()
                tr_top3 = (tt3 == t.unsqueeze(-1)).any(-1).float().mean().item()
            vl, vt, vt3 = evaluate(model, val, device)
            el = time.time() - t0
            cur_lr = opt.param_groups[0]["lr"]
            wandb.log({"train/loss": last_train_loss,
                       "train/top1_acc": tr_top1, "train/top3_acc": tr_top3,
                       "val/loss": vl, "val/top1": vt, "val/top3": vt3,
                       "lr": cur_lr, "step": step, "elapsed_s": el})
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step, f"{last_train_loss:.5f}", f"{tr_top1:.5f}",
                     f"{tr_top3:.5f}", f"{vl:.5f}", f"{vt:.5f}", f"{vt3:.5f}",
                     f"{cur_lr:.3e}", f"{el:.0f}"])
            print(f"[{tag}] step {step}/{args.steps} val_loss={vl:.4f} "
                  f"val_top1={vt:.4f} val_top3={vt3:.4f} "
                  f"train_top1={tr_top1:.4f} ({el:.0f}s)", flush=True)

        if args.sf_every and (step % args.sf_every == 0 or step == args.steps):
            stockfish_match(step)

    wandb.finish()
    print(f"[{tag}] DONE", flush=True)


if __name__ == "__main__":
    main()

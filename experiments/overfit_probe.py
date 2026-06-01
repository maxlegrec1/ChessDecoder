#!/usr/bin/env python3
"""Overfit probe: can the model drive train loss -> 0 on a SMALL fixed batch?

A healthy model must memorize a few hundred positions to ~0 policy loss / ~100%
acc within a few hundred steps. Use this to separate "real capacity/optimization
bug" from "data-limited / metric-saturated":
  - if it CANNOT overfit -> a bug (broken grad path, bad targets, precision floor)
  - if a bigger model overfits FASTER/LOWER than a small one -> capacity works
  - run with --fp8 to test whether FP8 precision caps the achievable loss

  CUDA_VISIBLE_DEVICES=0 uv run python experiments/overfit_probe.py CONFIG \
      [--fp8] [--bs 512] [--steps 600]
"""
import argparse, re
import torch, torch.nn as nn

from chessdecoder.dataloader.loader import get_dataloader
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.utils.training import load_config
from chessdecoder.utils.muon import build_optimizer
from chessdecoder.utils.fp8 import convert_model_to_fp8

IGNORE_INDEX = -100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config); mc, tc, dc = cfg["model"], cfg["training"], cfg["data"]
    dev = "cuda"
    model = ChessEncoder(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_layers=mc["num_layers"], d_ff=mc["d_ff"],
        attention_variant=mc.get("attention_variant", "geom"),
        policy_head=mc.get("policy_head", "cross_attn"),
        input_mode=mc.get("input_mode", "lc0_64")).to(dev)
    n = sum(p.numel() for p in model.parameters())/1e6
    if args.fp8:
        convert_model_to_fp8(model, recipe=tc.get("fp8_recipe", "tensorwise"))
    lr = args.lr if args.lr else tc["learning_rate"]
    opt = build_optimizer(model, tc.get("optimizer", "muon"), lr, tc["weight_decay"])

    # ONE fixed batch of valid-policy positions, memorized repeatedly.
    dl, _ = get_dataloader(dc["parquet_dir"], batch_size=args.bs, num_workers=2,
                           positions_per_game=1, cache_dir=dc.get("cache_dir"),
                           split="train", val_pct=tc.get("val_pct", 2))
    b = next(iter(dl))
    bid = b["board_ids"].reshape(-1, 68).to(dev)
    tgt = b["policy_tgt"].reshape(-1).to(dev)
    val = b["policy_valid"].reshape(-1).to(dev)
    tgt = torch.where(val, tgt, torch.full_like(tgt, IGNORE_INDEX))
    nvalid = int(val.sum())
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    print(f"model {n:.0f}M params | fp8={args.fp8} | lr={lr} | "
          f"fixed batch {bid.shape[0]} pos ({nvalid} valid) | overfitting {args.steps} steps")
    print(f"  {'step':>5} | {'pol_loss':>9} | {'train_acc':>9}")
    for s in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(bid)
            loss = ce(out["policy"].reshape(-1, move_vocab_size), tgt)
        if s % 50 == 0:
            with torch.no_grad():
                pred = out["policy"].reshape(-1, move_vocab_size).argmax(-1)
                acc = ((pred == tgt) & val).sum().float() / nvalid
            print(f"  {s:>5} | {loss.item():>9.4f} | {acc.item():>9.4f}")
        loss.backward(); opt.step()


if __name__ == "__main__":
    main()

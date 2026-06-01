#!/usr/bin/env python3
"""Gradient-flow probe: is the loss signal reaching every layer / head / embedding?

Builds the model from a training config, runs a few real fwd/bwd passes, and
reports grad-norm RMS grouped by module class and by encoder-layer depth. Use it
to spot vanishing/exploding gradients, dead modules, or a depth imbalance that
would cap loss/step.

Uses POLICY cross-entropy only (the dominant loss term, weight 5.0) — it
backprops through the whole encoder, the policy head, and the token/pos
embeddings, which is exactly the path a grad-flow probe needs. (The wdl head is
a small separate branch; omitting it keeps this self-contained and faithful to
the backbone gradient.) Run between training jobs:

  CUDA_VISIBLE_DEVICES=0 uv run python experiments/grad_flow.py \
      chessdecoder/train/config_bt4_baseline.yaml [--steps 4] [--ckpt PATH]
"""
import argparse
import re

import torch
import torch.nn as nn

from chessdecoder.dataloader.loader import get_dataloader
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.utils.training import load_config

IGNORE_INDEX = -100


def classify(name):
    if "tok_embedding" in name or "pos_embedding" in name:
        return "embedding"
    if "policy_head" in name:
        return "policy_head"
    if "wdl_head" in name:
        return "wdl_head"
    if "bias_module" in name:
        return "attn_bias"
    if "router" in name:
        return "moe_router"
    if "norm" in name:
        return "norm"
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "out_proj")):
        return "attn"
    if "mlp" in name:
        return "ffn"
    return "other"


def layer_idx(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def rms(norms):
    return (sum(n * n for n in norms) / len(norms)) ** 0.5 if norms else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bs", type=int, default=128,
                    help="probe batch (small; grad-flow is batch-size-independent). "
                         "MoE in eager (no fp8/compile) needs a small batch to fit.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mc, tc, dc = cfg["model"], cfg["training"], cfg["data"]
    dev = "cuda"

    model = ChessEncoder(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_layers=mc["num_layers"], d_ff=mc["d_ff"],
        attention_variant=mc.get("attention_variant", "geom"),
        policy_head=mc.get("policy_head", "cross_attn"),
        input_mode=mc.get("input_mode", "lc0_64"),
        ffn_type=mc.get("ffn_type", "dense"),
        moe_num_experts=mc.get("moe_num_experts", 8),
        moe_top_k=mc.get("moe_top_k", 2),
        moe_expert_d_ff=mc.get("moe_expert_d_ff", 768),
        moe_aux_loss_weight=mc.get("moe_aux_loss_weight", 1e-2),
        moe_router_noise=mc.get("moe_router_noise", 0.0),
    ).to(dev)
    if args.ckpt:
        # our own checkpoints; MoE ones embed the fp8 ScaledGroupedMMTensor
        # subclass, so weights_only=False is required (same as train.py resume).
        sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
        sd = sd.get("model_state_dict", sd)
        # subclass weights -> plain tensors for this bf16 probe model
        sd = {k: (v.to_local() if hasattr(v, "to_local") else
                  getattr(v, "_data", v)) for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"loaded {args.ckpt}")
    model.train()

    dl, _ = get_dataloader(dc["parquet_dir"], batch_size=args.bs,
                           num_workers=2, positions_per_game=dc.get("positions_per_game", 1),
                           cache_dir=dc.get("cache_dir"), split="train",
                           val_pct=tc.get("val_pct", 2))
    it = iter(dl)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    acc = {}  # param name -> [grad norms]
    nl = mc["num_layers"]
    for _ in range(args.steps):
        batch = next(it)
        bid = batch["board_ids"].to(dev)              # [B,N,68]
        B, N, _ = bid.shape
        pol_tgt = batch["policy_tgt"].to(dev)
        pol_val = batch["policy_valid"].to(dev)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(bid.reshape(B * N, 68))
            pol_logits = out["policy"].reshape(B, N, -1)
            tgt = torch.where(pol_val, pol_tgt,
                              torch.full_like(pol_tgt, IGNORE_INDEX))
            loss = ce(pol_logits.reshape(-1, move_vocab_size), tgt.reshape(-1))
        aux = model.moe_aux_loss()
        if aux is not None:
            loss = loss + aux
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                acc.setdefault(n, []).append(p.grad.detach().float().norm().item())

    groups = {}
    for n, norms in acc.items():
        groups.setdefault(classify(n), []).extend(norms)
    print(f"\n=== grad-norm RMS by group  ({args.config}, {args.steps} batches) ===")
    for g in sorted(groups, key=lambda k: -rms(groups[k])):
        print(f"  {g:12s}  rms_gradnorm = {rms(groups[g]):.3e}   ({len(groups[g])} params)")

    print("\n=== per-depth profile (RMS grad norm) ===")
    print(f"  {'layer':>5} | {'attn':>10} | {'ffn':>10}")
    for li in range(nl):
        a = rms([v for n, vs in acc.items() if layer_idx(n) == li
                 and classify(n) == "attn" for v in vs])
        f = rms([v for n, vs in acc.items() if layer_idx(n) == li
                 and classify(n) == "ffn" for v in vs])
        print(f"  {li:>5} | {a:>10.3e} | {f:>10.3e}")


if __name__ == "__main__":
    main()

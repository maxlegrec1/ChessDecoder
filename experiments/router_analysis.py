#!/usr/bin/env python3
"""MoE router probe: is the router actually leveraging all experts, and are its
logits in a healthy range (z-loss / router-LR diagnostics)?

Loads a trained MoE checkpoint, runs a real batch, and for every MoE layer reports:
  - router LOGIT scale: mean|logit|, max|logit|, and logsumexp (the quantity a
    z-loss penalizes). Large/growing logits -> over-confident router -> z-loss helps.
  - routing CONFIDENCE: mean top-1 softmax prob and entropy (low entropy = peaky).
  - EXPERT UTILIZATION: fraction of tokens whose top-1 (and top-k) is each expert,
    plus #dead experts (never selected) and a balance ratio (max/min load).
  - router WEIGHT norm.

  CUDA_VISIBLE_DEVICES=0 uv run python experiments/router_analysis.py CONFIG CKPT [--bs 256]
"""
import argparse, math
import torch
import torch.nn.functional as F

from chessdecoder.dataloader.loader import get_dataloader
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size
from chessdecoder.utils.training import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("ckpt")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config); mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]
    dev = "cuda"
    E = mc.get("moe_num_experts", 8); topk = mc.get("moe_top_k", 2)
    model = ChessEncoder(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_layers=mc["num_layers"], d_ff=mc["d_ff"],
        attention_variant=mc.get("attention_variant", "geom"),
        policy_head=mc.get("policy_head", "cross_attn"),
        input_mode=mc.get("input_mode", "lc0_64"), ffn_type="moe",
        moe_num_experts=E, moe_top_k=topk, moe_expert_d_ff=mc.get("moe_expert_d_ff", 768),
    ).to(dev)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): (v.to_local() if hasattr(v, "to_local")
          else getattr(v, "_data", v)) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt}  (missing {len(missing)}, unexpected {len(unexpected)})")
    model.eval()

    # capture each router's logits via forward hook on the router Linear
    logits_by_layer = {}
    routers = []
    for name, mod in model.named_modules():
        if name.endswith("mlp.router"):
            li = int(name.split("layers.")[1].split(".")[0])
            routers.append((li, name, mod))
            mod.register_forward_hook(
                lambda m, i, o, li=li: logits_by_layer.__setitem__(li, o.detach().float()))

    dl, _ = get_dataloader(dc["parquet_dir"], batch_size=args.bs, num_workers=2,
                           positions_per_game=1, cache_dir=dc.get("cache_dir"),
                           split="train", val_pct=tc.get("val_pct", 2))
    b = next(iter(dl))
    with torch.no_grad():
        model(b["board_ids"].reshape(-1, 68).to(dev))

    print(f"\nE={E} top_k={topk}  (uniform top-1 load = {1/E:.3f})")
    print(f"{'L':>2} | {'mean|lgt|':>9} {'max|lgt|':>8} {'lse':>6} | "
          f"{'top1prob':>8} {'entropy':>7}(max {math.log(E):.2f}) | "
          f"{'dead':>4} {'bal(max/min)':>12} | {'wnorm':>6}")
    for li in sorted(logits_by_layer):
        lg = logits_by_layer[li]                          # [T, E]
        probs = lg.softmax(-1)
        top1p, top1i = probs.max(-1)
        ent = -(probs * (probs + 1e-9).log()).sum(-1).mean().item()
        lse = torch.logsumexp(lg, -1).mean().item()
        # top-k utilization (fraction of tokens whose top-k includes each expert)
        topk_i = probs.topk(topk, -1).indices                      # [T,k]
        load = F.one_hot(top1i, E).float().mean(0)                 # top-1 load per expert
        dead = int((load < 1e-4).sum())
        nz = load[load > 1e-4]
        bal = (nz.max() / nz.min()).item() if len(nz) else float("inf")
        wnorm = dict(model.named_parameters())[
            [n for n,_ in model.named_parameters() if n.endswith(f"layers.{li}.mlp.router.weight")][0]
        ].norm().item()
        print(f"{li:>2} | {lg.abs().mean():>9.3f} {lg.abs().max():>8.2f} {lse:>6.2f} | "
              f"{top1p.mean():>8.3f} {ent:>7.3f}{'':9} | {dead:>4} {bal:>12.2f} | {wnorm:>6.2f}")
    # show one layer's full expert load vector
    li0 = sorted(logits_by_layer)[len(logits_by_layer)//2]
    load = F.one_hot(logits_by_layer[li0].softmax(-1).argmax(-1), E).float().mean(0)
    print(f"\nlayer {li0} top-1 expert load: " + " ".join(f"{x:.3f}" for x in load.tolist()))


if __name__ == "__main__":
    main()

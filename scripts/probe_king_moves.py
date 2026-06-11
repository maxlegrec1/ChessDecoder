"""Quality-suite probe: frozen trunk -> linear head -> # legal king moves.

Protocol (fixed, comparable across checkpoints):
  - positions: 60k train / 10k test from the HELD-OUT shard, seeded sample
  - label: count of legal moves from the stm king's square (castling included
    via python-chess from_square semantics), clamped to 0..10 (11 classes)
  - features: frozen AgentDecoder hidden at the last board token of
    [<root>, b19] (the ep slot)
  - head: single nn.Linear(768, 11), AdamW 1e-3, 5 epochs, batch 1024
  Report: test acc, MAE, majority-class baseline.

  CUDA_VISIBLE_DEVICES=0 uv run python scripts/probe_king_moves.py CKPT [CKPT2 ...]
"""
import glob
import sys

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from chessdecoder.agent.model import AgentDecoder
from chessdecoder.agent import patch_vocab as pv

DEV = "cuda"
import os
POOL = os.environ.get("PROBE_POOL", "last")
N_TRAIN, N_TEST = 60_000, 10_000
N_CLASSES = 11

VAL_SHARD = sorted(glob.glob("/mnt/2tb_2/decoder/parquet_files_decoder/*.parquet"))[-1]


def king_moves(board: chess.Board) -> int:
    k = board.king(board.turn)
    return min(sum(1 for m in board.legal_moves if m.from_square == k),
               N_CLASSES - 1)


def build_dataset():
    df = pd.read_parquet(VAL_SHARD, columns=["fen"])
    fens = df["fen"].sample(n=N_TRAIN + N_TEST + 2000, random_state=7).tolist()
    ids, ys = [], []
    for fen in fens:
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        ids.append([pv.ROOT] + pv.encode_board(b))
        ys.append(king_moves(b))
        if len(ids) >= N_TRAIN + N_TEST:
            break
    return (torch.tensor(ids[:N_TRAIN + N_TEST]),
            torch.tensor(ys[:N_TRAIN + N_TEST]))


@torch.no_grad()
def extract(model, ids):
    feats = []
    for i in range(0, len(ids), 256):
        chunk = ids[i:i + 256].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(chunk)
        if POOL == "mean":
            feats.append(h[:, 1:, :].mean(dim=1).float().cpu())
        else:
            feats.append(h[:, -1, :].float().cpu())
    return torch.cat(feats)


def probe(feats, ys):
    torch.manual_seed(0)
    Xtr, Xte = feats[:N_TRAIN].to(DEV), feats[N_TRAIN:].to(DEV)
    ytr, yte = ys[:N_TRAIN].to(DEV), ys[N_TRAIN:].to(DEV)
    head = nn.Linear(feats.shape[1], N_CLASSES).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(5):
        perm = torch.randperm(len(Xtr), device=DEV)
        for j in range(0, len(Xtr), 1024):
            sel = perm[j:j + 1024]
            loss = nn.functional.cross_entropy(head(Xtr[sel]), ytr[sel])
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        pred = head(Xte).argmax(-1)
    acc = (pred == yte).float().mean().item()
    mae = (pred - yte).abs().float().mean().item()
    return acc, mae


CACHE = "agent_data/probe_dataset_kingmoves.pt"


def main():
    import os
    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        ids, ys = d["ids"], d["ys"]
        print(f"probe dataset from cache ({len(ids):,})", flush=True)
    else:
        print("building probe dataset (held-out shard)...", flush=True)
        ids, ys = build_dataset()
        torch.save({"ids": ids, "ys": ys}, CACHE)
    maj = ys[N_TRAIN:].bincount().max().item() / N_TEST
    print(f"labels: dist={ys.bincount().tolist()}  majority-baseline={maj:.3f}",
          flush=True)
    for ckpt in sys.argv[1:]:
        sd = torch.load(ckpt, map_location=DEV, weights_only=False)
        step = sd.get("step", "?")
        v = sd["model_state_dict"]["tok_embedding.weight"].shape[0]
        model = AgentDecoder(vocab_size=v).to(DEV).eval()
        model.load_state_dict(sd["model_state_dict"])
        feats = extract(model, ids)
        acc, mae = probe(feats, ys)
        print(f"PROBE step={step}: king-moves acc={acc:.3f} mae={mae:.2f} "
              f"(majority {maj:.3f})", flush=True)
    # random-init control
    torch.manual_seed(123)
    model = AgentDecoder().to(DEV).eval()
    feats = extract(model, ids)
    acc, mae = probe(feats, ys)
    print(f"PROBE random-init control: acc={acc:.3f} mae={mae:.2f}", flush=True)


if __name__ == "__main__":
    main()

"""Generate the T5 distillation label corpus: fen -> oracle (q_bin, d_bin, top4).

Samples positions across the parquet shards (held-out shard excluded -> used
for val labels), labels them with the frozen oracle, writes parquet files:
  agent_data/t5_labels_train_*.parquet   (fen, q_bin, d_bin, m1..m4)
  agent_data/t5_labels_val.parquet

Exact top-4-legal fast path: top-48 raw logits on GPU, legality-check only
those 48 ucis. If >=4 legal are found among them, they are provably the true
top-4 legal (any legal move outside the top-48 has a lower logit than all 48).
Full legal scan as fallback (<4 found).

  CUDA_VISIBLE_DEVICES=0 uv run python scripts/gen_t5_labels.py [N_TRAIN=5000000]
"""
import glob
import os
import random
import sys

import chess
import numpy as np
import pandas as pd
import torch

from chessdecoder.agent.oracle import Oracle
from chessdecoder.agent import patch_vocab as pv
from chessdecoder.models.vocab import move_vocab, move_token_to_idx

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
N_VAL = 100_000
BATCH = 4096
OUT_DIR = "agent_data"
PARQUET_DIR = "/mnt/2tb_2/decoder/parquet_files_decoder"

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(0)

shards = sorted(glob.glob(f"{PARQUET_DIR}/*.parquet"))
VAL_SHARD = shards[-1]
train_shards = shards[:-1]
random.shuffle(train_shards)

oracle = Oracle()
ID2UCI = move_vocab  # raw policy index -> uci string


def top4_legal(board: chess.Board, logits_row: torch.Tensor,
               topk_ids: list[int]) -> list[int]:
    """Exact top-4 legal moves (agent-vocab ids) via the top-48 fast path."""
    found = []
    seen_moves = set()
    for idx in topk_ids:
        uci = ID2UCI[idx]
        try:
            mv = chess.Move.from_uci(uci)
        except chess.InvalidMoveError:
            continue
        if not board.is_legal(mv):
            # lc0 castling spelling: king-takes-rook (e1h1). Translate.
            if uci in ("e1h1", "e1a1", "e8h8", "e8a8"):
                std = {"e1h1": "e1g1", "e1a1": "e1c1",
                       "e8h8": "e8g8", "e8a8": "e8c8"}[uci]
                mv2 = chess.Move.from_uci(std)
                if board.is_legal(mv2) and mv2 not in seen_moves:
                    seen_moves.add(mv2)
                    found.append(pv.MOVE_BASE + idx)
                    if len(found) == 4:
                        return found
            continue
        if mv in seen_moves:
            continue
        seen_moves.add(mv)
        found.append(pv.MOVE_BASE + idx)
        if len(found) == 4:
            return found
    # fallback: full scan (also covers <4-legal positions)
    full = oracle._top_moves(board, logits_row)
    return full + [full[-1]] * (4 - len(full)) if full else []


def label_fens(fens: list[str]) -> list[tuple]:
    rows = []
    for i in range(0, len(fens), BATCH):
        chunk = fens[i:i + BATCH]
        pol, wdl = oracle._forward(chunk)
        q = (wdl[:, 0] - wdl[:, 2]).tolist()
        d = wdl[:, 1].tolist()
        topk = torch.topk(pol, 48, dim=-1).indices.tolist()
        for j, fen in enumerate(chunk):
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            moves = top4_legal(board, pol[j], topk[j])
            if len(moves) < 4:
                continue
            rows.append((fen, pv.q_to_bin(q[j]), pv.d_to_bin(d[j]), *moves))
        if (i // BATCH) % 50 == 0:
            print(f"  labeled {len(rows):,}", flush=True)
    return rows


def sample_and_label(shard_list, n_target, out_prefix):
    rows, file_i = [], 0
    per_shard = max(50_000, n_target // max(1, len(shard_list)) + 1)
    for sp in shard_list:
        if sum(1 for _ in ()) or len(rows) >= n_target:
            break
        df = pd.read_parquet(sp, columns=["fen"])
        fens = df["fen"].sample(n=min(per_shard, len(df)), random_state=0).tolist()
        rows.extend(label_fens(fens))
        print(f"shard {os.path.basename(sp)}: total {len(rows):,}/{n_target:,}",
              flush=True)
        if len(rows) >= 1_000_000 or len(rows) >= n_target:
            out = pd.DataFrame(rows[:n_target] if len(rows) >= n_target else rows,
                               columns=["fen", "q_bin", "d_bin",
                                        "m1", "m2", "m3", "m4"])
            path = f"{OUT_DIR}/{out_prefix}_{file_i:03d}.parquet"
            out.to_parquet(path)
            print(f"wrote {path} ({len(out):,} rows)", flush=True)
            n_target -= len(out)
            rows, file_i = [], file_i + 1
            if n_target <= 0:
                return
    if rows:
        out = pd.DataFrame(rows, columns=["fen", "q_bin", "d_bin",
                                          "m1", "m2", "m3", "m4"])
        out.to_parquet(f"{OUT_DIR}/{out_prefix}_{file_i:03d}.parquet")
        print(f"wrote final {len(out):,} rows", flush=True)


# ---------------------------------------------------------------------------
# Paired corpus (for t14_swing): consecutive game positions, both labeled.
# Run:  CUDA_VISIBLE_DEVICES=0 uv run python scripts/gen_t5_labels.py --paired [N]
# ---------------------------------------------------------------------------

def gen_paired(n_target: int, out_prefix: str, shard_list):
    rows = []
    for sp in shard_list:
        df = pd.read_parquet(sp, columns=["fen", "played_move", "game_id"])
        # consecutive rows within a game = (parent, move, child)
        fens, mvs, gids = (df["fen"].tolist(), df["played_move"].tolist(),
                          df["game_id"].tolist())
        cand = [(fens[i], mvs[i], fens[i + 1])
                for i in range(len(fens) - 1) if gids[i] == gids[i + 1]]
        random.shuffle(cand)
        cand = cand[:min(len(cand), n_target - len(rows) + 1000)]
        parents = [c[0] for c in cand]
        children = [c[2] for c in cand]
        lp = {}
        for i in range(0, len(parents), BATCH):
            _, wdl = oracle._forward(parents[i:i + BATCH])
            q = (wdl[:, 0] - wdl[:, 2]).tolist(); d = wdl[:, 1].tolist()
            for j in range(len(q)):
                lp[i + j] = (q[j], d[j])
        lc = {}
        for i in range(0, len(children), BATCH):
            _, wdl = oracle._forward(children[i:i + BATCH])
            q = (wdl[:, 0] - wdl[:, 2]).tolist(); d = wdl[:, 1].tolist()
            for j in range(len(q)):
                lc[i + j] = (q[j], d[j])
        for i, (pf, mu, cf) in enumerate(cand):
            qp, dp = lp[i]; qc, dc = lc[i]
            rows.append((pf, pv.q_to_bin(qp), pv.d_to_bin(dp), mu,
                         cf, pv.q_to_bin(qc), pv.d_to_bin(dc)))
        print(f"paired: {len(rows):,}/{n_target:,}", flush=True)
        if len(rows) >= n_target:
            break
    out = pd.DataFrame(rows[:n_target], columns=[
        "parent_fen", "q_p", "d_p", "move", "child_fen", "q_c", "d_c"])
    out.to_parquet(f"{OUT_DIR}/{out_prefix}.parquet")
    print(f"wrote {out_prefix} ({len(out):,})", flush=True)


if "--paired" in sys.argv:
    n = int(sys.argv[sys.argv.index("--paired") + 1]) \
        if len(sys.argv) > sys.argv.index("--paired") + 1 else 1_000_000
    gen_paired(50_000, "paired_labels_val", [VAL_SHARD])
    gen_paired(n, "paired_labels_train_000", train_shards)
    sys.exit(0)


print(f"labeling {N_VAL:,} val positions from {os.path.basename(VAL_SHARD)}",
      flush=True)
df = pd.read_parquet(VAL_SHARD, columns=["fen"])
val_rows = label_fens(df["fen"].sample(n=N_VAL, random_state=0).tolist())
pd.DataFrame(val_rows, columns=["fen", "q_bin", "d_bin", "m1", "m2", "m3", "m4"]
             ).to_parquet(f"{OUT_DIR}/t5_labels_val.parquet")
print(f"val done: {len(val_rows):,} rows", flush=True)

print(f"labeling {N_TRAIN:,} train positions", flush=True)
sample_and_label(train_shards, N_TRAIN, "t5_labels_train")
print("DONE", flush=True)

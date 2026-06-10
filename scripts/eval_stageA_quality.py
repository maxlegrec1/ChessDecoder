"""Stage-A pretrain quality probes (generation-mode, not teacher-forced).

A) world model under generation: greedy-decode the 19 board tokens after
   <root> b <line> m1..mk <probe>; exact-board rate, incl. depth
   extrapolation beyond the trained k<=6.
B) K=0 oracle agreement: generate (q,d,m1) after <root> b <oracle>;
   top-1 move agreement with the real oracle + q-bin MAE.

  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_stageA_quality.py CKPT [N=500]
"""
import sys

import chess
import pandas as pd
import torch

from chessdecoder.agent.model import AgentDecoder
from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.grammar import board_slot_mask
from chessdecoder.agent.tasks import apply_uci

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
DEV = "cuda"

model = AgentDecoder().to(DEV).eval()
sd = torch.load(CKPT, map_location=DEV, weights_only=False)
model.load_state_dict(sd["model_state_dict"])
print(f"loaded {CKPT} (step {sd.get('step')})", flush=True)

df = pd.read_parquet("agent_data/t5_labels_val.parquet")


@torch.no_grad()
def gen_tokens(prompts: list[list[int]], n_new: int, slot_masks=None) -> torch.Tensor:
    """Greedy decode n_new tokens for equal-length prompts (no KV cache —
    contexts are tiny). slot_masks: optional per-step bool mask [V]."""
    ids = torch.tensor(prompts, device=DEV)
    for j in range(n_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(ids)
        logits = model.logits_at(h[:, -1, :].float())
        if slot_masks is not None:
            logits = logits.masked_fill(~slot_masks[j].to(DEV), -1e30)
        ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
    return ids[:, -n_new:]


# ---------- A) generation-mode world model -------------------------------
board_masks = [board_slot_mask(i) for i in range(19)]
print("\nA) world model, generation mode (greedy, grammar-masked):")
for k in (1, 3, 6, 8, 10):
    prompts, targets = [], []
    rng = torch.Generator().manual_seed(k)
    fens = df["fen"].sample(n=N, random_state=k).tolist()
    for fen in fens:
        board = chess.Board(fen)
        work = board.copy()
        ucis = []
        ok = True
        for _ in range(k):
            legal = list(work.legal_moves)
            if not legal:
                ok = False
                break
            mv = legal[torch.randint(len(legal), (1,), generator=rng).item()]
            u = mv.uci()
            if u not in pv.MOVE_TO_ID:
                ok = False
                break
            ucis.append(u)
            work.push(mv)
        if not ok:
            continue
        prompts.append([pv.ROOT] + pv.encode_board(board) + [pv.LINE]
                       + [pv.MOVE_TO_ID[u] for u in ucis] + [pv.PROBE])
        targets.append(pv.encode_board(work))
        if len(prompts) >= 256:
            break
    out = gen_tokens(prompts, 19, board_masks)
    tgt = torch.tensor(targets, device=DEV)
    exact = (out == tgt).all(dim=1).float().mean().item()
    tok = (out == tgt).float().mean().item()
    tag = " (EXTRAPOLATION)" if k > 6 else ""
    print(f"  k={k:2d}: exact-board {exact:.3f}  per-token {tok:.4f}  "
          f"n={len(prompts)}{tag}", flush=True)

# ---------- B) K=0 oracle agreement --------------------------------------
print("\nB) K=0 generative oracle agreement:")
sample = df.sample(n=min(N * 4, len(df)), random_state=0)
prompts, truth = [], []
for fen, q, d, m1, *_ in sample.itertuples(index=False):
    prompts.append([pv.ROOT] + pv.encode_board(chess.Board(fen)) + [pv.ORACLE])
    truth.append((int(q), int(d), int(m1)))
qmask = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool); qmask[pv.QBIN_BASE:pv.QBIN_BASE+pv.N_QBIN] = True
dmask = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool); dmask[pv.DBIN_BASE:pv.DBIN_BASE+pv.N_DBIN] = True
mmask = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool); mmask[pv.MOVE_BASE:pv.MOVE_BASE+pv.N_MOVE] = True
agree, qerr, n = 0, 0.0, 0
for i in range(0, len(prompts), 256):
    chunk = prompts[i:i+256]
    out = gen_tokens(chunk, 3, [qmask, dmask, mmask])
    for j in range(len(chunk)):
        tq, td, tm = truth[i+j]
        qerr += abs((out[j, 0].item() - pv.QBIN_BASE) - tq)
        agree += int(out[j, 2].item() == tm)
        n += 1
print(f"  top-1 move agreement with oracle: {agree/n:.3f}  (n={n})")
print(f"  q-bin MAE (model's own q vs oracle): {qerr/n:.1f}/128")

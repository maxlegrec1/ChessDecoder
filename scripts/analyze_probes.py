"""What does the agent actually probe? Decode greedy-episode probes on
held-out roots and classify each: child-of-root (and of which root move —
oracle top-4 or other), 2-ply descendant, deeper/unreachable counterfactual,
piece distance from root. The emergent-behavior readout from the plan
(§ diagnostics): 'if probe-children-of-plausible-moves emerges from total
freedom, tree search was rediscovered'.

  uv run python scripts/analyze_probes.py <ckpt> [n_roots] [K]
"""
import sys
from collections import Counter

import chess
import pandas as pd
import torch

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.oracle import Oracle
from chessdecoder.agent.rl.engine import RolloutEngine
from chessdecoder.agent.rl.episodes import PREFIX_LEN, PROBE_TOKENS
from chessdecoder.agent.rl.rollout_proc import load_model


def classify(root: chess.Board, b: chess.Board, top4: set[str]) -> str:
    key = b.fen().rsplit(" ", 2)[0]
    for mv in root.legal_moves:
        root.push(mv)
        k1 = root.fen().rsplit(" ", 2)[0]
        if k1 == key:
            root.pop()
            return ("child_top4" if mv.uci() in top4 else "child_other")
        root.pop()
    # 2-ply scan (bounded)
    for mv in list(root.legal_moves):
        root.push(mv)
        for mv2 in list(root.legal_moves):
            root.push(mv2)
            if root.fen().rsplit(" ", 2)[0] == key:
                root.pop(); root.pop()
                return "grandchild"
            root.pop()
        root.pop()
    return "other"          # deeper, transposed, or counterfactual


def piece_distance(a: chess.Board, b: chess.Board) -> int:
    d = 0
    for sq in chess.SQUARES:
        if a.piece_at(sq) != b.piece_at(sq):
            d += 1
    return d


def main():
    ckpt, n_roots, K = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 256, \
        int(sys.argv[3]) if len(sys.argv) > 3 else 8
    suites = pd.read_parquet("agent_data/eval_suites.parquet")
    fens = suites[suites.suite == "sensitive"].fen.head(n_roots).tolist()
    model = load_model(ckpt)
    oracle = Oracle()
    B = 128
    eng = RolloutEngine(model, oracle, batch_size=B, k_budget=K,
                        dtype=torch.bfloat16)
    cats, dists, probes_used, invalid = Counter(), [], [], 0
    for i in range(0, len(fens), B):
        chunk = [chess.Board(f) for f in fens[i:i + B]]
        pad = B - len(chunk)
        eps = eng.rollout(chunk + [chunk[-1]] * pad, greedy=True)
        for root, e in zip(chunk, eps[:len(chunk)]):
            reply_top4 = set()
            r = oracle.query(root)
            from chessdecoder.agent.rl.reward import move_id_to_uci
            for mid in r.top_moves:
                u = move_id_to_uci(root, mid)
                if u:
                    reply_top4.add(u)
            probes_used.append(e.probes_valid + e.probes_invalid)
            invalid += e.probes_invalid
            j = PREFIX_LEN
            ids = e.ids
            while j < len(ids):
                t = ids[j]
                if t == pv.PROBE:
                    slots = ids[j + 1:j + 1 + pv.BOARD_LEN]
                    j += PROBE_TOKENS + 8
                    try:
                        b = pv.decode_board(slots)
                    except Exception:
                        b = None
                    if b is None or not b.is_valid():
                        cats["invalid"] += 1
                        continue
                    cats[classify(root, b, reply_top4)] += 1
                    dists.append(piece_distance(root, b))
                elif t == pv.ANSWER:
                    j += 2
                else:
                    j += 1
    total = sum(cats.values())
    print(f"roots {len(fens)}  K={K}  probes/ep mean "
          f"{sum(probes_used)/len(probes_used):.2f}")
    for c, n in cats.most_common():
        print(f"  {c:12s} {n:5d}  {n/total:.1%}")
    if dists:
        s = pd.Series(dists)
        print(f"piece distance: p50 {s.median():.0f}  p90 "
              f"{s.quantile(0.9):.0f}  mean {s.mean():.1f}")


if __name__ == "__main__":
    main()

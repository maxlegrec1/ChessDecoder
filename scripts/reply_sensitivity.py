"""Reply-conditioning diagnostic: fraction of final moves that CHANGE when
probe replies are replaced with random garbage. High = the answer policy
conditions on probe information; low = probes are ignored.

Reference points (sensitive suite, K=4 forced, greedy):
  pretrain base agent_35000 : 57.8% change
  grpo @31k (pre-audit-fix) : 14.8% change   (RL had unlearned conditioning)

  uv run python scripts/reply_sensitivity.py CKPT [n=128] [K=4]
"""
import random
import sys

import chess
import pandas as pd
import torch

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.oracle import Oracle, Reply
from chessdecoder.agent.rl.engine import RolloutEngine
from chessdecoder.agent.rl.reward import move_id_to_uci
from chessdecoder.agent.rl.rollout_proc import load_model


class CorruptOracle:
    """Real oracle for the prefix (first call), random garbage for probes."""

    def __init__(self, real):
        self.real = real
        self.rng = random.Random(7)
        self.first_call = True

    def query_batch(self, boards):
        if self.first_call:
            self.first_call = False
            return self.real.query_batch(boards)
        out = []
        for b in boards:
            legal = [pv.MOVE_TO_ID[k] for mv in b.legal_moves
                     for k in pv.move_keys(b, mv) if k in pv.MOVE_TO_ID][:8]
            self.rng.shuffle(legal)
            moves = (legal + legal)[:4] if legal else []
            out.append(Reply(q_bin=self.rng.randrange(128),
                             d_bin=self.rng.randrange(32), top_moves=moves))
        return out


def main():
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    suites = pd.read_parquet("agent_data/eval_suites.parquet")
    roots = [chess.Board(f)
             for f in suites[suites.suite == "sensitive"].fen.head(n)]
    model = load_model(ckpt)
    real = Oracle()
    B = len(roots)
    e_true = RolloutEngine(model, real, batch_size=B, k_budget=k,
                           dtype=torch.bfloat16).rollout(
        [b.copy() for b in roots], k_budgets=[k] * B,
        min_probes=[k] * B, greedy=True)
    e_corr = RolloutEngine(model, CorruptOracle(real), batch_size=B,
                           k_budget=k, dtype=torch.bfloat16).rollout(
        [b.copy() for b in roots], k_budgets=[k] * B,
        min_probes=[k] * B, greedy=True)
    changed = sum(move_id_to_uci(b, a.final_move)
                  != move_id_to_uci(b, c.final_move)
                  for b, a, c in zip(roots, e_true, e_corr))
    print(f"REPLY_SENSITIVITY {ckpt}: move changed under corrupted replies "
          f"{changed}/{len(roots)} ({changed/len(roots):.1%})")


if __name__ == "__main__":
    main()

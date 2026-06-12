"""Head-to-head: agent (K-probe episodes per move, greedy) vs oracle-greedy.

The conclusive L1 measurement: paired games (each start position played with
both colors) from held-out eval-suite positions, all games advanced in
lockstep so the batched RolloutEngine serves every agent move in parallel.
Adjudication at the move cap by oracle eval.

  uv run python scripts/h2h_agent_vs_oracle.py CKPT [n_positions=100] [K=16]
"""
import sys

import chess
import pandas as pd
import torch

from chessdecoder.agent.oracle import Oracle
from chessdecoder.agent.rl.engine import RolloutEngine
from chessdecoder.agent.rl.reward import move_id_to_uci
from chessdecoder.agent.rl.rollout_proc import load_model

CKPT = sys.argv[1]
N_POS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
B = 128
MAX_PLIES = 160


def main():
    suites = pd.read_parquet("agent_data/eval_suites.parquet")
    fens = pd.concat([suites[suites.suite == "quiet"].fen.head(N_POS // 2),
                      suites[suites.suite == "sensitive"].fen.head(
                          N_POS - N_POS // 2)]).tolist()
    model = load_model(CKPT)
    oracle = Oracle()
    eng = RolloutEngine(model, oracle, batch_size=B, k_budget=K,
                        dtype=torch.bfloat16)

    # paired games: (start, agent_color)
    games = [{"board": chess.Board(f), "agent_color": c, "result": None}
             for f in fens for c in (chess.WHITE, chess.BLACK)]

    def adjudicate(g):
        b = g["board"]
        out = b.outcome(claim_draw=True)
        if out is not None:
            if out.winner is None:
                return 0.5
            return 1.0 if out.winner == g["agent_color"] else 0.0
        r = oracle.query(b)                       # stm POV q
        from chessdecoder.agent import patch_vocab as pv
        q = (r.q_bin / 64.0) - 1.0                # bin -> approx q
        q_agent = q if b.turn == g["agent_color"] else -q
        if q_agent > 0.25:
            return 1.0
        if q_agent < -0.25:
            return 0.0
        return 0.5

    plies = 0
    while plies < MAX_PLIES:
        live = [g for g in games if g["result"] is None]
        if not live:
            break
        for g in live:
            out = g["board"].outcome(claim_draw=True)
            if out is not None:
                g["result"] = adjudicate(g)
        live = [g for g in games if g["result"] is None]
        # oracle moves (cheap, batched)
        opp = [g for g in live if g["board"].turn != g["agent_color"]]
        if opp:
            rs = oracle.query_batch([g["board"] for g in opp])
            for g, r in zip(opp, rs):
                u = move_id_to_uci(g["board"], r.top_moves[0])
                g["board"].push(chess.Move.from_uci(u))
        # agent moves (batched episodes)
        ag = [g for g in live if g["board"].turn == g["agent_color"]
              and g["board"].outcome(claim_draw=True) is None]
        for i in range(0, len(ag), B):
            chunk = ag[i:i + B]
            roots = [g["board"].copy() for g in chunk]
            pad = B - len(roots)
            eps = eng.rollout(roots + [roots[-1]] * pad,
                              k_budgets=[K] * B, greedy=True)
            for g, e in zip(chunk, eps[:len(chunk)]):
                u = move_id_to_uci(g["board"], e.final_move)
                g["board"].push(chess.Move.from_uci(u))
        plies += 1
        if plies % 20 == 0:
            done = sum(g["result"] is not None for g in games)
            print(f"ply {plies}: {done}/{len(games)} finished", flush=True)

    for g in games:
        if g["result"] is None:
            g["result"] = adjudicate(g)
    score = sum(g["result"] for g in games)
    n = len(games)
    w = sum(g["result"] == 1.0 for g in games)
    d = sum(g["result"] == 0.5 for g in games)
    l = n - w - d
    p = score / n
    import math
    se = math.sqrt(p * (1 - p) / n)
    print(f"\nAGENT vs ORACLE-GREEDY (K={K}, {n} games, paired): "
          f"+{w} ={d} -{l}  score {p:.3f} +/- {1.96*se:.3f} (95% CI)")
    elo = 400 * math.log10(p / (1 - p)) if 0 < p < 1 else float("inf")
    print(f"elo diff: {elo:+.0f}")


if __name__ == "__main__":
    main()

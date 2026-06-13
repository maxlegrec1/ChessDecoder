"""L2 in real games: PUCT-800 over the frozen oracle vs Stockfish UCI_Elo.

Same protocol as the L0 measurement (oracle-greedy 2486 [2458,2515] vs
SF2400: single sequential SF process, movetime, alternating colours, no
book — variety from SF's limited-strength randomisation). Agent games run
in parallel; each round batches all agent-to-move positions through one
search_batch_cpp call.

  uv run python scripts/eval_puct_vs_stockfish.py [--games 256] [--sims 800]
      [--elo 2400] [--sf-time 0.1]
"""
import argparse
import math

import chess
import chess.engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--elo", type=int, default=2400)
    ap.add_argument("--sf-time", type=float, default=0.1)
    a = ap.parse_args()

    from chessdecoder.agent.rl.oracle_engine import OracleEngine
    from chessdecoder.agent.rl.qref import search_batch_cpp
    eng = OracleEngine()
    sf = chess.engine.SimpleEngine.popen_uci("bin/stockfish")
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": a.elo})

    games = [{"board": chess.Board(), "us": chess.WHITE if g % 2 == 0
              else chess.BLACK, "result": None} for g in range(a.games)]

    def finish(g):
        out = g["board"].outcome(claim_draw=True)
        if out is None:
            return None
        if out.winner is None:
            return 0.5
        return 1.0 if out.winner == g["us"] else 0.0

    rounds = 0
    while any(g["result"] is None for g in games) and rounds < 300:
        for g in games:
            if g["result"] is None:
                g["result"] = finish(g)
        live = [g for g in games if g["result"] is None]
        # our moves: one batched 800-sim search over all our-turn positions
        ours = [g for g in live if g["board"].turn == g["us"]]
        if ours:
            res = search_batch_cpp(eng, [g["board"] for g in ours],
                                   sims=a.sims)
            for g, r in zip(ours, res):
                g["board"].push(chess.Move.from_uci(r.search_best))
                g["result"] = finish(g)
        # SF moves: sequential single process (protocol match)
        for g in [g for g in live if g["result"] is None
                  and g["board"].turn != g["us"]]:
            mv = sf.play(g["board"],
                         chess.engine.Limit(time=a.sf_time)).move
            g["board"].push(mv)
            g["result"] = finish(g)
        rounds += 1
        done = sum(g["result"] is not None for g in games)
        if rounds % 20 == 0:
            print(f"round {rounds}: {done}/{a.games} done", flush=True)

    for g in games:
        if g["result"] is None:
            g["result"] = 0.5                  # 300-round cap: call it a draw
    sf.quit()
    n = len(games)
    score = sum(g["result"] for g in games)
    w = sum(g["result"] == 1.0 for g in games)
    d = sum(g["result"] == 0.5 for g in games)
    p = score / n
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    def elo(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return a.elo + 400 * math.log10(x / (1 - x))
    print(f"\nPUCT-{a.sims} vs SF{a.elo}: +{w} ={d} -{n-w-d}  "
          f"score {p:.3f}")
    print(f"Elo {elo(p):.0f}  [{elo(p-1.96*se):.0f}, {elo(p+1.96*se):.0f}]"
          f"  (L0 oracle-greedy baseline: 2486 [2458, 2515])")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare raw NN Q-values vs MCTS-backed Q-values for root candidate moves.

For each candidate move, shows:
  - NN policy prior (from policy head)
  - Raw NN Q: value head eval of the child position (single forward pass, no search)
  - MCTS Q: backed-up minimax value after search
  - Visit count / fraction from MCTS
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mcts import LeelaMCTS


def main():
    engine_path = "trt/model_dynamic_leela.trt"

    # Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 — rich position with multiple candidates
    origin_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    history = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]
    # Position after 3.Bc4 Bc5: white to move
    # FEN: r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4

    simulations = 600
    max_variations = 10  # get as many as possible

    print(f"Origin FEN: {origin_fen}")
    print(f"History:    {' '.join(history)}")
    print(f"Simulations: {simulations}")
    print()

    mcts = LeelaMCTS(engine_path=engine_path, simulations=simulations)
    result = mcts.run_with_variations(
        origin_fen,
        history,
        max_variations=max_variations,
        max_variation_depth=10,
    )

    # Root NN value (same for all moves — it's the position eval, not per-move)
    root_wdl = result["value"]
    root_q = root_wdl[0] - root_wdl[2]  # W - L
    print(f"Root NN WDL: W={root_wdl[0]:.4f}  D={root_wdl[1]:.4f}  L={root_wdl[2]:.4f}  (Q={root_q:+.4f})")
    print(f"Best move (most visited): {result['action']}")
    print()

    # Build lookup: root_move -> first variation node's raw NN WDL
    raw_nn_q = {}  # move -> Q from root player's perspective
    var_visits = {}  # move -> (visit_count, visit_fraction)
    for var in result.get("variations", []):
        move = var["root_move"]
        nodes = var.get("nodes", [])
        if nodes:
            # First node is the child position (opponent's turn)
            # WDL is from side-to-move (opponent), so negate for root's perspective
            child_wdl = nodes[0]["wdl"]
            raw_nn_q[move] = -(child_wdl[0] - child_wdl[2])  # negate: opponent's W-L -> root's Q
        var_visits[move] = (var["visit_count"], var["visit_fraction"])

    # MCTS Q-values (already from root's perspective)
    mcts_q = result["q_values"]

    # NN policy priors
    policy = result["policy"]

    # Collect all candidate moves that have MCTS Q
    all_moves = sorted(mcts_q.keys(), key=lambda m: mcts_q[m], reverse=True)

    print(f"{'Move':<10} {'Policy':>8} {'NN Q':>8} {'MCTS Q':>8} {'Delta':>8} {'Visits':>8} {'Frac':>8}")
    print("-" * 72)
    for move in all_moves:
        mq = mcts_q[move]
        nn_q = raw_nn_q.get(move)
        pol = policy.get(move, 0.0)
        vc, vf = var_visits.get(move, (None, None))
        delta = f"{mq - nn_q:+.4f}" if nn_q is not None else "   n/a"
        nn_q_str = f"{nn_q:+.4f}" if nn_q is not None else "   n/a"
        vc_str = f"{vc:>8d}" if vc is not None else "     n/a"
        vf_str = f"{vf:>8.4f}" if vf is not None else "     n/a"
        print(f"{move:<10} {pol:>8.4f} {nn_q_str:>8} {mq:>+8.4f} {delta:>8} {vc_str} {vf_str}")


if __name__ == "__main__":
    main()

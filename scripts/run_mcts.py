"""Run Leela MCTS on a single FEN and print the variation output.

Usage:
    uv run python scripts/run_mcts.py
    uv run python scripts/run_mcts.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    uv run python scripts/run_mcts.py --simulations 200 --max-variations 3
"""

import argparse
import json

from chessdecoder.mcts import LeelaMCTS

DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DEFAULT_ENGINE = "trt/model_dynamic_leela.trt"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fen", default=DEFAULT_FEN)
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--max-variations", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--cpuct", type=float, default=1.5)
    args = parser.parse_args()

    print(f"FEN        : {args.fen}")
    print(f"Engine     : {args.engine}")
    print(f"Simulations: {args.simulations}")
    print()

    mcts = LeelaMCTS(engine_path=args.engine, simulations=args.simulations, cpuct=args.cpuct)
    result = mcts.run_with_variations(
        args.fen,
        max_variations=args.max_variations,
        max_variation_depth=args.max_depth,
    )

    v    = result['value']
    bv   = result.get('backed_up_value', v)
    print(f"Best move      : {result['action']}")
    print(f"Root raw   wdl : W={v[0]:.3f}  D={v[1]:.3f}  L={v[2]:.3f}  (Q={v[0]-v[2]:+.3f})")
    print(f"Root backed wdl: W={bv[0]:.3f}  D={bv[1]:.3f}  L={bv[2]:.3f}  (Q={bv[0]-bv[2]:+.3f})")
    print()

    print("Top policy moves:")
    for move, prob in list(result["policy"].items())[:10]:
        q = result["q_values"].get(move, float("nan"))
        print(f"  {move:<8}  prior={prob:.4f}  Q={q:+.3f}")
    print()

    variations = result.get("variations", [])
    print(f"Variations ({len(variations)}):")
    for i, var in enumerate(variations):
        root_move = var["root_move"]
        visits = var["visit_count"]
        frac = var["visit_fraction"]
        prior = var["prior"]
        nodes = var.get("nodes", [])
        print(f"\n  [{i+1}] {root_move}  visits={visits}  ({100*frac:.1f}%)  prior={prior:.4f}")
        for depth, node in enumerate(nodes):
            wdl    = node["wdl"]
            bu_wdl = node.get("backed_up_wdl", wdl)
            vc     = node["visit_count"]
            move   = node.get("move", "—")
            indent = "    " + "  " * depth
            print(f"{indent}fen        : {node['fen']}")
            print(f"{indent}raw    wdl : W={wdl[0]:.3f}  D={wdl[1]:.3f}  L={wdl[2]:.3f}  (Q={wdl[0]-wdl[2]:+.3f})")
            print(f"{indent}backed wdl : W={bu_wdl[0]:.3f}  D={bu_wdl[1]:.3f}  L={bu_wdl[2]:.3f}  (Q={bu_wdl[0]-bu_wdl[2]:+.3f})")
            print(f"{indent}visits     : {vc}")
            if move and depth < len(nodes) - 1:
                print(f"{indent}move       : {move}  →")


if __name__ == "__main__":
    main()

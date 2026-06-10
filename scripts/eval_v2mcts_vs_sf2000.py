"""Eval V2 + C++ PUCT MCTS against Stockfish UCI_Elo 2000.

Mirrors scripts/eval_v2_vs_sf2000.py but with the new C++ MCTS engine
(chessdecoder.mcts_v2.V2MCTS) replacing raw policy / one-ply VALUE search.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_v2mcts_vs_sf2000.py \\
        [N_GAMES] [SF_ELO] [SIMS] [CPUCT]
"""
import sys

from chessdecoder.eval.engine import PytorchModelAdapter
from chessdecoder.eval.elo_eval import model_vs_stockfish
from chessdecoder.mcts_v2 import V2MCTS


NGAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SF_ELO = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
CPUCT = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5

print(f"V2 + C++ MCTS vs SF Elo {SF_ELO} | {NGAMES} games | "
      f"sims={SIMS} cpuct={CPUCT} temp=0 (argmax visits)", flush=True)
mcts = V2MCTS(simulations=SIMS, cpuct=CPUCT, temperature=0.0,
              max_batch_leaves=16)


def mcts_predict(fen: str, _temp: float) -> str:
    r = mcts.search(fen)
    return r.action


adapter = PytorchModelAdapter(mcts_predict)

print(f"\n===== V2-MCTS({SIMS}) vs SF {SF_ELO} =====", flush=True)
wr, elo = model_vs_stockfish(
    model=adapter, model1_name=f"v2-mcts{SIMS}",
    num_games=NGAMES, temperature=0.0, elo=SF_ELO,
    pgn_dir="pgns",
)
print(f">>> v2-mcts({SIMS}): winrate={wr:.3f}  estimated_elo={elo}",
      flush=True)

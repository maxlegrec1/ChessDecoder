"""Eval the BT4 Leela checkpoint against Stockfish UCI_Elo 2000.

Mirrors `scripts/eval_v2_vs_sf2000.py` but with the Leela BT4 model.
Notes:
  - BT4 uses 112 input planes (8 history snapshots of 12 piece planes each +
    castling/STM aux planes). `get_move_from_fen_no_thinking` accepts either a
    FEN (history planes are synthesized as empty -> the position itself) or a
    UCI move list (history planes are filled by replaying the moves).
  - The eval harness has `PytorchModelAdapter.new_game()` /
    `record_move(fen_after, move_uci)` for history tracking, which lets us feed
    BT4 a real move history per game rather than degrading to one-frame FEN.
"""
import sys

import bulletchess
import chess
import torch

from chessdecoder.models.leela import BT4
from chessdecoder.eval.engine import PytorchModelAdapter
from chessdecoder.eval.elo_eval import model_vs_stockfish

CKPT = sys.argv[1] if len(sys.argv) > 1 else "chessdecoder/models/leela/model.pt"
NGAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ELO = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
DEV = "cuda"

print(f"Loading BT4 from {CKPT}", flush=True)
model = BT4().to(DEV).eval()
state = torch.load(CKPT, map_location=DEV, weights_only=False)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state, strict=True)
print(f"BT4 loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)  "
      f"games={NGAMES} vs SF Elo {ELO} temp0", flush=True)


# Track per-game move history so BT4 sees real history planes (not a
# repeated-current-position fallback). PytorchModelAdapter calls
# new_game() before each game and record_move(fen_after, move_uci) after every
# half-move (engine.py:215). We hook both via a tiny stateful callable.

_history: list[str] = []


def new_game():
    global _history
    _history = []


def record_move(fen_after: str, move_uci: str):
    """elo_eval calls this after every half-move with the resulting FEN."""
    _history.append(move_uci)


@torch.no_grad()
def leela_predict(fen: str, temperature: float) -> str:
    """elo_eval.play_game always starts from the standard initial position
    (bulletchess.Board() with no args), so the full _history list represents
    the game from move 1. Feed it to BT4 so the history planes are populated.
    For the very first call of a Model-as-White game, _history is empty —
    fall back to FEN-only."""
    if _history:
        return model.get_move_from_fen_no_thinking(list(_history), temperature)
    return model.get_move_from_fen_no_thinking(fen, temperature)


adapter = PytorchModelAdapter(leela_predict)
# elo_eval probes for `new_game` / `record_move` via hasattr; attach them.
adapter.new_game = new_game
adapter.record_move = record_move

print(f"\n===== Leela BT4 vs Stockfish {ELO} (temp 0, {NGAMES} games) =====",
      flush=True)
wr, elo = model_vs_stockfish(model=adapter, model1_name="leela-bt4",
                             num_games=NGAMES, temperature=0.0, elo=ELO,
                             pgn_dir="pgns")
print(f">>> leela-bt4: winrate={wr:.3f}  estimated_elo={elo}", flush=True)

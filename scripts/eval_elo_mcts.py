#!/usr/bin/env python3
"""
ELO evaluation of MCTS-based move selection strategies.

Two models:
  - MCTSRawNNModel: picks the root candidate with the highest raw NN Q (value head, no backup)
  - MCTSSearchQModel: picks the root candidate with the highest MCTS-backed Q (minimax backup)

Both run the same MCTS search; they differ only in which Q-value selects the final move.

Usage:
    uv run python scripts/eval_elo_mcts.py --simulations 600 --elo 1500 --num-games 200
    uv run python scripts/eval_elo_mcts.py --model mcts_q --simulations 800 --elo 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mcts import LeelaMCTS
from src.eval.elo_eval import model_vs_stockfish

# Pseudo-castling (model vocab) -> standard UCI
_PSEUDO_TO_STANDARD = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}


class _MCTSModelBase:
    """Base class that wraps LeelaMCTS and tracks move history across a game."""

    def __init__(
        self,
        engine_path: str = "trt/model_dynamic_leela.trt",
        simulations: int = 600,
        cpuct: float = 1.5,
        max_variations: int = 50,
        max_variation_depth: int = 1,
    ):
        self.simulations = simulations
        self.max_variations = max_variations
        self.max_variation_depth = max_variation_depth
        self._mcts = LeelaMCTS(
            engine_path=engine_path,
            simulations=simulations,
            cpuct=cpuct,
            temperature=1.0,
        )
        self._board: chess.Board | None = None
        self._history: list[str] = []
        self._origin_fen: str = chess.STARTING_FEN

    def _sync_to_fen(self, fen: str):
        """Bring internal board state in sync with the given FEN."""
        # First call
        if self._board is None:
            self._new_game(fen)
            return

        # Already in sync
        if self._fen_match(self._board.fen(), fen):
            return

        # Opponent played a move — find it
        for move in self._board.legal_moves:
            self._board.push(move)
            if self._fen_match(self._board.fen(), fen):
                self._history.append(move.uci())
                return
            self._board.pop()

        # Could not find move — new game
        self._new_game(fen)

    def _new_game(self, fen: str):
        """Start tracking a new game, reconstructing history from STARTING_FEN."""
        # Exact starting position (model plays white)
        if self._fen_match(fen, chess.STARTING_FEN):
            self._board = chess.Board()
            self._history = []
            self._origin_fen = chess.STARTING_FEN
            return

        # 1 move from starting position (model plays black, SF moved first)
        start = chess.Board()
        for move in start.legal_moves:
            start.push(move)
            if self._fen_match(start.fen(), fen):
                self._board = start.copy()
                self._history = [move.uci()]
                self._origin_fen = chess.STARTING_FEN
                return
            start.pop()

        # Fallback: non-standard start, use FEN directly
        self._board = chess.Board(fen)
        self._history = []
        self._origin_fen = fen

    @staticmethod
    def _fen_match(fen_a: str, fen_b: str) -> bool:
        """Compare FENs ignoring halfmove/fullmove counters."""
        return " ".join(fen_a.split()[:4]) == " ".join(fen_b.split()[:4])

    def _run_search(self, fen: str):
        """Run MCTS and return the result dict."""
        self._sync_to_fen(fen)
        result = self._mcts.run_with_variations(
            self._origin_fen,
            self._history,
            max_variations=self.max_variations,
            max_variation_depth=self.max_variation_depth,
        )
        return result

    def _select_move(self, result: dict, fen: str) -> str:
        """Override in subclass to pick the move."""
        raise NotImplementedError

    def predict_move(self, fen: str, temperature: float = 0.0) -> str | None:
        result = self._run_search(fen)
        move = self._select_move(result, fen)
        if move is None:
            return None
        # Convert pseudo-castling to standard UCI if needed
        move = _PSEUDO_TO_STANDARD.get(move, move)
        # Track our own move
        if self._board is not None:
            try:
                self._board.push(chess.Move.from_uci(move))
                self._history.append(move)
            except (chess.InvalidMoveError, chess.IllegalMoveError):
                pass
        return move


class MCTSRawNNModel(_MCTSModelBase):
    """Selects the root candidate move with the highest raw NN Q-value."""

    def _select_move(self, result: dict, fen: str) -> str | None:
        board = chess.Board(fen)
        legal_ucis = {m.uci() for m in board.legal_moves}

        # Build raw NN Q for each candidate from the first variation node
        best_move, best_q = None, -float("inf")
        for var in result.get("variations", []):
            move = _PSEUDO_TO_STANDARD.get(var["root_move"], var["root_move"])
            if move not in legal_ucis:
                continue
            nodes = var.get("nodes", [])
            if not nodes:
                continue
            # First node is opponent's turn — negate for root's perspective
            child_wdl = nodes[0]["wdl"]
            raw_q = -(child_wdl[0] - child_wdl[2])
            if raw_q > best_q:
                best_q = raw_q
                best_move = move

        # Fallback to MCTS action if no variations available
        if best_move is None:
            action = result.get("action")
            if action:
                best_move = _PSEUDO_TO_STANDARD.get(action, action)

        return best_move


class MCTSSearchQModel(_MCTSModelBase):
    """Selects the root candidate move with the highest MCTS-backed Q-value."""

    def _select_move(self, result: dict, fen: str) -> str | None:
        board = chess.Board(fen)
        legal_ucis = {m.uci() for m in board.legal_moves}

        q_values = result.get("q_values", {})
        best_move, best_q = None, -float("inf")
        for move, q in q_values.items():
            move_std = _PSEUDO_TO_STANDARD.get(move, move)
            if move_std not in legal_ucis:
                continue
            if q > best_q:
                best_q = q
                best_move = move_std

        # Fallback to MCTS action
        if best_move is None:
            action = result.get("action")
            if action:
                best_move = _PSEUDO_TO_STANDARD.get(action, action)

        return best_move


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MCTS move-selection strategies against Stockfish.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["nn_q", "mcts_q", "both"], default="both",
        help="Which model(s) to evaluate.",
    )
    parser.add_argument("--simulations", type=int, default=600)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--engine-path", type=str, default="trt/model_dynamic_leela.trt")
    parser.add_argument("--elo", type=int, default=1500, help="Stockfish ELO.")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    models_to_run = []
    if args.model in ("nn_q", "both"):
        models_to_run.append((
            "MCTS_RawNN_Q",
            MCTSRawNNModel(
                engine_path=args.engine_path,
                simulations=args.simulations,
                cpuct=args.cpuct,
            ),
        ))
    if args.model in ("mcts_q", "both"):
        models_to_run.append((
            "MCTS_Search_Q",
            MCTSSearchQModel(
                engine_path=args.engine_path,
                simulations=args.simulations,
                cpuct=args.cpuct,
            ),
        ))

    for name, model in models_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({args.simulations} sims vs Stockfish {args.elo})")
        print(f"{'='*60}\n")
        model_vs_stockfish(
            model,
            model1_name=name,
            num_games=args.num_games,
            temperature=args.temperature,
            elo=args.elo,
        )


if __name__ == "__main__":
    main()

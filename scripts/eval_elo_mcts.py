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

import chess

from chessdecoder.mcts import LeelaMCTS
from chessdecoder.eval.engine import MoveResult
from chessdecoder.eval.elo_eval import model_vs_stockfish
from chessdecoder.utils.uci import normalize_castling


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
        move = normalize_castling(move)
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
            move = normalize_castling(var["root_move"])
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
                best_move = normalize_castling(action)

        return best_move


class LeelaPolicyModel(_MCTSModelBase):
    """Picks the move with the highest raw Leela policy prior (no value head, no search)."""

    def predict_move(self, fen: str, temperature: float = 0.0) -> str | None:
        self._sync_to_fen(fen)

        result = self._mcts.run(self._origin_fen, self._history, simulations=0)
        policy = result.get("policy", {})
        board = chess.Board(fen)
        legal_ucis = {normalize_castling(m.uci()) for m in board.legal_moves}

        best_move, best_prior = None, -1.0
        for move, prior in policy.items():
            move_std = normalize_castling(move)
            if move_std in legal_ucis and prior > best_prior:
                best_prior = prior
                best_move = move_std

        if best_move is None:
            return None

        if self._board is not None:
            try:
                self._board.push(chess.Move.from_uci(best_move))
                self._history.append(best_move)
            except (chess.InvalidMoveError, chess.IllegalMoveError):
                pass

        return best_move

    def predict_moves(self, fens: list[str], temperature: float = 0.0) -> list[MoveResult]:
        return [MoveResult(move=self.predict_move(fen, temperature) or "") for fen in fens]

    def _select_move(self, result: dict, fen: str) -> str | None:
        raise NotImplementedError


class LeelaDepth1Model(_MCTSModelBase):
    """Picks the move with the highest raw Leela NN value at depth 1.

    Evaluates ALL legal child positions with 0 simulations (pure NN, no search),
    then selects the move whose resulting position has the lowest opponent Q.
    Equivalent to a minimax depth-1 with the Leela value head.
    """

    def predict_move(self, fen: str, temperature: float = 0.0) -> str | None:
        self._sync_to_fen(fen)

        board = chess.Board(fen)
        moves = [normalize_castling(m.uci()) for m in board.legal_moves]
        if not moves:
            return None

        # Evaluate every child position with 0 simulations (raw NN only)
        child_positions = [
            (self._origin_fen, self._history + [move]) for move in moves
        ]
        results = self._mcts.run_parallel(child_positions, simulations=0)

        best_move, best_q = None, -float("inf")
        for move, result in zip(moves, results):
            value = result.get("value")
            if value is None:
                continue
            q = -(value[0] - value[2])   # negate: child value is from opponent's POV
            if q > best_q:
                best_q = q
                best_move = move

        if best_move is None:
            return None

        # Track our own move in history
        if self._board is not None:
            try:
                self._board.push(chess.Move.from_uci(best_move))
                self._history.append(best_move)
            except (chess.InvalidMoveError, chess.IllegalMoveError):
                pass

        return best_move

    def predict_moves(self, fens: list[str], temperature: float = 0.0) -> list[MoveResult]:
        return [MoveResult(move=self.predict_move(fen, temperature) or "") for fen in fens]

    def _select_move(self, result: dict, fen: str) -> str | None:
        # Not used — predict_move is fully overridden
        raise NotImplementedError


class MCTSSearchQModel(_MCTSModelBase):
    """Selects the root candidate move with the highest MCTS-backed Q-value."""

    def _select_move(self, result: dict, fen: str) -> str | None:
        board = chess.Board(fen)
        legal_ucis = {m.uci() for m in board.legal_moves}

        q_values = result.get("q_values", {})
        best_move, best_q = None, -float("inf")
        for move, q in q_values.items():
            move_std = normalize_castling(move)
            if move_std not in legal_ucis:
                continue
            if q > best_q:
                best_q = q
                best_move = move_std

        # Fallback to MCTS action
        if best_move is None:
            action = result.get("action")
            if action:
                best_move = normalize_castling(action)

        return best_move


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MCTS move-selection strategies against Stockfish.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["nn_q", "mcts_q", "depth1", "policy", "both"], default="both",
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
    if args.model in ("depth1",):
        models_to_run.append((
            "Leela_Depth1",
            LeelaDepth1Model(engine_path=args.engine_path),
        ))
    if args.model in ("policy",):
        models_to_run.append((
            "Leela_Policy",
            LeelaPolicyModel(engine_path=args.engine_path),
        ))

    for name, model in models_to_run:
        print(f"\n{'='*60}")
        if "Depth1" in name:
            sims_str = "depth-1 value"
        elif "Policy" in name:
            sims_str = "policy argmax"
        else:
            sims_str = f"{args.simulations} sims"
        print(f"Evaluating: {name} ({sims_str} vs Stockfish {args.elo})")
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

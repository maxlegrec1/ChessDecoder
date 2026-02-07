#!/usr/bin/env python3
"""
Play Leela MCTS against Stockfish and estimate Elo.

Mimics src/eval/play_move.py structure: tqdm with running win rate,
streams PGN to disk after each game, prints estimated Elo at the end.

Usage:
    uv run python scripts/play_move.py
    uv run python scripts/play_move.py --stockfish-elo 1500 --games 50
"""

from __future__ import annotations

import argparse

import os
import sys
import traceback
from datetime import datetime
from math import log10
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

from src.mcts import LeelaMCTS

PGN_DIR = "./pgns"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leela MCTS vs Stockfish",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--cpuct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--engine-path", default="model_dynamic_leela.trt")
    p.add_argument("--stockfish", default="stockfish", help="Path to stockfish binary (default: find on PATH).")
    p.add_argument("--stockfish-time", type=float, default=0.1, help="Stockfish think time (seconds).")
    p.add_argument("--stockfish-elo", type=int, default=2800, help="Stockfish Elo limit.")
    p.add_argument("--games", type=int, default=100, help="Number of games to play.")
    return p.parse_args()


def estimate_elo(win_rate: float, stockfish_elo: int) -> float | str:
    if win_rate <= 0:
        return 0
    if win_rate >= 1:
        return stockfish_elo + 400
    if win_rate == 0.5:
        return stockfish_elo
    return stockfish_elo - 400 * log10((1 - win_rate) / win_rate)


def play_game(
    mcts: LeelaMCTS,
    sf: chess.engine.SimpleEngine,
    args: argparse.Namespace,
    mcts_is_white: bool,
    game_num: int,
) -> tuple[str, chess.pgn.Game]:
    board = chess.Board()
    history: list[str] = []

    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Leela MCTS vs Stockfish"
    pgn_game.headers["Site"] = "Local Machine"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_num)
    pgn_game.headers["White"] = "LeelaMCTS" if mcts_is_white else "Stockfish"
    pgn_game.headers["Black"] = "Stockfish" if mcts_is_white else "LeelaMCTS"
    if mcts_is_white:
        pgn_game.headers["BlackElo"] = str(args.stockfish_elo)
    else:
        pgn_game.headers["WhiteElo"] = str(args.stockfish_elo)
    node = pgn_game

    while not board.is_game_over(claim_draw=True):
        is_mcts_turn = (board.turn == chess.WHITE) == mcts_is_white

        if is_mcts_turn:
            try:
                result = mcts.run(
                    "",
                    history,
                    simulations=args.simulations,
                )
            except Exception as e:
                print(f"MCTS.run crashed at ply {board.ply()}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                r = "0-1" if board.turn == chess.WHITE else "1-0"
                pgn_game.headers["Result"] = r
                return r, pgn_game

            action = result.get("action")
            if not action:
                r = "0-1" if board.turn == chess.WHITE else "1-0"
                pgn_game.headers["Result"] = r
                return r, pgn_game

            try:
                move = board.parse_uci(action)
            except (chess.InvalidMoveError, chess.IllegalMoveError):
                print(
                    f"MCTS illegal move at ply {board.ply()}: {action!r} "
                    f"fen={board.fen()}",
                    file=sys.stderr,
                )
                r = "0-1" if board.turn == chess.WHITE else "1-0"
                pgn_game.headers["Result"] = r
                return r, pgn_game
        else:
            sf_result = sf.play(board, chess.engine.Limit(time=args.stockfish_time))
            move = sf_result.move
            if not move:
                r = "1-0" if board.turn == chess.WHITE else "0-1"
                pgn_game.headers["Result"] = r
                return r, pgn_game

        # History uses standard UCI — python-chess move.uci() is already standard
        node = node.add_variation(move)
        history.append(move.uci())
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    r = board.result(claim_draw=True)
    pgn_game.headers["Result"] = r
    return r, pgn_game


def main() -> None:
    args = parse_args()
    os.makedirs(PGN_DIR, exist_ok=True)
    pgn_filename = os.path.join(
        PGN_DIR,
        datetime.now().strftime("leela_mcts_vs_stockfish_%Y%m%d_%H%M%S.pgn"),
    )

    mcts = LeelaMCTS(
        engine_path=args.engine_path,
        simulations=args.simulations,
        cpuct=args.cpuct,
        temperature=args.temperature,
    )

    wins, draws = 0, 0
    pgn_file = open(pgn_filename, "w", encoding="utf-8")

    try:
        with chess.engine.SimpleEngine.popen_uci(args.stockfish) as sf:
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})

            pbar = tqdm(range(args.games), desc="MCTS vs SF")
            for i in pbar:
                mcts_is_white = (i % 2 == 0)
                result, pgn_game = play_game(mcts, sf, args, mcts_is_white, i + 1)

                if mcts_is_white and result == "1-0":
                    wins += 1
                elif not mcts_is_white and result == "0-1":
                    wins += 1
                elif result == "1/2-1/2":
                    draws += 1

                # Stream PGN to disk
                exporter = chess.pgn.FileExporter(pgn_file)
                pgn_game.accept(exporter)
                pgn_file.write("\n\n")
                pgn_file.flush()

                wr = (wins + 0.5 * draws) / (i + 1)
                pbar.set_description(
                    f"MCTS vs SF {args.stockfish_elo} (WR: {wr:.1%})"
                )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
    finally:
        pgn_file.close()

    n = args.games
    losses = n - wins - draws
    wr = (wins + 0.5 * draws) / n if n > 0 else 0
    elo = estimate_elo(wr, args.stockfish_elo)
    print(f"\nResults: +{wins} ={draws} -{losses} / {n}")
    print(f"Win rate: {wr:.1%}")
    print(f"Estimated Elo: {elo:.0f}" if isinstance(elo, float) else f"Estimated Elo: {elo}")
    print(f"PGN saved to {pgn_filename}")


if __name__ == "__main__":
    main()

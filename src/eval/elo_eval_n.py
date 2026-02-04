"""
Evaluation script for testing model vs Stockfish with n-move context.

This script evaluates the model's performance when using predict_move_n(),
which allows the model to condition on game history for move prediction.
"""

import torch
import chess
from chess import engine, pgn
import bulletchess
from bulletchess import CHECKMATE, DRAW
from datetime import datetime
from tqdm import tqdm
from math import log10
import os

STOCKFISH_PATH = "./stockfish-ubuntu-x86-64-avx2"


def get_stockfish_move(engine_obj, bullet_board):
    """Get Stockfish's move for the current position."""
    ch_board = chess.Board(bullet_board.fen())
    result = engine_obj.play(ch_board, engine.Limit(time=0.1))
    return bulletchess.Move.from_uci(result.move.uci())


def estimate_elo(win_rate, stockfish_elo):
    """Estimate model ELO based on win rate against Stockfish."""
    if win_rate == 0:
        return 0
    elif win_rate == 1:
        return 2500
    if win_rate == 0.5:
        return stockfish_elo
    if win_rate > 0 and win_rate < 1:
        return stockfish_elo - 400 * (log10((1 - win_rate) / win_rate))
    return "N/A"


def build_history_for_model(move_history, fen_history, max_n, model_is_white, current_move_number):
    """
    Build the history for predict_move_n based on the game state.

    Args:
        move_history: List of UCI moves played so far
        fen_history: List of FENs after each move (index i is FEN after move i)
        max_n: Maximum history context to use
        model_is_white: Whether the model is playing white
        current_move_number: 0-indexed move number (0 = first move of game)

    Returns:
        initial_fen: The starting position for context
        history: List of (board_fen, move_uci) tuples for predict_move_n
        effective_n: The actual n value being used

    Logic:
    - For white's first move (move 0): n=0, just initial position
    - For black's first move (move 1): n=1, have 1 board in history
    - For move k: n = k (capped at max_n)
    """
    # Starting FEN (before any moves)
    initial_fen = chess.STARTING_FEN

    # How many moves have been played before this turn
    moves_played = len(move_history)

    if moves_played == 0:
        # First move of game (white's turn), no history available
        return initial_fen, [], 0

    # Calculate effective n (how many (board, move) pairs we can include)
    # Each move in history gives us one (board_fen_after_move, move) pair
    effective_n = min(moves_played, max_n)

    # Build history: take the last effective_n moves
    # Each entry is (board_fen after move, move_uci)
    start_idx = moves_played - effective_n

    history = []
    for i in range(start_idx, moves_played):
        move_uci = move_history[i]
        board_fen_after = fen_history[i]
        history.append((board_fen_after, move_uci))

    # Determine the initial_fen for the context
    # If start_idx > 0, initial_fen is the board BEFORE the first move in our history
    if start_idx > 0:
        # FEN before move at start_idx is fen_history[start_idx - 1]
        # Wait, fen_history[i] is the FEN AFTER move i
        # So FEN before move start_idx is fen_history[start_idx - 1]
        initial_fen = fen_history[start_idx - 1]
    else:
        # We're using moves from the beginning
        initial_fen = chess.STARTING_FEN

    return initial_fen, history, effective_n


def play_game_n(player1_name, player2_name, engine_obj, model, max_n,
                temperature=0.1, stockfish_elo=None, game_num=1):
    """
    Play a game between model and Stockfish using n-move context.

    Args:
        player1_name: White player name ("Model" or "Stockfish")
        player2_name: Black player name ("Model" or "Stockfish")
        engine_obj: Stockfish engine object
        model: The chess decoder model
        max_n: Maximum history context for predict_move_n
        temperature: Sampling temperature for model
        stockfish_elo: Stockfish ELO setting
        game_num: Game number for PGN

    Returns:
        result: Game result string ("1-0", "0-1", or "1/2-1/2")
        pgn_game: PGN game object
    """
    board = bulletchess.Board()
    pgn_game = pgn.Game()
    pgn_game.headers["Event"] = f"{player1_name} vs {player2_name} (n={max_n})"
    pgn_game.headers["Site"] = "Local Machine"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_num)
    pgn_game.headers["White"] = player1_name
    pgn_game.headers["Black"] = player2_name

    if player1_name == "Stockfish" and stockfish_elo:
        pgn_game.headers["WhiteElo"] = str(stockfish_elo)
    if player2_name == "Stockfish" and stockfish_elo:
        pgn_game.headers["BlackElo"] = str(stockfish_elo)

    node = pgn_game

    # Track game history
    move_history = []  # List of UCI moves
    fen_history = []   # List of FENs after each move

    model_is_white = (player1_name == "Model")

    def _is_game_over(b):
        return (b in CHECKMATE) or (b in DRAW)

    move_number = 0  # 0-indexed

    while not _is_game_over(board):
        is_white_turn = (board.turn == bulletchess.WHITE)
        is_stockfish_turn = (is_white_turn and player1_name == "Stockfish") or \
                           (not is_white_turn and player2_name == "Stockfish")

        move = None

        if is_stockfish_turn:
            move = get_stockfish_move(engine_obj, board)
        else:
            # Model's turn - use predict_move_n
            current_fen = board.fen()

            # Build history for this move
            initial_fen, history, effective_n = build_history_for_model(
                move_history, fen_history, max_n, model_is_white, move_number
            )

            try:
                if len(history) == 0:
                    # No history available, use regular predict_move
                    move_uci = model.predict_move(current_fen, temperature=temperature)
                else:
                    # Use predict_move_n with history
                    move_uci = model.predict_move_n(
                        initial_fen, history, temperature=temperature
                    )
            except Exception as e:
                print(f"Model prediction error: {e}")
                print(f"FEN: {current_fen}, History length: {len(history)}")
                result = "0-1" if is_white_turn else "1-0"
                pgn_game.headers["Result"] = result
                return result, pgn_game

            if move_uci is None:
                print(f"Model failed to produce a move for FEN: {current_fen}")
                result = "0-1" if is_white_turn else "1-0"
                pgn_game.headers["Result"] = result
                return result, pgn_game

            try:
                move = bulletchess.Move.from_uci(move_uci)
                if move not in board.legal_moves():
                    print(f"Invalid move by Model: {move_uci} in {current_fen}")
                    result = "0-1" if is_white_turn else "1-0"
                    pgn_game.headers["Result"] = result
                    return result, pgn_game
            except ValueError:
                print(f"Invalid UCI string by Model: {move_uci} in {current_fen}")
                result = "0-1" if is_white_turn else "1-0"
                pgn_game.headers["Result"] = result
                return result, pgn_game

        # Apply move
        board.apply(move)
        move_uci = move.uci()

        # Update history
        move_history.append(move_uci)
        fen_history.append(board.fen())

        # Add to PGN
        pgn_move = chess.Move.from_uci(move_uci)
        node = node.add_variation(pgn_move)

        move_number += 1

    # Determine result
    if board in CHECKMATE:
        result = "0-1" if board.turn == bulletchess.WHITE else "1-0"
    elif board in DRAW:
        result = "1/2-1/2"
    else:
        result = "1/2-1/2"

    pgn_game.headers["Result"] = result
    return result, pgn_game


def save_pgn_game(pgn_game, pgn_filename):
    """Append a single PGN game to the file."""
    with open(pgn_filename, "a", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        pgn_game.accept(exporter)
        f.write("\n\n")


def model_vs_stockfish_n(model=None, model_name="run", num_games=1, max_n=5,
                         temperature=0.1, elo=1400, pgn_dir="pgns"):
    """
    Evaluate model vs Stockfish using n-move context prediction.

    Args:
        model: The chess decoder model
        model_name: Name for the model in PGN files
        num_games: Number of games to play
        max_n: Maximum history context for predict_move_n
        temperature: Sampling temperature
        elo: Stockfish ELO setting
        pgn_dir: Directory for saving PGN files

    Returns:
        win_rate: Model's win rate (wins + 0.5 * draws) / games
        estimated_elo: Estimated model ELO
    """
    assert num_games > 0
    assert isinstance(elo, int) and elo >= 1400
    assert max_n >= 0

    wins, draws = 0, 0
    games_played = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pgn_filename = os.path.join(pgn_dir, f"games_model_vs_stockfish_n{max_n}_{timestamp}.pgn")

    os.makedirs(pgn_dir, exist_ok=True)
    open(pgn_filename, "w", encoding="utf-8").close()
    print(f"Evaluating model with max_n={max_n} against Stockfish ELO {elo}")
    print(f"Saving games to {pgn_filename}")

    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine_obj:
            engine_obj.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

            pbar = tqdm(range(num_games), desc=f"Model (n={max_n}) vs Stockfish")
            for i in pbar:
                # Alternate who plays white
                if i % 2 == 0:
                    result, pgn_game = play_game_n(
                        "Model", "Stockfish", engine_obj, model, max_n,
                        temperature=temperature, stockfish_elo=elo, game_num=i+1
                    )
                    if result == "1-0":  # Model (White) wins
                        wins += 1
                    elif result == "1/2-1/2":
                        draws += 1
                else:
                    result, pgn_game = play_game_n(
                        "Stockfish", "Model", engine_obj, model, max_n,
                        temperature=temperature, stockfish_elo=elo, game_num=i+1
                    )
                    if result == "0-1":  # Model (Black) wins
                        wins += 1
                    elif result == "1/2-1/2":
                        draws += 1

                save_pgn_game(pgn_game, pgn_filename)
                games_played += 1

                current_win_rate = (wins + 0.5 * draws) / (i + 1)
                pbar.set_description(f"Model (n={max_n}) vs Stockfish (WR: {current_win_rate * 100:.1f}%)")

    except Exception as e:
        print(f"Error during game play: {e}")
        print(f"{games_played} games saved to {pgn_filename} before error.")

    print(f"Model wins: {wins}, Draws: {draws}, Losses: {num_games - wins - draws}")

    win_rate = 0
    estimated_model_elo = "N/A"
    if num_games > 0:
        win_rate = (wins + 0.5 * draws) / num_games
        estimated_model_elo = estimate_elo(win_rate, elo)
        print(f"Win rate against Stockfish ELO {elo}: {win_rate * 100:.2f}%")
        print(f"Estimated model ELO: {estimated_model_elo}")

    print(f"All {games_played} games saved to {pgn_filename}")
    return win_rate, estimated_model_elo


def compare_n_values(model=None, n_values=[0, 1, 2, 5, 10], num_games=20,
                     temperature=0.1, elo=1400, pgn_dir="pgns"):
    """
    Compare model performance across different n values.

    Args:
        model: The chess decoder model
        n_values: List of n values to test
        num_games: Number of games per n value
        temperature: Sampling temperature
        elo: Stockfish ELO setting
        pgn_dir: Directory for PGN files

    Returns:
        results: Dict mapping n -> (win_rate, estimated_elo)
    """
    results = {}

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"Testing with max_n = {n}")
        print('='*60)

        win_rate, est_elo = model_vs_stockfish_n(
            model=model,
            num_games=num_games,
            max_n=n,
            temperature=temperature,
            elo=elo,
            pgn_dir=pgn_dir
        )
        results[n] = (win_rate, est_elo)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Performance by n value")
    print('='*60)
    print(f"{'n':>5} | {'Win Rate':>10} | {'Est. ELO':>10}")
    print("-" * 30)
    for n in n_values:
        wr, ee = results[n]
        elo_str = f"{ee:.0f}" if isinstance(ee, (int, float)) else str(ee)
        print(f"{n:>5} | {wr*100:>9.1f}% | {elo_str:>10}")

    return results


if __name__ == "__main__":
    # Example usage - load model and run evaluation
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model vs Stockfish with n-move context")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--max_n", type=int, default=5, help="Maximum history context")
    parser.add_argument("--num_games", type=int, default=20, help="Number of games to play")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--elo", type=int, default=1400, help="Stockfish ELO")
    parser.add_argument("--pgn_dir", type=str, default="pgns", help="Directory for PGN files")
    parser.add_argument("--compare", action="store_true", help="Compare multiple n values")

    args = parser.parse_args()

    # Load model
    from src.models.model import ChessDecoder
    from src.models.vocab import vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ChessDecoder(vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.compare:
        compare_n_values(
            model=model,
            n_values=[0, 1, 2, 5, 10],
            num_games=args.num_games,
            temperature=args.temperature,
            elo=args.elo,
            pgn_dir=args.pgn_dir
        )
    else:
        model_vs_stockfish_n(
            model=model,
            num_games=args.num_games,
            max_n=args.max_n,
            temperature=args.temperature,
            elo=args.elo,
            pgn_dir=args.pgn_dir
        )

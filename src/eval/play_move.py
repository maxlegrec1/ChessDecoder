import torch
# Import python-chess strictly for engine and pgn support
import chess  # python-chess
from chess import engine, pgn
import bulletchess
from bulletchess import CHECKMATE, DRAW
from datetime import datetime # Added for timestamping
 
from tqdm import tqdm
from math import log10
import os

pgn_dir = "./pgns"
os.makedirs(pgn_dir,exist_ok=True)
STOCKFISH_PATH = "/mnt/2tb/ChessRL/stockfish/stockfish-ubuntu-x86-64-avx2"  # Adjust path if needed
STOCKFISH_ELO = 2500  # Set the desired ELO rating
N_GAMES = 200 # Reduced for faster testing, change back to 100 if needed

def get_stockfish_move(engine_obj, bullet_board):
    # Use python-chess board for Stockfish, then convert back to bulletchess.Move
    ch_board = chess.Board(bullet_board.fen())
    result = engine_obj.play(ch_board, engine.Limit(time=0.1))
    return bulletchess.Move.from_uci(result.move.uci())

def estimate_elo(win_rate, stockfish_elo):
    if win_rate == 0:
        return 0
    elif win_rate == 1:
        return stockfish_elo + 400
    if win_rate == 0.5: # Avoid log10(1) which is 0, leading to division by zero if win_rate makes (1-win_rate)/win_rate = 1
        return stockfish_elo
    if win_rate > 0 and win_rate < 1: # ensure win_rate is not 0 or 1 to avoid math errors with log10
        return stockfish_elo - 400 * (log10((1 - win_rate) / win_rate))
    return "N/A"


def play_game(player1_name, player2_name, engine, model1, model2, device, temp=0.1, stockfish_elo=None, game_num=1):
    board = bulletchess.Board()
    pgn_game = pgn.Game()
    pgn_game.headers["Event"] = f"{player1_name} vs {player2_name}"
    pgn_game.headers["Site"] = "Local Machine"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_num)
    pgn_game.headers["White"] = player1_name
    pgn_game.headers["Black"] = player2_name

    if player1_name == "Stockfish" and stockfish_elo:
        pgn_game.headers["WhiteElo"] = str(stockfish_elo)
    if player2_name == "Stockfish" and stockfish_elo:
        pgn_game.headers["BlackElo"] = str(stockfish_elo)

    node = pgn_game # Current node in the PGN game tree
    
    # Initialize move history tracking
    move_history = []
    

    # Helper: determine game-over state via bulletchess predicates
    def _is_game_over(b):
        return (b in CHECKMATE) or (b in DRAW)

    while not _is_game_over(board):
        current_player_name = ""
        model_to_use = None
        is_stockfish_turn = False
        
        if board.turn == bulletchess.WHITE:
            current_player_name = player1_name
            model_to_use = model1
        else:
            current_player_name = player2_name
            model_to_use = model2

        if board.turn == bulletchess.WHITE:
            if player1_name == "Stockfish":
                is_stockfish_turn = True
        else:
            if player2_name == "Stockfish":
                is_stockfish_turn = True

        move = None
        
        if is_stockfish_turn:
            move = get_stockfish_move(engine, board)
        else: # Model's turn
            # Policy-based move selection using move history list
            # move_uci = model_to_use.get_move_from_fen_no_thinking(move_history, T=temp, device=device)
            move_uci = model_to_use.get_move_from_fen_no_thinking(board.fen(), T=temp, device=device)
            if move_uci is None:
                print(f"Model ({current_player_name}) failed to produce a move for FEN: {board.fen()}")
                # Forfeit or error handling
                result = "0-1" if board.turn == bulletchess.WHITE else "1-0"
                pgn_game.headers["Result"] = result
                return result, pgn_game

            try:
                move = bulletchess.Move.from_uci(move_uci)
                if move not in board.legal_moves():
                    print(f"Invalid move by {current_player_name}: {move_uci} in {board.fen()}. Legal moves: {[m.uci() for m in board.legal_moves()]}")
                    # Forfeit due to illegal move
                    result = "0-1" if board.turn == bulletchess.WHITE else "1-0"
                    pgn_game.headers["Result"] = result
                    return result, pgn_game
            except ValueError:
                print(f"Invalid UCI string by {current_player_name}: {move_uci} in {board.fen()}")
                result = "0-1" if board.turn == bulletchess.WHITE else "1-0"
                pgn_game.headers["Result"] = result
                return result, pgn_game


        board.apply(move)
        # Update move history
        move_history.append(move.uci())
        # Convert bulletchess.Move to python-chess.Move for PGN
        pgn_move = chess.Move.from_uci(move.uci())
        node = node.add_variation(pgn_move) # Add move to PGN
            

    # Compute result via bulletchess state
    if board in CHECKMATE:
        # Side to move is checkmated; opponent wins
        result = "0-1" if board.turn == bulletchess.WHITE else "1-0"
    elif board in DRAW:
        result = "1/2-1/2"
    else:
        # Fallback (should not happen if loop condition is correct)
        result = "1/2-1/2"
    pgn_game.headers["Result"] = result
    return result, pgn_game

def main_model_vs_stockfish(model = None,model1_name = "run",device = "cuda",num_games = N_GAMES,temp = 0.1,elo = STOCKFISH_ELO):

    wins, draws = 0, 0
    all_pgn_games = [] # List to store PGN game objects

    pgn_filename = os.path.join(pgn_dir,datetime.now().strftime("games_model_vs_stockfish_%Y%m%d_%H%M%S.pgn"))

    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            
            pbar = tqdm(range(num_games), desc="Model vs Stockfish Games")
            for i in pbar:
                # Alternate who plays white
                if i % 2 == 0:
                    player1_model_name = "Model"
                    player2_stockfish_name = "Stockfish"
                    result, pgn_game = play_game(player1_model_name, player2_stockfish_name, engine, model, None, device, stockfish_elo=elo, game_num=i+1,temp=temp)
                    if result == "1-0": # Model (White) wins
                        wins += 1
                    elif result == "1/2-1/2":
                        draws += 1
                else:
                    player1_stockfish_name = "Stockfish"
                    player2_model_name = "Model"
                    result, pgn_game = play_game(player1_stockfish_name, player2_model_name, engine, None, model, device, stockfish_elo=elo, game_num=i+1,temp=temp)
                    if result == "0-1": # Model (Black) wins
                        wins +=1
                    elif result == "1/2-1/2":
                        draws += 1
                
                all_pgn_games.append(pgn_game)
                
                # Update progress bar description with current win rate
                current_win_rate = (wins + 0.5 * draws) / (i + 1)
                pbar.set_description(f"Model vs Stockfish Games (Win Rate: {current_win_rate * 100:.1f}%)")

    except Exception as e:
        print(f"An error occurred during game play: {e}")
        # Optionally save any games played so far
        if all_pgn_games:
            print(f"Saving {len(all_pgn_games)} games played so far to {pgn_filename}")
            with open(pgn_filename, "w", encoding="utf-8") as f:
                for pgn_game in all_pgn_games:
                    exporter = pgn.FileExporter(f)
                    pgn_game.accept(exporter)
                    f.write("\n\n") # Add some space between games


    print(f"Model wins: {wins}, Draws: {draws}, Losses: {num_games - wins - draws} out of {num_games} games.")
    if num_games > 0:
        win_rate = (wins + 0.5 * draws) / num_games
        estimated_elo = estimate_elo(win_rate, elo)
        print(f"Win rate against Stockfish ELO {elo}: {win_rate * 100:.2f}%")
        print(f"Estimated model ELO: {estimated_elo}")
    else:
        print("No games played.")

    # Save all games to a single PGN file
    with open(pgn_filename, "w", encoding="utf-8") as f:
        for pgn_game in all_pgn_games:
            exporter = chess.pgn.FileExporter(f)
            pgn_game.accept(exporter)
            f.write("\n\n") # Add some space between games for readability
    print(f"All games saved to {pgn_filename}")
    return win_rate, estimated_elo


if __name__ == "__main__":

    from src.models.small import BT4


    model1 = BT4(num_layers=10, d_model=512, d_ff=1024, num_heads=8).to("cuda")
    model1.load_state_dict(torch.load('src/models/small/model.pt'),strict=False)
    # model1.load_state_dict(torch.load('checkpoints/step_00001000.pt',weights_only=False)["model"])
    
    score = main_model_vs_stockfish(model=model2,model1_name="run",device="cuda",num_games=20,temp=1, elo=2500)
    print(score)
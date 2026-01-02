import chess
import pandas as pd
from src.models.vocab import token_to_idx

def fen_to_position_tokens(fen: str):
    board = chess.Board(fen)
    tokens = ["start_pos"]
    
    # Deterministic board scan: a1, b1, ..., h8
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "white" if piece.color == chess.WHITE else "black"
            piece_type = chess.piece_name(piece.piece_type)
            square_name = chess.square_name(square)
            tokens.append(f"{color}_{piece_type}_{square_name}")
            
    tokens.append("end_pos")
    
    # Castling rights
    rights = ""
    if board.has_kingside_castling_rights(chess.WHITE): rights += "K"
    if board.has_queenside_castling_rights(chess.WHITE): rights += "Q"
    if board.has_kingside_castling_rights(chess.BLACK): rights += "k"
    if board.has_queenside_castling_rights(chess.BLACK): rights += "q"
    
    if not rights:
        tokens.append("no_castling_rights")
    else:
        tokens.append(rights)
        
    # Side to move
    tokens.append("white_to_move" if board.turn == chess.WHITE else "black_to_move")
    
    return tokens

def game_to_token_ids(game_df):
    sequence = []
    wdl_data = [] # (index, best_move, wdl, is_valid_wdl)
    
    for _, row in game_df.iterrows():
        pos_tokens = fen_to_position_tokens(row['fen'])
        sequence.extend(pos_tokens)
        
        # The move token is where we want to predict the move AND the WDL
        if 'played_move' in row and row['played_move']:
            # Record the index of the move token in the sequence
            move_idx = len(sequence)
            sequence.append(row['played_move'])
            
            # Target for this index should be the best_move
            best_move = row['best_move']
            
            # Handle WDL NaNs
            win = row['win'] if pd.notna(row['win']) else 0.0
            draw = row['draw'] if pd.notna(row['draw']) else 0.0
            loss = row['loss'] if pd.notna(row['loss']) else 0.0
            
            is_valid_wdl = pd.notna(row['win']) and pd.notna(row['draw']) and pd.notna(row['loss'])
            
            wdl = [win, draw, loss]
            wdl_data.append((move_idx, best_move, wdl, is_valid_wdl))
            
    ids = [token_to_idx[t] for t in sequence]
    return ids, wdl_data

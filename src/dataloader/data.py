import chess
import pandas as pd
import random
from src.models.vocab import token_to_idx


def fen_to_position_tokens(fen: str):
    """
    Convert FEN to fixed-length position tokens.
    
    Output format (68 tokens total):
        - start_pos (1 token)
        - 64 board tokens in order a1, b1, c1, ..., h8 (each is either 'empty' or 'color_piece')
        - end_pos (1 token)
        - castling rights (1 token)
        - side to move (1 token)
    """
    board = chess.Board(fen)
    tokens = ["start_pos"]
    
    # Fixed 64 board tokens: a1, b1, c1, ..., h8
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "white" if piece.color == chess.WHITE else "black"
            piece_type = chess.piece_name(piece.piece_type)
            tokens.append(f"{color}_{piece_type}")
        else:
            tokens.append("empty")
    
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


def game_to_token_ids(game_df, skip_board_prob=0.0):
    sequence = []
    wdl_data = []  # (index, best_move, wdl, is_valid_wdl)
    block_boundaries = []  # [(start_idx, end_idx), ...] for each board block

    # Use itertuples() instead of iterrows() for ~10-100x faster iteration
    for i, row in enumerate(game_df.itertuples(index=False)):
        # Always include the first board, or include board with (1 - skip_board_prob)
        include_board = (i == 0) or (random.random() > skip_board_prob)

        if include_board:
            block_start_idx = len(sequence)  # Track block start
            pos_tokens = fen_to_position_tokens(row.fen)
            sequence.extend(pos_tokens)

        # The move token is where we want to predict the move AND the WDL
        played_move = getattr(row, 'played_move', None)
        if played_move:
            # Record the index of the move token in the sequence
            move_idx = len(sequence)
            sequence.append(played_move)

            if include_board:
                # Block includes start_pos through side_to_move (exclusive of move token)
                # This matches legacy behavior where mask[s_idx:m_idx, s_idx:m_idx] = True
                block_boundaries.append((block_start_idx, move_idx))

            # Target for this index should be the best_move
            best_move = row.best_move

            # Handle WDL NaNs
            win = row.win if pd.notna(row.win) else 0.0
            draw = row.draw if pd.notna(row.draw) else 0.0
            loss = row.loss if pd.notna(row.loss) else 0.0

            is_valid_wdl = pd.notna(row.win) and pd.notna(row.draw) and pd.notna(row.loss)

            wdl = [win, draw, loss]
            wdl_data.append((move_idx, best_move, wdl, is_valid_wdl))

    ids = [token_to_idx[t] for t in sequence]
    return ids, wdl_data, block_boundaries

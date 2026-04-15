import chess
import pandas as pd
import random
from chessdecoder.models.vocab import token_to_idx


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
    move_target_data = []  # (move_idx, best_move) — policy head supervision
    block_boundaries = []  # [(start_idx, end_idx), ...] for each board block
    value_data = []  # (wl_pos, d_pos, wl, d, is_valid_wdl)

    for i, row in enumerate(game_df.itertuples(index=False)):
        include_board = (i == 0) or (random.random() > skip_board_prob)

        if include_board:
            block_start_idx = len(sequence)
            pos_tokens = fen_to_position_tokens(row.fen)
            sequence.extend(pos_tokens)

        played_move = getattr(row, 'played_move', None)
        if played_move:
            move_idx = len(sequence)
            sequence.append(played_move)

            if include_board:
                block_boundaries.append((block_start_idx, move_idx))

            best_move = row.best_move
            move_target_data.append((move_idx, best_move))

            # Append WL and D placeholder tokens
            wl_pos = len(sequence)
            sequence.append("wl_value")
            d_pos = len(sequence)
            sequence.append("d_value")

            # WL/D targets: use the played move's Q/D values so the model learns
            # action-value estimation rather than a position-level state value.
            # played_q ∈ [-1, 1] is the Leela Q value for the move that was played;
            # played_d ∈ [0, 1] is its draw probability.
            # (Previously best_q / best_d were used here — an oversight.)
            raw_pq = getattr(row, 'played_q', None)
            raw_pd = getattr(row, 'played_d', None)
            is_valid_wdl = (raw_pq is not None and pd.notna(raw_pq) and
                            raw_pd is not None and pd.notna(raw_pd))
            wl = float(raw_pq) if is_valid_wdl else 0.0   # WL = win - loss = Q
            d  = float(raw_pd) if is_valid_wdl else 0.0   # D  = draw prob

            value_data.append((wl_pos, d_pos, wl, d, is_valid_wdl))

    ids = [token_to_idx[t] for t in sequence]
    return ids, move_target_data, block_boundaries, value_data

import chess

from chessdecoder.models.vocab import token_to_idx  # noqa: F401  (kept for clarity / downstream)


def fen_to_position_tokens(fen: str):
    """Convert FEN to fixed-length 68 position tokens:
    start_pos, 64 squares (a1..h8), end_pos, castling, side_to_move.
    """
    board = chess.Board(fen)
    tokens = ["start_pos"]

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "white" if piece.color == chess.WHITE else "black"
            piece_type = chess.piece_name(piece.piece_type)
            tokens.append(f"{color}_{piece_type}")
        else:
            tokens.append("empty")

    tokens.append("end_pos")

    rights = ""
    if board.has_kingside_castling_rights(chess.WHITE): rights += "K"
    if board.has_queenside_castling_rights(chess.WHITE): rights += "Q"
    if board.has_kingside_castling_rights(chess.BLACK): rights += "k"
    if board.has_queenside_castling_rights(chess.BLACK): rights += "q"

    tokens.append(rights if rights else "no_castling_rights")
    tokens.append("white_to_move" if board.turn == chess.WHITE else "black_to_move")

    return tokens

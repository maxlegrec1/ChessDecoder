import numpy as np
import torch
import bulletchess
from typing import List, Tuple, Optional
from .vocab import policy_index

# Precompute policy string to index mapping for O(1) lookups
policy_to_idx = {u: i for i, u in enumerate(policy_index)}


def _board_to_12_piece_planes(board: bulletchess.Board) -> np.ndarray:
    piece_types = [bulletchess.PAWN, bulletchess.KNIGHT, bulletchess.BISHOP, bulletchess.ROOK, bulletchess.QUEEN, bulletchess.KING]
    piece_colors = [bulletchess.WHITE, bulletchess.BLACK]

    planes = []
    for color in piece_colors:
        for piece_type in piece_types:
            mask = np.zeros((8, 8), dtype=np.float32)
            # Use board[color, piece_type] to get Bitboard, then iterate over squares
            bitboard = board[color, piece_type]
            for square in bitboard:
                # In bulletchess, squares have an index() method that returns 0-63
                square_idx = square.index()
                rank = square_idx // 8
                file = square_idx % 8
                mask[rank][file] = 1.0
            planes.append(mask)
    # Shape (8,8,12)
    return np.transpose(np.array(planes, dtype=np.float32), (1, 2, 0))


def _castling_planes(board: bulletchess.Board) -> np.ndarray:
    # Order must match existing model expectation via ustotheirs:
    # [WQ, WK, BQ, BK]
    wq = 1.0 if bulletchess.WHITE_QUEENSIDE in board.castling_rights else 0.0
    wk = 1.0 if bulletchess.WHITE_KINGSIDE in board.castling_rights else 0.0
    bq = 1.0 if bulletchess.BLACK_QUEENSIDE in board.castling_rights else 0.0
    bk = 1.0 if bulletchess.BLACK_KINGSIDE in board.castling_rights else 0.0
    planes = [
        np.full((8, 8), wq, dtype=np.float32),
        np.full((8, 8), wk, dtype=np.float32),
        np.full((8, 8), bq, dtype=np.float32),
        np.full((8, 8), bk, dtype=np.float32),
    ]
    return np.stack(planes, axis=0)  # (4,8,8)


def _mirror_board(board: bulletchess.Board) -> bulletchess.Board:
    """
    Fast mirror implementation for bulletchess.Board.
    Mirrors the board (flips ranks 1<->8, 2<->7, etc.) and flips colors.
    """
    # Create empty board
    mirrored = bulletchess.Board.empty()
    
    # Mirror all pieces
    for square in bulletchess.SQUARES:
        piece = board[square]
        if piece is not None:
            # Calculate mirrored square: flip rank (0-7 -> 7-0), keep file
            square_idx = square.index()
            rank = square_idx // 8
            file = square_idx % 8
            mirrored_rank = 7 - rank
            mirrored_idx = mirrored_rank * 8 + file
            mirrored_square = bulletchess.SQUARES[mirrored_idx]
            
            # Flip piece color
            mirrored_color = piece.color.opposite
            mirrored[mirrored_square] = bulletchess.Piece(mirrored_color, piece.piece_type)
    
    # Mirror castling rights: swap white<->black
    # Build castling rights by checking each type and creating CastlingRights
    new_castling_types = []
    if bulletchess.WHITE_KINGSIDE in board.castling_rights:
        new_castling_types.append(bulletchess.BLACK_KINGSIDE)
    if bulletchess.WHITE_QUEENSIDE in board.castling_rights:
        new_castling_types.append(bulletchess.BLACK_QUEENSIDE)
    if bulletchess.BLACK_KINGSIDE in board.castling_rights:
        new_castling_types.append(bulletchess.WHITE_KINGSIDE)
    if bulletchess.BLACK_QUEENSIDE in board.castling_rights:
        new_castling_types.append(bulletchess.WHITE_QUEENSIDE)
    
    # Build CastlingRights from list of types
    if new_castling_types:
        mirrored.castling_rights = bulletchess.CastlingRights(new_castling_types)
    else:
        mirrored.castling_rights = bulletchess.NO_CASTLING
    
    # Flip turn
    mirrored.turn = board.turn.opposite
    
    # Mirror en passant square if exists
    if board.en_passant_square is not None:
        ep_idx = board.en_passant_square.index()
        ep_rank = ep_idx // 8
        ep_file = ep_idx % 8
        mirrored_ep_rank = 7 - ep_rank
        mirrored_ep_idx = mirrored_ep_rank * 8 + ep_file
        mirrored.en_passant_square = bulletchess.SQUARES[mirrored_ep_idx]
    
    # Copy move counters
    mirrored.halfmove_clock = board.halfmove_clock
    mirrored.fullmove_number = board.fullmove_number
    
    return mirrored


def _build_snapshots(board: bulletchess.Board) -> List[bulletchess.Board]:
    # snapshots[0] is current, snapshots[1] one ply ago, ... up to 7 plies ago
    temp = board.copy()
    snaps: List[bulletchess.Board] = [temp.copy()]
    for _ in range(7):
        # Check if there are moves to undo by checking if undo() returns None
        try:
            temp.undo()
            snaps.append(temp.copy())
        except (IndexError, AttributeError):
            # No more moves to undo
            snaps.append(None)  # type: ignore
    return snaps


def encode_moves_to_tensor(uci_moves: List[str], starting_fen: Optional[str] = None) -> Tuple[torch.Tensor, np.ndarray]:
    board = bulletchess.Board.from_fen(starting_fen) if starting_fen is not None else bulletchess.Board()
    for mv in uci_moves:
        move = bulletchess.Move.from_uci(mv)
        board.apply(move)

    # Build history snapshots (current first)
    snapshots = _build_snapshots(board)

    # Always encode from white's perspective; mirror all snapshots if black to move
    mirror = (board.turn == bulletchess.BLACK)
    if mirror:
        snapshots = [_mirror_board(s) if s is not None else None for s in snapshots]

    # Assemble 112-channel tensor
    # 8 groups: each 12 piece planes + 1 blank = 13; total 104
    channels: List[np.ndarray] = []
    for i in range(8):
        if snapshots[i] is not None:
            planes12 = _board_to_12_piece_planes(snapshots[i])  # (8,8,12)
            channels.append(planes12)
        else:
            channels.append(np.zeros((8, 8, 12), dtype=np.float32))
        # blank plane
        channels.append(np.zeros((8, 8, 1), dtype=np.float32))

    # Special planes: WQ, WK, BQ, BK, is_black_to_move, blank, blank, ones
    current_for_flags = snapshots[0]
    assert current_for_flags is not None
    castling = _castling_planes(current_for_flags)  # (4,8,8)
    is_black_to_move = 1.0 if (board.turn == bulletchess.BLACK) else 0.0
    specials = [
        castling[0:1, :, :],  # WQ
        castling[1:2, :, :],  # WK
        castling[2:3, :, :],  # BQ
        castling[3:4, :, :],  # BK
        np.full((1, 8, 8), is_black_to_move, dtype=np.float32),
        np.zeros((1, 8, 8), dtype=np.float32),
        np.zeros((1, 8, 8), dtype=np.float32),
        np.ones((1, 8, 8), dtype=np.float32),
    ]

    # Concatenate to (8,8,112)
    stacked = np.concatenate(channels, axis=2)  # (8,8,104)
    specials_hwk = np.transpose(np.concatenate(specials, axis=0), (1, 2, 0))  # (8,8,8)
    final_hwk = np.concatenate([stacked, specials_hwk], axis=2)  # (8,8,112)

    # Convert to tensor (1,112,8,8)
    final_tensor = torch.from_numpy(final_hwk).permute(2, 0, 1).unsqueeze(0).float()

    # Legal moves mask built from mirrored board to match policy_index perspective
    board_for_mask = _mirror_board(board) if (board.turn == bulletchess.BLACK) else board.copy()
    lm = np.ones(1858, dtype=np.float32) * (-1000)
    
    # Collect all legal moves as UCI strings
    legal_moves_uci = set()
    for possible in board_for_mask.legal_moves():
        u = possible.uci()
        if u[-1] != 'n':
            legal_moves_uci.add(u)
        else:
            legal_moves_uci.add(u[:-1])
    
    # Mark all legal moves
    for u in legal_moves_uci:
        idx = policy_to_idx.get(u)
        if idx is not None:
            lm[idx] = 0
    
    # Add castling moves as king-to-rook-square moves ONLY if the corresponding 
    # standard castling move is actually legal (to verify castling is possible)
    # White kingside: e1h1 (king to rook square) if e1g1 is legal
    if "e1g1" in legal_moves_uci:
        castling_move = "e1h1"
        idx = policy_to_idx.get(castling_move)
        if idx is not None:
            lm[idx] = 0
    
    # White queenside: e1a1 (king to rook square) if e1c1 is legal
    if "e1c1" in legal_moves_uci:
        castling_move = "e1a1"
        idx = policy_to_idx.get(castling_move)
        if idx is not None:
            lm[idx] = 0
    
    # Black kingside: e8h8 (king to rook square) if e8g8 is legal
    if "e8g8" in legal_moves_uci:
        castling_move = "e8h8"
        idx = policy_to_idx.get(castling_move)
        if idx is not None:
            lm[idx] = 0
    
    # Black queenside: e8a8 (king to rook square) if e8c8 is legal
    if "e8c8" in legal_moves_uci:
        castling_move = "e8a8"
        idx = policy_to_idx.get(castling_move)
        if idx is not None:
            lm[idx] = 0

    return final_tensor, lm


def encode_fen_to_tensor(fen: str) -> Tuple[torch.Tensor, np.ndarray]:
    board = bulletchess.Board.from_fen(fen)

    # History: only current snapshot, others are zeros
    snapshots = [board.copy()] + [None] * 7

    # Mirror snapshots if black to move so encoding is from white's perspective
    if board.turn == bulletchess.BLACK:
        snapshots = [_mirror_board(s) if s is not None else None for s in snapshots]

    # Assemble 112-channel tensor
    channels: List[np.ndarray] = []
    for i in range(8):
        if snapshots[i] is not None:
            planes12 = _board_to_12_piece_planes(snapshots[i])
            channels.append(planes12)
        else:
            channels.append(np.zeros((8, 8, 12), dtype=np.float32))
        channels.append(np.zeros((8, 8, 1), dtype=np.float32))

    current_for_flags = snapshots[0]
    assert current_for_flags is not None
    castling = _castling_planes(current_for_flags)
    is_black_to_move = 1.0 if (board.turn == bulletchess.BLACK) else 0.0
    specials = [
        castling[0:1, :, :],
        castling[1:2, :, :],
        castling[2:3, :, :],
        castling[3:4, :, :],
        np.full((1, 8, 8), is_black_to_move, dtype=np.float32),
        np.zeros((1, 8, 8), dtype=np.float32),
        np.zeros((1, 8, 8), dtype=np.float32),
        np.ones((1, 8, 8), dtype=np.float32),
    ]

    stacked = np.concatenate(channels, axis=2)
    specials_hwk = np.transpose(np.concatenate(specials, axis=0), (1, 2, 0))
    final_hwk = np.concatenate([stacked, specials_hwk], axis=2)

    final_tensor = torch.from_numpy(final_hwk).permute(2, 0, 1).unsqueeze(0).float()

    # Legal moves mask from mirrored perspective when black to move
    board_for_mask = _mirror_board(board) if (board.turn == bulletchess.BLACK) else board.copy()
    lm = np.ones(1858, dtype=np.float32) * (-1000)
    
    # Collect all legal moves as UCI strings
    legal_moves_uci = set()
    for possible in board_for_mask.legal_moves():
        u = possible.uci()
        if u[-1] != 'n':
            legal_moves_uci.add(u)
        else:
            legal_moves_uci.add(u[:-1])
    
    # Mark all legal moves
    for u in legal_moves_uci:
        idx = policy_to_idx.get(u)
        if idx is not None:
            lm[idx] = 0
    
    # Add castling moves as king-to-rook-square moves ONLY if the corresponding 
    # standard castling move is actually legal (to verify castling is possible)
    # White kingside: e1h1 (king to rook square) if e1g1 is legal
    if "e1g1" in legal_moves_uci:
        castling_move = "e1h1"
        idx = policy_to_idx.get(castling_move)
        if idx is not None:
            lm[idx] = 0
    
    # White queenside: e1a1 (king to rook square) if e1c1 is legal
    if "e1c1" in legal_moves_uci:
        castling_move = "e1a1"
        idx = policy_to_idx.get(castling_move)
        if idx is not None:
            lm[idx] = 0
    
    # Black kingside: e8h8 (king to rook square) if e8g8 is legal
    if "e8g8" in legal_moves_uci:
        castling_move = "e8h8"
        idx = policy_to_idx.get(castling_move)
        if idx is not None:
            lm[idx] = 0
    
    # Black queenside: e8a8 (king to rook square) if e8c8 is legal
    if "e8c8" in legal_moves_uci:
        castling_move = "e8a8"
        idx = policy_to_idx.get(castling_move)
        if idx is not None:
            lm[idx] = 0

    return final_tensor, lm


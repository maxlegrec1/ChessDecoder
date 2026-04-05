"""UCI move normalization helpers.

The model vocabulary encodes castling as king-takes-rook (e.g. e1h1 for
white O-O), while standard UCI uses king-to-target-square (e1g1). Both
directions are needed: `normalize_castling` converts model output →
standard UCI (for board legality checks, comparison with Stockfish,
etc.), and `to_model_uci` converts standard UCI → model vocab (for
encoding training targets).
"""

PSEUDO_TO_STANDARD = {
    "e1h1": "e1g1",
    "e1a1": "e1c1",
    "e8h8": "e8g8",
    "e8a8": "e8c8",
}

STANDARD_TO_PSEUDO = {v: k for k, v in PSEUDO_TO_STANDARD.items()}


def normalize_castling(move: str) -> str:
    """Convert a king-takes-rook castling move to standard UCI, passthrough otherwise."""
    return PSEUDO_TO_STANDARD.get(move, move)


def to_model_uci(move: str) -> str:
    """Convert standard UCI castling to the king-takes-rook form used by the model vocab."""
    return STANDARD_TO_PSEUDO.get(move, move)

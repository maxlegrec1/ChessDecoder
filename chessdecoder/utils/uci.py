"""UCI move normalization helpers."""

# The model emits king-takes-rook castling moves (e.g. e1h1 for white O-O),
# but standard UCI uses king-to-target-square (e1g1). Map pseudo → standard.
PSEUDO_TO_STANDARD = {
    "e1h1": "e1g1",
    "e1a1": "e1c1",
    "e8h8": "e8g8",
    "e8a8": "e8c8",
}


def normalize_castling(move: str) -> str:
    """Convert a king-takes-rook castling move to standard UCI, passthrough otherwise."""
    return PSEUDO_TO_STANDARD.get(move, move)

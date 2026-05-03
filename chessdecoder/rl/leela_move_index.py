"""Mapping between UCI moves and the 1858-element Leela policy index.

The HuggingFace ChessFENS dataset (and other Leela-derived data) stores
policy as a 1858-float array, where index `i` corresponds to the UCI move
at line `i+1` of `chessdecoder/cpp/mcts/leela_policy_index.txt`. The vocab
is **white-perspective**: black-to-move positions are mirrored (rank 1↔8)
before the policy is computed, and the move predicted by the model must
be mirrored back to the same convention before lookup.

Castling is encoded as king-takes-rook (e1h1, e1a1, e8h8, e8a8) in this
vocab — same convention as `chessdecoder.utils.uci.to_model_uci`.

Engine-agnostic: pure python, no torch, no cutlass dependency. Safe to
land on `main`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from chessdecoder.utils.uci import to_model_uci


_LEELA_VOCAB_PATH = (
    Path(__file__).resolve().parents[1] / "cpp" / "mcts" / "leela_policy_index.txt"
)


@lru_cache(maxsize=1)
def _load_vocab() -> tuple[list[str], dict[str, int]]:
    """Load the 1858 UCI move strings and the inverse map."""
    with open(_LEELA_VOCAB_PATH) as f:
        moves = [ln.strip() for ln in f if ln.strip()]
    if len(moves) != 1858:
        raise RuntimeError(
            f"Expected 1858 moves in {_LEELA_VOCAB_PATH}, got {len(moves)}"
        )
    return moves, {m: i for i, m in enumerate(moves)}


def leela_vocab() -> list[str]:
    """Return the white-perspective UCI vocabulary (length 1858)."""
    return _load_vocab()[0]


def _mirror_rank_uci(move: str) -> str:
    """Flip ranks 1↔8, 2↔7, ... in the from/to squares of a UCI move.
    Pass non-UCI strings through unchanged."""
    if len(move) < 4:
        return move
    out = list(move)
    for i in (1, 3):
        c = out[i]
        if "1" <= c <= "8":
            out[i] = chr(ord("0") + (9 - (ord(c) - ord("0"))))
    return "".join(out)


def _side_to_move_is_black(fen: str) -> bool:
    parts = fen.split()
    return len(parts) >= 2 and parts[1] == "b"


def policy_idx_for_move(move: str, fen: str) -> int | None:
    """Return the Leela policy index for `move` played from `fen`, or None
    if the move can't be mapped (e.g. underpromotions outside the vocab)."""
    _, vocab = _load_vocab()
    # Normalize king-target castling (e1g1) → king-takes-rook (e1h1) for
    # vocab lookup. Pass-through for non-castling moves.
    canon = to_model_uci(move)
    # Mirror to white-perspective if black to move.
    if _side_to_move_is_black(fen):
        canon = _mirror_rank_uci(canon)
    return vocab.get(canon)


def policy_prob(move: str, fen: str, policy_array) -> float:
    """Probability the Leela teacher assigned to `move` at `fen`.

    `policy_array` is a 1858-float array (e.g. numpy / list). Illegal
    entries are stored as -1 in the upstream dataset; this function maps
    those (and unknown moves) to 0.0.
    """
    idx = policy_idx_for_move(move, fen)
    if idx is None:
        return 0.0
    p = float(policy_array[idx])
    return max(0.0, p)

"""Shared helpers for task generators."""
from __future__ import annotations

import chess

from chessdecoder.agent import patch_vocab as pv

_CASTLE_FIX = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}


def apply_uci(board: chess.Board, uci: str) -> bool:
    """Apply a corpus move string in place (lc0 castling + bare knight-promo
    spellings handled). False if illegal (corpus glitch / ambiguous token)."""
    try:
        mv = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return False
    if not board.is_legal(mv):
        fixed = _CASTLE_FIX.get(uci)
        if fixed is None:
            # bare knight-promo token: 'a7a8' meaning a7a8n
            if len(uci) == 4 and uci[1] in "27" and uci[3] in "18":
                mv2 = chess.Move.from_uci(uci + "n")
                if board.is_legal(mv2):
                    board.push(mv2)
                    return True
            return False
        mv = chess.Move.from_uci(fixed)
        if not board.is_legal(mv):
            return False
    board.push(mv)
    return True


def qbin_centered(qbin: int) -> float:
    """q-bin -> centered scalar in (-1, 1) for comparisons."""
    return (qbin + 0.5) / pv.N_QBIN * 2.0 - 1.0


def stm_of(board_ids: list[int]) -> int:
    """0 = white to move, 1 = black, from a 19-token board encoding."""
    return board_ids[17] - pv.STM_BASE


def sample_line(games, rng, k: int, random_walk_frac: float = 0.2):
    """(start_board, [k ucis], game, idx) from a real game, or a random-legal
    walk with prob random_walk_frac. None,...*3 on repeated bad luck."""
    for _ in range(10):
        g = games[rng.randrange(len(games))]
        if len(g) <= k:
            continue
        i = rng.randrange(len(g) - k)
        try:
            board = chess.Board(g[i][0])
        except Exception:
            continue
        if rng.random() < random_walk_frac:
            walk, b = [], board.copy(stack=False)
            for _ in range(k):
                legal = list(b.legal_moves)
                if not legal:
                    break
                mv = rng.choice(legal)
                walk.append(mv.uci())
                b.push(mv)
            if len(walk) == k:
                return board, walk, g, i
            continue
        return board, [g[i + j][1] for j in range(k)], g, i
    return None, None, None, None


def encode_line_tokens(board: chess.Board, ucis: list[str]):
    """Validate+apply a uci line; returns (move_tokens, final_board) or
    (None, None)."""
    work = board.copy(stack=False)
    toks = []
    for u in ucis:
        t = pv.uci_to_token(u)
        if t is None or not apply_uci(work, u):
            return None, None
        toks.append(t)
    return toks, work

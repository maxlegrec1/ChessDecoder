"""Board-local tasks: world model, policy priors, board structure."""
from __future__ import annotations

import chess

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.tasks import Example, Task, register
from chessdecoder.agent.tasks.helpers import (apply_uci, encode_line_tokens,
                                              sample_line)


def _line_example(board, ucis, tid):
    b0 = pv.encode_board(board)
    mv_toks, work = encode_line_tokens(board, ucis)
    if mv_toks is None:
        return None
    b1 = pv.encode_board(work)
    ids = [pv.ROOT] + b0 + [pv.LINE] + mv_toks + [pv.PROBE] + b1
    n_ctx = 1 + 19 + 1 + len(mv_toks) + 1
    return Example(ids, [False] * n_ctx + [True] * 19, tid, (n_ctx, 19))


@register
class Copy(Task):
    name, tid, requires = "t1_copy", 1, ("games",)

    def make(self, src, rng):
        board, _, _, _ = sample_line(src.games, rng, 1)
        if board is None:
            return None
        b = pv.encode_board(board)
        return Example([pv.ROOT] + b + [pv.PROBE] + b,
                       [False] * 21 + [True] * 19, self.tid, (21, 19))


@register
class Apply(Task):
    name, tid, requires = "t2_apply", 2, ("games",)

    def make(self, src, rng):
        board, ucis, _, _ = sample_line(src.games, rng, 1)
        if board is None:
            return None
        return _line_example(board, ucis, self.tid)


@register
class ApplyLine(Task):
    name, tid, requires = "t3_line", 3, ("games",)

    def make(self, src, rng):
        k = rng.randint(2, 10)
        board, ucis, _, _ = sample_line(src.games, rng, k)
        if board is None:
            return None
        return _line_example(board, ucis, self.tid)


@register
class LegalSample(Task):
    """3 uniform-random legal moves — teaches the legality manifold."""
    name, tid, requires = "t7_legal", 7, ("games",)

    def make(self, src, rng):
        board, _, _, _ = sample_line(src.games, rng, 1)
        if board is None:
            return None
        legal = [m.uci() for m in board.legal_moves]
        toks = [pv.uci_to_token(u) for u in legal]
        toks = [t for t in toks if t is not None]
        if not toks:
            return None
        picks = [toks[rng.randrange(len(toks))] for _ in range(3)] \
            if len(toks) < 3 else rng.sample(toks, 3)
        ids = [pv.ROOT] + pv.encode_board(board) + [pv.LEGAL] + picks
        return Example(ids, [False] * 21 + [True] * 3, self.tid)


@register
class PlayedMove(Task):
    """Behavioral cloning of the data distribution (NOT the oracle)."""
    name, tid, requires = "t8_played", 8, ("games",)

    def make(self, src, rng):
        board, ucis, _, _ = sample_line(src.games, rng, 1, random_walk_frac=0.0)
        if board is None:
            return None
        t = pv.uci_to_token(ucis[0])
        if t is None:
            return None
        ids = [pv.ROOT] + pv.encode_board(board) + [pv.PLAYED, t]
        return Example(ids, [False] * 21 + [True], self.tid)


@register
class LastMove(Task):
    """Which move produced this position (provenance reading)."""
    name, tid, requires = "t10_lastmove", 10, ("games",)

    def make(self, src, rng):
        for _ in range(10):
            g = src.games[rng.randrange(len(src.games))]
            if len(g) < 3:
                continue
            i = rng.randrange(1, len(g))
            t = pv.uci_to_token(g[i - 1][1])
            if t is None:
                continue
            try:
                b = pv.encode_board(chess.Board(g[i][0]))
            except Exception:
                continue
            return Example([pv.ROOT] + b + [pv.LASTMOVE, t],
                           [False] * 21 + [True], self.tid)
        return None


@register
class Horizon(Task):
    """Plies until the game ended (clamped to the NUM region)."""
    name, tid, requires = "t11_horizon", 11, ("games",)

    def make(self, src, rng):
        g = src.games[rng.randrange(len(src.games))]
        i = rng.randrange(len(g))
        try:
            b = pv.encode_board(chess.Board(g[i][0]))
        except Exception:
            return None
        h = min(len(g) - i, pv.N_NUM - 1)
        return Example([pv.ROOT] + b + [pv.HORIZON, pv.num_token(h)],
                       [False] * 21 + [True], self.tid)


@register
class PathBetween(Task):
    """bA + bB (1-4 plies apart) -> the connecting move sequence."""
    name, tid, requires = "t12_path", 12, ("games",)

    def make(self, src, rng):
        k = rng.randint(1, 4)
        board, ucis, _, _ = sample_line(src.games, rng, k)
        if board is None:
            return None
        mv_toks, work = encode_line_tokens(board, ucis)
        if mv_toks is None:
            return None
        ids = ([pv.ROOT] + pv.encode_board(board) + [pv.TARGET]
               + pv.encode_board(work) + [pv.LINE] + mv_toks)
        n_ctx = 1 + 19 + 1 + 19 + 1
        return Example(ids, [False] * n_ctx + [True] * k, self.tid, (n_ctx, k))


@register
class Opening(Task):
    """Game's first 4 plies from any position (late = opening-family prior)."""
    name, tid, requires = "t17_opening", 17, ("games",)

    def make(self, src, rng):
        for _ in range(10):
            g = src.games[rng.randrange(len(src.games))]
            if len(g) < 6:
                continue
            toks = [pv.uci_to_token(g[j][1]) for j in range(4)]
            if any(t is None for t in toks):
                continue
            i = rng.randrange(len(g))
            try:
                b = pv.encode_board(chess.Board(g[i][0]))
            except Exception:
                continue
            return Example([pv.ROOT] + b + [pv.OPENING] + toks,
                           [False] * 21 + [True] * 4, self.tid)
        return None


@register
class FillBlank(Task):
    """Masked-patch completion (board MAE): 4-8 of 16 patches masked; answer
    = the original patches in ascending slot order."""
    name, tid, requires = "t18_fill", 18, ("games",)

    def make(self, src, rng):
        board, _, _, _ = sample_line(src.games, rng, 1)
        if board is None:
            return None
        b = pv.encode_board(board)
        n_mask = rng.randint(4, 8)
        slots = sorted(rng.sample(range(16), n_mask))
        masked = list(b)
        answer = []
        for s in slots:
            answer.append(b[s])
            masked[s] = pv.MASK
        ids = [pv.ROOT] + masked + [pv.FILL] + answer
        n_ctx = 1 + 19 + 1
        return Example(ids, [False] * n_ctx + [True] * n_mask, self.tid,
                       (n_ctx, n_mask))

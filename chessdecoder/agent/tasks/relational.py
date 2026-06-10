"""Long-context relational tasks: cross-board, cross-value reasoning."""
from __future__ import annotations

import chess

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.tasks import Example, Task, register
from chessdecoder.agent.tasks.helpers import qbin_centered, stm_of


def _blocks(rows):
    """rows of (b19, q, d, _) -> (ids, loss) context of probe blocks."""
    ids, loss = [], []
    for b, q, d, _ in rows:
        ids += [pv.PROBE] + b + [pv.ORACLE, pv.QBIN_BASE + q, pv.DBIN_BASE + d]
        loss += [False] * 23
    return ids, loss


@register
class Aggregate(Task):
    """<best> raw-bin argmax, or 3x value recall."""
    name, tid, requires = "t4_agg", 4, ("labels",)

    def make(self, src, rng):
        n = rng.randint(4, 10)
        rows = src.next_labels(n)
        ids, loss = _blocks(rows)
        qbins = [r[1] for r in rows]
        if rng.random() < 0.5:
            ids += [pv.BEST, pv.num_token(max(range(n), key=lambda i: qbins[i]))]
            loss += [False, True]
        else:
            for j in rng.sample(range(n), k=min(3, n)):
                ids += [pv.RECALL, pv.num_token(j), pv.QBIN_BASE + qbins[j]]
                loss += [False, False, True]
        return Example(ids, loss, self.tid)


@register
class AtDistance(Task):
    """Apply a line to the ROOT after distractor probe blocks."""
    name, tid, requires = "t6_distance", 6, ("games", "labels")

    def make(self, src, rng):
        from chessdecoder.agent.tasks.helpers import encode_line_tokens, sample_line
        k = rng.randint(1, 4)
        board, ucis, _, _ = sample_line(src.games, rng, k)
        if board is None:
            return None
        mv_toks, work = encode_line_tokens(board, ucis)
        if mv_toks is None:
            return None
        ids = [pv.ROOT] + pv.encode_board(board)
        loss = [False] * 20
        bids, bloss = _blocks(src.next_labels(rng.randint(3, 8)))
        ids += bids
        loss += bloss
        ids += [pv.LINE] + mv_toks + [pv.PROBE]
        loss += [False] * (1 + k + 1)
        n_ctx = len(ids)
        ids += pv.encode_board(work)
        loss += [True] * 19
        return Example(ids, loss, self.tid, (n_ctx, 19))


@register
class TrajectoryTrack(Task):
    """2-3 interleaved game threads; <next> j -> index of board j's successor
    block (3 queries per example)."""
    name, tid, requires = "t13_traj", 13, ("games",)

    def make(self, src, rng):
        n_games = rng.randint(2, 3)
        per = rng.randint(3, 4)
        threads = []
        for _ in range(n_games):
            for _try in range(10):
                g = src.games[rng.randrange(len(src.games))]
                if len(g) <= per:
                    continue
                i = rng.randrange(len(g) - per)
                try:
                    boards = [pv.encode_board(chess.Board(g[i + j][0]))
                              for j in range(per)]
                except Exception:
                    continue
                threads.append(boards)
                break
        if len(threads) < 2:
            return None
        # interleave (thread, step) in random order, tracking block indices
        order = [(t, s) for t, th in enumerate(threads) for s in range(len(th))]
        rng.shuffle(order)
        # keep temporal order within each thread
        seen = {t: 0 for t in range(len(threads))}
        seq = []
        for t, _ in order:
            seq.append((t, seen[t]))
            seen[t] += 1
        seq = [(t, s) for t, s in seq if s < len(threads[t])]
        block_of = {}
        ids, loss = [], []
        for bi, (t, s) in enumerate(seq):
            block_of[(t, s)] = bi
            ids += [pv.PROBE] + threads[t][s]
            loss += [False] * 20
        # queries: blocks with an in-context successor
        cands = [(t, s) for (t, s) in block_of if (t, s + 1) in block_of]
        if not cands:
            return None
        for t, s in rng.sample(cands, k=min(3, len(cands))):
            ids += [pv.NEXT, pv.num_token(block_of[(t, s)]),
                    pv.num_token(block_of[(t, s + 1)])]
            loss += [False, False, True]
        return Example(ids, loss, self.tid)


@register
class ValueSwing(Task):
    """Parent->child labeled pairs; <swing> -> index of the pair whose move
    changed the (parent-POV) eval the most."""
    name, tid, requires = "t14_swing", 14, ("paired",)

    def make(self, src, rng):
        n = rng.randint(3, 6)
        pairs = src.next_paired(n)
        ids, loss = [], []
        swings = []
        for bp, qp, dp, mt, bc, qc, dc in pairs:
            ids += ([pv.PROBE] + bp + [pv.ORACLE, pv.QBIN_BASE + qp,
                     pv.DBIN_BASE + dp, pv.LINE, mt]
                    + [pv.PROBE] + bc + [pv.ORACLE, pv.QBIN_BASE + qc,
                       pv.DBIN_BASE + dc])
            loss += [False] * (23 + 2 + 23)
            # child q is from the child's stm POV = opponent of parent stm
            swings.append(abs(-qbin_centered(qc) - qbin_centered(qp)))
        ids += [pv.SWING, pv.num_token(max(range(n), key=lambda i: swings[i]))]
        loss += [False, True]
        return Example(ids, loss, self.tid)


@register
class BestForColor(Task):
    """<bestw>/<bestb>: best board FOR A COLOR — requires stm-aware q flip."""
    name, tid, requires = "t15_bestcolor", 15, ("labels",)

    def make(self, src, rng):
        n = rng.randint(4, 10)
        rows = src.next_labels(n)
        ids, loss = _blocks(rows)
        # white-POV value of each board: q if white to move else -q
        wq = [qbin_centered(r[1]) if stm_of(r[0]) == 0 else -qbin_centered(r[1])
              for r in rows]
        ids += [pv.BESTW, pv.num_token(max(range(n), key=lambda i: wq[i]))]
        loss += [False, True]
        ids += [pv.BESTB, pv.num_token(min(range(n), key=lambda i: wq[i]))]
        loss += [False, True]
        return Example(ids, loss, self.tid)


@register
class Reachability(Task):
    """<reach>: is bB reachable from bA within ~4 plies? (game pair = yes,
    cross-game pair = no)."""
    name, tid, requires = "t16_reach", 16, ("games",)

    def make(self, src, rng):
        from chessdecoder.agent.tasks.helpers import sample_line
        pos = rng.random() < 0.5
        if pos:
            k = rng.randint(1, 4)
            board, ucis, _, _ = sample_line(src.games, rng, k,
                                            random_walk_frac=0.0)
            if board is None:
                return None
            from chessdecoder.agent.tasks.helpers import encode_line_tokens
            toks, work = encode_line_tokens(board, ucis)
            if toks is None:
                return None
            bA, bB = pv.encode_board(board), pv.encode_board(work)
        else:
            b1, _, _, _ = sample_line(src.games, rng, 1)
            b2, _, _, _ = sample_line(src.games, rng, 1)
            if b1 is None or b2 is None:
                return None
            bA, bB = pv.encode_board(b1), pv.encode_board(b2)
        ids = [pv.ROOT] + bA + [pv.TARGET] + bB + [pv.REACH,
                                                   pv.num_token(int(pos))]
        return Example(ids, [False] * 41 + [True], self.tid)

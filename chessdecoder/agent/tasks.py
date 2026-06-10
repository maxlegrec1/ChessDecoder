"""Stage-A pretraining task generators (T1-T6) + stream packer.

Every example is formatted with the SAME tokens episodes use, so Stage A
literally pretrains episode sub-patterns:

  T1 copy            <root> b19 <probe> b19                       loss on 2nd board
  T2 apply           <root> b19 <line> m <probe> b'19             loss on b'
  T3 apply-line      <root> b19 <line> m1..mk <probe> b'19        k in 2..6
  T4a best-of-stream n x (<probe> b19 <oracle> q d) <best> idx    loss on idx
  T4b value-recall   n x (<probe> b19 <oracle> q d) <recall> idx q  loss on q
  T5 distillation    <root> b19 <oracle> q d m1 m2 m3 m4          loss on reply
  T6 at-distance     <root> b19  n x (<probe> b19 <oracle> q d)
                     <line> m1..mk <probe> b'19                   line applies to ROOT

Sources: real-game lines from the parquet shards (T2/T3/T6, ~20% replaced by
random-legal walks for off-distribution coverage + rare-patch training), and
the oracle-labeled corpus agent_data/t5_labels_*.parquet (T4/T5/T6
distractors — real boards with real oracle values).

Workers emit packed streams: ids[S], loss_mask[S], task_id[S], example_id[S]
(unique per board-answer span, -1 elsewhere — enables exact-board metrics),
input_pos[S] (random RoPE offset per stream).
"""
from __future__ import annotations

import glob
import random

import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from chessdecoder.agent import patch_vocab as pv

STREAM_LEN = 4096
MAX_POS = 8192          # model RoPE table size; offsets sampled in the slack

# task-id constants
T1, T2, T3, T4, T5, T6 = 1, 2, 3, 4, 5, 6
# Per-EXAMPLE sampling mix. Note loss-TOKEN share differs (board answers are
# 19 tokens, T5 replies 6, T4 1-3): this mix yields roughly
# T2+T3 ~55%, T5 ~18%, T1/T6 ~13%, T4 ~3% of gradient tokens.
TASK_MIX = [(T1, 0.03), (T2, 0.20), (T3, 0.22), (T4, 0.15), (T5, 0.35), (T6, 0.05)]

_CASTLE_FIX = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}


def apply_uci(board: chess.Board, uci: str) -> bool:
    """Apply a corpus move string (handles lc0 castling spelling). False if
    illegal (corpus glitch) — caller skips the line."""
    try:
        mv = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return False
    if not board.is_legal(mv):
        fixed = _CASTLE_FIX.get(uci)
        if fixed is None:
            return False
        mv = chess.Move.from_uci(fixed)
        if not board.is_legal(mv):
            return False
    board.push(mv)
    return True


class Example:
    __slots__ = ("ids", "loss", "task", "eid_span")

    def __init__(self, ids, loss, task, eid_span=None):
        self.ids, self.loss, self.task = ids, loss, task
        self.eid_span = eid_span      # (start, len) of the board-answer span


# ---------------------------------------------------------------------------
# Generators (each returns Example or None on bad data)
# ---------------------------------------------------------------------------

def make_t1(board: chess.Board) -> Example:
    b = pv.encode_board(board)
    ids = [pv.ROOT] + b + [pv.PROBE] + b
    loss = [False] * (1 + 19 + 1) + [True] * 19
    return Example(ids, loss, T1, (21, 19))


def _line_example(board: chess.Board, ucis: list[str], task: int) -> Example | None:
    b0 = pv.encode_board(board)
    work = board.copy(stack=False)
    mv_toks = []
    for u in ucis:
        if u not in pv.MOVE_TO_ID or not apply_uci(work, u):
            return None
        mv_toks.append(pv.MOVE_TO_ID[u])
    b1 = pv.encode_board(work)
    ids = [pv.ROOT] + b0 + [pv.LINE] + mv_toks + [pv.PROBE] + b1
    n_ctx = 1 + 19 + 1 + len(mv_toks) + 1
    loss = [False] * n_ctx + [True] * 19
    return Example(ids, loss, task, (n_ctx, 19))


def make_t2(board: chess.Board, uci: str) -> Example | None:
    return _line_example(board, [uci], T2)


def make_t3(board: chess.Board, ucis: list[str]) -> Example | None:
    return _line_example(board, ucis, T3)


def make_t5(label_row) -> Example:
    """label_row: (board_ids19, q_bin, d_bin, m1..m4) pre-encoded."""
    b, q, d, ms = label_row
    ids = [pv.ROOT] + b + [pv.ORACLE, pv.QBIN_BASE + q, pv.DBIN_BASE + d] + ms
    loss = [False] * (1 + 19 + 1) + [True] * (2 + 4)
    return Example(ids, loss, T5)


def make_t4(label_rows, rng: random.Random) -> Example:
    """label_rows: list of (board_ids19, q_bin, d_bin, _ms). Two variants."""
    ids, loss = [], []
    qbins = []
    for b, q, d, _ in label_rows:
        ids += [pv.PROBE] + b + [pv.ORACLE, pv.QBIN_BASE + q, pv.DBIN_BASE + d]
        loss += [False] * (1 + 19 + 3)
        qbins.append(q)
    if rng.random() < 0.5:                      # T4a best index
        target = int(np.argmax(qbins))
        ids += [pv.BEST, pv.num_token(target)]
        loss += [False, True]
    else:                                       # T4b value recall (3 queries)
        for j in rng.sample(range(len(qbins)), k=min(3, len(qbins))):
            ids += [pv.RECALL, pv.num_token(j), pv.QBIN_BASE + qbins[j]]
            loss += [False, False, True]
    return Example(ids, loss, T4)


def make_t6(board: chess.Board, ucis: list[str], distractors,
            rng: random.Random) -> Example | None:
    b0 = pv.encode_board(board)
    work = board.copy(stack=False)
    mv_toks = []
    for u in ucis:
        if u not in pv.MOVE_TO_ID or not apply_uci(work, u):
            return None
        mv_toks.append(pv.MOVE_TO_ID[u])
    b1 = pv.encode_board(work)
    ids = [pv.ROOT] + b0
    loss = [False] * 20
    for db, dq, dd, _ in distractors:
        ids += [pv.PROBE] + db + [pv.ORACLE, pv.QBIN_BASE + dq, pv.DBIN_BASE + dd]
        loss += [False] * 23
    ids += [pv.LINE] + mv_toks + [pv.PROBE]
    loss += [False] * (1 + len(mv_toks) + 1)
    n_ctx = len(ids)
    ids += b1
    loss += [True] * 19
    return Example(ids, loss, T6, (n_ctx, 19))


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

class AgentTaskDataset(IterableDataset):
    """Generates packed Stage-A streams on the fly.

    Per worker: holds one game shard (fen/played_move grouped by game) and a
    slice of the label corpus in memory; regenerates from a fresh shard when
    exhausted. Deterministic per (seed, worker, shard-epoch).
    """

    def __init__(self, parquet_dir: str, label_glob: str, seed: int = 0,
                 stream_len: int = STREAM_LEN, exclude_last_shard: bool = True,
                 random_walk_frac: float = 0.2):
        shards = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
        self.shards = shards[:-1] if exclude_last_shard else shards
        self.label_files = sorted(glob.glob(label_glob))
        assert self.shards and self.label_files, "missing data files"
        self.seed = seed
        self.stream_len = stream_len
        self.random_walk_frac = random_walk_frac

    # -- data loading per worker -------------------------------------------
    def _load_games(self, rng) -> list[list[tuple[str, str]]]:
        path = rng.choice(self.shards)
        df = pd.read_parquet(path, columns=["fen", "played_move", "game_id"])
        games, cur, cur_gid = [], [], None
        for fen, mv, gid in zip(df["fen"], df["played_move"], df["game_id"]):
            if gid != cur_gid:
                if len(cur) > 1:
                    games.append(cur)
                cur, cur_gid = [], gid
            cur.append((fen, mv))
        if len(cur) > 1:
            games.append(cur)
        rng.shuffle(games)
        return games

    def _load_labels(self, rng) -> list[tuple]:
        path = rng.choice(self.label_files)
        df = pd.read_parquet(path)
        rows = []
        for fen, q, d, m1, m2, m3, m4 in df.itertuples(index=False):
            try:
                b = pv.encode_board(chess.Board(fen))
            except Exception:
                continue
            rows.append((b, int(q), int(d), [int(m1), int(m2), int(m3), int(m4)]))
        rng.shuffle(rows)
        return rows

    # -- example sampling ----------------------------------------------------
    def _sample_line(self, games, rng, k: int):
        """(start_board, [k ucis]) from a real game, or a random-legal walk."""
        for _ in range(10):
            g = games[rng.randrange(len(games))]
            if len(g) <= k:
                continue
            i = rng.randrange(len(g) - k)
            try:
                board = chess.Board(g[i][0])
            except Exception:
                continue
            if rng.random() < self.random_walk_frac:
                walk, b = [], board.copy(stack=False)
                for _ in range(k):
                    legal = list(b.legal_moves)
                    if not legal:
                        break
                    mv = rng.choice(legal)
                    walk.append(mv.uci())
                    b.push(mv)
                if len(walk) == k:
                    return board, walk
                continue
            return board, [g[i + j][1] for j in range(k)]
        return None, None

    def _make_example(self, games, labels, label_pos, rng):
        r, acc = rng.random(), 0.0
        task = T5
        for t, w in TASK_MIX:
            acc += w
            if r < acc:
                task = t
                break
        if task == T1:
            board, _ = self._sample_line(games, rng, 1)
            return make_t1(board) if board else None
        if task in (T2, T3):
            k = 1 if task == T2 else rng.randint(2, 6)
            board, ucis = self._sample_line(games, rng, k)
            if board is None:
                return None
            return make_t2(board, ucis[0]) if task == T2 else make_t3(board, ucis)
        if task == T4:
            n = rng.randint(4, 10)
            rows = [labels[(label_pos[0] + i) % len(labels)] for i in range(n)]
            label_pos[0] += n
            return make_t4(rows, rng)
        if task == T5:
            row = labels[label_pos[0] % len(labels)]
            label_pos[0] += 1
            return make_t5(row)
        # T6
        k = rng.randint(1, 4)
        board, ucis = self._sample_line(games, rng, k)
        if board is None:
            return None
        n = rng.randint(3, 8)
        rows = [labels[(label_pos[0] + i) % len(labels)] for i in range(n)]
        label_pos[0] += n
        return make_t6(board, ucis, rows, rng)

    # -- iteration -----------------------------------------------------------
    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = random.Random(self.seed * 9973 + wid * 7919)
        games = self._load_games(rng)
        labels = self._load_labels(rng)
        label_pos = [0]
        examples_served = 0
        next_eid = wid * 100_000_000

        S = self.stream_len
        while True:
            ids = np.full(S, pv.PAD, dtype=np.int64)
            loss = np.zeros(S, dtype=bool)
            task = np.zeros(S, dtype=np.int8)
            eid = np.full(S, -1, dtype=np.int64)
            cursor = 0
            while cursor < S - 32:
                ex = self._make_example(games, labels, label_pos, rng)
                if ex is None:
                    continue
                n = len(ex.ids)
                if cursor + n > S:
                    break
                ids[cursor:cursor + n] = ex.ids
                loss[cursor:cursor + n] = ex.loss
                task[cursor:cursor + n] = np.where(ex.loss, ex.task, 0)
                if ex.eid_span is not None:
                    s0, sl = ex.eid_span
                    eid[cursor + s0: cursor + s0 + sl] = next_eid
                    next_eid += 1
                cursor += n
                examples_served += 1
                if examples_served % 200_000 == 0:   # refresh shard slice
                    games = self._load_games(rng)
            offset = rng.randrange(0, MAX_POS - S + 1)
            pos = np.arange(offset, offset + S, dtype=np.int64)
            yield (torch.from_numpy(ids), torch.from_numpy(loss),
                   torch.from_numpy(task), torch.from_numpy(eid),
                   torch.from_numpy(pos))


def build_val_streams(parquet_dir: str, val_labels_path: str, n_streams: int,
                      seed: int = 1234) -> list:
    """Fixed held-out val streams: games from the LAST shard + val labels."""
    ds = AgentTaskDataset.__new__(AgentTaskDataset)
    shards = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
    ds.shards = shards[-1:]
    ds.label_files = [val_labels_path]
    ds.seed = seed
    ds.stream_len = STREAM_LEN
    ds.random_walk_frac = 0.2
    out = []
    for i, item in enumerate(iter(ds)):
        out.append(item)
        if i + 1 >= n_streams:
            break
    return out

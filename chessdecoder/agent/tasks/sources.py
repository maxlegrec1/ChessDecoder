"""Shared per-worker data sources, provisioned once and rotated centrally.

Tasks declare ``requires`` from: "games" (real-game shard), "labels" (oracle
label corpus rows), "paired" (parent/move/child oracle-labeled pairs).
"""
from __future__ import annotations

import glob

import chess
import pandas as pd

from chessdecoder.agent import patch_vocab as pv

ROTATE_EVERY = 200_000   # examples between source refreshes


class Sources:
    def __init__(self, needs: set[str], parquet_dir: str, label_glob: str,
                 paired_glob: str, rng, last_shard_only: bool = False):
        self._needs = needs
        self._rng = rng
        shards = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
        self._shards = shards[-1:] if last_shard_only else shards[:-1]
        self._label_files = sorted(glob.glob(label_glob)) if label_glob else []
        self._paired_files = sorted(glob.glob(paired_glob)) if paired_glob else []
        if "games" in needs:
            assert self._shards, "no game shards"
        if "labels" in needs:
            assert self._label_files, "no label files"
        if "paired" in needs:
            assert self._paired_files, ("no paired-label files — run "
                                        "scripts/gen_t5_labels.py --paired")
        self.games: list | None = None
        self.labels: list | None = None
        self.label_pos = 0
        self.paired: list | None = None
        self.paired_pos = 0
        self.refresh()

    # -- loading -------------------------------------------------------------
    def _load_games(self):
        df = pd.read_parquet(self._rng.choice(self._shards),
                             columns=["fen", "played_move", "game_id"])
        games, cur, cur_gid = [], [], None
        for fen, mv, gid in zip(df["fen"], df["played_move"], df["game_id"]):
            if gid != cur_gid:
                if len(cur) > 1:
                    games.append(cur)
                cur, cur_gid = [], gid
            cur.append((fen, mv))
        if len(cur) > 1:
            games.append(cur)
        self._rng.shuffle(games)
        return games

    def _load_labels(self):
        df = pd.read_parquet(self._rng.choice(self._label_files))
        rows = []
        for fen, q, d, m1, m2, m3, m4 in df.itertuples(index=False):
            try:
                b = pv.encode_board(chess.Board(fen))
            except Exception:
                continue
            rows.append((b, int(q), int(d), [int(m1), int(m2), int(m3), int(m4)]))
        self._rng.shuffle(rows)
        return rows

    def _load_paired(self):
        df = pd.read_parquet(self._rng.choice(self._paired_files))
        rows = []
        for pf, qp, dp, mu, cf, qc, dc in df.itertuples(index=False):
            t = pv.uci_to_token(mu)
            if t is None:
                continue
            try:
                bp = pv.encode_board(chess.Board(pf))
                bc = pv.encode_board(chess.Board(cf))
            except Exception:
                continue
            rows.append((bp, int(qp), int(dp), t, bc, int(qc), int(dc)))
        self._rng.shuffle(rows)
        return rows

    def refresh(self):
        if "games" in self._needs:
            self.games = self._load_games()
        if "labels" in self._needs:
            self.labels = self._load_labels()
            self.label_pos = 0
        if "paired" in self._needs:
            self.paired = self._load_paired()
            self.paired_pos = 0

    # -- cursors ---------------------------------------------------------------
    def next_labels(self, n: int):
        out = [self.labels[(self.label_pos + i) % len(self.labels)]
               for i in range(n)]
        self.label_pos += n
        return out

    def next_paired(self, n: int):
        out = [self.paired[(self.paired_pos + i) % len(self.paired)]
               for i in range(n)]
        self.paired_pos += n
        return out

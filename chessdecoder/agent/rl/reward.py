"""Rewarder: Q_ref regret lookup + per-episode metrics. A library used inside
the rollout process — never a separate service (reward = table lookup).

R = Q_ref(root, a_agent) - max_a Q_ref(root, a)      (regret, <= 0)
    - invalid_eps * probes_invalid                    (config, default 0.01)

Metrics attached per episode (consumed by the trainer's wandb writer):
beat_greedy (Q_ref(a_agent) > Q_ref(oracle_greedy)), match_search_best,
match_corpus_best, probe validity stats.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import chess
import numpy as np
import pandas as pd

from chessdecoder.agent import patch_vocab as pv

QREF_DIR = "agent_data/qref"


def _key(fen: str) -> str:
    return fen.rsplit(" ", 2)[0]


@dataclass
class RootRef:
    moves: list[str]
    q: np.ndarray
    oracle_greedy: str
    search_best: str
    corpus_best: str | None


class QRefTable:
    def __init__(self, qref_dir: str = QREF_DIR):
        self.dir = qref_dir
        self._table: dict[str, RootRef] = {}
        self._loaded: set[str] = set()
        self.reload()

    def reload(self) -> int:
        """Pick up new qref shards; returns number of roots added."""
        added = 0
        for f in sorted(glob.glob(f"{self.dir}/qref_*.parquet")):
            if f in self._loaded:
                continue
            df = pd.read_parquet(f)
            for r in df.itertuples(index=False):
                self._table[_key(r.fen)] = RootRef(
                    moves=list(r.moves), q=np.asarray(r.q, dtype=np.float32),
                    oracle_greedy=r.oracle_greedy, search_best=r.search_best,
                    corpus_best=getattr(r, "corpus_best", None))
                added += 1
            self._loaded.add(f)
        return added

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, fen: str) -> bool:
        return _key(fen) in self._table

    def get(self, fen: str) -> RootRef | None:
        return self._table.get(_key(fen))

    def roots(self) -> list[str]:
        return list(self._table.keys())


def move_id_to_uci(root: chess.Board, move_id: int) -> str | None:
    """Agent MOVE-region id -> python-chess uci on this board (handles the
    lc0 castling spelling)."""
    for mv in root.legal_moves:
        for k in pv.move_keys(root, mv):
            if pv.MOVE_TO_ID.get(k) == move_id:
                return mv.uci()
    return None


def score_episode(ep, ref: RootRef, root: chess.Board,
                  invalid_eps: float = 0.01) -> dict:
    """Returns reward + metrics. ep: rl.episodes.Episode (final_move set)."""
    uci = move_id_to_uci(root, ep.final_move)
    assert uci is not None, "grammar guarantees a legal final move"
    i = ref.moves.index(uci)
    q_best = float(ref.q.max())
    q_agent = float(ref.q[i])
    ig = ref.moves.index(ref.oracle_greedy)
    q_greedy = float(ref.q[ig])
    reward = (q_agent - q_best) - invalid_eps * ep.probes_invalid
    return dict(
        reward=reward,
        regret=q_agent - q_best,
        q_agent=q_agent,
        beat_greedy=q_agent > q_greedy + 1e-6,
        match_greedy=uci == ref.oracle_greedy,
        match_search_best=uci == ref.search_best,
        match_corpus_best=uci == ref.corpus_best,
        probes_valid=ep.probes_valid,
        probes_invalid=ep.probes_invalid,
        final_uci=uci,
    )

"""Episode grammar: state machine + per-state token masks.

Used at generation time (RL / eval / scripted teacher). Pretraining is
teacher-forced so it only needs the formats, not the masks — but the masks
are also handy for masked-decoding evaluation of Stage-A tasks.

Episode shape (markdowns/01-search-agent-plan.md §2.2, <go> dropped — boards
are fixed-length so the state machine knows where they end):

  HARNESS:  <root> b19 <oracle> q d m1 m2 m3 m4 [budget]
  loop:
    AGENT:  <probe> b19            (19 slots, per-slot region masks)
    HARNESS:<oracle> q d m1..m4 [budget]   or   <invalid> [budget]
  AGENT:    <answer> move           (masked to legal root moves)
"""
from __future__ import annotations

from enum import Enum, auto

import chess
import torch

from chessdecoder.agent import patch_vocab as pv


class S(Enum):
    VERB = auto()          # expect <probe> or <answer>
    BOARD = auto()         # inside the 19 board slots (slot index tracked)
    ANSWER_MOVE = auto()   # expect a legal root move token
    DONE = auto()


def _range_mask(base: int, n: int, size: int) -> torch.Tensor:
    m = torch.zeros(size, dtype=torch.bool)
    m[base:base + n] = True
    return m


# Static per-slot masks (built once).
MASK_PATCH = _range_mask(pv.PATCH_BASE, pv.N_PATCH, pv.VOCAB_SIZE)
MASK_CASTLE = _range_mask(pv.CASTLE_BASE, pv.N_CASTLE, pv.VOCAB_SIZE)
MASK_STM = _range_mask(pv.STM_BASE, pv.N_STM, pv.VOCAB_SIZE)
MASK_EP = _range_mask(pv.EP_BASE, pv.N_EP, pv.VOCAB_SIZE)


def board_slot_mask(slot: int) -> torch.Tensor:
    if slot < 16:
        return MASK_PATCH
    return (MASK_CASTLE, MASK_STM, MASK_EP)[slot - 16]


class EpisodeGrammar:
    """Per-episode decoding state. The harness owns one per live episode."""

    def __init__(self, root: chess.Board, budget: int, min_probes: int = 0):
        self.root = root
        self.budget = budget
        self.min_probes = min(min_probes, budget)
        self.used = 0
        self.state = S.VERB
        self.slot = 0
        self._legal_root_mask = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool)
        for mv in root.legal_moves:
            for k in pv.move_keys(root, mv):
                if k in pv.MOVE_TO_ID:
                    self._legal_root_mask[pv.MOVE_TO_ID[k]] = True

    def mask(self) -> torch.Tensor:
        """Legal next-token mask for the agent's current slot."""
        if self.state == S.VERB:
            m = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool)
            if self.budget > 0:
                m[pv.PROBE] = True
            if self.used >= self.min_probes:
                m[pv.ANSWER] = True
            return m
        if self.state == S.BOARD:
            return board_slot_mask(self.slot)
        if self.state == S.ANSWER_MOVE:
            return self._legal_root_mask
        raise RuntimeError("episode finished")

    def step(self, tok: int) -> None:
        """Advance on an AGENT-emitted token. Harness-injected tokens
        (oracle replies) do not pass through here."""
        if self.state == S.VERB:
            if tok == pv.PROBE:
                self.state, self.slot = S.BOARD, 0
            elif tok == pv.ANSWER:
                self.state = S.ANSWER_MOVE
            else:
                raise ValueError(f"bad verb token {tok}")
        elif self.state == S.BOARD:
            self.slot += 1
            if self.slot == pv.BOARD_LEN:
                self.state = S.VERB        # harness injects reply, burns budget
        elif self.state == S.ANSWER_MOVE:
            self.state = S.DONE
        else:
            raise RuntimeError("episode finished")

    def on_probe_resolved(self) -> None:
        """Harness calls this after injecting the oracle reply / <invalid>."""
        self.budget -= 1
        self.used += 1

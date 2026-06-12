"""Episode structure shared by the rollout engine and the GRPO trainer.

Token layout (all rows lockstep, one token per step, fixed-length segments):

  prefix  (28, injected): <root> b19 <oracle> q d m1..m4 budget
  probe   (20, agent):    <probe> b19
  reply   ( 8, injected): <oracle> q d m1..m4 budget   |  <invalid> budget PAD*6
  answer  ( 2, agent):    <answer> move
  pad     (.., injected): PAD until the batch finishes

The trainer never trusts stored masks: ``replay`` re-derives, purely from the
token ids + root fen, which positions are agent-emitted and which grammar mask
governed each of them. tests/test_rl_engine.py asserts replay == what the
engine actually used at sampling time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import torch

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.grammar import EpisodeGrammar, board_slot_mask

PREFIX_LEN = 1 + pv.BOARD_LEN + pv.REPLY_LEN + 1          # 28
REPLY_TOKENS = pv.REPLY_LEN + 1                            # 8 (incl budget)
PROBE_TOKENS = 1 + pv.BOARD_LEN                            # 20
ANSWER_TOKENS = 2


def episode_len(k_budget: int) -> int:
    """Worst-case token count for an episode with K probes."""
    return PREFIX_LEN + k_budget * (PROBE_TOKENS + REPLY_TOKENS) + ANSWER_TOKENS


# mask kinds at agent positions (see replay)
VERB, SLOT0 = -1, 0          # SLOT0..SLOT18 = board slots, ANSWER_MV = -2
ANSWER_MV = -2


@dataclass
class Episode:
    root_fen: str
    k_budget: int
    min_probes: int = 0
    ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)   # behavior logprobs
    agent: list[bool] = field(default_factory=list)        # agent-emitted?
    probes_valid: int = 0
    probes_invalid: int = 0
    probes_repeat: int = 0          # informationless re-queries (no penalty)
    final_move: int | None = None                          # agent MOVE id
    done: bool = False


def replay(ids: list[int], root_fen: str, k_budget: int,
           min_probes: int = 0) -> list[tuple[int, int, int, bool]]:
    """Parse a finished episode -> [(position, mask_kind, budget_left,
    answer_ok)] for every agent-emitted position. mask_kind: VERB (-1),
    ANSWER_MV (-2), or board slot index 0..18. budget_left / answer_ok
    rebuild the exact VERB mask (answer_ok = min-probe quota met). Raises
    on malformed episodes (engine bug)."""
    out = []
    pos = PREFIX_LEN
    budget = k_budget
    used = 0
    min_p = min(min_probes, k_budget)
    while pos < len(ids):
        t = ids[pos]
        if t == pv.PAD:                       # trailing pad after answer
            pos += 1
            continue
        out.append((pos, VERB, budget, used >= min_p))
        if t == pv.PROBE:
            assert budget > 0, "probe after budget exhausted"
            for s in range(pv.BOARD_LEN):
                out.append((pos + 1 + s, s, budget, True))
            pos += PROBE_TOKENS
            r = ids[pos]
            assert r in (pv.ORACLE, pv.INVALID), f"bad reply token {r}"
            pos += REPLY_TOKENS
            budget -= 1
            used += 1
        elif t == pv.ANSWER:
            assert used >= min_p, "answer before min-probe quota"
            out.append((pos + 1, ANSWER_MV, budget, True))
            pos += ANSWER_TOKENS
        else:
            raise AssertionError(f"bad verb token {t} at {pos}")
    return out


def mask_for(kind: int, grammar_or_root) -> torch.Tensor:
    """Mask tensor for a mask kind. grammar_or_root: EpisodeGrammar (engine
    path) or a root chess.Board (trainer path, ANSWER_MV only)."""
    if kind == ANSWER_MV:
        if isinstance(grammar_or_root, EpisodeGrammar):
            return grammar_or_root._legal_root_mask
        m = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool)
        for mv in grammar_or_root.legal_moves:
            for k in pv.move_keys(grammar_or_root, mv):
                if k in pv.MOVE_TO_ID:
                    m[pv.MOVE_TO_ID[k]] = True
        return m
    if kind == VERB:
        raise ValueError("VERB mask depends on remaining budget; build inline")
    return board_slot_mask(kind)


def verb_mask(budget: int, answer_ok: bool = True) -> torch.Tensor:
    m = torch.zeros(pv.VOCAB_SIZE, dtype=torch.bool)
    if budget > 0:
        m[pv.PROBE] = True
    if answer_ok:
        m[pv.ANSWER] = True
    return m

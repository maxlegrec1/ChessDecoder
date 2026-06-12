"""RolloutEngine — batched, grammar-constrained episode generation.

Design: STRICT LOCKSTEP. Every row appends exactly one token per step —
either a sampled agent token or the next token of a pending injection queue
(oracle replies, pads). All rows therefore share one sequence length, so
torchtune's batch-uniform KVCache and a single causal mask are exact.

Correctness contracts (tests/test_rl_engine.py):
  1. stored behavior logprobs == teacher-forced recompute (KV on and off)
  2. every sampled token satisfies the grammar mask replayed from the ids
  3. injected oracle tokens == an independent eager Oracle query
  4. throughput floor
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import torch

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.grammar import EpisodeGrammar, S
from chessdecoder.agent.model import AgentDecoder
from chessdecoder.agent.rl.episodes import (ANSWER_TOKENS, PREFIX_LEN,
                                            PROBE_TOKENS, REPLY_TOKENS,
                                            Episode, episode_len)


def _root_prefix(board: chess.Board, reply, budget: int) -> list[int]:
    """<root> b19 <oracle> q d m1..m4 budget  (PREFIX_LEN tokens)."""
    return ([pv.ROOT] + pv.encode_board(board) + reply.tokens()
            + [pv.num_token(budget)])


def _reply_tokens(reply, budget_left: int) -> list[int]:
    return reply.tokens() + [pv.num_token(budget_left)]


def _invalid_tokens(budget_left: int) -> list[int]:
    return ([pv.INVALID, pv.num_token(budget_left)]
            + [pv.PAD] * (REPLY_TOKENS - 2))


class RolloutEngine:
    def __init__(self, model: AgentDecoder, oracle, device: str = "cuda",
                 batch_size: int = 128, max_len: int | None = None,
                 k_budget: int = 16, temperature: float = 1.0,
                 dtype: torch.dtype = torch.bfloat16):
        self.model = model.to(device).eval()
        self.oracle = oracle
        self.device = device
        self.B = batch_size
        self.K = k_budget
        self.T = temperature
        self.dtype = dtype
        self.max_len = max_len or episode_len(k_budget) + 8
        self.model.setup_caches(batch_size, dtype, self.max_len)
        # full causal mask buffer [max_len, max_len], sliced per step
        self._causal = torch.tril(torch.ones(self.max_len, self.max_len,
                                             dtype=torch.bool, device=device))

    @torch.no_grad()
    def rollout(self, roots: list[chess.Board],
                k_budgets: list[int] | None = None,
                min_probes: list[int] | None = None,
                greedy: bool = False) -> list[Episode]:
        """One lockstep batch of episodes. len(roots) must equal B (repeat
        roots G times upstream for GRPO groups). min_probes[i] masks
        <answer> until that many probes are used (probe-collapse counter)."""
        assert len(roots) == self.B
        ks = k_budgets or [self.K] * self.B
        mps = min_probes or [0] * self.B
        replies = self.oracle.query_batch(roots)
        eps = [Episode(root_fen=b.fen(), k_budget=k, min_probes=mp)
               for b, k, mp in zip(roots, ks, mps)]
        grams = [EpisodeGrammar(b, k, mp)
                 for b, k, mp in zip(roots, ks, mps)]
        # honest repeat semantics: re-probing an already-answered position
        # (incl. the root, answered in the prefix) returns no new information
        seen_keys = [{b.fen().rsplit(" ", 2)[0]} for b in roots]
        queues: list[list[int]] = [[] for _ in range(self.B)]   # pending injections
        pend_board: list[list[int] | None] = [None] * self.B    # probe slots buf

        for i in range(self.B):
            eps[i].ids = _root_prefix(roots[i], replies[i], ks[i])
            eps[i].logprobs = [0.0] * PREFIX_LEN
            eps[i].agent = [False] * PREFIX_LEN

        ids = torch.tensor([e.ids for e in eps], dtype=torch.int64,
                           device=self.device)
        self.model.reset_caches()
        pos = torch.arange(PREFIX_LEN, device=self.device).unsqueeze(0) \
                   .expand(self.B, -1)
        mask = self._causal[:PREFIX_LEN].unsqueeze(0) \
                   .expand(self.B, -1, -1).contiguous()
        h = self._fwd(ids, pos, mask)
        cur = PREFIX_LEN

        while cur < self.max_len:
            logits = self.model.logits_at(h[:, -1].float())       # [B,V]
            # grammar masks for rows that sample this step
            step_tok = torch.full((self.B,), pv.PAD, dtype=torch.int64)
            step_lp = torch.zeros(self.B)
            step_agent = [False] * self.B
            sample_rows = []
            masks = []
            for i in range(self.B):
                if queues[i] or eps[i].done:
                    continue
                sample_rows.append(i)
                masks.append(grams[i].mask())
            if sample_rows:
                rows = torch.tensor(sample_rows, device=self.device)
                m = torch.stack(masks).to(self.device)
                lg = logits[rows].masked_fill(~m, float("-inf"))
                if greedy:
                    choice = lg.argmax(-1)
                    lp = torch.log_softmax(lg, -1).gather(
                        1, choice.unsqueeze(1)).squeeze(1)
                else:
                    probs = torch.softmax(lg / self.T, -1)
                    choice = torch.multinomial(probs, 1).squeeze(1)
                    lp = torch.log_softmax(lg / self.T, -1).gather(
                        1, choice.unsqueeze(1)).squeeze(1)
                choice_l = choice.cpu().tolist()
                lp_l = lp.cpu().tolist()
                for j, i in enumerate(sample_rows):
                    step_tok[i] = choice_l[j]
                    step_lp[i] = lp_l[j]
                    step_agent[i] = True
            # injected rows
            for i in range(self.B):
                if eps[i].done and not queues[i]:
                    continue              # PAD default
                if queues[i]:
                    step_tok[i] = queues[i].pop(0)

            tok_l = step_tok.tolist()
            probe_done_rows = []
            for i in range(self.B):
                t = tok_l[i]
                eps[i].ids.append(t)
                eps[i].logprobs.append(float(step_lp[i]))
                eps[i].agent.append(step_agent[i])
                if not step_agent[i]:
                    continue
                g = grams[i]
                if g.state == S.BOARD:
                    pend_board[i].append(t)
                g.step(t)
                if t == pv.PROBE:
                    pend_board[i] = []
                elif g.state == S.VERB and pend_board[i] is not None \
                        and len(pend_board[i]) == pv.BOARD_LEN:
                    probe_done_rows.append(i)
                elif g.state == S.DONE:
                    eps[i].final_move = t
                    eps[i].done = True

            if probe_done_rows:
                boards, valid_rows = [], []
                for i in probe_done_rows:
                    b = self._decode_probe(pend_board[i])
                    pend_board[i] = None
                    g = grams[i]
                    g.on_probe_resolved()
                    if b is None:
                        eps[i].probes_invalid += 1
                        queues[i] = _invalid_tokens(g.budget)
                    elif b.fen().rsplit(" ", 2)[0] in seen_keys[i]:
                        eps[i].probes_repeat += 1
                        queues[i] = _invalid_tokens(g.budget)
                    else:
                        seen_keys[i].add(b.fen().rsplit(" ", 2)[0])
                        eps[i].probes_valid += 1
                        boards.append(b)
                        valid_rows.append(i)
                if boards:
                    rs = self.oracle.query_batch(boards)
                    for i, r in zip(valid_rows, rs):
                        queues[i] = _reply_tokens(r, grams[i].budget)

            if all(e.done and not q for e, q in zip(eps, queues)):
                break
            step_ids = step_tok.to(self.device).unsqueeze(1)
            pos = torch.full((self.B, 1), cur, dtype=torch.int64,
                             device=self.device)
            # .contiguous() is load-bearing: sdpa silently misreads the
            # zero-stride expanded slice in the 1-token path
            mask = self._causal[cur].unsqueeze(0).unsqueeze(0) \
                       .expand(self.B, 1, -1).contiguous()
            h = self._fwd(step_ids, pos, mask)
            cur += 1
        return eps

    def _fwd(self, ids, pos, mask):
        if self.dtype == torch.bfloat16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(ids, input_pos=pos, mask=mask)
        return self.model(ids, input_pos=pos, mask=mask)

    @staticmethod
    def _decode_probe(slots: list[int]) -> chess.Board | None:
        """19 board tokens -> Board, or None if undecodable / illegal /
        terminal (terminal probes carry no move info -> treated invalid)."""
        try:
            b = pv.decode_board(slots)
        except Exception:
            return None
        if b is None or not b.is_valid():
            return None
        if b.is_game_over():
            return None
        return b

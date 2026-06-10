"""Frozen oracle: position -> (q_bin, d_bin, top-4 legal moves).

Wraps the 30M ChessEncoder baseline (Elo 2486 [2458, 2515] vs SF2400 — see
markdowns/01-search-agent-plan.md §0.1). The oracle is the ONLY source of
chess evaluation in the search-agent system; the agent queries it through the
harness and never sees gradients from it.

Reply move ids live in the agent vocab's MOVE region with the same index
order as the oracle's 1924-way policy head. Castling is scored over both
vocab spellings (uci + lc0 king-takes-rook) and reported in python-chess uci
form when present in the vocab.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import torch

from chessdecoder.dataloader.loader import fen_to_ids
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size, move_token_to_idx
from chessdecoder.agent import patch_vocab as pv

ORACLE_CKPT = ("checkpoints/oracle_30M/oracle-30M_20260609_235451/"
               "checkpoint_246000.pt")
ORACLE_DIMS = dict(embed_dim=512, num_heads=8, num_layers=10, d_ff=1152)
TOP_K = 4


@dataclass
class Reply:
    q_bin: int                  # 0..127, from the side-to-move's POV
    d_bin: int                  # 0..31
    top_moves: list[int]        # agent-vocab MOVE-region ids, best first

    def tokens(self) -> list[int]:
        """<oracle> q d m1..m4 (REPLY_LEN tokens; budget appended by harness)."""
        return ([pv.ORACLE, pv.QBIN_BASE + self.q_bin, pv.DBIN_BASE + self.d_bin]
                + self.top_moves)


class Oracle:
    def __init__(self, ckpt: str = ORACLE_CKPT, device: str = "cuda",
                 memo_max: int = 2_000_000):
        self.device = device
        m = ChessEncoder(vocab_size=vocab_size, attention_variant="geom",
                         policy_head="cross_attn", input_mode="lc0_64",
                         ffn_type="dense", **ORACLE_DIMS).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing, f"oracle load missing keys: {missing[:5]}"
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m
        self.n_queries = 0          # budget-relevant (unique) evaluations
        self.n_memo_hits = 0
        self._memo: dict[str, Reply] = {}
        self._memo_max = memo_max

    # -- low level ----------------------------------------------------------
    @torch.no_grad()
    def _forward(self, fens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = np.empty((len(fens), 68), dtype=np.int32)
        for i, f in enumerate(fens):
            fen_to_ids(f, ids[i])
        bid = torch.from_numpy(ids.astype(np.int64)).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(bid)
        pol = out["policy"].float()                       # [B, 1924]
        wdl = self.model.mean_wdl(out["wdl"].float())     # [B, 3] (W, D, L)
        return pol, wdl

    @staticmethod
    def _top_moves(board: chess.Board, logits: torch.Tensor) -> list[int]:
        """Top-K *legal* moves by policy logit, agent-vocab ids, best first."""
        scored = []
        lg = logits.tolist()
        for mv in board.legal_moves:
            keys = pv.move_keys(board, mv)
            best_key, best_lp = None, -1e30
            for k in keys:
                idx = move_token_to_idx.get(k)
                if idx is not None and lg[idx] > best_lp:
                    best_lp, best_key = lg[idx], k
            if best_key is not None:
                scored.append((best_lp, best_key))
        scored.sort(key=lambda x: -x[0])
        out = [pv.MOVE_TO_ID[k] for _, k in scored[:TOP_K]]
        while out and len(out) < TOP_K:           # rare: <4 legal moves
            out.append(out[-1])
        return out

    # -- public -------------------------------------------------------------
    @torch.no_grad()
    def query_batch(self, boards: list[chess.Board]) -> list[Reply]:
        """Batched oracle query with memoization on FEN (sans counters)."""
        keys = [b.fen().rsplit(" ", 2)[0] for b in boards]
        fresh_idx = [i for i, k in enumerate(keys) if k not in self._memo]
        if fresh_idx:
            fens = [boards[i].fen() for i in fresh_idx]
            pol, wdl = self._forward(fens)
            q = (wdl[:, 0] - wdl[:, 2]).tolist()
            d = wdl[:, 1].tolist()
            for j, i in enumerate(fresh_idx):
                r = Reply(q_bin=pv.q_to_bin(q[j]), d_bin=pv.d_to_bin(d[j]),
                          top_moves=self._top_moves(boards[i], pol[j]))
                if len(self._memo) < self._memo_max:
                    self._memo[keys[i]] = r
            self.n_queries += len(fresh_idx)
        self.n_memo_hits += len(boards) - len(fresh_idx)
        return [self._memo.get(k) or self._uncached(boards[i])
                for i, k in enumerate(keys)]

    def _uncached(self, board: chess.Board) -> Reply:
        pol, wdl = self._forward([board.fen()])
        q = (wdl[0, 0] - wdl[0, 2]).item()
        return Reply(pv.q_to_bin(q), pv.d_to_bin(wdl[0, 1].item()),
                     self._top_moves(board, pol[0]))

    def query(self, board: chess.Board) -> Reply:
        return self.query_batch([board])[0]

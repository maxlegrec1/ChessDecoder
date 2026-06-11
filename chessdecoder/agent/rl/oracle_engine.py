"""OracleEngine — the fast batched oracle for RL rollouts and PUCT.

Same contract as chessdecoder.agent.oracle.Oracle (which stays the eager
correctness reference) but built for throughput:

- bf16 + torch.compile(mode="reduce-overhead") -> CUDA-graphed forward
- fixed batch buckets (pad to 32/128/256) so compiled shapes are reused
- top-4 legal extraction on GPU (masked top-k over the 1924-way policy)
  instead of the per-board python loop
- exposes eval_batch() (raw q/d/policy) for the PUCT q_ref producer

Parity with the eager Oracle is enforced by tests/test_rl_oracle_engine.py
(tie-order tolerance on top-4 only).
"""
from __future__ import annotations

import chess
import numpy as np
import torch

from chessdecoder.dataloader.loader import fen_to_ids
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size, move_token_to_idx
from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.oracle import ORACLE_CKPT, ORACLE_DIMS, TOP_K, Reply

BUCKETS = (32, 128, 256)

# oracle policy index -> agent MOVE-region id (same uci key strings)
_IDX_TO_AGENT = np.full(1924, -1, dtype=np.int64)
for _k, _idx in move_token_to_idx.items():
    _aid = pv.MOVE_TO_ID.get(_k)
    if _aid is not None:
        _IDX_TO_AGENT[_idx] = _aid


def _legal_indices(board: chess.Board) -> tuple[list[int], list[int], list[int]]:
    """(policy indices, agent ids, legal-move group ids) for all legal moves;
    castling contributes both vocab spellings under one group id (GPU top-k
    picks the better one, dedupe collapses the pair)."""
    idxs, aids, gids = [], [], []
    for g, mv in enumerate(board.legal_moves):
        for k in pv.move_keys(board, mv):
            idx = move_token_to_idx.get(k)
            if idx is not None and _IDX_TO_AGENT[idx] >= 0:
                idxs.append(idx)
                aids.append(int(_IDX_TO_AGENT[idx]))
                gids.append(g)
    return idxs, aids, gids


class OracleEngine:
    def __init__(self, ckpt: str = ORACLE_CKPT, device: str = "cuda",
                 memo_max: int = 2_000_000, compile_model: bool = True):
        self.device = device
        m = ChessEncoder(vocab_size=vocab_size, attention_variant="geom",
                         policy_head="cross_attn", input_mode="lc0_64",
                         ffn_type="dense", **ORACLE_DIMS).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        missing, _ = m.load_state_dict(sd, strict=False)
        assert not missing, f"oracle load missing keys: {missing[:5]}"
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m
        self._fwd = self._fwd_eager
        if compile_model:
            self._compiled = torch.compile(self._fwd_eager,
                                           mode="reduce-overhead")
            self._fwd = self._compiled
        self.n_queries = 0
        self.n_memo_hits = 0
        self._memo: dict[str, Reply] = {}
        self._memo_max = memo_max

    # -- forward -------------------------------------------------------------
    def _fwd_eager(self, bid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(bid)
        pol = out["policy"].float()
        wdl = self.model.mean_wdl(out["wdl"].float())
        return pol, wdl

    @torch.no_grad()
    def _forward_padded(self, ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """ids [N,68] -> (policy [N,1924], wdl [N,3]); pads N up to a bucket."""
        n = ids.shape[0]
        bucket = next((b for b in BUCKETS if b >= n), None)
        if bucket is None:                      # huge batch: chunk by max bucket
            outs = [self._forward_padded(ids[i:i + BUCKETS[-1]])
                    for i in range(0, n, BUCKETS[-1])]
            return (torch.cat([o[0] for o in outs]),
                    torch.cat([o[1] for o in outs]))
        if bucket > n:
            pad = np.repeat(ids[:1], bucket - n, axis=0)
            ids = np.concatenate([ids, pad], axis=0)
        bid = torch.from_numpy(ids.astype(np.int64)).to(self.device,
                                                        non_blocking=True)
        pol, wdl = self._fwd(bid)
        # clone: reduce-overhead reuses output buffers across calls
        return pol[:n].clone(), wdl[:n].clone()

    # -- public --------------------------------------------------------------
    @torch.no_grad()
    def eval_batch(self, fens: list[str]) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
        """Raw outputs for PUCT: (q [N] stm-POV, d [N], policy logits
        [N,1924]) as GPU tensors."""
        ids = np.empty((len(fens), 68), dtype=np.int32)
        for i, f in enumerate(fens):
            fen_to_ids(f, ids[i])
        pol, wdl = self._forward_padded(ids)
        self.n_queries += len(fens)
        return wdl[:, 0] - wdl[:, 2], wdl[:, 1], pol

    @torch.no_grad()
    def query_batch(self, boards: list[chess.Board]) -> list[Reply]:
        """Drop-in for Oracle.query_batch (same Reply, FEN memo)."""
        keys = [b.fen().rsplit(" ", 2)[0] for b in boards]
        fresh = [i for i, k in enumerate(keys) if k not in self._memo]
        if fresh:
            fens = [boards[i].fen() for i in fresh]
            q, d, pol = self.eval_batch(fens)
            # GPU top-k over legal-masked logits (topk 8 then dedupe by move
            # to collapse castling double-spellings)
            masked = torch.full_like(pol, -1e30)
            agent_of = torch.zeros(pol.shape, dtype=torch.int64,
                                   device=pol.device)
            group_of = torch.full(pol.shape, -1, dtype=torch.int64,
                                  device=pol.device)
            # one flattened H2D transfer instead of 3 tiny copies per board
            rows, idxs_all, aids_all, gids_all = [], [], [], []
            for j, i in enumerate(fresh):
                idxs, aids, gids = _legal_indices(boards[i])
                rows.extend([j] * len(idxs))
                idxs_all.extend(idxs)
                aids_all.extend(aids)
                gids_all.extend(gids)
            flat = torch.from_numpy(np.array([rows, idxs_all, aids_all,
                                              gids_all], dtype=np.int64)
                                    ).to(pol.device, non_blocking=True)
            rr, tt, aa, gg = flat[0], flat[1], flat[2], flat[3]
            masked[rr, tt] = pol[rr, tt]
            agent_of[rr, tt] = aa
            group_of[rr, tt] = gg
            kk = min(2 * TOP_K, masked.shape[1])
            top = masked.topk(kk, dim=1)
            top_agent = torch.gather(agent_of, 1, top.indices).cpu().tolist()
            top_group = torch.gather(group_of, 1, top.indices).cpu().tolist()
            top_val = top.values.cpu().tolist()
            qs, ds = q.cpu().tolist(), d.cpu().tolist()
            for j, i in enumerate(fresh):
                moves, seen_g = [], set()
                for aid, g, v in zip(top_agent[j], top_group[j], top_val[j]):
                    if v <= -1e29:
                        break
                    if g in seen_g:
                        continue
                    seen_g.add(g)
                    moves.append(aid)
                    if len(moves) == TOP_K:
                        break
                while moves and len(moves) < TOP_K:
                    moves.append(moves[-1])
                r = Reply(q_bin=pv.q_to_bin(qs[j]), d_bin=pv.d_to_bin(ds[j]),
                          top_moves=moves)
                if len(self._memo) < self._memo_max:
                    self._memo[keys[i]] = r
        self.n_memo_hits += len(boards) - len(fresh)
        return [self._memo[k] for k in keys]

    def query(self, board: chess.Board) -> Reply:
        return self.query_batch([board])[0]


if __name__ == "__main__":   # latency bench: uv run python -m chessdecoder.agent.rl.oracle_engine
    import time
    import pandas as pd
    import glob as _glob
    eng = OracleEngine()
    f = sorted(_glob.glob("/mnt/2tb_2/decoder/parquet_files_decoder/*.parquet"))[0]
    fens = pd.read_parquet(f, columns=["fen"]).fen.head(4096).tolist()
    for warm in range(3):
        eng.eval_batch(fens[:256])
    torch.cuda.synchronize()
    for bs in (32, 128, 256):
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < 3:
            i = n % 3000
            eng.eval_batch(fens[i:i + bs])
            n += bs
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"eval_batch bs={bs}: {n/dt:,.0f} pos/s ({1000*dt/(n/bs):.2f} ms/batch)")
    boards = [chess.Board(f) for f in fens[:256]]
    eng._memo.clear()
    t0 = time.perf_counter()
    eng.query_batch(boards)
    torch.cuda.synchronize()
    print(f"query_batch(256, cold): {1000*(time.perf_counter()-t0):.1f} ms")

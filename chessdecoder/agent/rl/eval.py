"""Fixed eval suites + the L0/L1/L2 table.

Suites are built ONCE from the held-out corpus shard (the qref producer
never touches it): 800-sim reference search on sensitive + quiet roots.
Evaluation then compares, at matched probe budget K:
  L0  oracle greedy            (no search, no agent)
  L1  agent, greedy decode, K probes
  L2  PUCT over the oracle with K sims (equal unique-oracle-call budget)
All scored by regret under the 800-sim reference Q of the suite itself.
"""
from __future__ import annotations

import glob
import os

import chess
import numpy as np
import pandas as pd
import torch

SUITE_PATH = "agent_data/eval_suites.parquet"
HELDOUT_SHARD = sorted(glob.glob(
    "/mnt/2tb_2/decoder/parquet_files_decoder/*.parquet"))[-1]


def build_suites(n_per_suite: int = 1000, sims: int = 800,
                 seed: int = 0) -> pd.DataFrame:
    from chessdecoder.agent.rl.oracle_engine import OracleEngine
    from chessdecoder.agent.rl.qref import search_batch
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(HELDOUT_SHARD,
                         columns=["fen", "best_move", "root_q", "orig_q"])
    df = df.drop_duplicates("fen")
    gain = (df.root_q - df.orig_q).abs().fillna(0.0)
    # candidate pools: search-sensitive proxy = top-quartile |gain|, quiet =
    # bottom quartile; final sensitive label comes from the reference search
    hi = df[gain >= gain.quantile(0.75)].sample(3 * n_per_suite,
                                                random_state=seed)
    lo = df[gain <= gain.quantile(0.25)].sample(2 * n_per_suite,
                                                random_state=seed)
    engine = OracleEngine()
    rows = []
    for pool, label in ((hi, "sensitive"), (lo, "quiet")):
        kept = 0
        boards, metas = [], []
        for fen, bm, *_ in pool.itertuples(index=False):
            if kept >= n_per_suite:
                break
            try:
                b = chess.Board(fen)
            except Exception:
                continue
            if b.is_game_over() or not b.is_valid():
                continue
            boards.append(b)
            metas.append(bm)
            if len(boards) == 256 or kept + len(boards) >= n_per_suite:
                for r, cb in zip(search_batch(engine, boards, sims=sims),
                                 metas):
                    sensitive = r.search_best != r.oracle_greedy
                    if label == "sensitive" and not sensitive:
                        continue
                    rows.append(dict(
                        suite=label, fen=r.fen, moves=r.moves, q=r.q,
                        visits=r.visits, oracle_greedy=r.oracle_greedy,
                        search_best=r.search_best, corpus_best=cb,
                        search_sensitive=sensitive))
                    kept += 1
                boards, metas = [], []
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SUITE_PATH), exist_ok=True)
    out.to_parquet(SUITE_PATH)
    return out


def _regret(ref_moves, ref_q, uci) -> float:
    q = np.asarray(ref_q)
    return float(q[list(ref_moves).index(uci)] - q.max())


def eval_model(ckpt: str, ks=(0, 4, 16), batch_size: int = 128,
               suites: pd.DataFrame | None = None,
               max_roots: int | None = None) -> pd.DataFrame:
    """L0/L1/L2 regret table for one checkpoint."""
    from chessdecoder.agent.rl.engine import RolloutEngine
    from chessdecoder.agent.rl.oracle_engine import OracleEngine
    from chessdecoder.agent.rl.qref import search_batch
    from chessdecoder.agent.rl.reward import move_id_to_uci
    from chessdecoder.agent.rl.rollout_proc import load_model

    if suites is None:
        suites = pd.read_parquet(SUITE_PATH)
    model = load_model(ckpt)
    oracle = OracleEngine()
    results = []
    for suite in suites.suite.unique():
        sdf = suites[suites.suite == suite]
        if max_roots:
            sdf = sdf.head(max_roots)
        boards = [chess.Board(f) for f in sdf.fen]
        refs = list(sdf.itertuples(index=False))
        # L0
        l0 = np.mean([_regret(r.moves, r.q, r.oracle_greedy) for r in refs])
        results.append(dict(suite=suite, policy="L0_oracle_greedy", k=0,
                            regret=l0, beat_greedy=0.0))
        for k in ks:
            # L1: agent
            eng = RolloutEngine(model, oracle, batch_size=batch_size,
                                k_budget=max(k, 1), dtype=torch.bfloat16)
            regs, beats, equals = [], [], []
            for i in range(0, len(boards), batch_size):
                chunk = boards[i:i + batch_size]
                pad = batch_size - len(chunk)
                eps = eng.rollout(chunk + [chunk[-1]] * pad,
                                  k_budgets=[k] * batch_size, greedy=True)
                for b, e, r in zip(chunk, eps[:len(chunk)], refs[i:i + batch_size]):
                    uci = move_id_to_uci(b, e.final_move)
                    reg = _regret(r.moves, r.q, uci)
                    g = _regret(r.moves, r.q, r.oracle_greedy)
                    regs.append(reg)
                    beats.append(reg > g + 1e-9)
                    equals.append(abs(reg - g) <= 1e-9)
            results.append(dict(suite=suite, policy="L1_agent", k=k,
                                regret=float(np.mean(regs)),
                                beat_greedy=float(np.mean(beats)),
                                equal_greedy=float(np.mean(equals))))
            del eng
            torch.cuda.empty_cache()
            # L2: PUCT at k sims (k=0 == L0)
            if k > 0:
                regs2, beats2 = [], []
                for i in range(0, len(boards), 256):
                    chunk = boards[i:i + 256]
                    rs = search_batch(oracle, chunk, sims=k)
                    for r2, r in zip(rs, refs[i:i + 256]):
                        reg = _regret(r.moves, r.q, r2.search_best)
                        regs2.append(reg)
                        beats2.append(reg > _regret(r.moves, r.q,
                                                    r.oracle_greedy) + 1e-9)
                results.append(dict(suite=suite, policy="L2_puct", k=k,
                                    regret=float(np.mean(regs2)),
                                    beat_greedy=float(np.mean(beats2))))
    return pd.DataFrame(results)

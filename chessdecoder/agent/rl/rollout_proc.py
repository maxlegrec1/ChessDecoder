"""Rollout process (B in the architecture): generates scored episode groups.

  uv run python -m chessdecoder.agent.rl.rollout_proc chessdecoder/agent/rl/config_grpo.yaml

Loop: hot-reload trainer weights -> sample G-grouped roots from the QRef
table (sensitive_frac from search-sensitive roots) -> lockstep rollout ->
score -> write groups to the shm buffer. Backpressure: sleeps while the
buffer is deeper than buffer_max_depth.
"""
from __future__ import annotations

import os
import sys
import time

import chess
import numpy as np
import torch
import yaml

from chessdecoder.agent.model import AgentDecoder
from chessdecoder.agent.rl.buffer import GroupBuffer, load_weights_if_newer
from chessdecoder.agent.rl.engine import RolloutEngine
from chessdecoder.agent.rl.oracle_engine import OracleEngine
from chessdecoder.agent.rl.reward import QRefTable, score_episode


def load_model(ckpt: str) -> AgentDecoder:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    m = AgentDecoder(vocab_size=sd["tok_embedding.weight"].shape[0])
    m.load_state_dict(sd)
    return m


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))["rollout"]
    # salt the seed per process start: a fixed seed replays the same root
    # sequence after every restart (same bug class as the pretrain resume
    # memorization bump), triple-sampling early roots across segments
    salt = (os.getpid() * 31 + int(time.time())) % 1_000_000
    seed = cfg.get("seed", 0) + salt
    print(f"rollout seed {seed} (salt {salt})", flush=True)
    torch.manual_seed(seed)
    model = load_model(cfg["base_ckpt"])
    oracle = OracleEngine()
    qref = QRefTable(cfg["qref_dir"])
    while len(qref) < cfg["min_roots"]:
        print(f"waiting for qref roots: {len(qref)}/{cfg['min_roots']}",
              flush=True)
        time.sleep(60)
        qref.reload()
    B, G = cfg["batch_size"], cfg["group_size"]
    assert B % G == 0
    engine = RolloutEngine(model, oracle, batch_size=B,
                           k_budget=cfg["probe_budget"],
                           temperature=cfg["temperature"],
                           dtype=torch.bfloat16)
    buf = GroupBuffer(cfg["buffer_dir"])
    rng = np.random.default_rng(seed)
    version, n_batches = 0, 0
    sens, quiet = qref.split_roots()
    print(f"rollout up: {len(qref)} roots ({len(sens)} sensitive)", flush=True)

    while True:
        upd = load_weights_if_newer(cfg["weights_dir"], version)
        if upd is not None:
            sd, version = upd
            model.load_state_dict({k: v.to("cuda") for k, v in sd.items()},
                                  strict=True)
            print(f"weights -> v{version}", flush=True)
        if n_batches % 20 == 19 and qref.reload():
            sens, quiet = qref.split_roots()
        while buf.depth() >= cfg["buffer_max_depth"]:
            time.sleep(1.0)

        n_groups = B // G
        n_s = int(round(n_groups * cfg["sensitive_frac"]))
        keys = list(rng.choice(sens, size=min(n_s, len(sens)), replace=False))
        keys += list(rng.choice(quiet or sens,
                                size=n_groups - len(keys), replace=False))
        roots = [chess.Board(k + " 0 1") for k in keys for _ in range(G)]
        # (K, min_probes) sampled jointly per group: trains the whole
        # test-time-compute curve on-distribution (budget tokens K<16 were
        # OOD at eval and produced a regret sink at K=8)
        bq = cfg.get("budget_quota_choices")
        if bq:
            picks = [bq[int(rng.integers(len(bq)))] for _ in range(n_groups)]
            group_k = [int(p[0]) for p in picks]
            group_mp = [int(p[1]) for p in picks]
        else:
            group_k = [cfg["probe_budget"]] * n_groups
            mp_choices = cfg.get("min_probes_choices", [0])
            group_mp = [int(rng.choice(mp_choices)) for _ in range(n_groups)]
        ks_ = [k for k in group_k for _ in range(G)]
        mps = [mp for mp in group_mp for _ in range(G)]
        t0 = time.perf_counter()
        eps = engine.rollout(roots, k_budgets=ks_, min_probes=mps)
        dt = time.perf_counter() - t0

        for gi in range(n_groups):
            grp = eps[gi * G:(gi + 1) * G]
            ref = qref.get(grp[0].root_fen)
            scores = [score_episode(e, ref, chess.Board(e.root_fen),
                                    invalid_eps=cfg["invalid_eps"],
                                    corpus_bonus=cfg.get("corpus_bonus", 0.0))
                      for e in grp]
            L = len(grp[0].ids)
            buf.write(dict(
                root_fen=grp[0].root_fen,
                version=version,
                temperature=cfg["temperature"],
                k_budget=grp[0].k_budget,
                min_probes=group_mp[gi],
                ids=torch.tensor([e.ids for e in grp], dtype=torch.int32),
                logprobs=torch.tensor([e.logprobs for e in grp]),
                agent=torch.tensor([e.agent for e in grp]),
                rewards=torch.tensor([s["reward"] for s in scores]),
                metrics=scores,
                gen_seconds=dt / n_groups,
                tokens=sum(len(e.ids) for e in grp),
            ), version)
        n_batches += 1
        if n_batches % 10 == 0:
            toks = sum(len(e.ids) for e in eps)
            print(f"batch {n_batches}: {toks/dt:,.0f} tok/s, buffer depth "
                  f"{buf.depth()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

"""Stream packer + datasets over the task registry."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.tasks import REGISTRY
from chessdecoder.agent.tasks.sources import Sources, ROTATE_EVERY

STREAM_LEN = 4096
MAX_POS = 8192

# example-probability mix (loss-token shares verified by tests)
DEFAULT_MIX = {
    "t1_copy": 0.02, "t2_apply": 0.11, "t3_line": 0.15, "t4_agg": 0.08,
    "t5_distill": 0.20, "t6_distance": 0.04, "t7_legal": 0.06,
    "t8_played": 0.06, "t10_lastmove": 0.03, "t11_horizon": 0.03,
    "t12_path": 0.08, "t13_traj": 0.03, "t14_swing": 0.02,
    "t15_bestcolor": 0.02, "t16_reach": 0.01, "t17_opening": 0.02,
    "t18_fill": 0.04,
}


class AgentTaskDataset(IterableDataset):
    def __init__(self, parquet_dir: str, label_glob: str,
                 paired_glob: str = "", task_mix: dict | None = None,
                 seed: int = 0, stream_len: int = STREAM_LEN,
                 last_shard_only: bool = False):
        mix = dict(task_mix or DEFAULT_MIX)
        unknown = set(mix) - set(REGISTRY)
        assert not unknown, f"unknown tasks in mix: {unknown}"
        total = sum(mix.values())
        self.mix = [(name, w / total) for name, w in mix.items() if w > 0]
        self.parquet_dir, self.label_glob = parquet_dir, label_glob
        self.paired_glob = paired_glob
        self.seed = seed
        self.stream_len = stream_len
        self.last_shard_only = last_shard_only

    def _needs(self) -> set[str]:
        out: set[str] = set()
        for name, _ in self.mix:
            out.update(REGISTRY[name].requires)
        return out

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = random.Random(self.seed * 9973 + wid * 7919)
        src = Sources(self._needs(), self.parquet_dir, self.label_glob,
                      self.paired_glob, rng,
                      last_shard_only=self.last_shard_only)
        tasks = {name: REGISTRY[name]() for name, _ in self.mix}
        names = [n for n, _ in self.mix]
        weights = [w for _, w in self.mix]
        served = 0
        next_eid = wid * 100_000_000
        S = self.stream_len

        while True:
            ids = np.full(S, pv.PAD, dtype=np.int64)
            loss = np.zeros(S, dtype=bool)
            task = np.zeros(S, dtype=np.int8)
            eid = np.full(S, -1, dtype=np.int64)
            cursor = 0
            while cursor < S - 32:
                name = rng.choices(names, weights)[0]
                ex = tasks[name].make(src, rng)
                if ex is None:
                    continue
                n = len(ex.ids)
                assert len(ex.loss) == n, f"{name}: ids/loss length mismatch"
                if cursor + n > S:
                    break
                ids[cursor:cursor + n] = ex.ids
                loss[cursor:cursor + n] = ex.loss
                task[cursor:cursor + n] = np.where(ex.loss, ex.task_id, 0)
                if ex.eid_span is not None:
                    s0, sl = ex.eid_span
                    eid[cursor + s0: cursor + s0 + sl] = next_eid
                    next_eid += 1
                cursor += n
                served += 1
                if served % ROTATE_EVERY == 0:
                    src.refresh()
            offset = rng.randrange(0, MAX_POS - S + 1)
            pos = np.arange(offset, offset + S, dtype=np.int64)
            yield (torch.from_numpy(ids), torch.from_numpy(loss),
                   torch.from_numpy(task), torch.from_numpy(eid),
                   torch.from_numpy(pos))


def build_val_streams(parquet_dir: str, val_labels_path: str, n_streams: int,
                      paired_glob: str = "", task_mix: dict | None = None,
                      seed: int = 1234) -> list:
    """Fixed held-out streams: games from the LAST shard + val labels."""
    ds = AgentTaskDataset(parquet_dir, val_labels_path,
                          paired_glob=paired_glob, task_mix=task_mix,
                          seed=seed, last_shard_only=True)
    out = []
    for i, item in enumerate(iter(ds)):
        out.append(item)
        if i + 1 >= n_streams:
            break
    return out

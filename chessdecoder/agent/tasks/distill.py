"""Oracle distillation."""
from __future__ import annotations

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.tasks import Example, Task, register


@register
class OracleDistill(Task):
    """Predict the oracle's reply: q, d, top-4 moves."""
    name, tid, requires = "t5_distill", 5, ("labels",)

    def make(self, src, rng):
        b, q, d, ms = src.next_labels(1)[0]
        ids = [pv.ROOT] + b + [pv.ORACLE, pv.QBIN_BASE + q, pv.DBIN_BASE + d] + ms
        return Example(ids, [False] * 21 + [True] * 6, self.tid)

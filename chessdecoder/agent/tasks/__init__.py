"""Pretraining task registry.

A task is a class with a stable integer ``tid``, a declared set of data
``requires`` (provisioned once per worker by Sources), and a ``make`` method
producing an Example. Adding a task = one class + one line in the config's
``task_mix``; metrics and val streams pick it up automatically via the
registry.

Loss-mask contract (tested by tests/test_agent_tasks.py): loss is True only
on the example's ANSWER tokens; all context (boards given, oracle replies in
blocks, query markers) is conditioning with zero gradient. We deliberately do
NOT train full next-token prediction.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Example:
    ids: list[int]
    loss: list[bool]
    task_id: int
    eid_span: tuple[int, int] | None = None   # (start, len) of an exact-span


class Task:
    name: str = ""
    tid: int = 0
    requires: tuple[str, ...] = ()

    def make(self, src, rng) -> Example | None:
        raise NotImplementedError


REGISTRY: dict[str, type[Task]] = {}


def register(cls: type[Task]) -> type[Task]:
    assert cls.name and cls.tid, f"{cls} needs name+tid"
    assert cls.name not in REGISTRY
    assert cls.tid not in {c.tid for c in REGISTRY.values()}, f"dup tid {cls.tid}"
    REGISTRY[cls.name] = cls
    return cls


TASK_NAMES: dict[int, str] = {}


def _finalize():
    TASK_NAMES.update({c.tid: c.name for c in REGISTRY.values()})


# Import task modules (registration side effects), then freeze the name map.
from chessdecoder.agent.tasks import board_local, distill, relational  # noqa: E402,F401
_finalize()

from chessdecoder.agent.tasks.helpers import apply_uci                 # noqa: E402,F401
from chessdecoder.agent.tasks.stream import (                          # noqa: E402,F401
    AgentTaskDataset, build_val_streams, STREAM_LEN, MAX_POS)

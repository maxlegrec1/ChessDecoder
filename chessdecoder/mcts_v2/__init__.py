"""Python wrapper around the V2 C++ PUCT MCTS engine.

Mirrors the LeelaMCTS API but loads the V2 BoardForward TorchScript instead
of a Leela TRT engine. Each MCTS node is evaluated by feeding the position's
68 board tokens through V2's encoder + WDL head + decoder + policy head
(no history, no thinking trace — the "first-board policy" mode).

Usage:

    from chessdecoder.mcts_v2 import V2MCTS

    mcts = V2MCTS(
        export_dir="exports/v2",          # has board_forward.ts + vocab.txt
        simulations=800,
        cpuct=1.5,
        temperature=0.0,                  # 0 = argmax visits
        max_batch_leaves=16,
    )
    result = mcts.search("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(result["action"])               # e.g. "g1f3"
    print(result["policy"])               # [(uci, visit_frac), ...] sorted desc
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import _v2_inference_cpp as _cpp


@dataclass
class V2MCTSResult:
    action: str
    policy: list[tuple[str, float]]    # (uci, visit_fraction)
    q_values: list[tuple[str, float]]  # (uci, mean Q)
    root_wdl: tuple[float, float, float]
    sims_done: int


class V2MCTS:
    def __init__(
        self,
        export_dir: str = "exports/v2",
        *,
        simulations: int = 800,
        cpuct: float = 1.5,
        temperature: float = 0.0,
        max_batch_leaves: int = 16,
        device: str = "cuda:0",
    ) -> None:
        ts_path = os.path.join(export_dir, "board_forward.ts")
        vocab_path = os.path.join(export_dir, "vocab.txt")
        if not os.path.isfile(ts_path) or not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"Need board_forward.ts + vocab.txt in {export_dir}. "
                "Run scripts/export_v2_board_forward.py first."
            )
        self._vocab = _cpp.Vocab.from_json(vocab_path)
        self._net = _cpp.BoardForward(ts_path, device)
        self._cfg = _cpp.MctsConfig()
        self._cfg.simulations = simulations
        self._cfg.cpuct = cpuct
        self._cfg.temperature = temperature
        self._cfg.max_batch_leaves = max_batch_leaves
        self._engine = _cpp.V2Mcts(self._net, self._vocab, self._cfg)

    def search(
        self,
        fen: str,
        *,
        simulations: int | None = None,
        cpuct: float | None = None,
        temperature: float | None = None,
    ) -> V2MCTSResult:
        if any(x is not None for x in (simulations, cpuct, temperature)):
            cfg = _cpp.MctsConfig()
            cfg.simulations = simulations if simulations is not None else self._cfg.simulations
            cfg.cpuct = cpuct if cpuct is not None else self._cfg.cpuct
            cfg.temperature = temperature if temperature is not None else self._cfg.temperature
            cfg.max_batch_leaves = self._cfg.max_batch_leaves
            engine = _cpp.V2Mcts(self._net, self._vocab, cfg)
        else:
            engine = self._engine
        r = engine.search(fen)
        # sort policy + q_values by visit count desc for ergonomics
        policy_sorted = sorted(r.policy, key=lambda x: -x[1])
        q_dict = dict(r.q_values)
        q_sorted = [(uci, q_dict.get(uci, 0.0)) for uci, _ in policy_sorted]
        return V2MCTSResult(
            action=r.action,
            policy=policy_sorted,
            q_values=q_sorted,
            root_wdl=(r.root_w, r.root_d, r.root_l),
            sims_done=r.sims_done,
        )

    __call__ = search


__all__ = ["V2MCTS", "V2MCTSResult"]

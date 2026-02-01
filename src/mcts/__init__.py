from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import _inference_cpp as _cpp  # type: ignore[attr-defined]


def _coerce_history(history: Iterable[str]) -> list[str]:
    return list(history)


@dataclass
class _BaseMCTSConfig:
    engine_path: str
    simulations: int
    cpuct: float
    temperature: float


class _BaseMCTS:
    def __init__(
        self,
        engine_path: str,
        *,
        simulations: int = 800,
        cpuct: float = 1.5,
        temperature: float = 1.0,
    ) -> None:
        self._config = _BaseMCTSConfig(
            engine_path=engine_path,
            simulations=simulations,
            cpuct=cpuct,
            temperature=temperature,
        )

    def _prepare_call(
        self,
        *,
        simulations: int | None,
        cpuct: float | None,
        temperature: float | None,
        engine_path: str | None,
    ) -> _BaseMCTSConfig:
        cfg = self._config
        return _BaseMCTSConfig(
            engine_path=engine_path or cfg.engine_path,
            simulations=simulations if simulations is not None else cfg.simulations,
            cpuct=cpuct if cpuct is not None else cfg.cpuct,
            temperature=temperature if temperature is not None else cfg.temperature,
        )

    @staticmethod
    def _format_result(payload: Mapping[str, object]) -> dict[str, object]:
        action = payload.get("action")
        policy_pairs = payload.get("policy") or []
        q_values_raw = payload.get("q_values") or {}
        value = payload.get("value")

        policy: MutableMapping[str, float] = {}
        for entry in policy_pairs:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                move, prob = entry
                policy[str(move)] = float(prob)

        q_values = {str(move): float(q) for move, q in dict(q_values_raw).items()}

        return {
            "action": action,
            "policy": policy,
            "q_values": q_values,
            "value": tuple(value) if value is not None else None,
        }

class LeelaMCTS(_BaseMCTS):
    def __init__(
        self,
        engine_path: str = "leela_minibatch.trt",
        *,
        simulations: int = 800,
        cpuct: float = 1.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            engine_path,
            simulations=simulations,
            cpuct=cpuct,
            temperature=temperature,
        )

    def run(
        self,
        fen: str,
        history: Sequence[str] = (),
        *,
        simulations: int | None = None,
        cpuct: float | None = None,
        temperature: float | None = None,
        engine_path: str | None = None,
    ) -> dict[str, object]:
        cfg = self._prepare_call(
            simulations=simulations,
            cpuct=cpuct,
            temperature=temperature,
            engine_path=engine_path,
        )
        payload = _cpp.leela_mcts_search(
            fen,
            _coerce_history(history),
            cfg.simulations,
            cfg.cpuct,
            cfg.temperature,
            cfg.engine_path,
        )
        return self._format_result(payload)

    __call__ = run

    def run_parallel(
        self,
        positions: list[tuple[str, list[str]]],
        *,
        simulations: int | None = None,
        cpuct: float | None = None,
        temperature: float | None = None,
        engine_path: str | None = None,
        max_batch_size: int = 256,
        max_variations: int = 5,
        max_variation_depth: int = 20,
    ) -> list[dict[str, object]]:
        cfg = self._prepare_call(
            simulations=simulations,
            cpuct=cpuct,
            temperature=temperature,
            engine_path=engine_path,
        )
        fens = [p[0] for p in positions]
        histories = [list(p[1]) for p in positions]
        raw = _cpp.leela_mcts_search_parallel(
            fens,
            histories,
            cfg.simulations,
            cfg.cpuct,
            cfg.temperature,
            cfg.engine_path,
            max_batch_size,
            max_variations,
            max_variation_depth,
        )
        return [
            self._format_result(r) | {"variations": r.get("variations", [])}
            for r in raw
        ]

    def run_with_variations(
        self,
        fen: str,
        history: Sequence[str] = (),
        *,
        simulations: int | None = None,
        cpuct: float | None = None,
        temperature: float | None = None,
        engine_path: str | None = None,
        max_variations: int = 5,
        max_variation_depth: int = 20,
    ) -> dict[str, object]:
        cfg = self._prepare_call(
            simulations=simulations,
            cpuct=cpuct,
            temperature=temperature,
            engine_path=engine_path,
        )
        payload = _cpp.leela_mcts_search_with_variations(
            fen,
            _coerce_history(history),
            cfg.simulations,
            cfg.cpuct,
            cfg.temperature,
            cfg.engine_path,
            max_variations,
            max_variation_depth,
        )
        result = self._format_result(payload)
        result["variations"] = payload.get("variations", [])
        return result


__all__ = [
    "LeelaMCTS",
]


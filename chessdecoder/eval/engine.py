"""Model-agnostic engine abstraction for chess move prediction.

Defines a single `MovePredictor` Protocol that all engine types satisfy,
plus adapters that bridge the two C++ engines and pure-Python callables.

Design rationale
----------------
`BatchedInferenceEngine` and `ThinkingInferenceEngine` have fundamentally
different performance profiles:

  BatchedInferenceEngine  — N FENs in one batched GPU op; pre-allocated CUDA
                            graph buffers always pad to max_batch_size.
                            → optimal_batch_size = max_batch_size (e.g. 32)
                            → right tool for: offline bulk eval (tactics, CPL)

  ThinkingInferenceEngine — 1 FEN at a time; no batch padding overhead.
                            → optimal_batch_size = 1
                            → right tool for: sequential game play (ELO)

Callers that want to be engine-agnostic can pass `batch_size=engine.optimal_batch_size`
to `evaluate_puzzles` / `evaluate_positions` and get the efficient path
automatically.

Usage
-----
    from chessdecoder.eval.engine import build_batched_engine, build_thinking_engine

    # Bulk offline eval (tactics, CPL)
    engine = build_batched_engine("exports/export_282k", batch_size=32)
    results = evaluate_positions(engine, positions, batch_size=engine.optimal_batch_size)

    # Sequential game-play eval (ELO)
    engine = build_thinking_engine("exports/export_282k")
    move = engine.predict_moves([fen], 0.0)[0].move

    # Stub for unit tests (no C++ required)
    engine = PytorchModelAdapter(lambda fen, temp: "e2e4")
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from chessdecoder.utils.uci import normalize_castling


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass
class MoveResult:
    """Unified result returned by every MovePredictor implementation.

    `move` is always populated (UCI string, castling already normalised).
    The remaining fields are only populated by BatchedEngineAdapter; they
    default to empty lists for ThinkingEngineAdapter and PytorchModelAdapter.
    """
    move: str
    token_ids: list[int] = field(default_factory=list)
    wl_entries: list[tuple[int, float]] = field(default_factory=list)
    d_entries: list[tuple[int, float]] = field(default_factory=list)
    move_log_probs: list[tuple[int, float]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


@runtime_checkable
class MovePredictor(Protocol):
    """Common interface for all chess move-prediction engines.

    Attributes:
        optimal_batch_size: How many FENs per `predict_moves` call is most
            efficient for this engine.  Pass this as `batch_size` to
            `evaluate_puzzles` / `evaluate_positions` for the best throughput.
    """
    optimal_batch_size: int

    def predict_moves(
        self,
        fens: list[str],
        temperature: float,
    ) -> list[MoveResult]:
        """Predict one move per FEN.

        Args:
            fens: List of FEN strings (length ≤ optimal_batch_size is best).
            temperature: Sampling temperature; 0.0 = greedy argmax.

        Returns:
            One MoveResult per input FEN, in the same order.
        """
        ...


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


class BatchedEngineAdapter:
    """Wraps `_decoder_inference_cpp.BatchedInferenceEngine`.

    All five temperature attributes are set to 0.0 on construction
    (greedy / argmax inference).  Converts C++ `BatchedResult` objects
    to the unified `MoveResult`.

    `optimal_batch_size` equals `batch_size` passed to the constructor —
    the CUDA-graph buffers are pre-allocated for exactly that many sequences.
    Sending fewer FENs still works (the C++ engine pads internally) but
    wastes GPU work.
    """

    def __init__(self, export_dir: str, batch_size: int = 32) -> None:
        import _decoder_inference_cpp as cpp  # noqa: PLC0415 — lazy import (GPU dep)
        self.optimal_batch_size = batch_size
        self._engine = cpp.BatchedInferenceEngine(
            f"{export_dir}/backbone.pt",
            f"{export_dir}/weights",
            f"{export_dir}/vocab.json",
            f"{export_dir}/config.json",
            batch_size,
        )
        for attr in ("board_temperature", "think_temperature",
                     "policy_temperature", "wl_temperature", "d_temperature"):
            setattr(self._engine, attr, 0.0)

    def predict_moves(
        self,
        fens: list[str],
        temperature: float = 0.0,
    ) -> list[MoveResult]:
        raw = self._engine.predict_moves(fens, temperature)
        return [
            MoveResult(
                move=normalize_castling(r.move) if r.move else "",
                token_ids=list(r.token_ids),
                wl_entries=list(r.wl_entries),
                d_entries=list(r.d_entries),
                move_log_probs=list(r.move_log_probs),
            )
            for r in raw
        ]


class ThinkingEngineAdapter:
    """Wraps `_decoder_inference_cpp.ThinkingInferenceEngine`.

    Implements `predict_moves` as a sequential loop over FENs — one C++
    `predict_move` call per FEN.  This is the engine's natural operating
    mode; `optimal_batch_size = 1` tells callers not to bundle FENs.

    Only `move` is populated in returned `MoveResult`s; token_ids etc.
    are available via the engine's getter methods if needed.
    """

    optimal_batch_size: int = 1

    def __init__(self, export_dir: str) -> None:
        import _decoder_inference_cpp as cpp  # noqa: PLC0415
        self._engine = cpp.ThinkingInferenceEngine(
            f"{export_dir}/backbone.pt",
            f"{export_dir}/weights",
            f"{export_dir}/vocab.json",
            f"{export_dir}/config.json",
        )
        for attr in ("board_temperature", "think_temperature",
                     "policy_temperature", "wl_temperature", "d_temperature"):
            setattr(self._engine, attr, 0.0)

    def predict_moves(
        self,
        fens: list[str],
        temperature: float = 0.0,
    ) -> list[MoveResult]:
        return [
            MoveResult(move=normalize_castling(
                self._engine.predict_move(f, temperature) or ""
            ))
            for f in fens
        ]


class PytorchModelAdapter:
    """Wraps any callable ``(fen: str, temperature: float) -> str | None``.

    Intended for unit tests, CPU-only use, and legacy Python models —
    no C++ dependency required.  `optimal_batch_size = 1`.
    """

    optimal_batch_size: int = 1

    def __init__(self, fn: Callable[[str, float], str | None]) -> None:
        self._fn = fn

    def predict_moves(
        self,
        fens: list[str],
        temperature: float = 0.0,
    ) -> list[MoveResult]:
        return [
            MoveResult(move=normalize_castling(self._fn(f, temperature) or ""))
            for f in fens
        ]


# --------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------


def build_batched_engine(
    export_dir: str,
    batch_size: int = 32,
) -> BatchedEngineAdapter:
    """Create a greedy `BatchedEngineAdapter` for bulk offline eval.

    Suitable for tactics accuracy, CPL, and any eval that queries many
    positions in parallel.

    Args:
        export_dir: Path to the exported model directory
            (must contain backbone.pt, weights/, vocab.json, config.json).
        batch_size: CUDA-graph batch size; match to the eval script's
            `--batch-size` argument.
    """
    return BatchedEngineAdapter(export_dir, batch_size)


def build_thinking_engine(export_dir: str) -> ThinkingEngineAdapter:
    """Create a greedy `ThinkingEngineAdapter` for sequential game-play eval.

    Suitable for ELO estimation (Stockfish games) where positions are
    evaluated one at a time in game order.

    Args:
        export_dir: Path to the exported model directory.
    """
    return ThinkingEngineAdapter(export_dir)

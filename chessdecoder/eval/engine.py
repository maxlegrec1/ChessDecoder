"""Model-agnostic engine abstraction for chess move prediction.

Defines a single `MovePredictor` Protocol that all engine types satisfy,
plus adapters that bridge the two C++ engines and pure-Python callables.

Design rationale
----------------
`ThinkingBatchedInferenceEngine` and `ThinkingSingleInferenceEngine` have fundamentally
different performance profiles:

  ThinkingBatchedInferenceEngine  — N FENs in one batched GPU op; pre-allocated CUDA
                            graph buffers always pad to max_batch_size.
                            → optimal_batch_size = max_batch_size (e.g. 32)
                            → right tool for: offline bulk eval (tactics, CPL)

  ThinkingSingleInferenceEngine — 1 FEN at a time; no batch padding overhead.
                            → optimal_batch_size = 1
                            → right tool for: sequential game play (ELO)

Callers that want to be engine-agnostic can pass `batch_size=engine.optimal_batch_size`
to `evaluate_puzzles` / `evaluate_positions` and get the efficient path
automatically.

Usage
-----
    from chessdecoder.eval.engine import build_thinking_batched_engine, build_thinking_single_engine

    # Bulk offline eval (tactics, CPL)
    engine = build_thinking_batched_engine("exports/export_282k", batch_size=32)
    results = evaluate_positions(engine, positions, batch_size=engine.optimal_batch_size)

    # Sequential game-play eval (ELO)
    engine = build_thinking_single_engine("exports/export_282k")
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


class ThinkingBatchedEngineAdapter:
    """Wraps `_decoder_inference_cpp.ThinkingBatchedInferenceEngine`.

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
        self._engine = cpp.ThinkingBatchedInferenceEngine(
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


class ThinkingSingleEngineAdapter:
    """Wraps `_decoder_inference_cpp.ThinkingSingleInferenceEngine`.

    Implements `predict_moves` as a sequential loop over FENs — one C++
    `predict_move` call per FEN.  This is the engine's natural operating
    mode; `optimal_batch_size = 1` tells callers not to bundle FENs.

    Only `move` is populated in returned `MoveResult`s; token_ids etc.
    are available via the engine's getter methods if needed.
    """

    optimal_batch_size: int = 1

    def __init__(self, export_dir: str) -> None:
        import _decoder_inference_cpp as cpp  # noqa: PLC0415
        self._engine = cpp.ThinkingSingleInferenceEngine(
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


class NonThinkingModelAdapter:
    """Batched GPU inference for non-thinking (pretrain/finetune) checkpoints.

    Loads a `ChessDecoder` checkpoint directly — no TorchScript export
    needed.  All board positions are exactly 68 tokens, so N FENs can be
    stacked into a single (N, 68) batch and processed in one forward pass.

    `optimal_batch_size` should be set to whatever fills the GPU without
    OOM (default 64).  Pass this as `batch_size` to `evaluate_puzzles` /
    `evaluate_positions`.
    """

    def __init__(
        self,
        checkpoint_path: str,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        import chess  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from chessdecoder.export.common import load_model  # noqa: PLC0415
        from chessdecoder.dataloader.data import fen_to_position_tokens  # noqa: PLC0415
        from chessdecoder.models.vocab import (  # noqa: PLC0415
            token_to_idx, idx_to_token,
            move_idx_to_full_idx, move_token_to_idx,
        )

        self.optimal_batch_size = batch_size
        self._device = device
        self._torch = torch
        self._chess = chess
        self._fen_to_tokens = fen_to_position_tokens
        self._token_to_idx = token_to_idx
        self._idx_to_token = idx_to_token
        self._move_idx_to_full = move_idx_to_full_idx
        self._move_token_to_idx = move_token_to_idx

        print(f"[NonThinkingModelAdapter] Loading {checkpoint_path} ...")
        model, _ = load_model(checkpoint_path, device="cpu")
        model.to(device)
        if device == "cuda":
            model.half()
        model.eval()
        self._model = model
        print(f"[NonThinkingModelAdapter] Ready (batch={batch_size})")

    _CASTLING_TO_VOCAB = {
        "e1g1": "e1h1", "e1c1": "e1a1",
        "e8g8": "e8h8", "e8c8": "e8a8",
    }
    _CASTLING_FROM_VOCAB = {v: k for k, v in _CASTLING_TO_VOCAB.items()}

    def predict_moves(
        self,
        fens: list[str],
        temperature: float = 0.0,
    ) -> list[MoveResult]:
        torch = self._torch
        chess = self._chess

        # Build (B, 68) token-id tensor
        all_ids = [
            [self._token_to_idx[t] for t in self._fen_to_tokens(fen)]
            for fen in fens
        ]
        input_ids = torch.tensor(all_ids, dtype=torch.long, device=self._device)
        B, S = input_ids.shape
        block_id = torch.zeros(B, S, dtype=torch.long, device=self._device)

        with torch.no_grad():
            h = self._model(input_ids, mask_type="prefix", block_id=block_id)
            # policy_head at last position (stm token = index S-1)
            logits_all = self._model.policy_head(h[:, -1, :]).float()  # (B, move_vocab_size)

        results = []
        for i, fen in enumerate(fens):
            logits = logits_all[i]

            # Mask out illegal moves
            board = chess.Board(fen)
            legal_sub_idxs = []
            for move in board.legal_moves:
                uci = move.uci()
                uci = self._CASTLING_TO_VOCAB.get(uci, uci)
                if uci in self._move_token_to_idx:
                    legal_sub_idxs.append(self._move_token_to_idx[uci])
            if legal_sub_idxs:
                mask = torch.full_like(logits, float("-inf"))
                mask[legal_sub_idxs] = 0.0
                logits = logits + mask

            if temperature == 0.0:
                sub_idx = torch.argmax(logits).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                sub_idx = torch.multinomial(probs, 1).item()

            full_idx = self._move_idx_to_full[sub_idx]
            move_str = self._idx_to_token[full_idx]
            move_str = self._CASTLING_FROM_VOCAB.get(move_str, move_str)
            results.append(MoveResult(move=move_str))

        return results


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


def build_thinking_batched_engine(
    export_dir: str,
    batch_size: int = 32,
) -> ThinkingBatchedEngineAdapter:
    """Create a greedy `ThinkingBatchedEngineAdapter` for bulk offline eval.

    Suitable for tactics accuracy, CPL, and any eval that queries many
    positions in parallel.

    Args:
        export_dir: Path to the exported model directory
            (must contain backbone.pt, weights/, vocab.json, config.json).
        batch_size: CUDA-graph batch size; match to the eval script's
            `--batch-size` argument.
    """
    return ThinkingBatchedEngineAdapter(export_dir, batch_size)


def build_thinking_single_engine(export_dir: str) -> ThinkingSingleEngineAdapter:
    """Create a greedy `ThinkingSingleEngineAdapter` for sequential game-play eval.

    Suitable for ELO estimation (Stockfish games) where positions are
    evaluated one at a time in game order.

    Args:
        export_dir: Path to the exported model directory.
    """
    return ThinkingSingleEngineAdapter(export_dir)


def build_nonthinker_engine(
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = "cuda",
) -> NonThinkingModelAdapter:
    """Create a batched engine for non-thinking (pretrain/finetune) checkpoints.

    Loads the `ChessDecoder` directly from a `.pt` checkpoint — no
    TorchScript export needed.  All inputs are 68 tokens so the full
    `batch_size` is forwarded in one GPU op.

    Args:
        checkpoint_path: Path to a `checkpoint_*.pt` file.
        batch_size: Number of FENs to forward in parallel (default 64).
        device: "cuda" or "cpu".
    """
    return NonThinkingModelAdapter(checkpoint_path, batch_size=batch_size, device=device)

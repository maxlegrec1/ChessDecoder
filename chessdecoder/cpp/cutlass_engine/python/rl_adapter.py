"""RL adapter: present a libtorch-compatible interface over the CUTLASS engine.

The libtorch ThinkingBatchedInferenceEngine exposes:
    engine = cpp.ThinkingBatchedInferenceEngine(backbone_pt, weights_dir,
                                                vocab_json, config_json, B)
    engine.think_temperature = ...
    engine.policy_temperature = ...
    engine.board_temperature = ...
    engine.wl_temperature = ...
    engine.d_temperature = ...
    results = engine.predict_moves(fens, fallback_temperature)
        # → list[BatchedResult]:
        #     r.move:               str
        #     r.token_ids:          list[int]
        #     r.move_log_probs:     list[(pos, lp)]
        #     r.wl_entries:         list[(pos, value)]
        #     r.d_entries:          list[(pos, value)]
        #     r.wl_bucket_indices:  list[(pos, idx)]
        #     r.d_bucket_indices:   list[(pos, idx)]
        #     r.wl_log_probs:       list[(pos, lp)]
        #     r.d_log_probs:        list[(pos, lp)]

The CUTLASS engine emits parallel arrays (move_positions/move_log_probs,
wl_positions/wl_indices/wl_values/wl_log_probs, etc.) plus extra args on
predict_moves_thinking(fens, temp, max_seq_len, max_iters). This adapter
wraps both gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import _cutlass_decoder_cpp as _ce


# Mirrors the libtorch BatchedResult shape exactly.
@dataclass
class BatchedResultLike:
    move: str
    token_ids: list[int]
    move_log_probs: list[tuple[int, float]]
    wl_entries: list[tuple[int, float]]
    d_entries: list[tuple[int, float]]
    wl_bucket_indices: list[tuple[int, int]]
    d_bucket_indices: list[tuple[int, int]]
    wl_log_probs: list[tuple[int, float]]
    d_log_probs: list[tuple[int, float]]


def _convert(r) -> BatchedResultLike:
    """Translate cutlass RolloutResult → libtorch-shape BatchedResult."""
    return BatchedResultLike(
        move=r.move,
        token_ids=list(r.token_ids),
        move_log_probs=list(zip(list(r.move_positions), list(r.move_log_probs))),
        wl_entries=list(zip(list(r.wl_positions), list(r.wl_values))),
        d_entries=list(zip(list(r.d_positions), list(r.d_values))),
        wl_bucket_indices=list(zip(list(r.wl_positions), list(r.wl_indices))),
        d_bucket_indices=list(zip(list(r.d_positions), list(r.d_indices))),
        wl_log_probs=list(zip(list(r.wl_positions), list(r.wl_log_probs))),
        d_log_probs=list(zip(list(r.d_positions), list(r.d_log_probs))),
    )


class CutlassRLEngine:
    """Drop-in for libtorch ThinkingBatchedInferenceEngine in chessdecoder/rl/rollout.py.

    Constructor signature matches libtorch:
        CutlassRLEngine(backbone_pt, weights_dir, vocab_json, config_json, batch_size)
    Note: backbone_pt is unused by the cutlass engine (kept for API parity).
    """

    # Defaults match RL config.yaml; can be overridden via attributes after construction.
    DEFAULT_MAX_SEQ_LEN = 4096
    DEFAULT_MAX_ITERS = 64

    def __init__(self, backbone_pt: str, weights_dir: str,
                 vocab_json: str, config_json: str, batch_size: int):
        self._engine = _ce.ThinkingEngine(
            backbone_pt="",  # cutlass engine ignores this
            weights_dir=weights_dir,
            vocab_json=vocab_json,
            config_json=config_json,
            batch_size=batch_size,
        )
        self.max_seq_len = self.DEFAULT_MAX_SEQ_LEN
        self.max_iters = self.DEFAULT_MAX_ITERS

    # libtorch-style temperature setters (libtorch exposes them as Python
    # attributes that internally call setters; we mirror that).
    @property
    def think_temperature(self): return self._think_t
    @think_temperature.setter
    def think_temperature(self, v: float):
        self._think_t = v
        self._engine.think_temperature = v

    @property
    def policy_temperature(self): return self._policy_t
    @policy_temperature.setter
    def policy_temperature(self, v: float):
        self._policy_t = v
        self._engine.policy_temperature = v

    @property
    def board_temperature(self): return self._board_t
    @board_temperature.setter
    def board_temperature(self, v: float):
        self._board_t = v
        self._engine.board_temperature = v

    @property
    def wl_temperature(self): return self._wl_t
    @wl_temperature.setter
    def wl_temperature(self, v: float):
        self._wl_t = v
        self._engine.wl_temperature = v

    @property
    def d_temperature(self): return self._d_t
    @d_temperature.setter
    def d_temperature(self, v: float):
        self._d_t = v
        self._engine.d_temperature = v

    def predict_moves(self, fens: list[str], fallback_temperature: float):
        """Drop-in match for libtorch's predict_moves(fens, temp).

        Returns a list of BatchedResultLike dataclasses, one per FEN, in
        submission order.
        """
        raw = self._engine.predict_moves_thinking(
            fens, fallback_temperature, self.max_seq_len, self.max_iters)
        return [_convert(r) for r in raw]


def build_engine_for_rl(export_dir: str, batch_size: int,
                       config) -> CutlassRLEngine:
    """Construct a CutlassRLEngine matching the libtorch _build_engine pattern."""
    eng = CutlassRLEngine(
        backbone_pt=str(Path(export_dir) / "backbone.pt"),  # ignored
        weights_dir=str(Path(export_dir) / "weights"),
        vocab_json=str(Path(export_dir) / "vocab.json"),
        config_json=str(Path(export_dir) / "config.json"),
        batch_size=batch_size,
    )
    eng.think_temperature = config.think_temperature
    eng.policy_temperature = config.policy_temperature
    eng.board_temperature = config.board_temperature
    eng.wl_temperature = config.wl_temperature
    eng.d_temperature = config.d_temperature
    return eng

"""Rollout generation using the batched C++ inference engine.

The batched engine runs in-process. With Phase 0's pre-allocated KV cache
and the B200's 191 GB VRAM, we keep both the training model and the
inference engine resident simultaneously — no subprocess fork, no
model.cpu()/model.to() shuttling, no per-step CUDA-graph rebuild.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from chessdecoder.rl.config import GRPOConfig


@dataclass
class RolloutResult:
    fen: str
    final_move: str
    token_ids: list[int]
    wl_entries: list[tuple[int, float]]
    d_entries: list[tuple[int, float]]
    # (prediction_position, log_prob) pairs for every sampled move token
    # (thinking moves + final move). Position matches thinking_move_mask /
    # final_move_mask positions in sequence.py::parse_rollout.
    move_log_probs: list[tuple[int, float]]
    # WL/D bucket sampling — recorded so GRPO can reinforce wl_head/d_head.
    # Position is the wl_value/d_value token position; parse_rollout
    # subtracts 1 to align with the hidden state that produced the sample.
    wl_bucket_indices: list[tuple[int, int]]
    d_bucket_indices: list[tuple[int, int]]
    wl_log_probs: list[tuple[int, float]]
    d_log_probs: list[tuple[int, float]]
    num_tokens: int


def _build_engine(export_dir: str, config: GRPOConfig, batch_size: int):
    """Construct a fresh CUTLASS inference engine and apply temperatures."""
    import sys
    sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/python")
    from rl_adapter import build_engine_for_rl
    return build_engine_for_rl(export_dir, batch_size, config)


def generate_rollouts(
    export_dir: str,
    fens: list[str],
    config: GRPOConfig,
) -> list[list[RolloutResult]]:
    """Generate G rollouts per FEN using the batched C++ engine in-process.

    Args:
        export_dir: path to exported TorchScript model.
        fens: list of B FEN strings.
        config: GRPO config.

    Returns:
        Nested list [B][G] of RolloutResults.
    """
    G = config.group_size
    ibs = config.inference_batch_size

    # Flatten FEN×G
    all_fens = [fen for fen in fens for _ in range(G)]

    engine = _build_engine(export_dir, config, ibs)
    raw_results = []
    try:
        for start in range(0, len(all_fens), ibs):
            chunk = all_fens[start:start + ibs]
            raw = engine.predict_moves(chunk, config.think_temperature)
            for r in raw:
                raw_results.append({
                    "move": r.move,
                    "token_ids": list(r.token_ids),
                    "wl_entries": list(r.wl_entries),
                    "d_entries": list(r.d_entries),
                    "move_log_probs": list(r.move_log_probs),
                    "wl_bucket_indices": list(r.wl_bucket_indices),
                    "d_bucket_indices": list(r.d_bucket_indices),
                    "wl_log_probs": list(r.wl_log_probs),
                    "d_log_probs": list(r.d_log_probs),
                })
            print(f"  [rollout] {start + len(chunk)}/{len(all_fens)} done", flush=True)
    finally:
        # Drop the engine — caller may want the GPU back for training even
        # though Phase 0's pre-allocated KV makes the footprint stable. With
        # B200 + Phase 3 training co-resident, this could be persisted across
        # outer steps; left as a future optimization (Phase 3.1 update_weights).
        del engine
        torch.cuda.empty_cache()

    # Convert to RolloutResults and reshape [B][G]
    all_results = []
    for i, r in enumerate(raw_results):
        fen_idx = i // G
        all_results.append(RolloutResult(
            fen=fens[fen_idx],
            final_move=r["move"],
            token_ids=r["token_ids"],
            wl_entries=[tuple(e) for e in r["wl_entries"]],
            d_entries=[tuple(e) for e in r["d_entries"]],
            move_log_probs=[tuple(e) for e in r.get("move_log_probs", [])],
            wl_bucket_indices=[tuple(e) for e in r.get("wl_bucket_indices", [])],
            d_bucket_indices=[tuple(e) for e in r.get("d_bucket_indices", [])],
            wl_log_probs=[tuple(e) for e in r.get("wl_log_probs", [])],
            d_log_probs=[tuple(e) for e in r.get("d_log_probs", [])],
            num_tokens=len(r["token_ids"]),
        ))

    grouped = []
    for fen_idx in range(len(fens)):
        grouped.append(all_results[fen_idx * G:(fen_idx + 1) * G])

    return grouped


# ---------------------------------------------------------------------------
# Model export utility
# ---------------------------------------------------------------------------

def export_model(model: torch.nn.Module, config: dict, export_dir: str | Path):
    """Export current model weights for the CUTLASS engine.

    The cutlass engine reads raw FP16 weight blobs (no TorchScript backbone),
    plus vocab.json + config.json. This is a thin wrapper around
    `export_for_cutlass` that handles DDP-wrapped models and eval/train mode.

    Args:
        model: ChessDecoder (possibly wrapped in DDP).
        config: full config dict (must have "model" key).
        export_dir: directory to write weights/, vocab.json, config.json.
    """
    import sys

    sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/python")
    from export_for_cutlass import export_for_cutlass

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()
    try:
        export_for_cutlass(raw_model, config, export_dir)
    finally:
        if was_training:
            raw_model.train()

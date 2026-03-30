"""Rollout generation using the batched C++ inference engine.

The batched engine runs in the main process — no subprocess workers needed.
Before rollouts, training models are offloaded to CPU to free GPU memory.
After rollouts, the engine is destroyed and GPU memory is reclaimed for training.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from src.rl.config import GRPOConfig


@dataclass
class RolloutResult:
    fen: str
    final_move: str
    token_ids: list[int]
    wl_entries: list[tuple[int, float]]
    d_entries: list[tuple[int, float]]
    num_tokens: int


def generate_rollouts(
    export_dir: str,
    fens: list[str],
    config: GRPOConfig,
) -> list[list[RolloutResult]]:
    """Generate G rollouts per FEN using the batched C++ engine.

    Creates the engine, runs all rollouts, destroys the engine.
    The engine owns ~2-14GB GPU memory depending on batch size,
    so it must not coexist with training models on GPU.

    Args:
        export_dir: path to exported TorchScript model.
        fens: list of B FEN strings.
        config: GRPO config (uses group_size, inference_batch_size, temperatures).

    Returns:
        Nested list [B][G] of RolloutResults.
    """
    import _decoder_inference_cpp as cpp

    G = config.group_size
    B = len(fens)
    ibs = config.inference_batch_size

    engine = cpp.BatchedInferenceEngine(
        str(Path(export_dir) / "backbone.pt"),
        str(Path(export_dir) / "weights"),
        str(Path(export_dir) / "vocab.json"),
        str(Path(export_dir) / "config.json"),
        ibs,
    )
    engine.think_temperature = config.think_temperature
    engine.policy_temperature = config.policy_temperature
    engine.board_temperature = config.board_temperature

    # Flatten all FEN×G combinations
    all_fens = [fen for fen in fens for _ in range(G)]
    total = len(all_fens)

    # Process in chunks of inference_batch_size
    all_results: list[RolloutResult] = []
    for start in range(0, total, ibs):
        chunk = all_fens[start:start + ibs]
        raw = engine.predict_moves(chunk, config.think_temperature)
        for i, r in enumerate(raw):
            fen_idx = (start + i) // G
            all_results.append(RolloutResult(
                fen=fens[fen_idx],
                final_move=r.move,
                token_ids=list(r.token_ids),
                wl_entries=list(r.wl_entries),
                d_entries=list(r.d_entries),
                num_tokens=len(r.token_ids),
            ))

    # Destroy engine, free GPU
    del engine
    torch.cuda.empty_cache()

    # Reshape into [B][G]
    grouped: list[list[RolloutResult]] = []
    for fen_idx in range(B):
        group = all_results[fen_idx * G:(fen_idx + 1) * G]
        grouped.append(group)

    return grouped


# ---------------------------------------------------------------------------
# Model export utility
# ---------------------------------------------------------------------------

def export_model(model: torch.nn.Module, config: dict, export_dir: str | Path):
    """Export current model weights for C++ engine consumption.

    Args:
        model: ChessDecoder (possibly wrapped in DDP).
        config: full config dict (must have "model" key).
        export_dir: directory to write backbone.pt, weights/, vocab.json, config.json.
    """
    from src.export.common import export_head_weights, export_vocab, export_config
    from src.export.backbone_causal import from_chess_decoder
    from src.export.export_torchscript import export_torchscript

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()

    export_vocab(export_dir)
    export_config(config, raw_model, export_dir)
    export_head_weights(raw_model, export_dir / "weights")

    causal_backbone = from_chess_decoder(raw_model)
    export_torchscript(causal_backbone, export_dir, config)
    del causal_backbone
    torch.cuda.empty_cache()

    if was_training:
        raw_model.train()

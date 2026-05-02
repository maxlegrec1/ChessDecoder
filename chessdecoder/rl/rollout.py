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
    """Construct a fresh inference engine and apply temperatures.

    Set RL_ENGINE=cutlass to use the CUTLASS-based engine (35-60% faster
    on B200). Default is the libtorch ThinkingBatchedInferenceEngine.
    """
    import os
    backend = os.environ.get("RL_ENGINE", "libtorch").lower()

    if backend == "cutlass":
        import sys
        sys.path.insert(0, "/workspace/ChessDecoder/chessdecoder/cpp/cutlass_engine/python")
        from rl_adapter import build_engine_for_rl
        return build_engine_for_rl(export_dir, batch_size, config)

    import _decoder_inference_cpp as cpp
    engine = cpp.ThinkingBatchedInferenceEngine(
        str(Path(export_dir) / "backbone.pt"),
        str(Path(export_dir) / "weights"),
        str(Path(export_dir) / "vocab.json"),
        str(Path(export_dir) / "config.json"),
        batch_size,
    )
    engine.think_temperature = config.think_temperature
    engine.policy_temperature = config.policy_temperature
    engine.board_temperature = config.board_temperature
    engine.wl_temperature = config.wl_temperature
    engine.d_temperature = config.d_temperature
    return engine


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
    """Export current model weights for C++ engine consumption.

    Saves a temporary checkpoint, then runs TorchScript tracing in a subprocess
    to guarantee complete GPU memory cleanup (libtorch retains internal references
    that survive Python-side del/gc).

    Args:
        model: ChessDecoder (possibly wrapped in DDP).
        config: full config dict (must have "model" key).
        export_dir: directory to write backbone.pt, weights/, vocab.json, config.json.
    """
    import json
    import subprocess
    import sys
    import tempfile

    from chessdecoder.export.common import export_head_weights, export_vocab, export_config

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()

    # Export vocab, config, and head weights (no GPU allocation needed)
    export_vocab(export_dir)
    export_config(config, raw_model, export_dir)
    export_head_weights(raw_model, export_dir / "weights")

    # Save model state + config to temp files for subprocess
    tmp_ckpt = Path(tempfile.mktemp(suffix=".pt"))
    tmp_cfg = Path(tempfile.mktemp(suffix=".json"))
    torch.save(raw_model.state_dict(), tmp_ckpt)
    with open(tmp_cfg, "w") as f:
        json.dump({"config": config, "vocab_size": raw_model.tok_embedding.num_embeddings}, f)

    if was_training:
        raw_model.train()

    # Run TorchScript tracing in subprocess (isolates libtorch GPU memory)
    proc = subprocess.run(
        [sys.executable, "-m", "chessdecoder.rl.rollout", "--export",
         str(tmp_ckpt), str(tmp_cfg), str(export_dir)],
        capture_output=False,
        timeout=120,
    )
    tmp_ckpt.unlink(missing_ok=True)
    tmp_cfg.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"TorchScript export subprocess failed with code {proc.returncode}")


def _run_export_subprocess(ckpt_path: str, cfg_path: str, export_dir: str):
    """Subprocess entry point: load checkpoint, trace to TorchScript, save."""
    import json

    from chessdecoder.models.model import ChessDecoder
    from chessdecoder.export.backbone_causal import from_chess_decoder
    from chessdecoder.export.export_torchscript import export_torchscript

    with open(cfg_path) as f:
        payload = json.load(f)
    config = payload["config"]
    vocab_size = payload["vocab_size"]
    mc = config["model"]

    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=mc["embed_dim"],
        num_heads=mc["num_heads"],
        num_layers=mc["num_layers"],
        max_seq_len=mc["max_seq_len"],
        d_ff=mc.get("d_ff"),
        n_buckets=mc.get("n_buckets", 100),
        value_hidden_size=mc.get("value_hidden_size", 256),
        num_fourier_freq=mc.get("num_fourier_freq", 128),
        wl_sigma=mc.get("wl_sigma", 0.4),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    backbone = from_chess_decoder(model)
    del model
    export_torchscript(backbone, Path(export_dir), config)


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] == "--export":
        # python chessdecoder/rl/rollout.py --export <ckpt> <cfg> <export_dir>
        _run_export_subprocess(argv[1], argv[2], argv[3])
    else:
        raise SystemExit(
            "rollout.py is no longer a rollout subprocess entry point — "
            "rollouts run in-process via generate_rollouts(). "
            "Use --export <ckpt> <cfg> <export_dir> for TorchScript tracing."
        )

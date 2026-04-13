"""Rollout generation using the batched C++ inference engine.

The batched engine runs in the main process — no subprocess workers needed.
Before rollouts, training models are offloaded to CPU to free GPU memory.
After rollouts, the engine is destroyed and GPU memory is reclaimed for training.
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
    num_tokens: int


def _run_rollouts_subprocess(export_dir: str, fens_json_path: str, results_path: str):
    """Subprocess entry point: create engine, run rollouts, save results to disk."""
    import json
    import _decoder_inference_cpp as cpp

    with open(fens_json_path) as f:
        data = json.load(f)
    all_fens = data["fens"]
    ibs = data["inference_batch_size"]
    think_temp = data["think_temperature"]
    policy_temp = data["policy_temperature"]
    board_temp = data["board_temperature"]

    engine = cpp.ThinkingBatchedInferenceEngine(
        str(Path(export_dir) / "backbone.pt"),
        str(Path(export_dir) / "weights"),
        str(Path(export_dir) / "vocab.json"),
        str(Path(export_dir) / "config.json"),
        ibs,
    )
    engine.think_temperature = think_temp
    engine.policy_temperature = policy_temp
    engine.board_temperature = board_temp

    all_results = []
    for start in range(0, len(all_fens), ibs):
        chunk = all_fens[start:start + ibs]
        raw = engine.predict_moves(chunk, think_temp)
        for r in raw:
            all_results.append({
                "move": r.move,
                "token_ids": list(r.token_ids),
                "wl_entries": list(r.wl_entries),
                "d_entries": list(r.d_entries),
                "move_log_probs": list(r.move_log_probs),
            })
        print(f"  [rollout] {start + len(chunk)}/{len(all_fens)} done", flush=True)

    with open(results_path, "w") as f:
        json.dump(all_results, f)


def generate_rollouts(
    export_dir: str,
    fens: list[str],
    config: GRPOConfig,
) -> list[list[RolloutResult]]:
    """Generate G rollouts per FEN using the batched C++ engine in a subprocess.

    Runs the engine in a subprocess for complete GPU memory isolation —
    libtorch's CUDA allocator retains internal references that survive
    Python-side cleanup, so only process termination fully frees GPU memory.

    Args:
        export_dir: path to exported TorchScript model.
        fens: list of B FEN strings.
        config: GRPO config.

    Returns:
        Nested list [B][G] of RolloutResults.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    G = config.group_size

    # Flatten FEN×G
    all_fens = [fen for fen in fens for _ in range(G)]

    # Write input data
    tmp_dir = Path(tempfile.mkdtemp(prefix="rollout_"))
    fens_path = tmp_dir / "fens.json"
    results_path = tmp_dir / "results.json"

    with open(fens_path, "w") as f:
        json.dump({
            "fens": all_fens,
            "inference_batch_size": config.inference_batch_size,
            "think_temperature": config.think_temperature,
            "policy_temperature": config.policy_temperature,
            "board_temperature": config.board_temperature,
        }, f)

    # Run in subprocess with expandable_segments to avoid CUDA fragmentation.
    # Inherits CUDA_VISIBLE_DEVICES from parent so subprocess uses the same GPU.
    env = os.environ.copy()
    # PYTORCH_ALLOC_CONF is the new name (PyTorch 2.5+); keep the old one for
    # backwards compat with older torch builds. Setting both avoids the
    # "PYTORCH_CUDA_ALLOC_CONF is deprecated" warning on newer versions.
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    proc = subprocess.run(
        [sys.executable, "-m", "chessdecoder.rl.rollout",
         str(export_dir), str(fens_path), str(results_path)],
        capture_output=False,
        timeout=1800,
        env=env,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Rollout subprocess failed with code {proc.returncode}")

    # Read results
    with open(results_path) as f:
        raw_results = json.load(f)

    # Clean up
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

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
        # python chessdecoder/rl/rollout.py <export_dir> <fens_json> <results_json>
        _run_rollouts_subprocess(argv[0], argv[1], argv[2])

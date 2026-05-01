"""Compute the pretrain cpp_pt_best_acc metric for a single checkpoint.

Uses the BATCHED C++ inference engine — orders of magnitude faster than the
single-position engine that finetune.cpp_eval uses during training (the
single engine exists only because it has to share the GPU with the training
model; standalone eval has no such constraint).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_cpp_pretrain.py \
        --checkpoint checkpoints/finetune-thinking-v1_20260422_135004/checkpoint_step_326000.pt \
        --config chessdecoder/rl/config.yaml

Position sampling matches the RL training loop (same data dir, eval split,
seed offset = +2), so the resulting pt_best_acc is directly comparable to
the cpp_pt_best_acc trace in wandb.
"""

import argparse
import glob
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch

from chessdecoder.dataloader.sampling import load_pretrain_positions
from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size
from chessdecoder.rl.config import GRPOConfig
from chessdecoder.rl.rollout import export_model
from chessdecoder.utils.training import load_pretrained_checkpoint
from chessdecoder.utils.uci import normalize_castling


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="chessdecoder/rl/config.yaml")
    p.add_argument("--num-positions", type=int, default=None,
                   help="Override config.num_eval_positions")
    p.add_argument("--batch-size", type=int, default=200,
                   help="Batched-engine CUDA-graph batch size")
    args = p.parse_args()

    cfg = GRPOConfig.from_yaml(args.config)
    n = args.num_positions or cfg.num_eval_positions
    B = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} num_positions={n} batch_size={B}")

    # ── Build + load model just to export it ─────────────────────────────────
    mc = cfg.model
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_layers=mc["num_layers"], max_seq_len=mc["max_seq_len"],
        d_ff=mc.get("d_ff"), n_buckets=mc.get("n_buckets", 100),
        value_hidden_size=mc.get("value_hidden_size", 256),
        num_fourier_freq=mc.get("num_fourier_freq", 128),
        wl_sigma=mc.get("wl_sigma", 0.4),
    ).to(device)
    load_pretrained_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    export_dir = Path(tempfile.mkdtemp(prefix="cpp_eval_"))
    try:
        print(f"Exporting model to {export_dir} ...")
        export_model(model, {"model": cfg.model}, export_dir)

        # Free the training model — the batched engine now owns the GPU.
        del model
        torch.cuda.empty_cache()

        # ── Sample positions ─────────────────────────────────────────────────
        all_pt = sorted(glob.glob(os.path.join(cfg.pretrain_parquet_dir, "*.parquet")))
        n_train = int(len(all_pt) * cfg.pretrain_train_split)
        if n_train == len(all_pt) and len(all_pt) > 1:
            n_train -= 1
        eval_files = all_pt[n_train:]
        print(f"Pretrain split: {n_train} train / {len(eval_files)} eval files")

        positions = load_pretrain_positions(
            cfg.pretrain_parquet_dir, n, cfg.eval_seed + 2, files=eval_files,
        )
        print(f"Sampled {len(positions)} pretrain eval positions")

        # ── Build batched engine and run ─────────────────────────────────────
        import _decoder_inference_cpp as cpp
        engine = cpp.ThinkingBatchedInferenceEngine(
            str(export_dir / "backbone.pt"),
            str(export_dir / "weights"),
            str(export_dir / "vocab.json"),
            str(export_dir / "config.json"),
            B,
        )
        for attr in ("board_temperature", "think_temperature",
                     "policy_temperature", "wl_temperature", "d_temperature"):
            setattr(engine, attr, 0.0)

        correct = 0
        completed = 0
        t0 = time.time()
        N = len(positions)
        for start in range(0, N, B):
            chunk = positions[start:start + B]
            fens = [pos["fen"] for pos in chunk]
            results = engine.predict_moves(fens, 0.0)
            for pos, r in zip(chunk, results):
                move = r.move
                if move:
                    completed += 1
                    if normalize_castling(move) == pos["best_move"]:
                        correct += 1
            print(f"  [eval] {start + len(chunk)}/{N} done "
                  f"(elapsed {time.time()-t0:.1f}s)", flush=True)

        elapsed = time.time() - t0
        acc = correct / max(completed, 1)
        print()
        print(f"pt_best_acc = {acc:.4f} ({correct}/{completed} completed of {N})")
        print(f"elapsed: {elapsed:.1f}s ({N / elapsed:.1f} fen/s)")

    finally:
        shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

"""Encoder-only chess training loop.

ChessEncoder takes the 68 board tokens of a single position and predicts
``best_move`` (policy CE) and the position's WDL (soft CE on the 2-D-simplex
categorical). Per-game grouping survives in the loader so each batch is a mix
of plies from many games; the encoder itself sees every position independently
(``[B, N, 68] -> [B*N, 68]``).

Run:  CUDA_VISIBLE_DEVICES=0 uv run python chessdecoder/train/train.py \
          [chessdecoder/train/config.yaml] \
          [--set training.learning_rate=3e-4 ...] [--max-steps 1500]

The ``--set`` overrides take dotted keys and YAML-parsed values, so sweeps can
launch the same entrypoint with different LR / optimizer / model size without
generating one yaml file per run. ``--max-steps`` caps the training loop.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import wandb
import yaml
from datetime import datetime
from tqdm import tqdm

from chessdecoder.dataloader.loader import get_dataloader
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process, get_device,
    average_gradients, barrier, print_rank0)
from chessdecoder.utils.fp8 import (
    convert_model_to_fp8, compile_fp8_hot_path, count_fp8_linears)
from chessdecoder.utils.muon import build_optimizer
from chessdecoder.utils.training import (
    load_config, save_training_checkpoint, init_wandb_with_resume)

IGNORE_INDEX = -100


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?",
                   default="chessdecoder/train/config.yaml")
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   help="dotted-key override, e.g. --set training.learning_rate=3e-4")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after this many gradient steps (sweep budget cap)")
    return p.parse_args()


def _apply_override(config: dict, dotted: str) -> None:
    key, _, raw = dotted.partition("=")
    parts = key.split(".")
    d = config
    for p in parts[:-1]:
        d = d[p]
    # yaml.safe_load handles ints/bools/null/strings cleanly but the YAML 1.1
    # spec doesn't recognize bare scientific notation (``1e-3`` -> str ``"1e-3"``,
    # ``1.0e-3`` -> float). Retry as float so sweep grids stay readable.
    val = yaml.safe_load(raw)
    if isinstance(val, str):
        try:
            val = float(raw)
        except ValueError:
            pass
    d[parts[-1]] = val


def train():
    args = _parse_args()
    config = load_config(args.config)
    for kv in args.overrides:
        _apply_override(config, kv)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps

    rank, local_rank, world_size = setup_distributed()
    device = get_device(local_rank)
    print_rank0(f"Using device: {device}, world_size: {world_size}")

    seed = config["training"].get("seed", 42)
    start_epoch, step, resume_epoch_step = 0, 0, 0

    mc = config["model"]
    model = ChessEncoder(
        vocab_size=vocab_size,
        embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_layers=mc["num_layers"], seq_len=mc.get("seq_len", 68),
        d_ff=mc["d_ff"],
        attention_variant=mc.get("attention_variant", "baseline"),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"Model: {n_params/1e6:.2f}M params")

    dataloader, dataset = get_dataloader(
        config["data"]["parquet_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        positions_per_game=config["data"]["positions_per_game"],
        seed=seed, rank=rank, world_size=world_size)

    use_fp8 = config["training"].get("use_fp8", False)
    use_amp = config["training"].get("use_amp", False)
    # FP8 pairs with bf16 autocast (same range as fp32 -> no GradScaler).
    autocast_dtype = torch.bfloat16 if use_fp8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_fp8))

    # ----- model state must load BEFORE FP8 conversion -----
    resume_from = config["training"].get("resume_from")
    ck = None
    if resume_from:
        ckpts = sorted([f for f in os.listdir(resume_from)
                        if f.startswith("checkpoint_") and f.endswith(".pt")],
                       key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        if ckpts:
            latest = os.path.join(resume_from, ckpts[-1])
            print_rank0(f"Resuming from checkpoint: {latest}")
            ck = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict({k.replace("_orig_mod.", ""): v
                                   for k, v in ck["model_state_dict"].items()})
            start_epoch, step = ck["epoch"], ck["step"]
            resume_epoch_step = ck.get("epoch_step", 0)

    if use_fp8:
        recipe = config["training"].get("fp8_recipe", "tensorwise")
        convert_model_to_fp8(model, recipe=recipe)
        nfp8, nrest = count_fp8_linears(model)
        print_rank0(f"FP8: converted {nfp8} Linear modules to Float8 "
                    f"(recipe={recipe}); {nrest} remained bf16/fp32.")
        if config["training"].get("fp8_compile", True):
            compile_fp8_hot_path(model)
            print_rank0("FP8: torch.compile applied to encoder stack")

    optimizer = build_optimizer(
        model,
        config["training"].get("optimizer", "adamw"),
        config["training"]["learning_rate"],
        config["training"]["weight_decay"])
    print_rank0(f"Optimizer: {config['training'].get('optimizer', 'adamw')} "
                f"lr={config['training']['learning_rate']} "
                f"wd={config['training']['weight_decay']} "
                f"autocast_dtype={autocast_dtype}")

    if ck is not None:
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        for pg in optimizer.param_groups:
            pg["lr"] = config["training"]["learning_rate"]
        # Only restore the GradScaler state when both the saved scaler and the
        # current scaler are enabled (FP8 mode saves an empty {}).
        if scaler.is_enabled() and ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        if config["training"].get("reset_data_iteration", False):
            resume_epoch_step = 0
            print_rank0("reset_data_iteration: starting data iteration fresh "
                        "(resumed weights/optimizer/step only)")
        run_dir = resume_from
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config["training"]["checkpoint_dir"],
                               f"{config['run_name']}_{ts}")
        if is_main_process():
            os.makedirs(run_dir, exist_ok=True)
        barrier()
    print_rank0(f"Checkpoints -> {run_dir}")

    if is_main_process():
        init_wandb_with_resume(project=config["project_name"],
                               run_name=config["run_name"], config=config,
                               checkpoint_dir=run_dir)

    model.train()
    w_pol = config["loss"]["policy_weight"]
    w_wdl = config["loss"]["wdl_weight"]
    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    grad_clip = config["training"].get("grad_clip", 10.0)
    log_every = config["training"]["log_every_n_steps"]
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # pos/s = (B * N) / wall_clock between log windows.
    pos_per_step = config["data"]["batch_size"] * config["data"]["positions_per_game"]
    t_window = time.time()
    steps_in_window = 0
    max_steps = config["training"].get("max_steps")

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        print_rank0(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        dataset.epoch = epoch
        skip = resume_epoch_step
        resume_epoch_step = 0
        epoch_step = 0

        for batch in tqdm(dataloader, disable=not is_main_process()):
            epoch_step += 1
            if epoch_step <= skip:
                continue

            bid = batch["board_ids"].to(device)            # [B,N,68]
            B, N, _ = bid.shape
            # FP8 + torch.compile requires a fixed batch size (K = B*N*S must
            # be statically divisible by 16). Drop partial last batches.
            if use_fp8 and B != config["data"]["batch_size"]:
                continue
            pol_tgt = batch["policy_tgt"].to(device)       # [B,N]
            pol_val = batch["policy_valid"].to(device)
            wdl_tgt = batch["wdl_tgt"].to(device)          # [B,N,N_CELLS]
            wdl_mean = batch["wdl_mean"].to(device)        # [B,N,3]
            wdl_val = batch["wdl_valid"].to(device)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                out = model(bid.reshape(B * N, 68))
                pol_logits = out["policy"].reshape(B, N, -1)
                wdl_logits = out["wdl"].reshape(B, N, -1)

                pmask = pol_val
                policy_loss = ce(
                    pol_logits.reshape(-1, move_vocab_size),
                    torch.where(pmask, pol_tgt,
                                torch.full_like(pol_tgt, IGNORE_INDEX)).reshape(-1))

                vmask = wdl_val
                logp = torch.log_softmax(wdl_logits.float(), -1)
                wdl_ce = -(wdl_tgt * logp).sum(-1)                          # [B,N]
                wdl_loss = (wdl_ce * vmask).sum() / (vmask.sum() + 1e-8)

                total = w_pol * policy_loss + w_wdl * wdl_loss

            (scaler.scale(total / grad_accum)).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                average_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            steps_in_window += 1
            if step % log_every == 0 and is_main_process():
                with torch.no_grad():
                    move_correct = (pol_logits.argmax(-1) == pol_tgt) & pmask
                    move_acc = move_correct.sum() / (pmask.sum() + 1e-8)

                    vf = vmask.float()
                    vn = vf.sum() + 1e-8
                    wdl_p = model.mean_wdl(wdl_logits)                       # [B,N,3]
                    q_pred = wdl_p[..., 0] - wdl_p[..., 2]
                    q_tgt = wdl_mean[..., 0] - wdl_mean[..., 2]
                    q_mae = ((q_pred - q_tgt).abs() * vf).sum() / vn
                    d_mae = ((wdl_p[..., 1] - wdl_mean[..., 1]).abs() * vf).sum() / vn
                    wdl_acc = ((wdl_p.argmax(-1) == wdl_mean.argmax(-1))
                               & vmask).sum() / vn
                    pcat = torch.softmax(wdl_logits.float(), -1)
                    wdl_entropy = ((-(pcat * (pcat + 1e-9).log()).sum(-1))
                                   * vf).sum() / vn

                    policy_logit_max = pol_logits.detach().abs().max()
                    value_logit_max = wdl_logits.detach().abs().max()

                    now = time.time()
                    dt = max(now - t_window, 1e-6)
                    steps_per_s = steps_in_window / dt
                    pos_per_s = steps_per_s * pos_per_step
                    t_window = now
                    steps_in_window = 0

                print(f"Step {step}: loss {total.item():.4f} "
                      f"(pol {policy_loss.item():.4f} wdl {wdl_loss.item():.4f}) "
                      f"move_acc={move_acc.item():.3f} wdl_acc={wdl_acc.item():.3f} "
                      f"q_mae={q_mae.item():.3f} "
                      f"pos/s={pos_per_s:.0f} steps/s={steps_per_s:.2f}")
                wandb.log({
                    "train/total_loss": total.item(),
                    "train/move_loss": policy_loss.item(),
                    "train/wdl_loss": wdl_loss.item(),
                    "train/move_acc": move_acc.item(),
                    "train/wdl_acc": wdl_acc.item(),
                    "train/q_mae": q_mae.item(),
                    "train/d_mae": d_mae.item(),
                    "train/wdl_entropy": wdl_entropy.item(),
                    "train/policy_logit_max": policy_logit_max.item(),
                    "train/value_logit_max": value_logit_max.item(),
                    "train/pos_per_s": pos_per_s,
                    "train/steps_per_s": steps_per_s,
                    "train/epoch": epoch, "train/step": step,
                })

            del pol_logits, wdl_logits, out, total
            step += 1

            save_every = config["training"].get("save_every_n_steps")
            if save_every and step % save_every == 0 and is_main_process():
                save_training_checkpoint(
                    os.path.join(run_dir, f"checkpoint_{step}.pt"),
                    model=model, optimizer=optimizer, scaler=scaler, step=step,
                    extra_state={"epoch": epoch, "epoch_step": epoch_step,
                                 "config": config})
                barrier()

            if max_steps is not None and step >= max_steps:
                print_rank0(f"Reached max_steps={max_steps}, stopping.")
                cleanup_distributed()
                return

        if is_main_process():
            save_training_checkpoint(
                os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                model=model, optimizer=optimizer, scaler=scaler, step=step,
                extra_state={"epoch": epoch + 1, "epoch_step": 0, "config": config})
        barrier()

    cleanup_distributed()


if __name__ == "__main__":
    train()

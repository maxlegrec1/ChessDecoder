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
from chessdecoder.models.value_buckets import N_CELLS, project_targets
from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process, get_device,
    average_gradients, barrier, print_rank0)
from chessdecoder.utils.fp8 import (
    convert_model_to_fp8, compile_fp8_hot_path, count_fp8_linears,
    convert_moe_experts_to_fp8)
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
        input_mode=mc.get("input_mode", "default"),
        policy_head=mc.get("policy_head", "linear"),
        ffn_type=mc.get("ffn_type", "dense"),
        moe_num_experts=mc.get("moe_num_experts", 8),
        moe_top_k=mc.get("moe_top_k", 2),
        moe_expert_d_ff=mc.get("moe_expert_d_ff"),
        moe_aux_loss_weight=mc.get("moe_aux_loss_weight", 1e-2),
        moe_capacity_factor=mc.get("moe_capacity_factor"),
        moe_router_noise=mc.get("moe_router_noise", 0.0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"Model: {n_params/1e6:.2f}M params")

    # MFU bookkeeping. Approximate forward+backward FLOPs per token as
    # ``6 * N_matmul`` (Kaplan/Chinchilla convention) where N_matmul is the
    # total weight count of the model's nn.Linear modules — embedding lookups
    # and norms are gathers/elementwise and don't add tensorcore FLOPs.
    n_matmul = sum(m.in_features * m.out_features
                   for m in model.modules() if isinstance(m, nn.Linear))
    # Tokens per training step: (batch_size * positions_per_game) boards x
    # 68 board tokens each (the encoder processes every position
    # independently after the reshape inside the loop).
    tokens_per_step = (config["data"]["batch_size"]
                       * config["data"]["positions_per_game"] * 68)
    flops_per_step = 6 * n_matmul * tokens_per_step
    # Peak tensorcore FLOPs of the device the GEMMs actually run on. For an
    # RTX 4090: 165.2 TFLOPs/s in bf16, 660.6 in FP8 (E4M3, dense). Override
    # via ``training.peak_tflops`` if you're on different hardware.
    default_peak_tflops = 660.6 if config["training"].get("use_fp8", False) else 165.2
    peak_flops = config["training"].get("peak_tflops",
                                        default_peak_tflops) * 1e12
    print_rank0(f"FLOPs/step (approx): {flops_per_step/1e12:.2f} TF; "
                f"peak: {peak_flops/1e12:.0f} TF/s")

    val_pct = config["training"].get("val_pct", 2)
    max_shards = config["data"].get("max_shards")
    dataloader, dataset = get_dataloader(
        config["data"]["parquet_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        positions_per_game=config["data"]["positions_per_game"],
        seed=seed, rank=rank, world_size=world_size,
        cache_dir=config["data"].get("cache_dir"),
        split="train", val_pct=val_pct, max_shards=max_shards)

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
        if config["training"].get("moe_fp8", True):
            nmoe = convert_moe_experts_to_fp8(model)
            if nmoe:
                print_rank0(f"FP8: converted {nmoe} MoE expert blocks to rowwise "
                            f"float8 grouped-GEMM.")
        if config["training"].get("fp8_compile", True):
            compile_fp8_hot_path(model)
            print_rank0("FP8: torch.compile applied to encoder stack")

    adamw_lr_mult = config["training"].get("adamw_lr_mult", 1.0)
    optimizer = build_optimizer(
        model,
        config["training"].get("optimizer", "adamw"),
        config["training"]["learning_rate"],
        config["training"]["weight_decay"],
        adamw_lr_mult=adamw_lr_mult)
    print_rank0(f"Optimizer: {config['training'].get('optimizer', 'adamw')} "
                f"lr={config['training']['learning_rate']} "
                f"adamw_lr_mult={adamw_lr_mult} "
                f"wd={config['training']['weight_decay']} "
                f"autocast_dtype={autocast_dtype}")

    if ck is not None:
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        for pg in optimizer.param_groups:
            base = config["training"]["learning_rate"]
            pg["lr"] = base * adamw_lr_mult if pg.get("kind") == "adam" else base
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

    # ---- held-out validation (leakage-free game-hash split) ----
    # Materialize a FIXED set of val batches once so every eval scores the
    # exact same held-out positions (clean train-vs-val curve).
    val_every = config["training"].get("val_every_n_steps", 1000)
    n_val_batches = config["training"].get("val_batches", 8)
    val_data = []
    if is_main_process() and n_val_batches > 0:
        v_loader, _ = get_dataloader(
            config["data"]["parquet_dir"],
            batch_size=config["data"]["batch_size"],
            num_workers=min(4, config["data"].get("num_workers", 0)),
            positions_per_game=config["data"]["positions_per_game"],
            seed=seed, rank=rank, world_size=world_size,
            cache_dir=config["data"].get("cache_dir"),
            split="val", val_pct=val_pct, max_shards=max_shards)
        vit = iter(v_loader)
        for _ in range(n_val_batches):
            try:
                val_data.append(next(vit))
            except StopIteration:
                break
        del v_loader, vit
        print_rank0(f"Val: {len(val_data)} fixed batches "
                    f"({len(val_data) * config['data']['batch_size']} positions, "
                    f"{val_pct}% game-hash holdout)")

    @torch.no_grad()
    def run_val(val_data):
        model.eval()
        agg = dict(total_loss=0.0, move_loss=0.0, wdl_loss=0.0,
                   move_acc=0.0, wdl_acc=0.0, q_mae=0.0)
        nb = 0
        for vb in val_data:
            bid = vb["board_ids"].to(device, non_blocking=True)
            Bv, Nv, _ = bid.shape
            pol_tgt = vb["policy_tgt"].to(device, non_blocking=True)
            pol_val = vb["policy_valid"].to(device, non_blocking=True)
            wdl_mean = vb["wdl_mean"].to(device, non_blocking=True)
            wdl_val = vb["wdl_valid"].to(device, non_blocking=True)
            q_in = vb["q"].to(device, non_blocking=True)
            d_in = vb["d"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype,
                                enabled=use_amp):
                out = model(bid.reshape(Bv * Nv, 68))
                pol_logits = out["policy"].reshape(Bv, Nv, -1)
                wdl_logits = out["wdl"].reshape(Bv, Nv, -1)
                vflat = wdl_val.reshape(-1)
                wdl_tgt = torch.zeros(Bv * Nv, N_CELLS, device=device,
                                      dtype=torch.float32)
                if vflat.any():
                    wdl_tgt[vflat] = project_targets(
                        q_in.reshape(-1)[vflat], d_in.reshape(-1)[vflat])
                wdl_tgt = wdl_tgt.reshape(Bv, Nv, N_CELLS)
                pmask = pol_val
                policy_loss = ce(
                    pol_logits.reshape(-1, move_vocab_size),
                    torch.where(pmask, pol_tgt,
                                torch.full_like(pol_tgt, IGNORE_INDEX)).reshape(-1))
                vmask = wdl_val
                logp = torch.log_softmax(wdl_logits.float(), -1)
                wdl_loss = (-(wdl_tgt * logp).sum(-1) * vmask).sum() / (vmask.sum() + 1e-8)
                total = w_pol * policy_loss + w_wdl * wdl_loss
            move_acc = ((pol_logits.argmax(-1) == pol_tgt) & pmask).sum() / (pmask.sum() + 1e-8)
            wdl_p = model.mean_wdl(wdl_logits)
            vf = vmask.float(); vn = vf.sum() + 1e-8
            q_pred = wdl_p[..., 0] - wdl_p[..., 2]
            q_tgt = wdl_mean[..., 0] - wdl_mean[..., 2]
            agg["q_mae"] += (((q_pred - q_tgt).abs() * vf).sum() / vn).item()
            agg["wdl_acc"] += (((wdl_p.argmax(-1) == wdl_mean.argmax(-1)) & vmask).sum() / vn).item()
            agg["total_loss"] += total.item()
            agg["move_loss"] += policy_loss.item()
            agg["wdl_loss"] += wdl_loss.item()
            agg["move_acc"] += move_acc.item()
            nb += 1
        model.train()
        return {k: v / max(nb, 1) for k, v in agg.items()}

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

            bid = batch["board_ids"].to(device, non_blocking=True)     # [B,N,68]
            B, N, _ = bid.shape
            # FP8 + torch.compile requires a fixed batch size (K = B*N*S must
            # be statically divisible by 16). Drop partial last batches.
            if use_fp8 and B != config["data"]["batch_size"]:
                continue
            pol_tgt = batch["policy_tgt"].to(device, non_blocking=True)
            pol_val = batch["policy_valid"].to(device, non_blocking=True)
            wdl_mean = batch["wdl_mean"].to(device, non_blocking=True)  # [B,N,3]
            wdl_val = batch["wdl_valid"].to(device, non_blocking=True)
            q_in = batch["q"].to(device, non_blocking=True)             # [B,N]
            d_in = batch["d"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                out = model(bid.reshape(B * N, 68))
                pol_logits = out["policy"].reshape(B, N, -1)
                wdl_logits = out["wdl"].reshape(B, N, -1)
                # One GPU call instead of 2048 per-yield CPU calls (see
                # loader._gather_game). Zero-fills invalid rows so the
                # downstream ``vmask`` zeroing still applies cleanly.
                valid_flat = wdl_val.reshape(-1)
                wdl_tgt = torch.zeros(B * N, N_CELLS,
                                      device=device, dtype=torch.float32)
                if valid_flat.any():
                    wdl_tgt[valid_flat] = project_targets(
                        q_in.reshape(-1)[valid_flat],
                        d_in.reshape(-1)[valid_flat])
                wdl_tgt = wdl_tgt.reshape(B, N, N_CELLS)

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
                # MoE load-balancing aux loss (None for the dense FFN). Already
                # scaled by moe_aux_loss_weight inside each expert block.
                moe_aux = model.moe_aux_loss()
                if moe_aux is not None:
                    total = total + moe_aux

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
                    mfu = (flops_per_step * steps_in_window) / (dt * peak_flops)
                    t_window = now
                    steps_in_window = 0

                print(f"Step {step}: loss {total.item():.4f} "
                      f"(pol {policy_loss.item():.4f} wdl {wdl_loss.item():.4f}) "
                      f"move_acc={move_acc.item():.3f} wdl_acc={wdl_acc.item():.3f} "
                      f"q_mae={q_mae.item():.3f} "
                      f"pos/s={pos_per_s:.0f} steps/s={steps_per_s:.2f} "
                      f"mfu={mfu*100:.1f}%")
                # Cumulative compute since wandb.init: useful as a x-axis
                # alongside ``_runtime`` for scaling-law fits. ``step`` here
                # counts micro-batches (one forward+backward), so
                # ``step * flops_per_step`` is total FLOPs spent.
                cumulative_tf = (step + 1) * flops_per_step / 1e12
                log_dict = {
                    "train/total_loss": total.item(),
                    "train/move_loss": policy_loss.item(),
                    "train/wdl_loss": wdl_loss.item(),
                    "train/moe_aux_loss": (moe_aux.item()
                                           if moe_aux is not None else 0.0),
                    "train/move_acc": move_acc.item(),
                    "train/wdl_acc": wdl_acc.item(),
                    "train/q_mae": q_mae.item(),
                    "train/d_mae": d_mae.item(),
                    "train/wdl_entropy": wdl_entropy.item(),
                    "train/policy_logit_max": policy_logit_max.item(),
                    "train/value_logit_max": value_logit_max.item(),
                    "train/pos_per_s": pos_per_s,
                    "train/steps_per_s": steps_per_s,
                    "train/mfu": mfu,
                    "train/cumulative_tflops": cumulative_tf,
                    "train/epoch": epoch, "train/step": step,
                }
                # held-out val on the fixed split (val_every must be a multiple
                # of log_every so it lands inside this logging window).
                if val_data and step % val_every == 0:
                    tr_acc = move_acc.item()
                    vm = run_val(val_data)
                    log_dict.update({f"val/{k}": v for k, v in vm.items()})
                    log_dict["val/move_acc_gap"] = tr_acc - vm["move_acc"]
                    print(f"  [val @ {step}] train move_acc={tr_acc:.3f}  "
                          f"val move_acc={vm['move_acc']:.3f} "
                          f"(gap {tr_acc - vm['move_acc']:+.3f}) "
                          f"wdl_acc={vm['wdl_acc']:.3f} q_mae={vm['q_mae']:.3f}")
                wandb.log(log_dict)

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

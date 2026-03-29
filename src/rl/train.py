"""GRPO reinforcement learning training loop for ChessDecoder.

Usage:
    uv run python -m src.rl.train --config src/rl/config.yaml
"""

import argparse
import os
import random
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import wandb

from src.models.model import ChessDecoder
from src.models.vocab import vocab_size
from src.finetune.train import load_pretrained_checkpoint
from src.finetune.cpp_eval import (
    load_variation_positions,
    load_pretrain_positions,
    evaluate as evaluate_cpp_selfplay,
)
from src.utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process, get_device,
    average_gradients, barrier, print_rank0,
)

from src.rl.config import GRPOConfig
from src.rl.rollout import RolloutEngine, RolloutResult, export_model
from src.rl.sequence import parse_rollout, collate_rollouts
from src.rl.log_probs import compute_ref_log_probs, compute_current_log_probs, compute_policy_entropy
from src.rl.rewards import CompositeReward
from src.rl.grpo import compute_group_advantages, grpo_loss
from src.rl.metrics import GRPOMetrics


def _make_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_position_pool(config: GRPOConfig) -> list[dict]:
    """Load positions with ground truth for RL training.

    Uses pretrain data (Stockfish best_move) for a large, diverse pool
    that the model hasn't been directly finetuned on.
    """
    positions = load_pretrain_positions(
        config.pretrain_parquet_dir,
        n=50000,
        seed=config.eval_seed,
    )
    print_rank0(f"Loaded {len(positions)} pretrain positions for RL training")
    return positions


def _sample_batch(pool: list[dict], batch_size: int, rng: random.Random) -> list[dict]:
    """Sample a batch of positions from the pool."""
    return rng.sample(pool, min(batch_size, len(pool)))


def _save_checkpoint(model, ref_model, optimizer, scaler, config, step, checkpoint_dir):
    """Save a training checkpoint."""
    path = Path(checkpoint_dir) / f"checkpoint_rl_step_{step}.pt"
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": {
            "model": config.model,
            "grpo": {
                "group_size": config.group_size,
                "clip_epsilon": config.clip_epsilon,
                "kl_coeff": config.kl_coeff,
            },
        },
    }, path)
    print_rank0(f"  Saved checkpoint: {path}")


def train():
    parser = argparse.ArgumentParser(description="GRPO RL training for ChessDecoder")
    parser.add_argument("--config", default="src/rl/config.yaml")
    args = parser.parse_args()

    config = GRPOConfig.from_yaml(args.config)

    rank, local_rank, world_size = setup_distributed()
    device = get_device(local_rank)
    print_rank0(f"GRPO training | device={device} world_size={world_size}")

    # ── Model ────────────────────────────────────────────────────────────
    mc = config.model

    def _build_model():
        return ChessDecoder(
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
        ).to(device)

    model = _build_model()
    load_pretrained_checkpoint(model, config.pretrain_checkpoint, device)
    print_rank0(f"Loaded checkpoint: {config.pretrain_checkpoint}")

    # Reference model (frozen copy)
    ref_model = _build_model()
    raw_model = model.module if hasattr(model, "module") else model
    ref_model.load_state_dict(raw_model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print_rank0("Reference model initialized (frozen)")

    # ── Optimizer / scheduler / scaler ────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, config.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # ── Data ──────────────────────────────────────────────────────────────
    position_pool = _load_position_pool(config)
    rng = random.Random(config.eval_seed)

    # Eval positions (fixed set for periodic C++ evaluation)
    eval_var_positions = load_variation_positions(
        config.variation_parquet_dir, config.num_eval_positions, config.eval_seed + 1,
    )
    eval_pt_positions = load_pretrain_positions(
        config.pretrain_parquet_dir, config.num_eval_positions, config.eval_seed + 2,
    )

    # ── Reward function ───────────────────────────────────────────────────
    reward_fn = CompositeReward({
        "move_quality": config.reward_move_quality_weight,
        "format": config.reward_format_weight,
        "coherence": config.reward_coherence_weight,
    })

    # ── Checkpoint dir ────────────────────────────────────────────────────
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.checkpoint_dir) / f"{config.run_name}_{run_timestamp}"
    if is_main_process():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────
    if is_main_process():
        wandb.init(
            project=config.project_name,
            name=f"{config.run_name}_{run_timestamp}",
            config={
                "grpo": {
                    "group_size": config.group_size,
                    "clip_epsilon": config.clip_epsilon,
                    "kl_coeff": config.kl_coeff,
                    "ppo_epochs": config.ppo_epochs,
                    "max_kl": config.max_kl,
                },
                "training": {
                    "lr": config.learning_rate,
                    "mini_batch_size": config.mini_batch_size,
                    "grad_accum_steps": config.grad_accum_steps,
                    "max_grad_norm": config.max_grad_norm,
                },
                "rollout": {
                    "batch_size": config.rollout_batch_size,
                    "think_temperature": config.think_temperature,
                    "policy_temperature": config.policy_temperature,
                },
                "model": config.model,
            },
        )

    metrics = GRPOMetrics()

    # ── Initial model export ──────────────────────────────────────────────
    export_dir = Path(tempfile.mkdtemp(prefix="grpo_export_"))
    print_rank0(f"Exporting model to {export_dir} ...")
    export_model(model, {"model": config.model}, export_dir)
    print_rank0("Initial export complete")

    # ── Rollout engine ────────────────────────────────────────────────────
    rollout_engine = RolloutEngine(str(export_dir), config)
    print_rank0(f"Rollout engine started ({config.num_workers} workers)")

    # ── Training loop ─────────────────────────────────────────────────────
    max_seq_len = mc["max_seq_len"]
    G = config.group_size

    try:
        for outer_step in range(1, config.num_outer_steps + 1):
            step_start = time.time()
            metrics.reset()

            # 1. Sample positions
            batch_positions = _sample_batch(position_pool, config.rollout_batch_size, rng)
            fens = [p["fen"] for p in batch_positions]
            B = len(fens)

            # 2. Generate rollouts
            print_rank0(f"Step {outer_step}: generating {B}x{G} rollouts ...")
            t0 = time.time()
            grouped_rollouts = rollout_engine.generate(fens)
            rollout_time = time.time() - t0
            print_rank0(f"  Rollouts done in {rollout_time:.1f}s")

            # 3. Compute rewards
            grouped_rewards: list[list[tuple[float, dict[str, float]]]] = []
            reward_tensor = torch.zeros(B, G)
            for fen_idx, (group, gt) in enumerate(zip(grouped_rollouts, batch_positions)):
                fen_rewards = []
                for sample_idx, rollout in enumerate(group):
                    total_r, components = reward_fn(rollout.final_move, rollout.token_ids, gt)
                    fen_rewards.append((total_r, components))
                    reward_tensor[fen_idx, sample_idx] = total_r
                grouped_rewards.append(fen_rewards)

            # Log rollout metrics
            metrics.log_rollout_batch(grouped_rollouts, grouped_rewards, batch_positions, rollout_time)

            # 4. Compute group-relative advantages
            advantages = compute_group_advantages(reward_tensor)  # [B, G]
            advantages_flat = advantages.reshape(-1).to(device)   # [B*G]

            # 5. Parse rollouts into tensors
            parsed = []
            for group in grouped_rollouts:
                for rollout in group:
                    parsed.append(parse_rollout(
                        rollout.token_ids, rollout.wl_entries,
                        rollout.d_entries, max_seq_len,
                    ))
            all_batch = collate_rollouts(parsed, device)

            # 6. Compute reference log-probs (once)
            ref_lp = compute_ref_log_probs(ref_model, all_batch, config.use_amp)  # [B*G]

            # 7. Compute old policy log-probs (once, detached)
            with torch.no_grad():
                old_lp = compute_current_log_probs(model, all_batch, config.use_amp).detach()

            # 8. PPO inner loop
            N = B * G
            model.train()
            inner_step_count = 0

            kl_exceeded = False
            for ppo_epoch in range(config.ppo_epochs):
                # Shuffle indices for mini-batching
                perm = torch.randperm(N, device=device)

                for mb_start in range(0, N, config.mini_batch_size):
                    mb_end = min(mb_start + config.mini_batch_size, N)
                    mb_idx = perm[mb_start:mb_end]

                    # Slice mini-batch
                    mb_batch = {}
                    for k, v in all_batch.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == N:
                            mb_batch[k] = v[mb_idx]
                        else:
                            mb_batch[k] = v
                    mb_advantages = advantages_flat[mb_idx]
                    mb_old_lp = old_lp[mb_idx]
                    mb_ref_lp = ref_lp[mb_idx]

                    # Forward with gradients
                    mb_lp = compute_current_log_probs(model, mb_batch, config.use_amp)

                    loss, info = grpo_loss(
                        mb_lp, mb_old_lp, mb_ref_lp, mb_advantages,
                        config.clip_epsilon, config.kl_coeff,
                    )

                    scaled_loss = loss / config.grad_accum_steps
                    scaler.scale(scaled_loss).backward()

                    inner_step_count += 1
                    if inner_step_count % config.grad_accum_steps == 0:
                        scaler.unscale_(optimizer)
                        average_gradients(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                    metrics.log_training_step(info)

                    # Early stop if KL too high (checked per mini-batch)
                    if info["kl"] > config.max_kl:
                        print_rank0(f"  Early stop PPO at epoch {ppo_epoch+1}: KL={info['kl']:.4f} > {config.max_kl}")
                        kl_exceeded = True
                        break

                if kl_exceeded:
                    break

            # Flush any remaining gradients
            if inner_step_count % config.grad_accum_steps != 0:
                scaler.unscale_(optimizer)
                average_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # ── Logging ───────────────────────────────────────────────────
            step_time = time.time() - step_start
            current_lr = optimizer.param_groups[0]["lr"]

            if is_main_process() and outer_step % config.log_every == 0:
                # Entropy is expensive (full prefix forward on all rollouts)
                # Only compute every eval_every steps
                if outer_step % config.eval_every == 0:
                    with torch.no_grad():
                        entropy = compute_policy_entropy(model, all_batch, config.use_amp)
                    metrics.log_entropy(entropy)
                metrics.to_wandb(outer_step, lr=current_lr)
                print_rank0(
                    f"Step {outer_step} | "
                    f"reward={sum(r for r, _ in grouped_rewards[0]) / G:.3f} | "
                    f"kl={info['kl']:.4f} | "
                    f"clip={info['clip_fraction']:.3f} | "
                    f"time={step_time:.1f}s"
                )

            # ── Checkpoint ────────────────────────────────────────────────
            if is_main_process() and outer_step % config.save_every == 0:
                _save_checkpoint(model, ref_model, optimizer, scaler, config, outer_step, checkpoint_dir)

            # ── C++ evaluation ────────────────────────────────────────────
            eval_results = None
            if is_main_process() and outer_step % config.eval_every == 0:
                print_rank0(f"  Running C++ evaluation ...")
                eval_results = evaluate_cpp_selfplay(
                    model, {"model": config.model}, eval_var_positions, eval_pt_positions, outer_step,
                )
                if eval_results:
                    metrics.to_wandb(outer_step, lr=current_lr, eval_results=eval_results)

            # ── Re-export model for next rollout batch ────────────────────
            new_export_dir = Path(tempfile.mkdtemp(prefix="grpo_export_"))
            export_model(model, {"model": config.model}, new_export_dir)
            rollout_engine.reload(str(new_export_dir))
            # Clean up old export
            shutil.rmtree(export_dir, ignore_errors=True)
            export_dir = new_export_dir

            barrier()

    except KeyboardInterrupt:
        print_rank0("\nInterrupted. Saving checkpoint ...")
        if is_main_process():
            _save_checkpoint(model, ref_model, optimizer, scaler, config, outer_step, checkpoint_dir)

    finally:
        rollout_engine.shutdown()
        shutil.rmtree(export_dir, ignore_errors=True)
        if is_main_process():
            wandb.finish()
        cleanup_distributed()
        print_rank0("Training complete.")


if __name__ == "__main__":
    train()

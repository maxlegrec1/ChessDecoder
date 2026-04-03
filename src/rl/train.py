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
from src.rl.rollout import generate_rollouts, RolloutResult, export_model
from src.rl.sequence import parse_rollout, collate_rollouts
from src.rl.log_probs import compute_ref_log_probs, compute_current_log_probs, compute_policy_entropy
from src.rl.rewards import CompositeReward
from src.rl.grpo import compute_group_advantages, normalize_advantages_minibatch, grpo_loss
from src.rl.metrics import GRPOMetrics


def _make_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PositionStream:
    """Streams training positions from pretrain parquet files.

    Each call to sample_batch() loads B positions from the current file.
    When a file is exhausted, advances to the next one.  When all files
    have been used, reshuffles and cycles back to the first file.
    """

    def __init__(self, parquet_dir: str, seed: int):
        import glob
        self._files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self._files:
            raise RuntimeError(f"No parquet files found in {parquet_dir}")
        self._rng = random.Random(seed)
        self._rng.shuffle(self._files)
        self._file_idx = 0
        self._buffer: list[dict] = []
        self._positions_served = 0
        print_rank0(f"PositionStream: {len(self._files)} parquet files from {parquet_dir}")

    def _load_next_file(self):
        """Load and shuffle positions from the next parquet file."""
        import pandas as pd
        from src.finetune.cpp_eval import (
            _normalize_castling, _filter_standard_games, _sample_one_per_game,
        )

        if self._file_idx >= len(self._files):
            # All files consumed — reshuffle and restart
            self._rng.shuffle(self._files)
            self._file_idx = 0
            print_rank0("  PositionStream: cycled through all files, reshuffling")

        fname = self._files[self._file_idx]
        self._file_idx += 1

        df = pd.read_parquet(fname, columns=["fen", "best_move", "game_id", "ply"])
        df = _filter_standard_games(df)
        sampled = _sample_one_per_game(df, self._rng.randint(0, 2**31))
        sampled = sampled.sample(frac=1, random_state=self._rng.randint(0, 2**31)).reset_index(drop=True)

        self._buffer = [
            {"fen": row["fen"], "best_move": _normalize_castling(row["best_move"])}
            for _, row in sampled.iterrows()
        ]
        print_rank0(f"  PositionStream: loaded {len(self._buffer)} positions from {Path(fname).name}")

    def sample_batch(self, batch_size: int) -> list[dict]:
        """Load a fresh parquet file and return batch_size positions from it.

        Every call loads a new file for maximum data diversity.
        Remaining positions in the file are discarded.
        """
        self._load_next_file()
        result = self._buffer[:batch_size]
        self._buffer = []  # discard remainder — next call loads a new file
        self._positions_served += len(result)
        return result

    def state_dict(self) -> dict:
        """Serializable state for checkpointing."""
        return {
            "file_idx": self._file_idx,
            "rng_state": self._rng.getstate(),
            "positions_served": self._positions_served,
            "buffer_len": len(self._buffer),
        }

    def load_state_dict(self, state: dict):
        """Restore state from checkpoint.  Replays file loading to recover buffer."""
        self._rng.setstate(state["rng_state"])
        self._file_idx = state["file_idx"]
        self._positions_served = state["positions_served"]
        # Buffer is not saved (too large); replay the last file load to recover it.
        # After resume the first sample_batch will load the current file fresh.
        self._buffer = []
        print_rank0(f"  PositionStream: restored at file {self._file_idx}/{len(self._files)}, "
                    f"{self._positions_served} positions served")


def _save_checkpoint(model, ref_model, optimizer, scaler, config, step,
                     checkpoint_dir, position_stream=None):
    """Save a training checkpoint."""
    path = Path(checkpoint_dir) / f"checkpoint_rl_step_{step}.pt"
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": {
            "model": config.model,
            "grpo": {
                "group_size": config.group_size,
                "clip_epsilon_low": config.clip_epsilon_low,
                "clip_epsilon_high": config.clip_epsilon_high,
                "kl_coeff": config.kl_coeff,
            },
        },
    }
    if position_stream is not None:
        state["position_stream"] = position_stream.state_dict()
    torch.save(state, path)
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
    load_pretrained_checkpoint(model, config.finetuned_checkpoint, device)
    print_rank0(f"Loaded finetuned checkpoint: {config.finetuned_checkpoint}")

    # Reference model (frozen copy — always from the finetuned checkpoint,
    # NOT from a resumed RL checkpoint, so KL is measured against the original)
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

    # ── Resume from RL checkpoint ─────────────────────────────────────────
    start_step = 0
    _resume_stream_state = None
    if config.resume_from:
        rl_ckpts = sorted(
            [f for f in os.listdir(config.resume_from)
             if f.startswith("checkpoint_rl_step_") and f.endswith(".pt")],
            key=lambda x: os.path.getmtime(os.path.join(config.resume_from, x)),
        )
        if rl_ckpts:
            latest = os.path.join(config.resume_from, rl_ckpts[-1])
            print_rank0(f"Resuming from RL checkpoint: {latest}")
            rl_checkpoint = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(rl_checkpoint["model_state_dict"])
            optimizer.load_state_dict(rl_checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(rl_checkpoint["scaler_state_dict"])
            start_step = rl_checkpoint["step"]
            _resume_stream_state = rl_checkpoint.get("position_stream")
            # Advance scheduler to match
            for _ in range(start_step):
                scheduler.step()
            # Override LR in case config changed
            for pg in optimizer.param_groups:
                pg["lr"] = config.learning_rate
            print_rank0(f"  Resumed at step {start_step}")
        else:
            print_rank0(f"WARNING: resume_from={config.resume_from} but no checkpoints found")

    # ── Data ──────────────────────────────────────────────────────────────
    position_stream = PositionStream(config.pretrain_parquet_dir, config.eval_seed)
    if _resume_stream_state is not None:
        position_stream.load_state_dict(_resume_stream_state)

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

    # ── WandB (resume existing run if ID file exists) ──────────────────
    if is_main_process():
        wandb_run_id = None
        # On resume, look for wandb ID in the resumed checkpoint dir
        if config.resume_from:
            old_id_path = Path(config.resume_from) / "wandb_run_id.txt"
            if old_id_path.exists():
                wandb_run_id = old_id_path.read_text().strip()
                print_rank0(f"Resuming wandb run: {wandb_run_id}")

        wandb.init(
            project=config.project_name,
            name=f"{config.run_name}_{run_timestamp}",
            config={
                "grpo": {
                    "group_size": config.group_size,
                    "clip_epsilon_low": config.clip_epsilon_low,
                    "clip_epsilon_high": config.clip_epsilon_high,
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
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )

        # Save wandb run ID for future resume
        wandb_id_path = checkpoint_dir / "wandb_run_id.txt"
        if not wandb_run_id:
            wandb_id_path.write_text(wandb.run.id)
            print_rank0(f"Saved wandb run ID to {wandb_id_path}")
        else:
            # Copy ID to new checkpoint dir so next resume finds it
            wandb_id_path.write_text(wandb_run_id)


    metrics = GRPOMetrics()

    # ── Initial model export ──────────────────────────────────────────────
    export_dir = Path(tempfile.mkdtemp(prefix="grpo_export_"))
    print_rank0(f"Exporting model to {export_dir} ...")
    export_model(model, {"model": config.model}, export_dir)
    print_rank0("Initial export complete")

    # ── Training loop ─────────────────────────────────────────────────────
    # GPU memory lifecycle per outer step:
    #   1. Offload training models to CPU, free all GPU memory
    #   2. Run batched C++ rollouts (engine owns full GPU)
    #   3. Destroy engine, free GPU
    #   4. Reload models to GPU, run training
    #   5. Export updated model (while models on GPU), repeat
    max_seq_len = mc["max_seq_len"]
    G = config.group_size

    try:
        for outer_step in range(start_step + 1, config.num_outer_steps + 1):
            step_start = time.time()
            metrics.reset()

            # 1. Sample positions from streaming parquet files
            batch_positions = position_stream.sample_batch(config.rollout_batch_size)
            fens = [p["fen"] for p in batch_positions]
            B = len(fens)

            # 2. Offload models to CPU so subprocess gets full GPU
            model.cpu()
            ref_model.cpu()
            torch.cuda.empty_cache()

            # 3. Generate rollouts in subprocess (complete GPU memory isolation)
            print_rank0(f"Step {outer_step}: generating {B}x{G} rollouts (batch={config.inference_batch_size}) ...")
            t0 = time.time()
            grouped_rollouts = generate_rollouts(str(export_dir), fens, config)
            rollout_time = time.time() - t0
            total_rollout_tok = sum(r.num_tokens for group in grouped_rollouts for r in group)
            print_rank0(f"  Rollouts: {rollout_time:.1f}s, {total_rollout_tok} tok, "
                        f"{total_rollout_tok/rollout_time:.0f} tok/s")

            # 4. Reload models to GPU for training
            model.to(device)
            ref_model.to(device)

            # 5. Compute rewards
            grouped_rewards: list[list[tuple[float, dict[str, float]]]] = []
            reward_tensor = torch.zeros(B, G)
            for fen_idx, (group, gt) in enumerate(zip(grouped_rollouts, batch_positions)):
                fen_rewards = []
                for sample_idx, rollout in enumerate(group):
                    total_r, components = reward_fn(rollout.final_move, rollout.token_ids, gt)
                    fen_rewards.append((total_r, components))
                    reward_tensor[fen_idx, sample_idx] = total_r
                grouped_rewards.append(fen_rewards)

            metrics.log_rollout_batch(grouped_rollouts, grouped_rewards, batch_positions, rollout_time)

            # 6. Compute group-relative advantages and filter non-diverse groups
            advantages, diverse_mask = compute_group_advantages(reward_tensor)
            # diverse_mask: [B] bool — True for groups with varying rewards

            # 7. Parse rollouts into tensors, filtering out non-diverse groups
            parsed = []
            advantages_kept = []
            for fen_idx, group in enumerate(grouped_rollouts):
                if not diverse_mask[fen_idx]:
                    continue  # skip groups where all rewards are identical
                for sample_idx, rollout in enumerate(group):
                    parsed.append(parse_rollout(
                        rollout.token_ids, rollout.wl_entries,
                        rollout.d_entries, max_seq_len,
                    ))
                    advantages_kept.append(advantages[fen_idx, sample_idx].item())

            if len(parsed) == 0:
                print_rank0("  All groups non-diverse, skipping training step")
                continue

            all_batch = collate_rollouts(parsed, device)
            advantages_flat = torch.tensor(advantages_kept, device=device)
            N = len(parsed)
            n_diverse = diverse_mask.sum().item()
            print_rank0(f"  Diverse groups: {n_diverse}/{B} "
                        f"({n_diverse * G} sequences for training)")

            # 8. Single-pass GRPO update
            model.train()
            inner_step_count = 0
            perm = torch.randperm(N, device=device)

            kl_exceeded = False
            for mb_start in range(0, N, config.mini_batch_size):
                mb_end = min(mb_start + config.mini_batch_size, N)
                mb_idx = perm[mb_start:mb_end]

                mb_batch = {k: v[mb_idx] for k, v in all_batch.items()
                            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == N}
                # Mini-batch advantage normalization
                mb_advantages = normalize_advantages_minibatch(advantages_flat[mb_idx])

                with torch.no_grad():
                    mb_ref_lp, mb_move_mask = compute_ref_log_probs(ref_model, mb_batch, config.use_amp)
                with torch.no_grad():
                    mb_old_lp, _ = compute_current_log_probs(model, mb_batch, config.use_amp)
                    mb_old_lp = mb_old_lp.detach()

                mb_lp, _ = compute_current_log_probs(model, mb_batch, config.use_amp)

                loss, info = grpo_loss(
                    mb_lp, mb_old_lp, mb_ref_lp, mb_move_mask,
                    mb_advantages, config.clip_epsilon_low,
                    config.clip_epsilon_high, config.kl_coeff, G,
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

                if info["kl"] > config.max_kl:
                    print_rank0(f"  Early stop: KL={info['kl']:.4f} > {config.max_kl}")
                    kl_exceeded = True
                    break

            # Flush remaining gradients
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
                if outer_step % config.eval_every == 0:
                    think_ents, final_ents = [], []
                    with torch.no_grad():
                        for mb_s in range(0, N, config.mini_batch_size):
                            mb_e = min(mb_s + config.mini_batch_size, N)
                            mb_b = {k: v[mb_s:mb_e] for k, v in all_batch.items()
                                    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == N}
                            te, fe = compute_policy_entropy(model, mb_b, config.use_amp)
                            think_ents.append(te)
                            final_ents.append(fe)
                    metrics.log_entropy((sum(think_ents) / len(think_ents),
                                         sum(final_ents) / len(final_ents)))
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
                _save_checkpoint(model, ref_model, optimizer, scaler, config, outer_step, checkpoint_dir, position_stream)

            # ── C++ evaluation ────────────────────────────────────────────
            if is_main_process() and outer_step % config.eval_every == 0:
                print_rank0(f"  Running C++ evaluation ...")
                eval_results = evaluate_cpp_selfplay(
                    model, {"model": config.model}, eval_var_positions, eval_pt_positions, outer_step,
                )
                if eval_results:
                    metrics.to_wandb(outer_step, lr=current_lr, eval_results=eval_results)

            # ── Export model for next rollout ──────────────────────────────
            new_export_dir = Path(tempfile.mkdtemp(prefix="grpo_export_"))
            export_model(model, {"model": config.model}, new_export_dir)
            shutil.rmtree(export_dir, ignore_errors=True)
            export_dir = new_export_dir

            barrier()

    except KeyboardInterrupt:
        print_rank0("\nInterrupted. Saving checkpoint ...")
        if is_main_process():
            _save_checkpoint(model, ref_model, optimizer, scaler, config, outer_step, checkpoint_dir, position_stream)

    finally:
        shutil.rmtree(export_dir, ignore_errors=True)
        if is_main_process():
            wandb.finish()
        cleanup_distributed()
        print_rank0("Training complete.")


if __name__ == "__main__":
    train()

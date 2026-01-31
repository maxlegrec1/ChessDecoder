"""
Finetuning training loop for ChessDecoder with thinking variations.

Two-pass architecture (same as pretraining):
  Pass 1 (Causal): board_head predicts board tokens
  Pass 2 (Prefix): policy_head predicts final moves, thinking_policy_head predicts variation moves,
                    wl_head/d_head predict values

Loads pretrained checkpoint, clones policy_head -> thinking_policy_head,
and expands embedding/heads for the new end_var token (vocab_size += 1).
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from tqdm import tqdm

from src.models.model import ChessDecoder
from src.models.vocab import vocab_size, token_to_idx
from finetune.loader import get_finetune_dataloader


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def soft_bucket_loss(logits, target_values, bucket_centers, valid_mask):
    """Soft CE loss: distribute target probability across two nearest buckets."""
    N = target_values.shape[0]
    n_buckets = bucket_centers.shape[0]

    if N == 0 or valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    diffs = target_values.unsqueeze(-1) - bucket_centers
    lower_idx = (diffs >= 0).long().sum(dim=-1) - 1
    lower_idx = lower_idx.clamp(0, n_buckets - 2)
    upper_idx = lower_idx + 1

    lower_centers = bucket_centers[lower_idx]
    upper_centers = bucket_centers[upper_idx]
    span = (upper_centers - lower_centers).clamp(min=1e-8)
    upper_weight = (target_values - lower_centers) / span
    upper_weight = upper_weight.clamp(0.0, 1.0)

    soft_labels = torch.zeros(N, n_buckets, device=logits.device)
    soft_labels.scatter_(1, lower_idx.unsqueeze(1), (1 - upper_weight).unsqueeze(1))
    soft_labels.scatter_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))

    loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    return (loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)


def load_pretrained_checkpoint(model, checkpoint_path, device, old_vocab_size=None):
    """
    Load pretrained checkpoint into the finetuning model.

    1. Clone policy_head weights -> thinking_policy_head
    2. Expand embedding and output heads for new end_var token (vocab_size += 1)
    3. Load with strict=False
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Checkpoint compatibility: rename old policy_head -> board_head if needed
    if "policy_head.weight" in state_dict and "board_head.weight" not in state_dict:
        state_dict["board_head.weight"] = state_dict.pop("policy_head.weight")
        state_dict["board_head.bias"] = state_dict.pop("policy_head.bias")

    # Remove old value head if present
    state_dict.pop("value_head.weight", None)
    state_dict.pop("value_head.bias", None)

    # Clone policy_head -> thinking_policy_head
    if "policy_head.weight" in state_dict:
        state_dict["thinking_policy_head.weight"] = state_dict["policy_head.weight"].clone()
        state_dict["thinking_policy_head.bias"] = state_dict["policy_head.bias"].clone()

    # Expand embedding and heads for new vocab token(s)
    if old_vocab_size is not None:
        new_vocab_size = vocab_size
        diff = new_vocab_size - old_vocab_size

        if diff > 0:
            # Expand tok_embedding
            if "tok_embedding.weight" in state_dict:
                old_emb = state_dict["tok_embedding.weight"]  # (old_V, E)
                embed_dim = old_emb.shape[1]
                new_rows = torch.randn(diff, embed_dim, device=old_emb.device) * 0.02
                state_dict["tok_embedding.weight"] = torch.cat([old_emb, new_rows], dim=0)

            # Expand all output heads that have vocab_size as first dim
            heads_to_expand = [
                "board_head.weight", "board_head.bias",
                "policy_head.weight", "policy_head.bias",
                "thinking_policy_head.weight", "thinking_policy_head.bias",
            ]
            for key in heads_to_expand:
                if key in state_dict:
                    old_param = state_dict[key]
                    if old_param.dim() == 2 and old_param.shape[0] == old_vocab_size:
                        # Weight matrix (V, E)
                        new_rows = torch.randn(diff, old_param.shape[1], device=old_param.device) * 0.02
                        state_dict[key] = torch.cat([old_param, new_rows], dim=0)
                    elif old_param.dim() == 1 and old_param.shape[0] == old_vocab_size:
                        # Bias vector (V,)
                        new_rows = torch.zeros(diff, device=old_param.device)
                        state_dict[key] = torch.cat([old_param, new_rows], dim=0)

    model.load_state_dict(state_dict, strict=False)
    return checkpoint


def train():
    config = load_config("finetune/config.yaml")

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["max_seq_len"],
        d_ff=config["model"].get("d_ff"),
        n_buckets=config["model"].get("n_buckets", 100),
        value_hidden_size=config["model"].get("value_hidden_size", 256),
        num_fourier_freq=config["model"].get("num_fourier_freq", 128),
        wl_sigma=config["model"].get("wl_sigma", 0.4),
    ).to(device)

    # Load pretrained checkpoint
    pretrain_checkpoint = config["training"]["pretrain_checkpoint"]
    # The pretrained model had vocab_size - 1 (before adding end_var)
    old_vocab_size = vocab_size - 1
    checkpoint = load_pretrained_checkpoint(model, pretrain_checkpoint, device, old_vocab_size=old_vocab_size)
    print(f"Loaded pretrained checkpoint from {pretrain_checkpoint}")

    # Check if resuming from a finetune checkpoint
    resume_from = config["training"].get("resume_from")
    start_epoch = 0
    step = 0

    if resume_from:
        checkpoint_files = [f for f in os.listdir(resume_from) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {resume_from}")

        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        latest_checkpoint = os.path.join(resume_from, checkpoint_files[-1])

        print(f"Resuming finetune from checkpoint: {latest_checkpoint}")
        ft_checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ft_checkpoint["model_state_dict"])
        start_epoch = ft_checkpoint["epoch"]
        step = ft_checkpoint["step"]
        print(f"Resumed from epoch {start_epoch}, step {step}")

    # Dataloader
    dataloader = get_finetune_dataloader(
        pretrain_parquet_dir=config["data"]["pretrain_parquet_dir"],
        variation_parquet_dir=config["data"]["variation_parquet_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        max_seq_len=config["data"]["max_seq_len"],
        variation_ratio=config["data"].get("variation_ratio", 0.2),
        max_variations=config["data"].get("max_variations", 3),
        max_depth=config["data"].get("max_depth", 5),
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Load optimizer state if resuming
    if resume_from and "optimizer_state_dict" in ft_checkpoint:
        optimizer.load_state_dict(ft_checkpoint["optimizer_state_dict"])

    # Learning rate scheduler with warmup
    warmup_steps = config["training"].get("warmup_steps", 500)
    grad_accum = config["training"].get("gradient_accumulation_steps", 1)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_idx["pad"], reduction='none')

    # Mixed precision
    use_amp = config["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if resume_from and "scaler_state_dict" in ft_checkpoint:
        scaler.load_state_dict(ft_checkpoint["scaler_state_dict"])

    # Checkpoint directory
    if resume_from:
        run_checkpoint_dir = resume_from
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_checkpoint_dir = os.path.join(
            config["training"]["checkpoint_dir"],
            f"{config['run_name']}_{timestamp}"
        )
        os.makedirs(run_checkpoint_dir, exist_ok=True)

    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    print(f"Mixed precision training: {'enabled' if use_amp else 'disabled'}")
    print(f"Gradient accumulation steps: {grad_accum}")
    print(f"Warmup steps: {warmup_steps}")

    # Initialize wandb
    wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config,
        resume="allow" if resume_from else None,
    )

    model.train()

    # Loss weights
    final_move_weight = config["loss"]["final_move_weight"]
    thinking_move_weight = config["loss"]["thinking_move_weight"]
    board_weight = config["loss"]["board_weight"]
    wl_weight = config["loss"].get("wl_weight", 1.0)
    d_weight = config["loss"].get("d_weight", 1.0)

    V = vocab_size

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            move_mask = batch["move_mask"].to(device)
            thinking_move_mask = batch["thinking_move_mask"].to(device)
            wl_positions = batch["wl_positions"].to(device)
            d_positions = batch["d_positions"].to(device)
            wl_targets = batch["wl_targets"].to(device)
            d_targets = batch["d_targets"].to(device)
            wdl_valid = batch["wdl_valid"].to(device)
            block_id = batch["block_id"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # === Pass 1: Causal masking for board generation ===
                h_causal = model(input_ids, mask_type="causal")
                board_logits = model.board_head(h_causal)

                # Board mask: valid positions, excluding move/thinking_move/wl/d positions
                valid_mask = target_ids != token_to_idx["pad"]
                board_mask = valid_mask & (~move_mask) & (~thinking_move_mask) & (~wl_positions) & (~d_positions)

                # Exclude positions before the first move (no causal context)
                any_move = move_mask | thinking_move_mask
                first_move_idx = any_move.int().argmax(dim=1)
                has_moves = any_move.any(dim=1)
                first_move_idx[~has_moves] = any_move.size(1)
                indices = torch.arange(any_move.size(1), device=device).unsqueeze(0)
                pre_first_move_mask = indices < first_move_idx.unsqueeze(1)
                board_mask = board_mask & (~pre_first_move_mask)

                ce_board = ce_loss_fn(board_logits.view(-1, V), target_ids.view(-1))
                ce_board = ce_board.view(target_ids.shape)
                board_loss = (ce_board * board_mask.float()).sum() / (board_mask.sum() + 1e-8)

                # === Pass 2: Prefix masking for move + value prediction ===
                wl_fourier_input = torch.zeros_like(wl_targets)
                d_fourier_input = torch.zeros_like(d_targets)

                if wl_positions.any():
                    wl_vals_at_pos = wl_targets[wl_positions]
                    wl_disc = model.discretize_to_bucket(wl_vals_at_pos, model.wl_bucket_centers)
                    wl_fourier_input[wl_positions] = wl_disc

                if d_positions.any():
                    d_vals_at_pos = d_targets[d_positions]
                    d_disc = model.discretize_to_bucket(d_vals_at_pos, model.d_bucket_centers)
                    d_fourier_input[d_positions] = d_disc

                h_prefix = model(
                    input_ids, mask_type="prefix", block_id=block_id,
                    wl_values=wl_fourier_input, d_values=d_fourier_input,
                    wl_positions=wl_positions, d_positions=d_positions,
                )

                # --- 1. Final/normal move prediction -> policy_head ---
                final_logits = model.policy_head(h_prefix)
                ce_final = ce_loss_fn(final_logits.view(-1, V), target_ids.view(-1))
                ce_final = ce_final.view(target_ids.shape)
                final_move_loss = (ce_final * move_mask.float()).sum() / (move_mask.sum() + 1e-8)

                # --- 2. Thinking move prediction -> thinking_policy_head ---
                think_logits = model.thinking_policy_head(h_prefix)
                ce_think = ce_loss_fn(think_logits.view(-1, V), target_ids.view(-1))
                ce_think = ce_think.view(target_ids.shape)
                thinking_move_loss = (ce_think * thinking_move_mask.float()).sum() / (thinking_move_mask.sum() + 1e-8)

                # --- 3. WL prediction at move token positions ---
                stm_nonzero = move_mask.nonzero(as_tuple=False)
                if stm_nonzero.shape[0] > 0:
                    wl_pred_batch = stm_nonzero[:, 0]
                    wl_pred_seq = stm_nonzero[:, 1] + 1
                    wl_pred_seq = wl_pred_seq.clamp(max=h_prefix.shape[1] - 1)
                    h_at_move = h_prefix[wl_pred_batch, wl_pred_seq]
                    wl_logits_final = model.wl_head(h_at_move)
                    wl_valid_flat = wdl_valid[move_mask]
                    wl_gt_flat = wl_targets[move_mask]
                    wl_loss = soft_bucket_loss(wl_logits_final, wl_gt_flat, model.wl_bucket_centers, wl_valid_flat)
                else:
                    wl_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    wl_logits_final = None

                # --- 4. D prediction at WL placeholder positions ---
                if wl_positions.any():
                    h_at_wl = h_prefix[wl_positions]
                    d_logits_final = model.d_head(h_at_wl)
                    d_valid_flat = wdl_valid[d_positions]
                    d_gt_flat = d_targets[d_positions]
                    d_loss = soft_bucket_loss(d_logits_final, d_gt_flat, model.d_bucket_centers, d_valid_flat)
                else:
                    d_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    d_logits_final = None

                # === Total loss ===
                total_loss = (
                    final_move_weight * final_move_loss +
                    thinking_move_weight * thinking_move_loss +
                    board_weight * board_loss +
                    wl_weight * wl_loss +
                    d_weight * d_loss
                )

            # Gradient accumulation
            loss = total_loss / grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # Metrics
            with torch.no_grad():
                # Final move accuracy
                preds_final = torch.argmax(final_logits, dim=-1)
                final_correct = (preds_final == target_ids) & move_mask
                final_move_acc = final_correct.sum() / (move_mask.sum() + 1e-8)

                # Thinking move accuracy
                preds_think = torch.argmax(think_logits, dim=-1)
                think_correct = (preds_think == target_ids) & thinking_move_mask
                thinking_move_acc = think_correct.sum() / (thinking_move_mask.sum() + 1e-8)

                # Board accuracy
                preds_board = torch.argmax(board_logits, dim=-1)
                board_correct = (preds_board == target_ids) & board_mask
                board_acc = board_correct.sum() / (board_mask.sum() + 1e-8)

                # WL MAE
                if wl_logits_final is not None and stm_nonzero.shape[0] > 0:
                    wl_probs = F.softmax(wl_logits_final.float(), dim=-1)
                    expected_wl = (wl_probs * model.wl_bucket_centers.unsqueeze(0)).sum(dim=-1)
                    wl_valid_f = wl_valid_flat.float()
                    wl_mae = ((expected_wl - wl_gt_flat).abs() * wl_valid_f).sum() / (wl_valid_f.sum() + 1e-8)
                else:
                    wl_mae = torch.tensor(0.0, device=device)

                # D MAE
                if d_logits_final is not None and wl_positions.any():
                    d_probs = F.softmax(d_logits_final.float(), dim=-1)
                    expected_d = (d_probs * model.d_bucket_centers.unsqueeze(0)).sum(dim=-1)
                    d_valid_f = d_valid_flat.float()
                    d_mae = ((expected_d - d_gt_flat).abs() * d_valid_f).sum() / (d_valid_f.sum() + 1e-8)
                else:
                    d_mae = torch.tensor(0.0, device=device)

                # Track ratio of thinking vs normal samples
                batch_has_thinking = thinking_move_mask.any(dim=1).sum().item()
                batch_has_normal = (move_mask.any(dim=1) & ~thinking_move_mask.any(dim=1)).sum().item()

            if step % config["training"]["log_every_n_steps"] == 0:
                print(f"Step {step}: Loss {total_loss.item():.4f} "
                      f"(FinalMove: {final_move_loss.item():.4f}, ThinkMove: {thinking_move_loss.item():.4f}, "
                      f"Board: {board_loss.item():.4f}, WL: {wl_loss.item():.4f}, D: {d_loss.item():.4f})")

                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/final_move_loss": final_move_loss.item(),
                    "train/thinking_move_loss": thinking_move_loss.item(),
                    "train/board_loss": board_loss.item(),
                    "train/wl_loss": wl_loss.item(),
                    "train/d_loss": d_loss.item(),
                    "train/final_move_acc": final_move_acc.item(),
                    "train/thinking_move_acc": thinking_move_acc.item(),
                    "train/board_acc": board_acc.item(),
                    "train/wl_mae": wl_mae.item(),
                    "train/d_mae": d_mae.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/batch_thinking_samples": batch_has_thinking,
                    "train/batch_normal_samples": batch_has_normal,
                })

            step += 1

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
        }
        checkpoint_path = os.path.join(run_checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(ckpt, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train()

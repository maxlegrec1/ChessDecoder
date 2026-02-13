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
from src.models.vocab import (vocab_size, token_to_idx, board_vocab_size, move_vocab_size,
                              board_idx_to_full_idx, move_idx_to_full_idx, board_token_to_idx)
from src.finetune.loader import get_finetune_dataloader, get_finetune_train_val_dataloaders


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def soft_bucket_loss(logits, target_values, bucket_centers, valid_mask):
    """Soft CE loss: distribute target probability across two nearest buckets."""
    N = target_values.shape[0]
    n_buckets = bucket_centers.shape[0]

    if N == 0 or valid_mask.sum() == 0:
        # Return zero without requires_grad to avoid memory leak from retaining computation graph
        return torch.tensor(0.0, device=logits.device)

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


def migrate_state_dict(state_dict):
    """Migrate old checkpoint: sub-vocab heads, expand embedding, clone thinking_policy_head."""
    # Expand tok_embedding if needed
    if "tok_embedding.weight" in state_dict:
        t = state_dict["tok_embedding.weight"]
        if t.shape[0] < vocab_size:
            pad = torch.zeros(vocab_size - t.shape[0], *t.shape[1:], dtype=t.dtype, device=t.device)
            state_dict["tok_embedding.weight"] = torch.cat([t, pad], dim=0)

    # board_head: extract rows for board sub-vocab from old full-vocab weights
    if "board_head.weight" in state_dict:
        old_w = state_dict["board_head.weight"]
        old_b = state_dict["board_head.bias"]
        if old_w.shape[0] > board_vocab_size:
            # Old checkpoint has full-vocab head — extract sub-vocab rows
            # Some board sub-vocab tokens may have indices >= old vocab size (e.g. generic_move)
            old_vocab_sz = old_w.shape[0]
            new_w = torch.zeros(board_vocab_size, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
            new_b = torch.zeros(board_vocab_size, dtype=old_b.dtype, device=old_b.device)
            for i, full_idx in enumerate(board_idx_to_full_idx):
                if full_idx < old_vocab_sz:
                    new_w[i] = old_w[full_idx]
                    new_b[i] = old_b[full_idx]
            state_dict["board_head.weight"] = new_w
            state_dict["board_head.bias"] = new_b

    # policy_head / thinking_policy_head: extract rows for move sub-vocab
    for head in ["policy_head", "thinking_policy_head"]:
        if f"{head}.weight" not in state_dict:
            continue
        old_w = state_dict[f"{head}.weight"]
        old_b = state_dict[f"{head}.bias"]
        if old_w.shape[0] > move_vocab_size:
            old_vocab_sz = old_w.shape[0]
            new_w = torch.zeros(move_vocab_size, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
            new_b = torch.zeros(move_vocab_size, dtype=old_b.dtype, device=old_b.device)
            for i, full_idx in enumerate(move_idx_to_full_idx):
                if full_idx < old_vocab_sz:
                    new_w[i] = old_w[full_idx]
                    new_b[i] = old_b[full_idx]
            state_dict[f"{head}.weight"] = new_w
            state_dict[f"{head}.bias"] = new_b

    # Clone policy_head -> thinking_policy_head (if not already present)
    if "thinking_policy_head.weight" not in state_dict and "policy_head.weight" in state_dict:
        state_dict["thinking_policy_head.weight"] = state_dict["policy_head.weight"].clone()
        state_dict["thinking_policy_head.bias"] = state_dict["policy_head.bias"].clone()

    return state_dict


def load_pretrained_checkpoint(model, checkpoint_path, device):
    """Load pretrained checkpoint, migrating vocab size and cloning thinking_policy_head."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = migrate_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=False)
    return checkpoint


def compute_accuracies(board_target_ids, board_logits, board_mask, preds_board,
                        move_target_ids, move_mask, thinking_move_mask,
                        final_logits, think_logits,
                        continue_var_mask, new_variation_mask):
    """Compute accuracy metrics from a batch. Returns dict of accuracy values."""
    accs = {}

    # Board accuracy
    board_correct = (preds_board == board_target_ids) & board_mask
    accs["board_acc"] = board_correct.sum().item() / (board_mask.sum().item() + 1e-8)

    # Final move accuracy
    preds_final = torch.argmax(final_logits, dim=-1)
    final_correct = (preds_final == move_target_ids) & move_mask
    accs["final_move_acc"] = final_correct.sum().item() / (move_mask.sum().item() + 1e-8)

    # Thinking move accuracy
    preds_think = torch.argmax(think_logits, dim=-1)
    think_correct = (preds_think == move_target_ids) & thinking_move_mask
    accs["thinking_move_acc"] = think_correct.sum().item() / (thinking_move_mask.sum().item() + 1e-8)

    # Structural token accuracies
    end_var_board_idx = board_token_to_idx["end_var"]
    end_think_board_idx = board_token_to_idx["end_think"]
    continue_var_board_idx = board_token_to_idx["continue_var"]
    new_variation_board_idx = board_token_to_idx["new_variation"]

    end_var_target_mask = (board_target_ids == end_var_board_idx) & board_mask
    end_think_target_mask = (board_target_ids == end_think_board_idx) & board_mask

    accs["end_var_acc"] = (preds_board[end_var_target_mask] == end_var_board_idx).float().mean().item() if end_var_target_mask.any() else 0.0
    accs["end_think_acc"] = (preds_board[end_think_target_mask] == end_think_board_idx).float().mean().item() if end_think_target_mask.any() else 0.0
    accs["continue_var_acc"] = (preds_board[continue_var_mask] == continue_var_board_idx).float().mean().item() if continue_var_mask.any() else 0.0
    accs["new_variation_acc"] = (preds_board[new_variation_mask] == new_variation_board_idx).float().mean().item() if new_variation_mask.any() else 0.0

    return accs


def validate(model, val_dataloader, val_batches, device, use_amp, config):
    """Run validation and return averaged accuracy metrics."""
    model.eval()
    IGNORE_INDEX = -100

    # Accumulate counts for weighted averaging
    counts = {}  # metric_name -> (correct_sum, total_sum)
    metric_keys = ["board_acc", "final_move_acc", "thinking_move_acc",
                   "end_var_acc", "end_think_acc", "continue_var_acc", "new_variation_acc"]
    for k in metric_keys:
        counts[k] = [0.0, 0.0]

    n_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
            if n_batches >= val_batches:
                break

            input_ids = batch["input_ids"].to(device)
            board_target_ids = batch["board_target_ids"].to(device)
            move_target_ids = batch["move_target_ids"].to(device)
            move_mask = batch["move_mask"].to(device)
            thinking_move_mask = batch["thinking_move_mask"].to(device)
            wl_positions = batch["wl_positions"].to(device)
            d_positions = batch["d_positions"].to(device)
            wl_targets = batch["wl_targets"].to(device)
            d_targets = batch["d_targets"].to(device)
            block_id = batch["block_id"].to(device)
            continue_var_mask = batch["continue_var_mask"].to(device)
            new_variation_mask = batch["new_variation_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # Fourier inputs
                wl_fourier_input = torch.zeros_like(wl_targets)
                d_fourier_input = torch.zeros_like(d_targets)
                if wl_positions.any():
                    wl_fourier_input[wl_positions] = model.discretize_to_bucket(
                        wl_targets[wl_positions], model.wl_bucket_centers)
                if d_positions.any():
                    d_fourier_input[d_positions] = model.discretize_to_bucket(
                        d_targets[d_positions], model.d_bucket_centers)

                # Pass 1: Causal
                h_causal = model(
                    input_ids, mask_type="causal",
                    wl_values=wl_fourier_input, d_values=d_fourier_input,
                    wl_positions=wl_positions, d_positions=d_positions,
                )
                board_logits = model.board_head(h_causal)

                # Board mask
                board_mask = board_target_ids != IGNORE_INDEX
                any_move = move_mask | thinking_move_mask
                first_move_idx = any_move.int().argmax(dim=1)
                has_moves = any_move.any(dim=1)
                first_move_idx[~has_moves] = any_move.size(1)
                indices = torch.arange(any_move.size(1), device=device).unsqueeze(0)
                pre_first_move_mask = indices < first_move_idx.unsqueeze(1)
                board_mask = board_mask & (~pre_first_move_mask)

                preds_board = torch.argmax(board_logits, dim=-1)

                # Pass 2: Prefix
                h_prefix = model(
                    input_ids, mask_type="prefix", block_id=block_id,
                    wl_values=wl_fourier_input, d_values=d_fourier_input,
                    wl_positions=wl_positions, d_positions=d_positions,
                )
                final_logits = model.policy_head(h_prefix)
                think_logits = model.thinking_policy_head(h_prefix)

            # Compute accuracies
            accs = compute_accuracies(
                board_target_ids, board_logits, board_mask, preds_board,
                move_target_ids, move_mask, thinking_move_mask,
                final_logits, think_logits,
                continue_var_mask, new_variation_mask,
            )

            # Accumulate with counts for proper weighted averaging
            counts["board_acc"][0] += (preds_board == board_target_ids)[board_mask].float().sum().item()
            counts["board_acc"][1] += board_mask.sum().item()

            preds_final = torch.argmax(final_logits, dim=-1)
            counts["final_move_acc"][0] += ((preds_final == move_target_ids) & move_mask).sum().item()
            counts["final_move_acc"][1] += move_mask.sum().item()

            preds_think = torch.argmax(think_logits, dim=-1)
            counts["thinking_move_acc"][0] += ((preds_think == move_target_ids) & thinking_move_mask).sum().item()
            counts["thinking_move_acc"][1] += thinking_move_mask.sum().item()

            end_var_board_idx = board_token_to_idx["end_var"]
            end_think_board_idx = board_token_to_idx["end_think"]
            continue_var_board_idx = board_token_to_idx["continue_var"]
            new_variation_board_idx = board_token_to_idx["new_variation"]

            ev_mask = (board_target_ids == end_var_board_idx) & board_mask
            counts["end_var_acc"][0] += (preds_board[ev_mask] == end_var_board_idx).sum().item() if ev_mask.any() else 0
            counts["end_var_acc"][1] += ev_mask.sum().item()

            et_mask = (board_target_ids == end_think_board_idx) & board_mask
            counts["end_think_acc"][0] += (preds_board[et_mask] == end_think_board_idx).sum().item() if et_mask.any() else 0
            counts["end_think_acc"][1] += et_mask.sum().item()

            counts["continue_var_acc"][0] += (preds_board[continue_var_mask] == continue_var_board_idx).sum().item() if continue_var_mask.any() else 0
            counts["continue_var_acc"][1] += continue_var_mask.sum().item()

            counts["new_variation_acc"][0] += (preds_board[new_variation_mask] == new_variation_board_idx).sum().item() if new_variation_mask.any() else 0
            counts["new_variation_acc"][1] += new_variation_mask.sum().item()

            n_batches += 1

    model.train()

    # Compute final averaged metrics
    results = {}
    for k in metric_keys:
        correct, total = counts[k]
        results[k] = correct / (total + 1e-8)

    print(f"  Validation ({n_batches} batches): "
          f"board={results['board_acc']:.4f}, final_move={results['final_move_acc']:.4f}, "
          f"think_move={results['thinking_move_acc']:.4f}, "
          f"end_var={results['end_var_acc']:.4f}, continue_var={results['continue_var_acc']:.4f}, "
          f"end_think={results['end_think_acc']:.4f}, new_var={results['new_variation_acc']:.4f}")

    return results


def train():
    config = load_config("src/finetune/config.yaml")

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
    checkpoint = load_pretrained_checkpoint(model, pretrain_checkpoint, device)
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

    # Dataloaders (train/val split)
    val_batches = config["training"].get("val_batches", 100)
    dataloader, val_dataloader = get_finetune_train_val_dataloaders(
        pretrain_parquet_dir=config["data"]["pretrain_parquet_dir"],
        variation_parquet_dir=config["data"]["variation_parquet_dir"],
        train_split=config["data"].get("train_split", 0.8),
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        max_seq_len=config["data"]["max_seq_len"],
        variation_ratio=config["data"].get("variation_ratio", 0.2),
        max_variations=config["data"].get("max_variations", 3),
        max_depth=config["data"].get("max_depth", 5),
        tau_base=config["data"].get("tau_base", 0.3),
        tau_alpha=config["data"].get("tau_alpha", 1.0),
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
    save_every_n_steps = config["training"].get("save_every_n_steps", 0)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    IGNORE_INDEX = -100
    board_ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
    move_ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')

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

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            board_target_ids = batch["board_target_ids"].to(device)
            move_target_ids = batch["move_target_ids"].to(device)
            move_mask = batch["move_mask"].to(device)
            thinking_move_mask = batch["thinking_move_mask"].to(device)
            wl_positions = batch["wl_positions"].to(device)
            d_positions = batch["d_positions"].to(device)
            wl_targets = batch["wl_targets"].to(device)
            d_targets = batch["d_targets"].to(device)
            wdl_valid = batch["wdl_valid"].to(device)
            block_id = batch["block_id"].to(device)
            first_is_not_best = batch["first_is_not_best"].to(device)
            pretrain_epoch = batch["pretrain_epoch"].to(device)
            continue_var_mask = batch["continue_var_mask"].to(device)
            new_variation_mask = batch["new_variation_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # === Prepare Fourier inputs (shared by both passes) ===
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

                # === Pass 1: Causal masking for board generation ===
                h_causal = model(
                    input_ids, mask_type="causal",
                    wl_values=wl_fourier_input, d_values=d_fourier_input,
                    wl_positions=wl_positions, d_positions=d_positions
                )
                board_logits = model.board_head(h_causal)

                # Board mask: non-ignored positions, excluding pre-first-move
                board_mask = board_target_ids != IGNORE_INDEX

                any_move = move_mask | thinking_move_mask
                first_move_idx = any_move.int().argmax(dim=1)
                has_moves = any_move.any(dim=1)
                first_move_idx[~has_moves] = any_move.size(1)
                indices = torch.arange(any_move.size(1), device=device).unsqueeze(0)
                pre_first_move_mask = indices < first_move_idx.unsqueeze(1)
                board_mask = board_mask & (~pre_first_move_mask)

                ce_board = board_ce_fn(board_logits.view(-1, board_vocab_size), board_target_ids.view(-1))
                ce_board = ce_board.view(board_target_ids.shape)
                board_loss = (ce_board * board_mask.float()).sum() / (board_mask.sum() + 1e-8)

                # === Pass 2: Prefix masking for move + value prediction ===
                h_prefix = model(
                    input_ids, mask_type="prefix", block_id=block_id,
                    wl_values=wl_fourier_input, d_values=d_fourier_input,
                    wl_positions=wl_positions, d_positions=d_positions,
                )

                # --- 1. Final/normal move prediction -> policy_head ---
                final_logits = model.policy_head(h_prefix)
                ce_final = move_ce_fn(final_logits.view(-1, move_vocab_size), move_target_ids.view(-1))
                ce_final = ce_final.view(move_target_ids.shape)
                final_move_loss = (ce_final * move_mask.float()).sum() / (move_mask.sum() + 1e-8)

                # --- 2. Thinking move prediction -> thinking_policy_head ---
                think_logits = model.thinking_policy_head(h_prefix)
                ce_think = move_ce_fn(think_logits.view(-1, move_vocab_size), move_target_ids.view(-1))
                ce_think = ce_think.view(move_target_ids.shape)
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
                    wl_loss = torch.tensor(0.0, device=device)
                    wl_logits_final = None

                # --- 4. D prediction at WL placeholder positions ---
                if wl_positions.any():
                    h_at_wl = h_prefix[wl_positions]
                    d_logits_final = model.d_head(h_at_wl)
                    d_valid_flat = wdl_valid[d_positions]
                    d_gt_flat = d_targets[d_positions]
                    d_loss = soft_bucket_loss(d_logits_final, d_gt_flat, model.d_bucket_centers, d_valid_flat)
                else:
                    d_loss = torch.tensor(0.0, device=device)
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
                # Final move accuracy (move sub-vocab space)
                preds_final = torch.argmax(final_logits, dim=-1)
                final_correct = (preds_final == move_target_ids) & move_mask
                final_move_acc = final_correct.sum() / (move_mask.sum() + 1e-8)

                # Thinking move accuracy (move sub-vocab space)
                preds_think = torch.argmax(think_logits, dim=-1)
                think_correct = (preds_think == move_target_ids) & thinking_move_mask
                thinking_move_acc = think_correct.sum() / (thinking_move_mask.sum() + 1e-8)

                # Board accuracy (per-token, board sub-vocab space)
                preds_board = torch.argmax(board_logits, dim=-1)
                board_correct = (preds_board == board_target_ids) & board_mask
                board_acc = board_correct.sum() / (board_mask.sum() + 1e-8)

                # Board accuracy (per-block, grouped by block_id)
                B, S = block_id.shape
                max_bid = block_id.max() + 1
                uid = block_id + (torch.arange(B, device=device) * max_bid).unsqueeze(1)
                flat_uid = uid.view(-1)
                flat_bc = board_correct.view(-1).float()
                flat_bm = board_mask.view(-1)
                bp = flat_bm.nonzero(as_tuple=True)[0]

                if bp.numel() > 0:
                    bp_uid = flat_uid[bp]
                    bp_bc = flat_bc[bp]
                    unique_uids, inv = bp_uid.unique(return_inverse=True)
                    n_unique = unique_uids.shape[0]
                    sum_correct = torch.zeros(n_unique, device=device).scatter_add_(0, inv, bp_bc)
                    sum_total = torch.zeros(n_unique, device=device).scatter_add_(0, inv, torch.ones_like(bp_bc))
                    board_total_acc = (sum_correct == sum_total).float().mean().item()

                    # Intra-block offset for sub-metrics
                    bp_pos = bp.float()
                    block_starts = torch.full((n_unique,), float('inf'), device=device)
                    block_starts.scatter_reduce_(0, inv, bp_pos, reduce="amin")
                    intra_idx = (bp_pos - block_starts[inv]).long()

                    # Squares (intra 1-64): per-board all-correct
                    sq = (intra_idx >= 1) & (intra_idx <= 64)
                    if sq.any():
                        sq_correct = torch.zeros(n_unique, device=device).scatter_add_(0, inv[sq], bp_bc[sq])
                        sq_total = torch.zeros(n_unique, device=device).scatter_add_(0, inv[sq], torch.ones_like(bp_bc[sq]))
                        has_sq = sq_total > 0
                        board_square_acc = ((sq_correct == sq_total) & has_sq).float().sum() / (has_sq.sum() + 1e-8)
                        board_square_acc = board_square_acc.item()
                    else:
                        board_square_acc = 0.0

                    # Castling (intra 65: end_pos predicts castling token)
                    castle = intra_idx == 65
                    board_castling_acc = bp_bc[castle].mean().item() if castle.any() else 0.0

                    # STM (intra 66: castling predicts STM token)
                    stm_metric = intra_idx == 66
                    board_stm_acc = bp_bc[stm_metric].mean().item() if stm_metric.any() else 0.0
                else:
                    board_total_acc = 0.0
                    board_square_acc = 0.0
                    board_castling_acc = 0.0
                    board_stm_acc = 0.0

                # end_var / end_think accuracy (derived from board_target_ids)
                end_var_board_idx = board_token_to_idx["end_var"]
                end_think_board_idx = board_token_to_idx["end_think"]
                end_var_target_mask = (board_target_ids == end_var_board_idx) & board_mask
                end_think_target_mask = (board_target_ids == end_think_board_idx) & board_mask
                end_var_acc = (preds_board[end_var_target_mask] == end_var_board_idx).float().mean().item() if end_var_target_mask.any() else 0.0
                end_think_acc = (preds_board[end_think_target_mask] == end_think_board_idx).float().mean().item() if end_think_target_mask.any() else 0.0

                # continue_var / new_variation accuracy (from metrics masks)
                continue_var_board_idx = board_token_to_idx["continue_var"]
                new_variation_board_idx = board_token_to_idx["new_variation"]
                if continue_var_mask.any():
                    continue_var_acc = (preds_board[continue_var_mask] == continue_var_board_idx).float().mean().item()
                else:
                    continue_var_acc = 0.0
                if new_variation_mask.any():
                    new_variation_acc = (preds_board[new_variation_mask] == new_variation_board_idx).float().mean().item()
                else:
                    new_variation_acc = 0.0

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

                # Reordered final move accuracy (first_is_not_best samples only)
                reorder_mask = move_mask & first_is_not_best.unsqueeze(1)
                if reorder_mask.sum() > 0:
                    reorder_correct = (preds_final == move_target_ids) & reorder_mask
                    final_move_acc_reordered = reorder_correct.sum().float() / reorder_mask.sum().float()
                else:
                    final_move_acc_reordered = torch.tensor(0.0, device=device)
                n_reordered = first_is_not_best.sum().item()

                # Track ratio of thinking vs normal samples
                batch_has_thinking = thinking_move_mask.any(dim=1).sum().item()
                batch_has_normal = (move_mask.any(dim=1) & ~thinking_move_mask.any(dim=1)).sum().item()

            if step % config["training"]["log_every_n_steps"] == 0:
                print(f"Step {step}: Loss {total_loss.item():.4f} "
                      f"(FinalMove: {final_move_loss.item():.4f}, ThinkMove: {thinking_move_loss.item():.4f}, "
                      f"Board: {board_loss.item():.4f}, "
                      f"WL: {wl_loss.item():.4f}, D: {d_loss.item():.4f})")

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
                    "train/board_total_acc": board_total_acc,
                    "train/board_square_acc": board_square_acc,
                    "train/board_castling_acc": board_castling_acc,
                    "train/board_stm_acc": board_stm_acc,
                    "train/end_var_acc": end_var_acc,
                    "train/end_think_acc": end_think_acc,
                    "train/continue_var_acc": continue_var_acc,
                    "train/new_variation_acc": new_variation_acc,
                    "train/wl_mae": wl_mae.item(),
                    "train/d_mae": d_mae.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/batch_thinking_samples": batch_has_thinking,
                    "train/batch_normal_samples": batch_has_normal,
                    "train/final_move_acc_reordered": final_move_acc_reordered.item(),
                    "train/n_reordered_samples": n_reordered,
                    "train/pretrain_epoch": pretrain_epoch.max().item(),
                    # Structural token accuracies (dedicated section)
                    "accuracies/train_end_var_acc": end_var_acc,
                    "accuracies/train_continue_var_acc": continue_var_acc,
                    "accuracies/train_end_think_acc": end_think_acc,
                    "accuracies/train_new_variation_acc": new_variation_acc,
                })

            step += 1

            # Save checkpoint every N steps
            if save_every_n_steps and step % save_every_n_steps == 0:
                ckpt = {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config,
                }
                checkpoint_path = os.path.join(run_checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save(ckpt, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # Validation at end of epoch
        print(f"Running validation for epoch {epoch + 1}...")
        val_results = validate(model, val_dataloader, val_batches, device, use_amp, config)
        wandb.log({
            "accuracies/val_end_var_acc": val_results["end_var_acc"],
            "accuracies/val_continue_var_acc": val_results["continue_var_acc"],
            "accuracies/val_end_think_acc": val_results["end_think_acc"],
            "accuracies/val_new_variation_acc": val_results["new_variation_acc"],
            "accuracies/val_board_acc": val_results["board_acc"],
            "accuracies/val_final_move_acc": val_results["final_move_acc"],
            "accuracies/val_thinking_move_acc": val_results["thinking_move_acc"],
            "train/epoch": epoch,
            "train/step": step,
        })

        # Save checkpoint at end of epoch
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

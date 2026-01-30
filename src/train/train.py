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
from src.dataloader.loader import get_dataloader


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def soft_bucket_loss(logits, target_values, bucket_centers, valid_mask):
    """Soft CE loss: distribute target probability across two nearest buckets via linear interpolation."""
    N = target_values.shape[0]
    n_buckets = bucket_centers.shape[0]

    if N == 0 or valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Find lower bucket: last bucket where center <= target
    diffs = target_values.unsqueeze(-1) - bucket_centers  # (N, B)
    lower_idx = (diffs >= 0).long().sum(dim=-1) - 1        # (N,)
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

    loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)  # (N,)
    return (loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)


def train():
    config = load_config("src/train/config.yaml")

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if resuming from checkpoint
    resume_from = config["training"].get("resume_from")
    start_epoch = 0
    step = 0

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

    # Dataloader
    dataloader = get_dataloader(
        config["data"]["parquet_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        max_seq_len=config["data"]["max_seq_len"],
        skip_board_prob=config["data"].get("skip_board_prob", 0.0)
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Loss function for board generation
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_idx["pad"], reduction='none')

    # Mixed precision training
    use_amp = config["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision training: {'enabled' if use_amp else 'disabled'}")

    # Resume from checkpoint if specified
    if resume_from:
        checkpoint_files = [f for f in os.listdir(resume_from) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {resume_from}")

        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        latest_checkpoint = os.path.join(resume_from, checkpoint_files[-1])

        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)

        state_dict = checkpoint["model_state_dict"]

        # Checkpoint compatibility: rename old policy_head -> board_head
        if "policy_head.weight" in state_dict and "board_head.weight" not in state_dict:
            state_dict["board_head.weight"] = state_dict.pop("policy_head.weight")
            state_dict["board_head.bias"] = state_dict.pop("policy_head.bias")
        # Remove old 3-class value head (incompatible shape)
        state_dict.pop("value_head.weight", None)
        state_dict.pop("value_head.bias", None)

        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        step = checkpoint["step"]

        print(f"Resumed from epoch {start_epoch}, step {step}")

        run_checkpoint_dir = resume_from
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_checkpoint_dir = os.path.join(
            config["training"]["checkpoint_dir"],
            f"{config['run_name']}_{timestamp}"
        )
        os.makedirs(run_checkpoint_dir, exist_ok=True)

    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")

    # Initialize wandb
    wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config,
        resume="allow" if resume_from else None,
    )

    model.train()

    # Loss weights
    move_weight = config["loss"]["move_weight"]
    board_weight = config["loss"]["board_weight"]
    wl_weight = config["loss"].get("wl_weight", 1.0)
    d_weight = config["loss"].get("d_weight", 1.0)

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            move_mask = batch["move_mask"].to(device)
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

                # Board mask: valid positions, excluding move/wl/d positions
                valid_mask = target_ids != token_to_idx["pad"]
                board_mask = valid_mask & (~move_mask) & (~wl_positions) & (~d_positions)

                # Exclude positions before the first move (pre-first-move tokens have no causal context)
                first_move_idx = move_mask.int().argmax(dim=1)  # (B,)
                has_moves = move_mask.any(dim=1)
                first_move_idx[~has_moves] = move_mask.size(1)
                indices = torch.arange(move_mask.size(1), device=device).unsqueeze(0)
                pre_first_move_mask = indices < first_move_idx.unsqueeze(1)
                board_mask = board_mask & (~pre_first_move_mask)

                ce_board = ce_loss_fn(board_logits.view(-1, vocab_size), target_ids.view(-1))
                ce_board = ce_board.view(target_ids.shape)
                board_loss = (ce_board * board_mask.float()).sum() / (board_mask.sum() + 1e-8)

                # === Pass 2: Prefix masking for move + value prediction ===
                # Discretize ground truth to nearest bucket centers for fourier injection
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
                    wl_positions=wl_positions, d_positions=d_positions
                )

                # --- 1. Move prediction at stm positions ---
                move_logits = model.policy_head(h_prefix)
                ce_move = ce_loss_fn(move_logits.view(-1, vocab_size), target_ids.view(-1))
                ce_move = ce_move.view(target_ids.shape)
                move_loss = (ce_move * move_mask.float()).sum() / (move_mask.sum() + 1e-8)

                # --- 2. WL prediction at move token positions (stm + 1) ---
                stm_nonzero = move_mask.nonzero(as_tuple=False)  # (N, 2)
                if stm_nonzero.shape[0] > 0:
                    wl_pred_batch = stm_nonzero[:, 0]
                    wl_pred_seq = stm_nonzero[:, 1] + 1  # move token = stm + 1
                    # Clamp to valid range
                    wl_pred_seq = wl_pred_seq.clamp(max=h_prefix.shape[1] - 1)
                    h_at_move = h_prefix[wl_pred_batch, wl_pred_seq]  # (N, E)
                    wl_logits = model.wl_head(h_at_move)  # (N, 100)
                    wl_valid_flat = wdl_valid[move_mask]  # (N,)
                    wl_gt_flat = wl_targets[move_mask]    # (N,)
                    wl_loss = soft_bucket_loss(wl_logits, wl_gt_flat, model.wl_bucket_centers, wl_valid_flat)
                else:
                    wl_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    wl_logits = None

                # --- 3. D prediction at WL placeholder positions (stm + 2) ---
                # Hidden states come from wl_positions, but ground truth D values are stored at d_positions
                if wl_positions.any():
                    h_at_wl = h_prefix[wl_positions]  # (M, E)
                    d_logits = model.d_head(h_at_wl)   # (M, 100)
                    d_valid_flat = wdl_valid[d_positions]
                    d_gt_flat = d_targets[d_positions]
                    d_loss = soft_bucket_loss(d_logits, d_gt_flat, model.d_bucket_centers, d_valid_flat)
                else:
                    d_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    d_logits = None

                # === Total loss ===
                total_loss = (
                    move_weight * move_loss +
                    board_weight * board_loss +
                    wl_weight * wl_loss +
                    d_weight * d_loss
                )

            # Scale loss for gradient accumulation
            loss = total_loss / config["training"].get("gradient_accumulation_steps", 1)
            scaler.scale(loss).backward()

            if (step + 1) % config["training"].get("gradient_accumulation_steps", 1) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Metrics
            with torch.no_grad():
                # Move accuracy
                preds_prefix = torch.argmax(move_logits, dim=-1)
                move_correct = (preds_prefix == target_ids) & move_mask
                move_acc = move_correct.sum() / (move_mask.sum() + 1e-8)

                # Board accuracy
                preds_causal = torch.argmax(board_logits, dim=-1)
                board_correct = (preds_causal == target_ids) & board_mask
                board_acc = board_correct.sum() / (board_mask.sum() + 1e-8)

                # WL MAE (expected value) and MSE (argmax bucket center)
                if wl_logits is not None and stm_nonzero.shape[0] > 0:
                    wl_probs = F.softmax(wl_logits.float(), dim=-1)
                    expected_wl = (wl_probs * model.wl_bucket_centers.unsqueeze(0)).sum(dim=-1)
                    wl_valid_f = wl_valid_flat.float()
                    wl_mae = ((expected_wl - wl_gt_flat).abs() * wl_valid_f).sum() / (wl_valid_f.sum() + 1e-8)
                    argmax_wl = model.wl_bucket_centers[wl_logits.argmax(dim=-1)]
                    wl_mse = (((argmax_wl - wl_gt_flat) ** 2) * wl_valid_f).sum() / (wl_valid_f.sum() + 1e-8)
                else:
                    wl_mae = torch.tensor(0.0, device=device)
                    wl_mse = torch.tensor(0.0, device=device)

                # D MAE (expected value) and MSE (argmax bucket center)
                if d_logits is not None and wl_positions.any():
                    d_probs = F.softmax(d_logits.float(), dim=-1)
                    expected_d = (d_probs * model.d_bucket_centers.unsqueeze(0)).sum(dim=-1)
                    d_valid_f = d_valid_flat.float()
                    d_mae = ((expected_d - d_gt_flat).abs() * d_valid_f).sum() / (d_valid_f.sum() + 1e-8)
                    argmax_d = model.d_bucket_centers[d_logits.argmax(dim=-1)]
                    d_mse = (((argmax_d - d_gt_flat) ** 2) * d_valid_f).sum() / (d_valid_f.sum() + 1e-8)
                else:
                    d_mae = torch.tensor(0.0, device=device)
                    d_mse = torch.tensor(0.0, device=device)

                # nth-move metrics
                max_track_moves = config["training"].get("max_track_nth_moves", 20)
                move_cumsum = move_mask.cumsum(dim=1)
                move_indices = (move_cumsum - 1).long()
                valid_nth_mask = move_mask & (move_indices < max_track_moves)

                if valid_nth_mask.any():
                    flat_move_indices = move_indices[valid_nth_mask]
                    flat_move_correct = move_correct[valid_nth_mask].float()

                    move_correct_by_nth = torch.zeros(max_track_moves, device=device)
                    count_by_nth = torch.zeros(max_track_moves, device=device)

                    move_correct_by_nth.scatter_add_(0, flat_move_indices, flat_move_correct)
                    count_by_nth.scatter_add_(0, flat_move_indices, torch.ones_like(flat_move_correct))

                    move_acc_by_nth = move_correct_by_nth / (count_by_nth + 1e-8)
                else:
                    move_acc_by_nth = torch.zeros(max_track_moves, device=device)
                    count_by_nth = torch.zeros(max_track_moves, device=device)

            if step % config["training"]["log_every_n_steps"] == 0:
                print(f"Step {step}: Loss {total_loss.item():.4f} "
                      f"(Move: {move_loss.item():.4f}, Board: {board_loss.item():.4f}, "
                      f"WL: {wl_loss.item():.4f}, D: {d_loss.item():.4f})")

                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/move_loss": move_loss.item(),
                    "train/board_loss": board_loss.item(),
                    "train/wl_loss": wl_loss.item(),
                    "train/d_loss": d_loss.item(),
                    "train/move_acc": move_acc.item(),
                    "train/board_acc": board_acc.item(),
                    "train/wl_mae": wl_mae.item(),
                    "train/wl_mse": wl_mse.item(),
                    "train/d_mae": d_mae.item(),
                    "train/d_mse": d_mse.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/grad_step": step / config["training"].get("gradient_accumulation_steps", 1)
                }

                for i in range(max_track_moves):
                    if count_by_nth[i] > 0:
                        log_dict[f"train/move_acc_nth/{i}"] = move_acc_by_nth[i].item()

                wandb.log(log_dict)

            step += 1

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
            }
            checkpoint_path = os.path.join(run_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train()

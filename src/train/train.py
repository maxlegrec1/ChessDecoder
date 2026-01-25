import os
import yaml
import torch
import torch.nn as nn
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

def train():
    config = load_config("src/train/config.yaml")
    
    # Initialize wandb
    wandb.init(project=config["project_name"], name=config["run_name"], config=config)
    
    # Create run-specific checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_checkpoint_dir = os.path.join(
        config["training"]["checkpoint_dir"],
        f"{config['run_name']}_{timestamp}"
    )
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["max_seq_len"],
        d_ff=config["model"].get("d_ff")
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
    
    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_idx["pad"], reduction='none')
    mse_loss_fn = nn.MSELoss(reduction='none')

    # Mixed precision training
    use_amp = config["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision training: {'enabled' if use_amp else 'disabled'}")

    step = 0
    model.train()
    
    # Enable anomaly detection
    # torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(config["training"]["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            wdl_targets = batch["wdl_targets"].to(device)
            wdl_mask = batch["wdl_mask"].to(device)
            block_id = batch["block_id"].to(device)

            # Mixed precision forward passes and loss computation
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # Pass 1: Causal masking for board prediction training
                # This ensures the model can generate boards autoregressively.
                policy_logits_causal, _ = model(input_ids, mask_type="causal")

                # Pass 2: Prefix masking for move and value prediction
                # This allows full bidirectional board context for better move selection.
                policy_logits_prefix, value_logits_prefix = model(
                    input_ids, mask_type="prefix", block_id=block_id
                )

                # Compute cross-entropy loss per position for both passes
                ce_loss_causal = ce_loss_fn(policy_logits_causal.view(-1, vocab_size), target_ids.view(-1))
                ce_loss_causal = ce_loss_causal.view(target_ids.shape)

                ce_loss_prefix = ce_loss_fn(policy_logits_prefix.view(-1, vocab_size), target_ids.view(-1))
                ce_loss_prefix = ce_loss_prefix.view(target_ids.shape)

                # Create masks
                mask = target_ids != token_to_idx["pad"]  # (B, T) - exclude padding

                # Find first move index for board mask (exclude pre-first-move tokens)
                first_move_idx = wdl_mask.int().argmax(dim=1)  # (B,)
                has_moves = wdl_mask.any(dim=1)  # (B,)
                first_move_idx[~has_moves] = wdl_mask.size(1)
                indices = torch.arange(wdl_mask.size(1), device=device).unsqueeze(0)
                pre_first_move_mask = indices < first_move_idx.unsqueeze(1)

                # Move Loss: only at move positions, using the Prefix (bidirectional) logits
                move_mask = wdl_mask & mask
                move_loss = (ce_loss_prefix * move_mask.float()).sum() / (move_mask.sum() + 1e-8)

                # Board Loss: at board positions, using the Causal logits (to prevent cheating)
                board_mask = (~wdl_mask) & mask & (~pre_first_move_mask)
                board_loss = (ce_loss_causal * board_mask.float()).sum() / (board_mask.sum() + 1e-8)

                # WDL Loss: at move positions, using the Prefix (bidirectional) logits
                wdl_loss_raw = mse_loss_fn(value_logits_prefix, wdl_targets)
                wdl_mask_expanded = wdl_mask.unsqueeze(-1).expand_as(wdl_loss_raw)
                wdl_loss = (wdl_loss_raw * wdl_mask_expanded).sum() / (wdl_mask_expanded.sum() + 1e-8)

                # Total Loss
                total_loss = (
                    config["loss"]["move_weight"] * move_loss +
                    config["loss"]["board_weight"] * board_loss +
                    config["loss"]["wdl_weight"] * wdl_loss
                )
            
            # if torch.isnan(total_loss):
            #     print(f"NaN loss detected at step {step}!")
            #     print(f"Policy Loss: {policy_loss.item()}")
            #     print(f"WDL Loss: {wdl_loss.item()}")
                
            #     if torch.isnan(wdl_loss):
            #         print("Investigating WDL NaN...")
            #         print(f"WDL Targets has NaNs: {torch.isnan(wdl_targets).any().item()}")
            #         print(f"Value Logits has NaNs: {torch.isnan(value_logits).any().item()}")
            #         print(f"WDL Mask sum: {wdl_mask.sum().item()}")
                    
            #         if torch.isnan(wdl_targets).any():
            #             print("Found NaNs in WDL targets!")
            #             # Print the specific row/indices
            #             nan_indices = torch.nonzero(torch.isnan(wdl_targets))
            #             print(f"NaN indices in targets: {nan_indices}")
                
            #     break
            
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
                # Accuracy
                preds_causal = torch.argmax(policy_logits_causal, dim=-1)
                preds_prefix = torch.argmax(policy_logits_prefix, dim=-1)
                
                # Move Accuracy (prefix)
                move_correct = (preds_prefix == target_ids) & wdl_mask
                move_acc = move_correct.sum() / (wdl_mask.sum() + 1e-8)
                
                # Board Accuracy (causal)
                board_correct = (preds_causal == target_ids) & board_mask
                board_acc = board_correct.sum() / (board_mask.sum() + 1e-8)
                
                # Overall Policy Acc (mixed)
                policy_acc = (move_correct.sum() + board_correct.sum()) / (move_mask.sum() + board_mask.sum() + 1e-8)
                
                # Value Accuracy: argmax of predicted value matches argmax of true value (at move positions)
                value_preds_argmax = torch.argmax(value_logits_prefix, dim=-1)  # [B, T]
                value_targets_argmax = torch.argmax(wdl_targets, dim=-1)  # [B, T]
                value_correct = (value_preds_argmax == value_targets_argmax) & wdl_mask
                value_acc = value_correct.sum() / (wdl_mask.sum() + 1e-8)
                
                # Move/Value accuracy by nth move in sequence
                # Compute move indices: 0 for 1st move, 1 for 2nd, etc.
                max_track_moves = config["training"].get("max_track_nth_moves", 20)
                move_cumsum = wdl_mask.cumsum(dim=1)  # [B, T], at move positions: 1, 2, 3, ...
                move_indices = (move_cumsum - 1).long()  # [B, T], at move positions: 0, 1, 2, ...
                
                # Only track moves with index < max_track_moves
                valid_nth_mask = wdl_mask & (move_indices < max_track_moves)
                
                if valid_nth_mask.any():
                    flat_move_indices = move_indices[valid_nth_mask]
                    flat_move_correct = move_correct[valid_nth_mask].float()
                    flat_value_correct = value_correct[valid_nth_mask].float()
                    
                    # Scatter-add for efficient aggregation
                    move_correct_by_nth = torch.zeros(max_track_moves, device=device)
                    value_correct_by_nth = torch.zeros(max_track_moves, device=device)
                    count_by_nth = torch.zeros(max_track_moves, device=device)
                    
                    move_correct_by_nth.scatter_add_(0, flat_move_indices, flat_move_correct)
                    value_correct_by_nth.scatter_add_(0, flat_move_indices, flat_value_correct)
                    count_by_nth.scatter_add_(0, flat_move_indices, torch.ones_like(flat_move_correct))
                    
                    # Compute accuracies per nth move
                    move_acc_by_nth = move_correct_by_nth / (count_by_nth + 1e-8)
                    value_acc_by_nth = value_correct_by_nth / (count_by_nth + 1e-8)
                else:
                    move_acc_by_nth = torch.zeros(max_track_moves, device=device)
                    value_acc_by_nth = torch.zeros(max_track_moves, device=device)
                    count_by_nth = torch.zeros(max_track_moves, device=device)
            
            if step % config["training"]["log_every_n_steps"] == 0:
                print(f"Step {step}: Loss {total_loss.item():.4f} (Move: {move_loss.item():.4f}, Board: {board_loss.item():.4f}, WDL: {wdl_loss.item():.4f})")
                
                # Base metrics
                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/move_loss": move_loss.item(),
                    "train/board_loss": board_loss.item(),
                    "train/wdl_loss": wdl_loss.item(),
                    "train/policy_acc": policy_acc.item(),
                    "train/move_acc": move_acc.item(),
                    "train/board_acc": board_acc.item(),
                    "train/value_acc": value_acc.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/grad_step": step / config["training"].get("gradient_accumulation_steps", 1)
                }
                
                # Add nth-move metrics (only for positions with sufficient samples)
                for i in range(max_track_moves):
                    if count_by_nth[i] > 0:
                        log_dict[f"train/move_acc_nth/{i}"] = move_acc_by_nth[i].item()
                        log_dict[f"train/value_acc_nth/{i}"] = value_acc_by_nth[i].item()
                
                wandb.log(log_dict)
                
            step += 1
            
        # Save checkpoint with optimizer state for resuming training
        if (epoch + 1) % 1 == 0:  # Save every epoch for now
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

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from src.models.encoder import ChessEncoder
from src.models.vocab import vocab_size, policy_index
from src.dataloader.encoder_loader import get_encoder_dataloader


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train():
    config = load_config("src/train/config_encoder.yaml")
    
    # Initialize wandb
    wandb.init(
        project=config["project_name"], 
        name=config["run_name"], 
        config=config
    )
    
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    # Model
    model = ChessEncoder(
        vocab_size=vocab_size,
        num_policy_tokens=len(policy_index),
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["max_seq_len"]
    ).to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Dataloader
    dataloader = get_encoder_dataloader(
        config["data"]["parquet_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        max_seq_len=config["data"]["max_seq_len"]
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Learning rate scheduler (optional)
    scheduler = None
    if config["training"].get("use_scheduler", False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"]
        )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    step = 0
    model.train()
    
    # Checkpoints directory
    checkpoint_dir = config["training"].get("checkpoint_dir", "src/train/checkpoints/encoder")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            
            # Forward pass
            policy_logits = model(input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = loss_fn(policy_logits, targets)
            
            # Scale for gradient accumulation
            grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Metrics
            with torch.no_grad():
                preds = torch.argmax(policy_logits, dim=-1)
                correct = (preds == targets).sum().item()
                batch_size = targets.size(0)
                
                epoch_loss += loss.item() * batch_size
                epoch_correct += correct
                epoch_total += batch_size
            
            # Logging
            if step % config["training"]["log_every_n_steps"] == 0:
                acc = correct / batch_size
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/move_acc": acc,
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/lr": optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint every N steps
            save_every_n_steps = config["training"].get("save_every_n_steps")
            if save_every_n_steps and (step + 1) % save_every_n_steps == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"encoder_step_{step + 1}.pt"
                )
                torch.save({
                    'step': step + 1,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
            
            step += 1
        
        # End of epoch stats
        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
        avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch/accuracy": avg_acc,
            "epoch/epoch": epoch + 1
        })
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint
        save_every = config["training"].get("save_every_n_epochs", 1)
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"encoder_epoch_{epoch + 1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_acc,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "encoder_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    wandb.finish()


if __name__ == "__main__":
    train()


import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from tqdm import tqdm

from chessdecoder.models.encoder import ChessEncoder
from chessdecoder.models.vocab import vocab_size, policy_index
from chessdecoder.dataloader.encoder_loader import get_encoder_dataloader
from chessdecoder.eval.elo_eval import model_vs_stockfish
from chessdecoder.eval.engine import PytorchModelAdapter
from chessdecoder.utils.training import (
    load_config,
    save_training_checkpoint,
    init_wandb_with_resume,
)


def run_stockfish_eval(model, eval_cfg: dict, run_dir: str, grad_step: int) -> dict:
    """Play ``num_games`` vs Stockfish and return {winrate, estimated_elo}."""
    was_training = model.training
    model.eval()
    try:
        adapter = PytorchModelAdapter(
            lambda fen, temp: model.predict_move(fen, temperature=temp, force_legal=True)
        )
        pgn_dir = os.path.join(run_dir, "eval_pgns", f"step_{grad_step}")
        winrate, estimated_elo = model_vs_stockfish(
            model=adapter,
            model1_name=f"encoder_step_{grad_step}",
            num_games=eval_cfg["num_games"],
            temperature=eval_cfg.get("temperature", 0.0),
            elo=eval_cfg["stockfish_elo"],
            pgn_dir=pgn_dir,
        )
    finally:
        if was_training:
            model.train()
    return {"winrate": winrate, "estimated_elo": estimated_elo}


def build_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Linear warmup followed by cosine decay to 0, stepped per gradient update."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    config = load_config("chessdecoder/train/config_encoder.yaml")

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model = ChessEncoder(
        vocab_size=vocab_size,
        num_policy_tokens=len(policy_index),
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["max_seq_len"],
        d_ff=config["model"].get("d_ff"),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    dataloader = get_encoder_dataloader(
        config["data"]["parquet_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        max_seq_len=config["data"]["max_seq_len"],
        match_decoder_sampling=config["data"].get("match_decoder_sampling", False),
        decoder_max_seq_len=config["data"].get("decoder_max_seq_len", 4096),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    warmup_steps = config["training"].get("warmup_steps", 0)
    max_steps = config["training"]["max_steps"]
    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    grad_clip = config["training"].get("grad_clip", 1.0)

    use_amp = config["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision training: {'enabled' if use_amp else 'disabled'}")

    loss_fn = nn.CrossEntropyLoss()

    # Resume / new run directory
    resume_from = config["training"].get("resume_from")
    step = 0
    grad_step = 0

    if resume_from:
        checkpoint_files = [
            f for f in os.listdir(resume_from)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {resume_from}")
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        latest = os.path.join(resume_from, checkpoint_files[-1])
        print(f"Resuming from: {latest}")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for pg in optimizer.param_groups:
            pg["lr"] = config["training"]["learning_rate"]
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt["step"]
        grad_step = ckpt.get("grad_step", step // grad_accum)
        run_dir = resume_from
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(
            config["training"]["checkpoint_dir"],
            f"{config['run_name']}_{timestamp}",
        )
        os.makedirs(run_dir, exist_ok=True)

    # Build scheduler AFTER optimizer state is loaded so last_epoch reflects resume.
    scheduler = build_lr_scheduler(optimizer, warmup_steps, max_steps)
    if grad_step > 0:
        scheduler.last_epoch = grad_step - 1
        scheduler._last_lr = [lr for lr in scheduler.get_lr()]
        for pg, lr in zip(optimizer.param_groups, scheduler._last_lr):
            pg["lr"] = lr

    print(f"Checkpoints → {run_dir}")

    init_wandb_with_resume(
        project=config["project_name"],
        run_name=config["run_name"],
        config=config,
        checkpoint_dir=run_dir,
    )

    model.train()
    pbar = tqdm(total=max_steps, initial=grad_step, desc="encoder")

    while grad_step < max_steps:
        for batch in dataloader:
            if grad_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                policy_logits = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(policy_logits, targets)

            scaler.scale(loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                grad_step += 1
                pbar.update(1)

            with torch.no_grad():
                logits_f = policy_logits.float()
                top3 = logits_f.topk(3, dim=-1).indices
                top1_correct = (top3[:, 0] == targets).float().mean().item()
                top3_correct = (top3 == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

            if step % config["training"]["log_every_n_steps"] == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    top1=f"{top1_correct:.3f}",
                    top3=f"{top3_correct:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
                wandb.log({
                    "train/loss": loss.item(),
                    "train/top1_acc": top1_correct,
                    "train/top3_acc": top3_correct,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/step": step,
                    "train/grad_step": grad_step,
                })

            step += 1

            just_stepped = (step % grad_accum == 0)

            save_every = config["training"].get("save_every_n_steps")
            if save_every and grad_step > 0 and grad_step % save_every == 0 and just_stepped:
                save_training_checkpoint(
                    os.path.join(run_dir, f"checkpoint_{grad_step}.pt"),
                    model=model, optimizer=optimizer, scaler=scaler, step=step,
                    extra_state={"grad_step": grad_step, "config": config},
                )

            eval_cfg = config.get("eval", {})
            eval_every = eval_cfg.get("every_n_steps")
            if eval_every and grad_step > 0 and grad_step % eval_every == 0 and just_stepped:
                pbar.write(f"[eval] step {grad_step}: {eval_cfg['num_games']} games vs Stockfish ELO {eval_cfg['stockfish_elo']}...")
                metrics = run_stockfish_eval(model, eval_cfg, run_dir, grad_step)
                wandb.log({
                    "eval/stockfish_winrate": metrics["winrate"],
                    "eval/stockfish_estimated_elo": metrics["estimated_elo"],
                    "eval/grad_step": grad_step,
                })
                pbar.write(f"[eval] winrate={metrics['winrate']:.3f} elo={metrics['estimated_elo']}")

    pbar.close()

    save_training_checkpoint(
        os.path.join(run_dir, f"checkpoint_final_{grad_step}.pt"),
        model=model, optimizer=optimizer, scaler=scaler, step=step,
        extra_state={"grad_step": grad_step, "config": config},
    )
    wandb.finish()


if __name__ == "__main__":
    train()

"""V2 pretraining loop (Phase C).

Same regime as V1 (one model, one optimizer, ordered game data, joint loss).
What changed is internal: V1's two passes (causal board CE + prefix
move/value) become one encoder pass (all boards of a game, parallel) → one
causal-decoder pass over the mixed [latents|move|wl|d] stream → heads, plus a
parallel absolute transition head replacing the autoregressive board_head.

Run:  CUDA_VISIBLE_DEVICES=0 uv run python chessdecoder/train/train_v2.py \
          [chessdecoder/train/config_v2.yaml]
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from tqdm import tqdm

from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.models.v2.model_v2 import (
    ChessDecoderV2, N_SQUARE_CLASSES, N_STM_CLASSES, N_CASTLING_CLASSES)
from chessdecoder.dataloader.loader_v2 import (
    get_v2_dataloader, assemble_decoder_inputs)
from chessdecoder.utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process, get_device,
    average_gradients, barrier, print_rank0)
from chessdecoder.utils.training import (
    load_config, soft_bucket_loss, save_training_checkpoint,
    init_wandb_with_resume)

IGNORE_INDEX = -100


def train():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "chessdecoder/train/config_v2.yaml"
    config = load_config(config_path)

    rank, local_rank, world_size = setup_distributed()
    device = get_device(local_rank)
    print_rank0(f"Using device: {device}, world_size: {world_size}")

    seed = config["training"].get("seed", 42)
    start_epoch, step, resume_epoch_step = 0, 0, 0

    mc = config["model"]
    model = ChessDecoderV2(
        vocab_size=vocab_size,
        embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_encoder_layers=mc["num_encoder_layers"],
        num_decoder_layers=mc["num_decoder_layers"],
        num_latents=mc["num_latents"],
        board_max_seq_len=mc.get("board_max_seq_len", 68),
        decoder_max_seq_len=mc["max_plies"] * (mc["num_latents"] + 3)
            if "max_plies" in mc else config["data"]["max_plies"] * (mc["num_latents"] + 3),
        d_ff=mc.get("d_ff"), n_buckets=mc.get("n_buckets", 100),
        value_hidden_size=mc.get("value_hidden_size", 256),
        num_fourier_freq=mc.get("num_fourier_freq", 128),
        wl_sigma=mc.get("wl_sigma", 0.4),
    ).to(device)

    dataloader, dataset = get_v2_dataloader(
        config["data"]["parquet_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        max_plies=config["data"]["max_plies"],
        seed=seed, rank=rank, world_size=world_size)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"])

    use_amp = config["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    resume_from = config["training"].get("resume_from")
    if resume_from:
        ckpts = sorted([f for f in os.listdir(resume_from)
                        if f.startswith("checkpoint_") and f.endswith(".pt")],
                       key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        latest = os.path.join(resume_from, ckpts[-1])
        print_rank0(f"Resuming from checkpoint: {latest}")
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        for pg in optimizer.param_groups:
            pg["lr"] = config["training"]["learning_rate"]
        scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch, step = ck["epoch"], ck["step"]
        resume_epoch_step = ck.get("epoch_step", 0)
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
    w_tr = config["loss"]["transition_weight"]
    w_wl = config["loss"]["wl_weight"]
    w_d = config["loss"]["d_weight"]
    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    grad_clip = config["training"].get("grad_clip", 10.0)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

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

            bid = batch["board_ids"].to(device)            # [B,P,68]
            B, P, _ = bid.shape
            move_full = batch["move_full"].to(device)       # [B,P]
            pol_tgt = batch["policy_tgt"].to(device)
            pol_val = batch["policy_valid"].to(device)
            wl = batch["wl"].to(device); d = batch["d"].to(device)
            wdl_val = batch["wdl_valid"].to(device)
            tsq = batch["trans_sq"].to(device)              # [B,P,64]
            tstm = batch["trans_stm"].to(device)
            tcas = batch["trans_cas"].to(device)
            tr_val = batch["trans_valid"].to(device)
            ply_mask = batch["ply_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                latents = model.encode_boards(
                    bid.reshape(B * P, 68)).reshape(B, P, -1, model.embed_dim)
                move_emb = model.tok_embedding(move_full)    # [B,P,E]
                seq, pos = assemble_decoder_inputs(
                    latents, move_emb, wl, d, model.fourier_encoder)
                h = model.decoder(seq)                       # [B, P*L, E]

                h_pol = h[:, pos["policy_pos"], :]           # [B,P,E]
                h_mv = h[:, pos["move_pos"], :]
                h_wl = h[:, pos["wl_pos"], :]

                # --- policy ---
                pol_logits = model.policy_head(h_pol)        # [B,P,move_vocab]
                pmask = pol_val & ply_mask
                policy_loss = ce(
                    pol_logits.reshape(-1, move_vocab_size),
                    torch.where(pmask, pol_tgt, torch.full_like(pol_tgt, IGNORE_INDEX)).reshape(-1))

                # --- value (WL at move pos, D at wl pos) — V1 mechanism ---
                vmask = (wdl_val & ply_mask).reshape(-1)
                wl_logits = model.wl_head(h_mv.reshape(-1, model.embed_dim))
                d_logits = model.d_head(h_wl.reshape(-1, model.embed_dim))
                wl_loss = soft_bucket_loss(wl_logits, wl.reshape(-1),
                                           model.wl_bucket_centers, vmask)
                d_loss = soft_bucket_loss(d_logits, d.reshape(-1),
                                          model.d_bucket_centers, vmask)

                # --- transition (parallel absolute world model) ---
                out = model.transition_head(
                    latents.reshape(B * P, -1, model.embed_dim),
                    move_emb.reshape(B * P, model.embed_dim))
                trm = tr_val.reshape(-1)
                sq_t = torch.where(trm.unsqueeze(1), tsq.reshape(B * P, 64),
                                   torch.full_like(tsq.reshape(B * P, 64), IGNORE_INDEX))
                stm_t = torch.where(trm, tstm.reshape(-1),
                                    torch.full_like(tstm.reshape(-1), IGNORE_INDEX))
                cas_t = torch.where(trm, tcas.reshape(-1),
                                    torch.full_like(tcas.reshape(-1), IGNORE_INDEX))
                trans_loss = (
                    ce(out["square"].reshape(-1, N_SQUARE_CLASSES), sq_t.reshape(-1))
                    + ce(out["stm"], stm_t)
                    + ce(out["castling"], cas_t))

                total = (w_pol * policy_loss + w_tr * trans_loss
                         + w_wl * wl_loss + w_d * d_loss)

            (scaler.scale(total / grad_accum)).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                average_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % config["training"]["log_every_n_steps"] == 0 and is_main_process():
                with torch.no_grad():
                    pa = ((pol_logits.argmax(-1) == pol_tgt) & pmask).sum() / (pmask.sum() + 1e-8)
                    sq_acc = ((out["square"].argmax(-1) == tsq.reshape(B * P, 64))
                              & trm.unsqueeze(1)).sum() / (trm.sum() * 64 + 1e-8)
                print(f"Step {step}: loss {total.item():.4f} "
                      f"(pol {policy_loss.item():.4f} trans {trans_loss.item():.4f} "
                      f"wl {wl_loss.item():.4f} d {d_loss.item():.4f})")
                wandb.log({
                    "train/total_loss": total.item(),
                    "train/policy_loss": policy_loss.item(),
                    "train/transition_loss": trans_loss.item(),
                    "train/wl_loss": float(wl_loss),
                    "train/d_loss": float(d_loss),
                    "train/policy_acc": pa.item(),
                    "train/transition_square_acc": sq_acc.item(),
                    "train/epoch": epoch, "train/step": step,
                    "train/grad_step": step / grad_accum,
                })

            del h, seq, latents, pol_logits, out, total
            step += 1

            save_every = config["training"].get("save_every_n_steps")
            if save_every and step % save_every == 0 and is_main_process():
                save_training_checkpoint(
                    os.path.join(run_dir, f"checkpoint_{step}.pt"),
                    model=model, optimizer=optimizer, scaler=scaler, step=step,
                    extra_state={"epoch": epoch, "epoch_step": epoch_step,
                                 "config": config})
                barrier()

        if is_main_process():
            save_training_checkpoint(
                os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                model=model, optimizer=optimizer, scaler=scaler, step=step,
                extra_state={"epoch": epoch + 1, "epoch_step": 0, "config": config})
        barrier()

    cleanup_distributed()


if __name__ == "__main__":
    train()

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
    load_config, save_training_checkpoint, init_wandb_with_resume)
from chessdecoder.utils.muon import build_optimizer

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

    optimizer = build_optimizer(
        model,
        config["training"].get("optimizer", "adamw"),
        config["training"]["learning_rate"],
        config["training"]["weight_decay"])
    print_rank0(f"Optimizer: {config['training'].get('optimizer', 'adamw')} "
                f"lr={config['training']['learning_rate']} "
                f"wd={config['training']['weight_decay']}")

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
    w_wdl = config["loss"]["wdl_weight"]
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
            wdl_tgt = batch["wdl_tgt"].to(device)           # [B,P,N_CELLS] soft cat
            wdl_mean = batch["wdl_mean"].to(device)         # [B,P,3] exact WDL
            wdl_val = batch["wdl_valid"].to(device)
            tsq = batch["trans_sq"].to(device)              # [B,P,64]
            tstm = batch["trans_stm"].to(device)
            tcas = batch["trans_cas"].to(device)
            tr_val = batch["trans_valid"].to(device)
            ply_mask = batch["ply_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                latents = model.encode_boards(
                    bid.reshape(B * P, 68)).reshape(B, P, -1, model.embed_dim)
                E = model.embed_dim

                # --- WDL: encoder-side 2-D-simplex categorical evaluator
                # (leak-free, multimodal-capable) ---
                ncell = wdl_tgt.shape[-1]
                wdl_logits = model.wdl_head(
                    latents.reshape(B * P, -1, E)).reshape(B, P, ncell)
                vmask = (wdl_val & ply_mask)                  # [B,P]
                logp = torch.log_softmax(wdl_logits.float(), dim=-1)
                wdl_ce = -(wdl_tgt * logp).sum(-1)            # [B,P] soft-CE
                wdl_loss = (wdl_ce * vmask).sum() / (vmask.sum() + 1e-8)

                # value token = Fourier(Q)+Fourier(D) from the exact target
                # WDL mean, teacher-forced into the stream (markdowns/12).
                value_emb = model.embed_wdl(
                    wdl_mean.reshape(-1, 3)).reshape(B, P, E)
                move_emb = model.tok_embedding(move_full)     # [B,P,E]
                seq, pos = assemble_decoder_inputs(latents, move_emb, value_emb)
                h = model.decoder(seq)                        # [B, P*L, E]

                # --- policy (read at the value slot: eval-aware move) ---
                pol_logits = model.policy_head(h[:, pos["policy_pos"], :])
                pmask = pol_val & ply_mask
                policy_loss = ce(
                    pol_logits.reshape(-1, move_vocab_size),
                    torch.where(pmask, pol_tgt, torch.full_like(pol_tgt, IGNORE_INDEX)).reshape(-1))

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
                         + w_wdl * wdl_loss)

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
                    # --- move / policy accuracy ---
                    move_correct = (pol_logits.argmax(-1) == pol_tgt) & pmask
                    pa = move_correct.sum() / (pmask.sum() + 1e-8)

                    # --- board (transition) reconstruction metrics ---
                    # In V2 the next-board prediction IS the transition head;
                    # these mirror V1's board_total/square/stm/castling acc.
                    bm = trm                                          # [B*P] valid
                    sq_ok = (out["square"].argmax(-1)
                             == tsq.reshape(B * P, 64))               # [B*P,64]
                    stm_ok = out["stm"].argmax(-1) == tstm.reshape(-1)
                    cas_ok = out["castling"].argmax(-1) == tcas.reshape(-1)
                    nb = bm.sum().clamp(min=1)
                    board_square_acc = (sq_ok & bm.unsqueeze(1)).sum() / (nb * 64)
                    board_stm_acc = (stm_ok & bm).sum() / nb
                    board_castling_acc = (cas_ok & bm).sum() / nb
                    board_total_acc = (
                        (sq_ok.all(dim=1) & stm_ok & cas_ok) & bm).sum() / nb
                    # V1 train/board_acc == per-cell board-token accuracy;
                    # V2 analogue = mean over the 66 transition cells / board.
                    board_acc = (((sq_ok & bm.unsqueeze(1)).sum()
                                  + (stm_ok & bm).sum()
                                  + (cas_ok & bm).sum()) / (nb * 66))

                    # --- WDL metrics: decode the simplex categorical to its
                    # mean and compare to the exact target WDL ---
                    vf = vmask.float()
                    vn = vf.sum() + 1e-8
                    wdl_p = model.wdl_head.mean_wdl(wdl_logits)          # [B,P,3]
                    q_pred = wdl_p[..., 0] - wdl_p[..., 2]               # Q = W-L
                    q_tgt = wdl_mean[..., 0] - wdl_mean[..., 2]
                    q_mae = ((q_pred - q_tgt).abs() * vf).sum() / vn
                    d_mae = ((wdl_p[..., 1] - wdl_mean[..., 1]).abs() * vf).sum() / vn
                    wdl_acc = ((wdl_p.argmax(-1) == wdl_mean.argmax(-1))
                               & vmask).sum() / vn
                    # categorical entropy = an uncertainty proxy
                    pcat = torch.softmax(wdl_logits.float(), -1)
                    wdl_entropy = ((-(pcat * (pcat + 1e-9).log()).sum(-1))
                                   * vf).sum() / vn

                    # --- max |logit| per head (logit-explosion watch) ---
                    policy_logit_max = pol_logits.detach().abs().max()
                    board_logit_max = torch.stack([
                        out["square"].detach().abs().max(),
                        out["stm"].detach().abs().max(),
                        out["castling"].detach().abs().max()]).max()
                    value_logit_max = wdl_logits.detach().abs().max()

                    # --- per-ply (nth-move) policy accuracy ---
                    max_track = config["training"].get("max_track_nth_moves", 20)
                    nth = {}
                    for i in range(min(max_track, P)):
                        cnt = pmask[:, i].sum()
                        if cnt > 0:
                            nth[f"train/move_acc_nth/{i}"] = (
                                move_correct[:, i].sum() / cnt).item()

                print(f"Step {step}: loss {total.item():.4f} "
                      f"(pol {policy_loss.item():.4f} trans {trans_loss.item():.4f} "
                      f"wdl {wdl_loss.item():.4f}) "
                      f"move_acc={pa.item():.3f} wdl_acc={wdl_acc.item():.3f} "
                      f"q_mae={q_mae.item():.3f} "
                      f"board[sq={board_square_acc.item():.3f} "
                      f"stm={board_stm_acc.item():.3f} cas={board_castling_acc.item():.3f} "
                      f"total={board_total_acc.item():.4f}]")
                wandb.log({
                    "train/total_loss": total.item(),
                    "train/move_loss": policy_loss.item(),
                    "train/board_loss": trans_loss.item(),
                    "train/wdl_loss": wdl_loss.item(),
                    "train/move_acc": pa.item(),
                    "train/board_acc": board_acc.item(),
                    "train/board_total_acc": board_total_acc.item(),
                    "train/board_square_acc": board_square_acc.item(),
                    "train/board_stm_acc": board_stm_acc.item(),
                    "train/board_castling_acc": board_castling_acc.item(),
                    "train/wdl_acc": wdl_acc.item(),
                    "train/q_mae": q_mae.item(),
                    "train/d_mae": d_mae.item(),
                    "train/wdl_entropy": wdl_entropy.item(),
                    "train/policy_logit_max": policy_logit_max.item(),
                    "train/board_logit_max": board_logit_max.item(),
                    "train/value_logit_max": value_logit_max.item(),
                    "train/epoch": epoch, "train/step": step,
                    "train/grad_step": step / grad_accum,
                    **nth,
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

"""V2 thinking-variation finetuning (Phase D).

Same joint objective as V2 pretraining + thinking-trace supervision, over the
spliced (board->k latents) stream. V2 needs no head surgery: the model
already has a first-class `thinking_policy_head` (V1 had to clone policy_head
and grow the vocab for `end_var` — gone here, the vocab is fixed).

`compute_finetune_v2_loss` is the unit-tested core; `train()` is the loop
around it (load a V2 pretrain checkpoint, iterate variation parquets).
"""
import os
import sys

import torch
import torch.nn as nn
from datetime import datetime

from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.models.v2.model_v2 import (
    ChessDecoderV2, board_tokens_to_transition_targets,
    N_SQUARE_CLASSES, N_STM_CLASSES, N_CASTLING_CLASSES)
from chessdecoder.dataloader.sequence_v2 import build_mixed_sequence
from chessdecoder.finetune.loader_v2 import (
    variation_to_v2_sample, flat_to_decoder_index)
from chessdecoder.utils.training import load_config, soft_bucket_loss

IGNORE = -100


def compute_finetune_v2_loss(model, plan, sup, ss_p: float = 0.0, device="cpu"):
    """One variation sample -> dict of scalar losses. ``ss_p`` is the
    scheduled-sampling probability: each transition's conditioning board
    latents are, w.p. ss_p, replaced by the model's own predicted-board
    latents (exposure-bias fix; ramp 0 -> p_max over finetuning)."""
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    seq, pos = build_mixed_sequence(model, plan, device=device)
    h = model.decoder(seq)[0]                                 # [S,E]
    f2d = flat_to_decoder_index(plan, pos)

    # thinking-policy: predict each variation/PV move from its predict_from pos
    tl = torch.tensor(0.0, device=device)
    if sup["thinking"]:
        idx = torch.tensor([f2d[p] for p, _ in sup["thinking"]], device=device)
        tgt = torch.tensor([m for _, m in sup["thinking"]], device=device)
        tl = ce(model.thinking_policy_head(h[idx]), tgt)

    # final move via policy_head at end_think
    fl = torch.tensor(0.0, device=device)
    if sup["final"] is not None:
        fp, fm = sup["final"]
        fl = ce(model.policy_head(h[f2d[fp]]).unsqueeze(0),
                torch.tensor([fm], device=device))

    # value: wl_head at the move pos (wl_flat-2), d_head at the wl pos
    wl_terms, d_terms = [], []
    for p, (val, valid, kind) in sup["value"].items():
        if not valid:
            continue
        if kind == "wl" and (p - 2) in f2d:
            wl_terms.append((f2d[p - 2], val))
        if kind == "d" and p in f2d:
            d_terms.append((f2d[p], val))
    wll = _bucket(model, model.wl_head, model.wl_bucket_centers, wl_terms, h, device)
    dl = _bucket(model, model.d_head, model.d_bucket_centers, d_terms, h, device)

    # transition: every (board, move, next_board) triple, scheduled-sampled
    trl = torch.tensor(0.0, device=device)
    trips = sup["transition_triples"]
    if trips:
        zc, me, tsq, tstm, tcas = [], [], [], [], []
        for si, sm, sn in trips:
            z_gt = pos["board_latents"][si]                   # [1,k,E]
            if ss_p > 0.0:
                mv_emb = model.tok_embedding(
                    torch.tensor([plan[sm].token_id], device=device))
                z_pred = model.encode_boards(
                    model.decode_transition(
                        model.transition_head(z_gt, mv_emb)))
                z_use = model.scheduled_sample_latents(z_gt, z_pred, ss_p)
            else:
                z_use = z_gt
            zc.append(z_use)
            me.append(model.tok_embedding(
                torch.tensor([plan[sm].token_id], device=device)))
            nb = torch.tensor([plan[sn].board_ids], device=device)
            a, b, c = board_tokens_to_transition_targets(nb)
            tsq.append(a); tstm.append(b); tcas.append(c)
        out = model.transition_head(torch.cat(zc, 0),
                                    torch.cat(me, 0))
        trl = (ce(out["square"].reshape(-1, N_SQUARE_CLASSES),
                  torch.cat(tsq, 0).reshape(-1))
               + ce(out["stm"], torch.cat(tstm, 0))
               + ce(out["castling"], torch.cat(tcas, 0)))

    return {"thinking": tl, "final": fl, "wl": wll, "d": dl, "transition": trl}


def _bucket(model, head, centers, terms, h, device):
    if not terms:
        return torch.tensor(0.0, device=device)
    idx = torch.tensor([i for i, _ in terms], device=device)
    val = torch.tensor([v for _, v in terms], device=device, dtype=torch.float32)
    valid = torch.ones(len(terms), dtype=torch.bool, device=device)
    return soft_bucket_loss(head(h[idx]), val, centers, valid)


def train():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "chessdecoder/finetune/config_v2.yaml"
    config = load_config(cfg_path)
    import glob, pandas as pd, wandb
    from chessdecoder.utils.training import save_training_checkpoint, init_wandb_with_resume

    device = "cuda"
    mc = config["model"]
    model = ChessDecoderV2(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_encoder_layers=mc["num_encoder_layers"],
        num_decoder_layers=mc["num_decoder_layers"], num_latents=mc["num_latents"],
        decoder_max_seq_len=mc.get("decoder_max_seq_len", 8192),
        d_ff=mc.get("d_ff")).to(device)

    pre = config["training"].get("pretrain_checkpoint")
    if pre:
        ck = torch.load(pre, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"loaded pretrain {pre}")

    opt = torch.optim.AdamW(model.parameters(),
                            lr=config["training"]["learning_rate"],
                            weight_decay=config["training"]["weight_decay"])
    L = config["loss"]
    files = sorted(glob.glob(os.path.join(config["data"]["variation_parquet_dir"], "*.parquet")))
    run = f"{config['run_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(config["training"]["checkpoint_dir"], run)
    os.makedirs(run_dir, exist_ok=True)
    init_wandb_with_resume(project=config["project_name"], run_name=config["run_name"],
                           config=config, checkpoint_dir=run_dir)
    model.train()
    step = 0
    ss_max = config["training"].get("scheduled_sampling_max", 0.25)
    ramp = config["training"].get("scheduled_sampling_ramp_steps", 20000)
    for epoch in range(config["training"]["num_epochs"]):
        for fp in files:
            df = pd.read_parquet(fp)
            for _, row in df.iterrows():
                ss_p = min(ss_max, ss_max * step / max(1, ramp))
                plan, sup = variation_to_v2_sample(
                    row.to_dict(), config["data"]["max_variations"],
                    config["data"]["max_depth"])
                losses = compute_finetune_v2_loss(model, plan, sup, ss_p, device)
                total = (L["thinking_move_weight"] * losses["thinking"]
                         + L["final_move_weight"] * losses["final"]
                         + L["wl_weight"] * losses["wl"]
                         + L["d_weight"] * losses["d"]
                         + L.get("transition_weight", 1.0) * losses["transition"])
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                if step % config["training"]["log_every_n_steps"] == 0:
                    wandb.log({f"train/{k}_loss": float(v) for k, v in losses.items()}
                              | {"train/total_loss": float(total),
                                 "train/ss_p": ss_p, "train/step": step})
                step += 1
                if step % config["training"]["save_every_n_steps"] == 0:
                    save_training_checkpoint(
                        os.path.join(run_dir, f"checkpoint_{step}.pt"),
                        model=model, optimizer=opt,
                        scaler=torch.amp.GradScaler("cuda", enabled=False),
                        step=step, extra_state={"epoch": epoch, "config": config})


if __name__ == "__main__":
    train()

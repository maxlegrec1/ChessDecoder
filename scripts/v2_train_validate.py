"""Bounded Phase-C validation: does the V2 loop actually train end-to-end on
real parquet_files_decoder data? Unit tests prove wiring; this proves the
joint loss goes down on real games. Not distributed, no wandb, ~N steps."""
import sys, time
import torch, torch.nn as nn

from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.models.v2.model_v2 import ChessDecoderV2, N_SQUARE_CLASSES
from chessdecoder.dataloader.loader_v2 import get_v2_dataloader, assemble_decoder_inputs
from chessdecoder.utils.training import load_config, soft_bucket_loss

IGNORE = -100
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
cfg = load_config("chessdecoder/train/config_v2.yaml")
dev = "cuda"
mc = cfg["model"]
m = ChessDecoderV2(
    vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
    num_encoder_layers=mc["num_encoder_layers"], num_decoder_layers=mc["num_decoder_layers"],
    num_latents=mc["num_latents"], board_max_seq_len=68,
    decoder_max_seq_len=cfg["data"]["max_plies"] * (mc["num_latents"] + 3),
    d_ff=mc["d_ff"]).to(dev)
m.train()
print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M  steps={STEPS}")

dl, ds = get_v2_dataloader(cfg["data"]["parquet_dir"], batch_size=cfg["data"]["batch_size"],
                           num_workers=4, max_plies=cfg["data"]["max_plies"])
opt = torch.optim.AdamW(m.parameters(), lr=cfg["training"]["learning_rate"],
                        weight_decay=cfg["training"]["weight_decay"])
scaler = torch.amp.GradScaler("cuda")
ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
losses = []
t0 = time.time()
it = iter(dl)
for step in range(STEPS):
    batch = next(it)
    bid = batch["board_ids"].to(dev); B, P, _ = bid.shape
    mf = batch["move_full"].to(dev)
    pt = batch["policy_tgt"].to(dev); pv = batch["policy_valid"].to(dev)
    wl = batch["wl"].to(dev); d = batch["d"].to(dev); wv = batch["wdl_valid"].to(dev)
    tsq = batch["trans_sq"].to(dev); trv = batch["trans_valid"].to(dev)
    pm = batch["ply_mask"].to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        lat = m.encode_boards(bid.reshape(B * P, 68)).reshape(B, P, -1, m.embed_dim)
        me = m.tok_embedding(mf)
        seq, pos = assemble_decoder_inputs(lat, me, wl, d, m.fourier_encoder)
        h = m.decoder(seq)
        pmask = pv & pm
        pol = m.policy_head(h[:, pos["policy_pos"], :])
        pl = ce(pol.reshape(-1, move_vocab_size),
                torch.where(pmask, pt, torch.full_like(pt, IGNORE)).reshape(-1))
        vm = (wv & pm).reshape(-1)
        wll = soft_bucket_loss(m.wl_head(h[:, pos["move_pos"], :].reshape(-1, m.embed_dim)),
                               wl.reshape(-1), m.wl_bucket_centers, vm)
        dl_ = soft_bucket_loss(m.d_head(h[:, pos["wl_pos"], :].reshape(-1, m.embed_dim)),
                               d.reshape(-1), m.d_bucket_centers, vm)
        out = m.transition_head(lat.reshape(B * P, -1, m.embed_dim), me.reshape(B * P, m.embed_dim))
        trm = trv.reshape(-1)
        sq_t = torch.where(trm.unsqueeze(1), tsq.reshape(B * P, 64),
                           torch.full_like(tsq.reshape(B * P, 64), IGNORE))
        trl = ce(out["square"].reshape(-1, N_SQUARE_CLASSES), sq_t.reshape(-1))
        tot = 5.0 * pl + trl + wll + dl_
    opt.zero_grad(set_to_none=True)
    scaler.scale(tot).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
    scaler.step(opt); scaler.update()
    losses.append(tot.item())
    if step % 20 == 0 or step == STEPS - 1:
        with torch.no_grad():
            pa = ((pol.argmax(-1) == pt) & pmask).sum() / (pmask.sum() + 1e-8)
            sa = ((out["square"].argmax(-1) == tsq.reshape(B * P, 64)) & trm.unsqueeze(1)).sum() / (trm.sum() * 64 + 1e-8)
        print(f"step {step:4d} | tot {tot.item():7.3f} pol {pl.item():6.3f} "
              f"trans {trl.item():6.3f} wl {float(wll):.3f} d {float(dl_):.3f} | "
              f"pol_acc {pa.item():.3f} sq_acc {sa.item():.3f} | "
              f"{(time.time()-t0)/(step+1):.2f}s/step mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
first = sum(losses[:10]) / 10
last = sum(losses[-10:]) / 10
print(f"\nmean loss first10={first:.3f} last10={last:.3f}  -> {'DOWN OK' if last < first else 'NOT DECREASING'}")

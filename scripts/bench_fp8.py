"""Bench: FP16 baseline vs FP8 (torchao Float8Linear + bf16 autocast).

Builds the V2 model at production dims, runs forward+backward+optimizer.step
on synthetic data for N steps in each mode, reports step/s + peak CUDA memory.
No real data, no checkpoints — pure throughput/memory measurement.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_fp8.py \\
        [warmup=10] [steps=50] [batch=256] [max_plies=10] [recipe=tensorwise]
"""
from __future__ import annotations
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessdecoder.models.vocab import vocab_size, move_vocab_size
from chessdecoder.models.v2.model_v2 import (
    ChessDecoderV2, N_SQUARE_CLASSES, N_STM_CLASSES, N_CASTLING_CLASSES)
from chessdecoder.models.v2.value_buckets import N_CELLS
from chessdecoder.dataloader.loader_v2 import assemble_decoder_inputs
from chessdecoder.utils.muon import build_optimizer
from chessdecoder.utils.fp8 import (
    convert_model_to_fp8, compile_fp8_hot_path, count_fp8_linears)


# --- config (matches chessdecoder/train/config_v2.yaml) ----------------------
WARMUP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 256
PLIES = int(sys.argv[4]) if len(sys.argv) > 4 else 10
RECIPE = sys.argv[5] if len(sys.argv) > 5 else "tensorwise"
EMBED, HEADS, ENC_L, DEC_L, K, D_FF = 1024, 16, 8, 10, 16, 1536
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def make_model():
    """Same dims as production training."""
    m = ChessDecoderV2(
        vocab_size=vocab_size, embed_dim=EMBED, num_heads=HEADS,
        num_encoder_layers=ENC_L, num_decoder_layers=DEC_L, num_latents=K,
        decoder_max_seq_len=PLIES * (K + 2), d_ff=D_FF).to(DEV)
    return m


def make_batch():
    """Synthetic batch matching the loader_v2 contract."""
    B, P = BATCH, PLIES
    bid = torch.randint(0, vocab_size, (B, P, 68), device=DEV)
    move_full = torch.randint(0, vocab_size, (B, P), device=DEV)
    pol_tgt = torch.randint(0, move_vocab_size, (B, P), device=DEV)
    pol_val = torch.ones(B, P, dtype=torch.bool, device=DEV)
    wdl_tgt = F.softmax(torch.randn(B, P, N_CELLS, device=DEV), dim=-1)
    wdl_mean = torch.tensor([0.4, 0.3, 0.3], device=DEV).expand(B, P, 3).contiguous()
    wdl_val = torch.ones(B, P, dtype=torch.bool, device=DEV)
    tsq = torch.randint(0, N_SQUARE_CLASSES, (B, P, 64), device=DEV)
    tstm = torch.randint(0, N_STM_CLASSES, (B, P), device=DEV)
    tcas = torch.randint(0, N_CASTLING_CLASSES, (B, P), device=DEV)
    tr_val = torch.ones(B, P, dtype=torch.bool, device=DEV)
    ply_mask = torch.ones(B, P, dtype=torch.bool, device=DEV)
    return dict(bid=bid, move_full=move_full, pol_tgt=pol_tgt, pol_val=pol_val,
                wdl_tgt=wdl_tgt, wdl_mean=wdl_mean, wdl_val=wdl_val,
                tsq=tsq, tstm=tstm, tcas=tcas, tr_val=tr_val, ply_mask=ply_mask)


def fwd_bwd(model, batch, autocast_dtype, use_amp):
    """One full training step worth of compute (forward + losses + backward).

    Mirrors the train_v2.py forward exactly so the bench reflects the actual
    hot path (encoder + decoder + transition + wdl heads + 3 CE losses).
    """
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    bid = batch["bid"]; B, P, _ = bid.shape
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
        E = model.embed_dim
        latents = model.encode_boards(bid.reshape(B * P, 68)).reshape(B, P, -1, E)
        wdl_logits = model.wdl_head(
            latents.reshape(B * P, -1, E)).reshape(B, P, N_CELLS)
        logp = torch.log_softmax(wdl_logits.float(), dim=-1)
        wdl_loss = -(batch["wdl_tgt"] * logp).sum(-1).mean()

        value_emb = model.embed_wdl(batch["wdl_mean"].reshape(-1, 3)).reshape(B, P, E)
        move_emb = model.tok_embedding(batch["move_full"])
        seq, pos = assemble_decoder_inputs(latents, move_emb, value_emb)
        h = model.decoder(seq)
        pol_logits = model.policy_head(h[:, pos["policy_pos"], :])
        policy_loss = ce(pol_logits.reshape(-1, move_vocab_size),
                         batch["pol_tgt"].reshape(-1))

        out = model.transition_head(latents.reshape(B * P, -1, E),
                                    move_emb.reshape(B * P, E))
        trans_loss = (
            ce(out["square"].reshape(-1, N_SQUARE_CLASSES),
               batch["tsq"].reshape(-1))
            + ce(out["stm"], batch["tstm"].reshape(-1))
            + ce(out["castling"], batch["tcas"].reshape(-1)))
        total = 5.0 * policy_loss + 1.0 * trans_loss + 1.0 * wdl_loss
    return total


def bench(label, *, use_fp8: bool, autocast_dtype: torch.dtype,
          use_amp: bool = True, compile_model: bool = False,
          recipe: str = "tensorwise") -> dict:
    print(f"\n===== {label} =====", flush=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    model = make_model().train()
    if use_fp8:
        convert_model_to_fp8(model, recipe=recipe)
        nfp8, nrest = count_fp8_linears(model)
        print(f"  FP8 ({recipe}): {nfp8} Float8Linear, {nrest} remained")
    if compile_model:
        # Submodule-level compile (encoder + decoder); whole-model
        # fullgraph=False doesn't fuse the FP8 cast+matmul.
        compile_fp8_hot_path(model)
        print("  torch.compile: applied to board_encoder + decoder")
    # GradScaler only makes sense with fp16 (overflow risk); bf16 + fp8 skip it.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and autocast_dtype == torch.float16
                         and not use_fp8))
    opt = build_optimizer(model, "muon", 1e-3, 0.01)
    batch = make_batch()

    use_scaler = scaler.is_enabled()
    # warmup
    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = fwd_bwd(model, batch, autocast_dtype, use_amp)
        (scaler.scale(loss) if use_scaler else loss).backward()
        if use_scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
    torch.cuda.synchronize()

    # measure
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = fwd_bwd(model, batch, autocast_dtype, use_amp)
        (scaler.scale(loss) if use_scaler else loss).backward()
        if use_scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = STEPS / dt
    peak_mb = torch.cuda.max_memory_allocated() / (1 << 20)
    print(f"  {STEPS} steps in {dt:.2f}s   -> {sps:.2f} step/s "
          f"({sps * BATCH:.0f} samples/s)   peak {peak_mb:.0f} MiB", flush=True)
    del model, opt, scaler, batch
    torch.cuda.empty_cache()
    return {"label": label, "steps": STEPS, "wall_s": dt, "step_per_s": sps,
            "peak_mib": peak_mb}


if __name__ == "__main__":
    print(f"bench_fp8: WARMUP={WARMUP} STEPS={STEPS} BATCH={BATCH} PLIES={PLIES}",
          flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}", flush=True)

    results = {}
    results["fp16"] = bench("FP16 (current baseline: fp16 autocast + GradScaler)",
                            use_fp8=False, autocast_dtype=torch.float16)
    results["bf16"] = bench("BF16 (no scaler, no fp8 — isolates autocast-dtype effect)",
                            use_fp8=False, autocast_dtype=torch.bfloat16)
    results[f"fp8_{RECIPE}_eager"] = bench(
        f"FP8 eager (torchao Float8Linear, no compile, {RECIPE})",
        use_fp8=True, autocast_dtype=torch.bfloat16, recipe=RECIPE)
    results[f"fp8_{RECIPE}_compile"] = bench(
        f"FP8 + submodule compile (encoder + decoder, {RECIPE})",
        use_fp8=True, autocast_dtype=torch.bfloat16, compile_model=True,
        recipe=RECIPE)

    print("\n===== SUMMARY =====")
    base = results["fp16"]["step_per_s"]
    for k, r in results.items():
        sps, mem = r["step_per_s"], r["peak_mib"]
        print(f"  {k:>22s}: {sps:6.2f} step/s  peak {mem:7.0f} MiB  "
              f"speedup vs fp16: {sps/base:.2f}x")

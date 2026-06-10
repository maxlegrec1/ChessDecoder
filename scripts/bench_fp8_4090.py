"""Benchmark fp16-AMP vs bf16+compile vs fp8+compile on the 30M oracle config.

Mirrors the train.py step (autocast dtype, GradScaler usage, fixed batch,
policy CE + WDL soft-CE losses) on synthetic data at the real batch size.
Run each mode in a fresh process:

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_fp8_4090.py fp16
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_fp8_4090.py bf16_compile
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_fp8_4090.py fp8
"""
import sys
import time

import torch
import torch.nn as nn

from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.value_buckets import N_CELLS
from chessdecoder.models.vocab import vocab_size, move_vocab_size, token_to_idx
from chessdecoder.utils.fp8 import (convert_model_to_fp8, compile_fp8_hot_path,
                                    count_fp8_linears)

MODE = sys.argv[1] if len(sys.argv) > 1 else "fp16"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
ACCUM = int(sys.argv[3]) if len(sys.argv) > 3 else 1
WARMUP, MEASURE = 12, 30

torch.manual_seed(0)
dev = "cuda"

model = ChessEncoder(vocab_size=vocab_size, embed_dim=512, num_heads=8,
                     num_layers=10, d_ff=1152, attention_variant="geom",
                     policy_head="cross_attn", input_mode="lc0_64",
                     ffn_type="dense").to(dev)

if MODE == "fp8":
    convert_model_to_fp8(model, recipe="tensorwise")
    nfp8, nrest = count_fp8_linears(model)
    print(f"FP8: {nfp8} Float8Linear, {nrest} bf16 Linears", flush=True)
    compile_fp8_hot_path(model)
    autocast_dtype = torch.bfloat16
elif MODE == "fp8_rowwise":
    convert_model_to_fp8(model, recipe="rowwise")
    nfp8, nrest = count_fp8_linears(model)
    print(f"FP8 rowwise: {nfp8} Float8Linear, {nrest} bf16 Linears", flush=True)
    compile_fp8_hot_path(model)
    autocast_dtype = torch.bfloat16
elif MODE == "fp8_ma":
    convert_model_to_fp8(model, recipe="tensorwise")
    nfp8, nrest = count_fp8_linears(model)
    print(f"FP8 max-autotune: {nfp8} Float8Linear, {nrest} bf16 Linears", flush=True)
    model.encoder = torch.compile(model.encoder, mode="max-autotune",
                                  dynamic=False)
    autocast_dtype = torch.bfloat16
elif MODE == "bf16_ma":
    model.encoder = torch.compile(model.encoder, mode="max-autotune",
                                  dynamic=False)
    autocast_dtype = torch.bfloat16
elif MODE == "bf16_compile":
    model.encoder = torch.compile(model.encoder, mode="default", dynamic=False)
    autocast_dtype = torch.bfloat16
elif MODE == "fp16":
    autocast_dtype = torch.float16
else:
    raise SystemExit(f"unknown mode {MODE}")

scaler = torch.amp.GradScaler("cuda", enabled=(MODE == "fp16"))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
ce = nn.CrossEntropyLoss()

# Synthetic batch — content irrelevant for speed; shapes/dtypes match train.
# Raw 68-token layout: [CLS, 64 squares, end_pos, castling, stm].
piece_ids = torch.randint(token_to_idx["white_pawn"],
                          token_to_idx["white_pawn"] + 13, (B, 68), device=dev)
pol_tgt = torch.randint(0, move_vocab_size, (B,), device=dev)
wdl_tgt = torch.softmax(torch.randn(B, N_CELLS, device=dev), dim=-1)

def step():
    """One optimizer step = ACCUM micro-batches (train.py-style accumulation)."""
    for i in range(ACCUM):
        with torch.autocast("cuda", dtype=autocast_dtype):
            out = model(piece_ids)
            pol_loss = ce(out["policy"], pol_tgt)
            logp = torch.log_softmax(out["wdl"].float(), dim=-1)
            wdl_loss = -(wdl_tgt * logp).sum(-1).mean()
            total = (5.0 * pol_loss + wdl_loss) / ACCUM
        scaler.scale(total).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

print(f"mode={MODE} B={B} accum={ACCUM} warming up ({WARMUP} steps, incl. compile)...",
      flush=True)
for _ in range(WARMUP):
    step()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(MEASURE):
    step()
torch.cuda.synchronize()
dt = (time.time() - t0) / MEASURE

print(f"RESULT mode={MODE} B={B} accum={ACCUM} (eff {B*ACCUM}): "
      f"{dt*1000:.1f} ms/opt-step  "
      f"{B*ACCUM*64/dt/1e6:.2f}M tok/s  "
      f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)

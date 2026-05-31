"""Throughput sweep: dense (compiled fp8) vs MoE-8e-top2 (eager fp8) across
model scales, to find where MoE's routing overhead stops dominating the expert
GEMMs. Same seq (64 lc0_64 tokens), same active FLOPs (expert_d_ff = d_ff/2).
Random data — we only measure fwd+bwd steps/s, not loss."""
import time, sys
import torch
from chessdecoder.models.model import ChessEncoder
from chessdecoder.models.vocab import vocab_size
from chessdecoder.utils.fp8 import (convert_model_to_fp8,
                                     convert_moe_experts_to_fp8)

torch.set_float32_matmul_precision("high")
DEV = "cuda"
BATCH = 512          # games; tokens/step = BATCH*64 (kept fixed across scales)
WARMUP, ITERS = 12, 25

# (embed_dim, num_layers, d_ff) — d_ff = 4*embed; expert_d_ff = d_ff/2 for MoE.
CONFIGS = [(640, 8, 2560), (1024, 8, 4096), (1536, 8, 6144), (2048, 8, 8192)]


def build(embed, layers, d_ff, moe):
    kw = dict(vocab_size=vocab_size, embed_dim=embed, num_heads=embed // 64,
              num_layers=layers, d_ff=d_ff, attention_variant="geom",
              policy_head="cross_attn", input_mode="lc0_64")
    if moe:
        kw.update(ffn_type="moe", moe_num_experts=8, moe_top_k=2,
                  moe_expert_d_ff=d_ff // 2,
                  moe_capacity_factor=1.25)  # fixed capacity -> compiles
    return ChessEncoder(**kw).to(DEV)


def bench(embed, layers, d_ff, moe, compile_dense):
    m = build(embed, layers, d_ff, moe)
    npar = sum(p.numel() for p in m.parameters())
    convert_model_to_fp8(m, "tensorwise")
    convert_moe_experts_to_fp8(m)
    if compile_dense:                       # compile BOTH dense and MoE now
        m.encoder = torch.compile(m.encoder, dynamic=False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    bid = torch.randint(0, vocab_size, (BATCH, 68), device=DEV).long()
    tgt = torch.randint(0, 1924, (BATCH,), device=DEV)
    def stepf():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            out = m(bid)
            loss = torch.nn.functional.cross_entropy(out["policy"], tgt)
            aux = m.moe_aux_loss()
            if aux is not None:
                loss = loss + aux
        loss.backward(); opt.step()
    for _ in range(WARMUP):
        stepf()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        stepf()
    torch.cuda.synchronize()
    sps = ITERS / (time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9
    del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return npar / 1e6, sps, sps * BATCH * 64, mem


print(f"{'embed':>6} {'dense_M':>8} {'moe_M':>7} {'dense_sps':>10} "
      f"{'moe_sps':>8} {'moe/dense':>10} {'dense_pos/s':>12} {'moe_pos/s':>10} {'mem_d/m_GB':>11}")
for embed, layers, d_ff in CONFIGS:
    try:
        dp, dsps, dpos, dmem = bench(embed, layers, d_ff, moe=False, compile_dense=True)
        mp, msps, mpos, mmem = bench(embed, layers, d_ff, moe=True, compile_dense=True)
        print(f"{embed:>6} {dp:>8.0f} {mp:>7.0f} {dsps:>10.2f} {msps:>8.2f} "
              f"{msps/dsps:>10.2f} {dpos:>12.0f} {mpos:>10.0f} {dmem:>5.0f}/{mmem:<5.0f}")
        sys.stdout.flush()
    except RuntimeError as e:
        print(f"{embed:>6}  OOM/err: {str(e)[:50]}"); torch.cuda.empty_cache()

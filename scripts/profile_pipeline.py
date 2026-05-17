"""Isolate the sweep bottleneck: dataloader (CPU parsing) vs model fwd/bwd (GPU).
Also benchmarks an alternative columnar parse path vs the current df.loc one.
"""
import sys, time, glob, random
import numpy as np
import torch

SHARD = sorted(glob.glob("/workspace/ChessDecoder/_cf_train/*.parquet"))[0]


def bench_current_dataloader(n_samples=4096):
    from chessdecoder.dataloader.encoder_loader import get_encoder_dataloader
    dl = get_encoder_dataloader("/workspace/ChessDecoder/_cf_train",
                                batch_size=512, num_workers=8, max_seq_len=68)
    it = iter(dl)
    # warm up one batch (worker spin-up + first parquet read excluded)
    next(it)
    t0 = time.time(); got = 0
    while got < n_samples:
        b = next(it); got += b["input_ids"].shape[0]
    dt = time.time() - t0
    print(f"[dataloader CURRENT] {got} samples in {dt:.2f}s = "
          f"{got/dt:,.0f} samp/s  (warm, 8 workers, bs512)")


def bench_parse_paths(n=20000):
    import pandas as pd
    import pyarrow.parquet as pq
    from chessdecoder.dataloader.data import fen_to_position_tokens
    from chessdecoder.models.vocab import token_to_idx, policy_to_idx

    # A) current path: pd.read_parquet + df.loc[idx] per row
    t0 = time.time()
    df = pd.read_parquet(SHARD)
    t_read = time.time() - t0
    idxs = df.index.tolist()[:n]
    t0 = time.time()
    ok = 0
    for idx in idxs:
        row = df.loc[idx]
        fen, bm = row["fen"], row["best_move"]
        if bm not in policy_to_idx:
            continue
        toks = fen_to_position_tokens(fen)
        _ = [token_to_idx[t] for t in toks]
        ok += 1
    t_cur = time.time() - t0
    print(f"[parse A: df.loc]      read={t_read:.2f}s  "
          f"{n} rows in {t_cur:.2f}s = {n/t_cur:,.0f} rows/s")

    # B) columnar: pyarrow -> python lists, positional iteration
    t0 = time.time()
    tbl = pq.read_table(SHARD, columns=["fen", "best_move"])
    fens = tbl.column("fen").to_pylist()
    bms = tbl.column("best_move").to_pylist()
    t_read2 = time.time() - t0
    t0 = time.time()
    ok = 0
    for i in range(n):
        fen, bm = fens[i], bms[i]
        if bm not in policy_to_idx:
            continue
        toks = fen_to_position_tokens(fen)
        _ = [token_to_idx[t] for t in toks]
        ok += 1
    t_col = time.time() - t0
    print(f"[parse B: columnar]    read={t_read2:.2f}s  "
          f"{n} rows in {t_col:.2f}s = {n/t_col:,.0f} rows/s  "
          f"-> {t_cur/t_col:.1f}x faster than A")


def bench_model(iters=30):
    from chessdecoder.models.v2.encoder_mode import ChessEncoderV2
    from chessdecoder.models.vocab import vocab_size, policy_index
    import torch.nn as nn
    dev = "cuda"
    m = ChessEncoderV2(vocab_size=vocab_size, num_policy_tokens=len(policy_index),
                       embed_dim=1024, num_heads=16, num_encoder_layers=10,
                       num_decoder_layers=2, num_latents=16, max_seq_len=68,
                       d_ff=1536).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    lf = nn.CrossEntropyLoss()
    x = torch.randint(0, vocab_size, (512, 68), device=dev)
    am = torch.ones(512, 68, dtype=torch.bool, device=dev)
    tg = torch.randint(0, len(policy_index), (512,), device=dev)
    for _ in range(5):  # warmup
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lf(m(x, attention_mask=am), tg)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lf(m(x, attention_mask=am), tg)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    per = dt / iters
    print(f"[model fwd+bwd]        {iters} micro-batches(512x68) in {dt:.2f}s "
          f"= {per*1000:.0f} ms/micro-batch  -> {per*4*1000:.0f} ms/opt-step "
          f"(accum=4, GPU-only, NOTE: contends with running sweep)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "parse"): bench_parse_paths()
    if which in ("all", "loader"): bench_current_dataloader()
    if which in ("all", "model"): bench_model()

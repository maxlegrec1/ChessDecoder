"""Profile C++ engine to identify bottlenecks."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import _decoder_inference_cpp as cpp

engine = cpp.ThinkingInferenceEngine(
    "export/backbone_causal.pt", "export/weights",
    "export/vocab.json", "export/config.json")

# Use the same 100 FENs from verify
import random, os
import pyarrow.parquet as pq

rng = random.Random(42)
data_dir = "/home/maxime/parquet_files_decoder/"
files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
fname = rng.choice(files)
t = pq.read_table(os.path.join(data_dir, fname), columns=["fen"])
indices = rng.sample(range(len(t)), min(300, len(t)))
fens_raw = [t.column("fen")[i].as_py() for i in indices]
seen = set()
fens = []
for f in fens_raw:
    if f not in seen:
        seen.add(f)
        fens.append(f)
    if len(fens) >= 20:
        break

# Warm up (2 calls to warm CUDA graphs)
engine.predict_move(fens[0], 0.0)
engine.predict_move(fens[1], 0.0)

# Enable profiling
engine.profiling = True
engine.reset_profile()

# Profile with CUDA sync to get accurate per-FEN timing
torch.cuda.synchronize()
total_tokens = 0
t_start = time.perf_counter()
per_fen = []

for i, fen in enumerate(fens):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.predict_move(fen, 0.0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ntok = len(list(engine.last_token_ids()))
    total_tokens += ntok
    elapsed = t1 - t0
    per_fen.append((ntok, elapsed))
    print(f"  FEN {i:2d}: {ntok:5d} tok in {elapsed*1000:8.1f} ms  ({ntok/elapsed:6.0f} tok/s)")

total_elapsed = time.perf_counter() - t_start
print(f"\nTotal: {total_tokens} tok in {total_elapsed:.2f}s ({total_tokens/total_elapsed:.0f} tok/s)")

# Print profiling breakdown
print(f"\n{'='*60}")
print(f"Profiling breakdown (accumulated over {len(fens)} FENs, {total_tokens} tokens):")
print(f"{'='*60}")

counters = {
    "prefix_init":    engine.prof_prefix_init,
    "board_prefill":  engine.prof_board_prefill,
    "board_catchup":  engine.prof_board_catchup,
    "board_gen":      engine.prof_board_gen,
    "prefix_block":   engine.prof_prefix_block,
    "prefix_incr":    engine.prof_prefix_incr,
    "causal_incr":    engine.prof_causal_incr,
    "head_eval":      engine.prof_head_eval,
}

total_profiled = sum(counters.values())
for name, val in sorted(counters.items(), key=lambda x: -x[1]):
    pct = val / total_elapsed * 100 if total_elapsed > 0 else 0
    pct_prof = val / total_profiled * 100 if total_profiled > 0 else 0
    print(f"  {name:20s}: {val*1000:8.1f} ms  ({pct:5.1f}% wall, {pct_prof:5.1f}% profiled)")

print(f"  {'--- total profiled':20s}: {total_profiled*1000:8.1f} ms  ({total_profiled/total_elapsed*100:5.1f}% wall)")
print(f"  {'--- wall clock':20s}: {total_elapsed*1000:8.1f} ms")
print(f"  {'--- unaccounted':20s}: {(total_elapsed-total_profiled)*1000:8.1f} ms  ({(total_elapsed-total_profiled)/total_elapsed*100:5.1f}% wall)")

# Per-token breakdown
print(f"\nPer-token breakdown (ms/tok):")
for name, val in sorted(counters.items(), key=lambda x: -x[1]):
    ms_per_tok = val * 1000 / total_tokens if total_tokens > 0 else 0
    print(f"  {name:20s}: {ms_per_tok:.4f} ms/tok")
print(f"  {'--- total profiled':20s}: {total_profiled*1000/total_tokens:.4f} ms/tok")
print(f"  {'--- wall clock':20s}: {total_elapsed*1000/total_tokens:.4f} ms/tok")

# Analyze: tok/s vs sequence length
per_fen.sort(key=lambda x: x[0])
print(f"\nBy sequence length:")
for ntok, elapsed in per_fen:
    ms_per_tok = elapsed * 1000 / ntok
    print(f"  {ntok:5d} tok: {ms_per_tok:.3f} ms/tok  ({ntok/elapsed:.0f} tok/s)")

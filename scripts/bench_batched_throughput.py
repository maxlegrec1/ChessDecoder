"""Quick throughput benchmark for batched engine at different batch sizes."""
import os, random, time
import pyarrow.parquet as pq
import _decoder_inference_cpp as cpp

EXPORT = "exports/base"
DATA_DIR = "/home/maxime/parquet_files_decoder/"

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
rng = random.Random(42)
fname = rng.choice(files)
table = pq.read_table(os.path.join(DATA_DIR, fname), columns=["fen"])
indices = rng.sample(range(len(table)), 100)
fens = [table.column("fen")[i].as_py() for i in indices]

# Single engine baseline
single = cpp.ThinkingInferenceEngine(
    f"{EXPORT}/backbone.pt", f"{EXPORT}/weights",
    f"{EXPORT}/vocab.json", f"{EXPORT}/config.json")
single.think_temperature = 1.0
single.policy_temperature = 1.0

# Warmup
single.predict_move(fens[0])

t0 = time.time()
for fen in fens[:20]:
    single.predict_move(fen)
t_single = time.time() - t0
print(f"Single x20: {t_single:.1f}s, {single.total_tokens / t_single:.0f} tok/s")

# Batched B=4
batched = cpp.BatchedInferenceEngine(
    f"{EXPORT}/backbone.pt", f"{EXPORT}/weights",
    f"{EXPORT}/vocab.json", f"{EXPORT}/config.json", 4)
batched.think_temperature = 1.0
batched.policy_temperature = 1.0

# Warmup
batched.predict_moves([fens[0]])

t0 = time.time()
total_toks = 0
for i in range(0, 20, 4):
    batch = fens[i:i+4]
    results = batched.predict_moves(batch)
    total_toks += sum(len(r.token_ids) for r in results)
t_batched = time.time() - t0
print(f"Batched B=4 x20: {t_batched:.1f}s, {total_toks / t_batched:.0f} tok/s")
print(f"Speedup: {t_single/t_batched:.2f}x")

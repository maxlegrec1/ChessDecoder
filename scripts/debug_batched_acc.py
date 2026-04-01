"""
Quick acc@k test comparing single engine to batched engine.
Uses few FENs for speed.
"""
import os
import random
import time
import pyarrow.parquet as pq
import _decoder_inference_cpp as cpp

EXPORT = "exports/base"
DATA_DIR = "/home/maxime/parquet_files_decoder/"
NUM_FENS = 20
K = 5
SEED = 42
TEMP = 1.5

def load_pairs(n, seed):
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    rng = random.Random(seed)
    fname = rng.choice(files)
    table = pq.read_table(os.path.join(DATA_DIR, fname), columns=["fen", "best_move"])
    indices = rng.sample(range(len(table)), min(n * 3, len(table)))
    seen, pairs = set(), []
    for i in indices:
        fen = table.column("fen")[i].as_py()
        if fen not in seen:
            seen.add(fen)
            pairs.append((fen, table.column("best_move")[i].as_py()))
        if len(pairs) >= n:
            break
    return pairs[:n]

pairs = load_pairs(NUM_FENS, SEED)
print(f"Loaded {len(pairs)} FENs, k={K}, temp={TEMP}")

# Single engine
single = cpp.ThinkingInferenceEngine(
    f"{EXPORT}/backbone.pt", f"{EXPORT}/weights",
    f"{EXPORT}/vocab.json", f"{EXPORT}/config.json")
single.think_temperature = TEMP
single.policy_temperature = TEMP

t0 = time.time()
correct_single = 0
for fen, best_move in pairs:
    moves = {single.predict_move(fen) for _ in range(K)}
    if best_move in moves:
        correct_single += 1
t_single = time.time() - t0
print(f"Single engine acc@{K}: {correct_single}/{NUM_FENS} = {correct_single/NUM_FENS:.1%}  ({t_single:.0f}s)")

# Batched engine (B=4)
batched = cpp.BatchedInferenceEngine(
    f"{EXPORT}/backbone.pt", f"{EXPORT}/weights",
    f"{EXPORT}/vocab.json", f"{EXPORT}/config.json", 4)
batched.think_temperature = TEMP
batched.policy_temperature = TEMP

t0 = time.time()
correct_batched = 0
for fen, best_move in pairs:
    chunk = [fen] * 4
    results = batched.predict_moves(chunk, TEMP)
    moves = {r.move for r in results[:4]}
    if best_move in moves:
        correct_batched += 1
t_batched = time.time() - t0
# Only 4 unique samples per FEN (due to dedup), but that's fine for comparison
print(f"Batched engine acc@4: {correct_batched}/{NUM_FENS} = {correct_batched/NUM_FENS:.1%}  ({t_batched:.0f}s)")

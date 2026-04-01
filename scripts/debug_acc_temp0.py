"""Compare single vs batched engine at temp=0 (pure accuracy test, no sampling noise)."""
import os, random, time
import pyarrow.parquet as pq
import _decoder_inference_cpp as cpp

EXPORT = "exports"
DATA_DIR = "/home/maxime/parquet_files_decoder/"
NUM_FENS = 50
SEED = 123

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
            seen.add(fen); pairs.append((fen, table.column("best_move")[i].as_py()))
        if len(pairs) >= n: break
    return pairs[:n]

pairs = load_pairs(NUM_FENS, SEED)

# Single engine at temp=0
single = cpp.ThinkingInferenceEngine(
    f"{EXPORT}/backbone.pt", f"{EXPORT}/weights",
    f"{EXPORT}/vocab.json", f"{EXPORT}/config.json")

t0 = time.time()
s_correct = sum(single.predict_move(fen, 0.0) == bm for fen, bm in pairs)
t_s = time.time() - t0
print(f"Single  temp=0 acc@1: {s_correct}/{NUM_FENS} = {s_correct/NUM_FENS:.1%}  ({t_s:.0f}s)")

# Batched engine at temp=0, B=1
batched = cpp.BatchedInferenceEngine(
    f"{EXPORT}/backbone.pt", f"{EXPORT}/weights",
    f"{EXPORT}/vocab.json", f"{EXPORT}/config.json", 1)

t0 = time.time()
b_correct = 0
b_agree = 0
for fen, bm in pairs:
    r = batched.predict_moves([fen], 0.0)[0]
    if r.move == bm: b_correct += 1
    s_move = single.predict_move(fen, 0.0)
    if r.move == s_move: b_agree += 1
t_b = time.time() - t0
print(f"Batched temp=0 acc@1: {b_correct}/{NUM_FENS} = {b_correct/NUM_FENS:.1%}  ({t_b:.0f}s)")
print(f"Batched vs Single agreement: {b_agree}/{NUM_FENS} = {b_agree/NUM_FENS:.1%}")

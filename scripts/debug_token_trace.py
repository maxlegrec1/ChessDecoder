"""Trace tokens for single vs batched to find where they diverge."""
import os, random
import pyarrow.parquet as pq
import _decoder_inference_cpp as cpp

EXPORT = "exports/base"

# Use one of the disagreeing FENs from the 5-FEN test
# FEN4 had different seq lengths (427 vs 712)
FEN = "r2qkb1r/ppp2ppp/2np1n2/4p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq"

single = cpp.ThinkingInferenceEngine(
    f"{EXPORT}/backbone.pt", f"{EXPORT}/weights",
    f"{EXPORT}/vocab.json", f"{EXPORT}/config.json")

batched = cpp.BatchedInferenceEngine(
    f"{EXPORT}/backbone.pt", f"{EXPORT}/weights",
    f"{EXPORT}/vocab.json", f"{EXPORT}/config.json", 1)

# temp=0 for determinism
s_move = single.predict_move(FEN, 0.0)
s_tokens = single.last_token_ids()
s_len = len(s_tokens)

r = batched.predict_moves([FEN], 0.0)[0]
b_tokens = r.token_ids
b_len = len(b_tokens)

print(f"Single: move={s_move}, len={s_len}")
print(f"Batched: move={r.move}, len={b_len}")
print()

# Find first divergence point
min_len = min(s_len, b_len)
first_diff = None
for i in range(min_len):
    if s_tokens[i] != b_tokens[i]:
        first_diff = i
        break

if first_diff is not None:
    print(f"First divergence at position {first_diff}:")
    lo = max(0, first_diff - 3)
    hi = min(min_len, first_diff + 5)
    print(f"  Single  [{lo}:{hi}]: {s_tokens[lo:hi]}")
    print(f"  Batched [{lo}:{hi}]: {b_tokens[lo:hi]}")
elif s_len != b_len:
    print(f"Tokens identical up to position {min_len}, but lengths differ ({s_len} vs {b_len})")
    # Show what's beyond
    if s_len > b_len:
        print(f"  Single extra: {s_tokens[b_len:b_len+10]}")
    else:
        print(f"  Batched extra: {b_tokens[s_len:s_len+10]}")
else:
    print("Tokens IDENTICAL!")

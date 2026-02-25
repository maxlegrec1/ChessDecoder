"""
Verify C++ TRT inference matches Python PyTorch inference.

Compares full COT token sequences and final moves on 100 FENs.
Both implementations must use temperature=0 (deterministic argmax).

Usage:
    uv run python scripts/verify_cpp_vs_python.py \
        --checkpoint checkpoint_step_32000.pt \
        --export-dir export
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
import torch

from src.models.vocab import idx_to_token


def load_fens(n=100, seed=42):
    """Load n diverse FENs from the training data."""
    import os
    data_dir = "/home/maxime/parquet_files_decoder/"
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))

    rng = random.Random(seed)
    # Read just one file, only the fen column, and sample rows
    fname = rng.choice(files)
    t = pq.read_table(os.path.join(data_dir, fname), columns=["fen"])
    total_rows = len(t)

    # Sample row indices
    indices = rng.sample(range(total_rows), min(n * 3, total_rows))
    fens_raw = [t.column("fen")[i].as_py() for i in indices]

    # Deduplicate
    seen = set()
    fens = []
    for f in fens_raw:
        if f not in seen:
            seen.add(f)
            fens.append(f)
        if len(fens) >= n:
            break

    return fens[:n]


def run_python(fens, checkpoint_path, device):
    """Run Python thinking inference on all FENs."""
    from scripts.test_evaluate_thinking import ThinkingModelWrapper
    from scripts.think import load_model

    print("  Loading model...", flush=True)
    model, max_seq_len = load_model(checkpoint_path, device)
    wrapper = ThinkingModelWrapper(model, device, max_seq_len, think_temperature=0.0)

    results = []
    total_tokens = 0
    t_start = time.time()
    for i, fen in enumerate(fens):
        move = wrapper.predict_move(fen, temperature=0.0)
        token_names = [idx_to_token[tid] for tid in wrapper.last_token_ids]
        ntok = len(wrapper.last_token_ids)
        total_tokens += ntok
        elapsed = time.time() - t_start
        tps = total_tokens / elapsed if elapsed > 0 else 0
        sys.stderr.write(f"\r  Python: {i+1}/{len(fens)} FENs | {total_tokens} tok | {tps:.0f} tok/s")
        sys.stderr.flush()
        results.append({
            "move": move,
            "token_ids": list(wrapper.last_token_ids),
            "token_names": token_names,
            "wl_entries": list(wrapper.last_wl_entries),
            "d_entries": list(wrapper.last_d_entries),
        })
    sys.stderr.write("\n")
    return results


def run_cpp(fens, export_dir):
    """Run C++ libtorch thinking inference on all FENs."""
    import _decoder_inference_cpp as cpp

    print("  Loading libtorch backbone...", flush=True)
    engine = cpp.ThinkingInferenceEngine(
        f"{export_dir}/backbone_causal.pt",
        f"{export_dir}/weights",
        f"{export_dir}/vocab.json",
        f"{export_dir}/config.json",
    )

    results = []
    total_tokens = 0
    t_start = time.time()
    for i, fen in enumerate(fens):
        move = engine.predict_move(fen, 0.0)
        tids = list(engine.last_token_ids())
        token_names = [engine.idx_to_token(t) for t in tids]
        ntok = len(tids)
        total_tokens += ntok
        elapsed = time.time() - t_start
        tps = total_tokens / elapsed if elapsed > 0 else 0
        sys.stderr.write(f"\r  C++: {i+1}/{len(fens)} FENs | {total_tokens} tok | {tps:.0f} tok/s")
        sys.stderr.flush()
        results.append({
            "move": move,
            "token_ids": tids,
            "token_names": token_names,
            "wl_entries": list(engine.last_wl_entries()),
            "d_entries": list(engine.last_d_entries()),
        })
    sys.stderr.write("\n")
    return results


def _cache_key(checkpoint_path, fens):
    """Deterministic cache key from checkpoint mtime+size and FEN list."""
    cp = Path(checkpoint_path)
    stat = cp.stat()
    raw = f"{cp.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{json.dumps(fens)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(checkpoint_path, fens):
    cache_dir = Path(__file__).resolve().parent / ".verify_cache"
    cache_dir.mkdir(exist_ok=True)
    key = _cache_key(checkpoint_path, fens)
    return cache_dir / f"python_results_{key}.json"


def save_python_cache(checkpoint_path, fens, results):
    path = _cache_path(checkpoint_path, fens)
    # Convert tuples to lists for JSON serialization
    serializable = []
    for r in results:
        serializable.append({
            "move": r["move"],
            "token_ids": r["token_ids"],
            "token_names": r["token_names"],
            "wl_entries": [[pos, val] for pos, val in r["wl_entries"]],
            "d_entries": [[pos, val] for pos, val in r["d_entries"]],
        })
    path.write_text(json.dumps(serializable))
    print(f"  Cached Python results to {path}")


def load_python_cache(checkpoint_path, fens):
    path = _cache_path(checkpoint_path, fens)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # Convert lists back to tuples for wl/d entries
    results = []
    for r in data:
        results.append({
            "move": r["move"],
            "token_ids": r["token_ids"],
            "token_names": r["token_names"],
            "wl_entries": [(pos, val) for pos, val in r["wl_entries"]],
            "d_entries": [(pos, val) for pos, val in r["d_entries"]],
        })
    return results


def find_first_diff(py_tokens, cpp_tokens):
    """Find position and tokens where sequences first diverge."""
    for i in range(min(len(py_tokens), len(cpp_tokens))):
        if py_tokens[i] != cpp_tokens[i]:
            return i, py_tokens[i], cpp_tokens[i]
    if len(py_tokens) != len(cpp_tokens):
        shorter = min(len(py_tokens), len(cpp_tokens))
        return shorter, "END" if shorter == len(py_tokens) else py_tokens[shorter], \
               "END" if shorter == len(cpp_tokens) else cpp_tokens[shorter]
    return None


def compare_values(py_entries, cpp_entries, label):
    """Compare WL/D value entries, return max absolute difference."""
    if len(py_entries) != len(cpp_entries):
        return float('inf'), f"count mismatch: {len(py_entries)} vs {len(cpp_entries)}"
    max_diff = 0.0
    for (py_pos, py_val), (cpp_pos, cpp_val) in zip(py_entries, cpp_entries):
        if py_pos != cpp_pos:
            return float('inf'), f"position mismatch at {label}: {py_pos} vs {cpp_pos}"
        diff = abs(py_val - cpp_val)
        max_diff = max(max_diff, diff)
    return max_diff, None


def main():
    parser = argparse.ArgumentParser(description="Verify C++ TRT vs Python PyTorch thinking inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-dir", default="export")
    parser.add_argument("--num-fens", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load FENs
    print(f"Loading {args.num_fens} FENs...")
    fens = load_fens(args.num_fens, args.seed)
    print(f"Loaded {len(fens)} unique FENs")

    # Run C++ first (faster)
    print(f"\n--- Running C++ TRT inference on {len(fens)} FENs ---")
    t0 = time.time()
    cpp_results = run_cpp(fens, args.export_dir)
    cpp_time = time.time() - t0
    cpp_total_tok = sum(len(r["token_ids"]) for r in cpp_results)
    print(f"C++ done in {cpp_time:.1f}s ({cpp_total_tok} tok, {cpp_total_tok/cpp_time:.0f} tok/s)")

    # Run Python (with caching)
    cached_py = load_python_cache(args.checkpoint, fens)
    if cached_py is not None:
        py_results = cached_py
        py_total_tok = sum(len(r["token_ids"]) for r in py_results)
        py_time = 0.0
        print(f"\n--- Python results loaded from cache ({py_total_tok} tok) ---")
    else:
        print(f"\n--- Running Python PyTorch inference on {len(fens)} FENs ---")
        t0 = time.time()
        py_results = run_python(fens, args.checkpoint, device)
        py_time = time.time() - t0
        py_total_tok = sum(len(r["token_ids"]) for r in py_results)
        print(f"Python done in {py_time:.1f}s ({py_total_tok} tok, {py_total_tok/py_time:.0f} tok/s)")
        save_python_cache(args.checkpoint, fens, py_results)

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS ({len(fens)} FENs)")
    print(f"{'='*80}")

    move_match = 0
    cot_match = 0
    cot_len_match = 0
    first_diverge_positions = []
    wl_diffs = []
    d_diffs = []
    move_mismatches = []

    for i, (fen, py, cpp) in enumerate(zip(fens, py_results, cpp_results)):
        # Compare final move
        moves_same = py["move"] == cpp["move"]
        if moves_same:
            move_match += 1
        else:
            move_mismatches.append((i, fen, py["move"], cpp["move"]))

        # Compare token sequence length
        if len(py["token_ids"]) == len(cpp["token_ids"]):
            cot_len_match += 1

        # Compare full COT tokens
        tokens_same = py["token_ids"] == cpp["token_ids"]
        if tokens_same:
            cot_match += 1
        else:
            diff = find_first_diff(py["token_names"], cpp["token_names"])
            if diff:
                pos, py_tok, cpp_tok = diff
                first_diverge_positions.append(pos)

        # Compare values (only if tokens match up to the value positions)
        wl_diff, wl_err = compare_values(py["wl_entries"], cpp["wl_entries"], "WL")
        d_diff, d_err = compare_values(py["d_entries"], cpp["d_entries"], "D")
        if wl_err is None:
            wl_diffs.append(wl_diff)
        if d_err is None:
            d_diffs.append(d_diff)

    print(f"\nFinal move match:  {move_match}/{len(fens)} ({100*move_match/len(fens):.1f}%)")
    print(f"COT exact match:   {cot_match}/{len(fens)} ({100*cot_match/len(fens):.1f}%)")
    print(f"COT length match:  {cot_len_match}/{len(fens)} ({100*cot_len_match/len(fens):.1f}%)")

    if wl_diffs:
        print(f"\nWL value diffs (when tokens match):")
        print(f"  Max: {max(wl_diffs):.6f}, Mean: {sum(wl_diffs)/len(wl_diffs):.6f}")
    if d_diffs:
        print(f"D value diffs (when tokens match):")
        print(f"  Max: {max(d_diffs):.6f}, Mean: {sum(d_diffs)/len(d_diffs):.6f}")

    if first_diverge_positions:
        print(f"\nFirst divergence position stats (when COT differs):")
        print(f"  Min: {min(first_diverge_positions)}, Max: {max(first_diverge_positions)}")
        print(f"  Mean: {sum(first_diverge_positions)/len(first_diverge_positions):.1f}")
        print(f"  Median: {sorted(first_diverge_positions)[len(first_diverge_positions)//2]}")

    if move_mismatches:
        print(f"\nMove mismatches (first 20):")
        for idx, fen, py_move, cpp_move in move_mismatches[:20]:
            py_len = len(py_results[idx]["token_ids"])
            cpp_len = len(cpp_results[idx]["token_ids"])
            diff = find_first_diff(py_results[idx]["token_names"], cpp_results[idx]["token_names"])
            div_pos = diff[0] if diff else "N/A"
            print(f"  [{idx:3d}] py={py_move:6s} cpp={cpp_move:6s} "
                  f"(py:{py_len} cpp:{cpp_len} tokens, div@{div_pos}) "
                  f"{fen[:50]}")

    # Detailed first-divergence analysis
    for i, (fen, py, cpp) in enumerate(zip(fens, py_results, cpp_results)):
        diff = find_first_diff(py["token_names"], cpp["token_names"])
        if diff:
            pos, py_tok, cpp_tok = diff
            print(f"\n--- Detailed divergence for FEN {i} at position {pos} ---")
            print(f"  FEN: {fen[:60]}")
            print(f"  Python token: {py_tok}")
            print(f"  C++    token: {cpp_tok}")
            # Show context: 5 tokens before divergence
            start = max(0, pos - 5)
            print(f"  Context (pos {start}..{pos}):")
            py_ctx = py["token_names"][start:pos+1]
            cpp_ctx = cpp["token_names"][start:pos+1]
            for j, (pt, ct) in enumerate(zip(py_ctx, cpp_ctx)):
                marker = " <<< DIVERGE" if start + j == pos else ""
                match = "==" if pt == ct else "!="
                print(f"    [{start+j:4d}] py={pt:20s} {match} cpp={ct:20s}{marker}")
            # Count how many structural tokens match
            struct_tokens = {"start_pos", "end_pos", "start_think", "end_think",
                            "end_var", "wl_value", "d_value", "white_to_move",
                            "black_to_move"}
            py_struct = [(j, t) for j, t in enumerate(py["token_names"]) if t in struct_tokens]
            cpp_struct = [(j, t) for j, t in enumerate(cpp["token_names"]) if t in struct_tokens]
            # Count matching structural token positions
            match_count = sum(1 for (pj, pt), (cj, ct) in zip(py_struct, cpp_struct)
                            if pj == cj and pt == ct)
            print(f"  Structural tokens: {match_count}/{min(len(py_struct), len(cpp_struct))} match")

    # Show a few exact matches in detail
    exact = [i for i in range(len(fens)) if py_results[i]["token_ids"] == cpp_results[i]["token_ids"]]
    if exact:
        print(f"\nSample exact COT matches (first 3):")
        for idx in exact[:3]:
            toks = py_results[idx]["token_names"]
            # Show structural tokens only
            structural = [(j, t) for j, t in enumerate(toks)
                         if t in ("start_pos", "end_pos", "start_think", "end_think",
                                  "end_var", "wl_value", "d_value", "white_to_move",
                                  "black_to_move") or (len(t) >= 4 and t[0] in "abcdefgh"
                                  and t[1] in "12345678" and t[2] in "abcdefgh")]
            moves_in_cot = [t for _, t in structural if len(t) >= 4 and t[0] in "abcdefgh"
                           and t[1] in "12345678" and t[2] in "abcdefgh"]
            print(f"  [{idx:3d}] {py_results[idx]['move']:6s} "
                  f"({len(toks)} tok) moves={moves_in_cot[:8]}")

    if py_time > 0:
        print(f"\nTiming: C++ {cpp_time:.1f}s ({cpp_total_tok/cpp_time:.0f} tok/s) vs "
              f"Python {py_time:.1f}s ({py_total_tok/py_time:.0f} tok/s) "
              f"({py_time/cpp_time:.1f}x slower)")
    else:
        print(f"\nTiming: C++ {cpp_time:.1f}s ({cpp_total_tok/cpp_time:.0f} tok/s) "
              f"(Python results from cache)")


if __name__ == "__main__":
    main()

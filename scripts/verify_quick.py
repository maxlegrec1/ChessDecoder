"""
Quick verification: 5 FENs, exact match between Python and C++ libtorch inference.

Usage:
    uv run python scripts/verify_quick.py --checkpoint checkpoint_step_32000.pt --export-dir export
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.vocab import idx_to_token


def _cache_key(checkpoint_path, fens):
    cp = Path(checkpoint_path)
    stat = cp.stat()
    raw = f"{cp.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{json.dumps(fens)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(checkpoint_path, fens):
    cache_dir = Path(__file__).resolve().parent / ".verify_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"python_results_{_cache_key(checkpoint_path, fens)}.json"


def save_python_cache(checkpoint_path, fens, results):
    path = _cache_path(checkpoint_path, fens)
    serializable = [{
        "move": r["move"],
        "token_ids": r["token_ids"],
        "token_names": r["token_names"],
        "wl_entries": [[p, v] for p, v in r["wl_entries"]],
        "d_entries": [[p, v] for p, v in r["d_entries"]],
    } for r in results]
    path.write_text(json.dumps(serializable))


def load_python_cache(checkpoint_path, fens):
    path = _cache_path(checkpoint_path, fens)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return [{
        "move": r["move"],
        "token_ids": r["token_ids"],
        "token_names": r["token_names"],
        "wl_entries": [(p, v) for p, v in r["wl_entries"]],
        "d_entries": [(p, v) for p, v in r["d_entries"]],
    } for r in data]

# 5 diverse FENs covering different game phases
TEST_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # starting position
    "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # open game
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/1BN2N2/PPPP1PPP/R1BQ1RK1 w - - 4 7",  # castled
    "8/5pk1/6p1/8/8/2B5/5PPP/6K1 w - - 0 40",  # endgame
    "r1bq1rk1/pp2nppp/2n1p3/3pP3/3P4/2N2N2/PP2BPPP/R1BQ1RK1 w - d6 0 10",  # French Defense
]


def run_python(fens, checkpoint_path, device):
    from scripts.test_evaluate_thinking import ThinkingModelWrapper
    from scripts.think import load_model

    model, max_seq_len = load_model(checkpoint_path, device)
    wrapper = ThinkingModelWrapper(model, device, max_seq_len, think_temperature=0.0)

    results = []
    for fen in fens:
        move = wrapper.predict_move(fen, temperature=0.0)
        results.append({
            "move": move,
            "token_ids": list(wrapper.last_token_ids),
            "token_names": [idx_to_token[t] for t in wrapper.last_token_ids],
            "wl_entries": list(wrapper.last_wl_entries),
            "d_entries": list(wrapper.last_d_entries),
        })
    return results


def run_cpp(fens, export_dir):
    import _decoder_inference_cpp as cpp

    engine = cpp.ThinkingInferenceEngine(
        f"{export_dir}/backbone_causal.pt",
        f"{export_dir}/weights",
        f"{export_dir}/vocab.json",
        f"{export_dir}/config.json",
    )

    results = []
    for fen in fens:
        move = engine.predict_move(fen, 0.0)
        tids = list(engine.last_token_ids())
        results.append({
            "move": move,
            "token_ids": tids,
            "token_names": [engine.idx_to_token(t) for t in tids],
            "wl_entries": list(engine.last_wl_entries()),
            "d_entries": list(engine.last_d_entries()),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-dir", default="export")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Running on {len(TEST_FENS)} test FENs...\n")

    # Run C++ first (faster)
    print("--- C++ libtorch inference ---")
    t0 = time.time()
    cpp_results = run_cpp(TEST_FENS, args.export_dir)
    cpp_time = time.time() - t0
    cpp_tok = sum(len(r["token_ids"]) for r in cpp_results)
    print(f"Done in {cpp_time:.1f}s ({cpp_tok} tokens, {cpp_tok/cpp_time:.0f} tok/s)\n")

    # Run Python (with caching)
    cached_py = load_python_cache(args.checkpoint, TEST_FENS)
    if cached_py is not None:
        py_results = cached_py
        py_tok = sum(len(r["token_ids"]) for r in py_results)
        py_time = 0.0
        print("--- Python results loaded from cache ---\n")
    else:
        print("--- Python PyTorch inference ---")
        t0 = time.time()
        py_results = run_python(TEST_FENS, args.checkpoint, device)
        py_time = time.time() - t0
        py_tok = sum(len(r["token_ids"]) for r in py_results)
        print(f"Done in {py_time:.1f}s ({py_tok} tokens, {py_tok/py_time:.0f} tok/s)\n")
        save_python_cache(args.checkpoint, TEST_FENS, py_results)

    # Compare
    # Note: prefix KV caching may cause tiny WL/D value differences (different FP16
    # GEMM kernels for incremental vs full-batch attention). Token sequences and moves
    # must match exactly; WL/D values use tolerance-based comparison.
    WL_D_TOL = 0.05  # bucket center spacing is ~0.067, so 0.05 catches adjacent bucket flips
    all_match = True
    max_wl_diff = 0.0
    max_d_diff = 0.0
    for i, (fen, py, cpp) in enumerate(zip(TEST_FENS, py_results, cpp_results)):
        move_ok = py["move"] == cpp["move"]
        cot_ok = py["token_ids"] == cpp["token_ids"]

        # Tolerance-based WL/D comparison
        wl_max = 0.0
        if len(py["wl_entries"]) == len(cpp["wl_entries"]):
            for (pp, pv), (cp, cv) in zip(py["wl_entries"], cpp["wl_entries"]):
                if pp != cp:
                    wl_max = float('inf')
                    break
                wl_max = max(wl_max, abs(pv - cv))
        else:
            wl_max = float('inf')
        wl_ok = wl_max <= WL_D_TOL

        d_max = 0.0
        if len(py["d_entries"]) == len(cpp["d_entries"]):
            for (pp, pv), (cp, cv) in zip(py["d_entries"], cpp["d_entries"]):
                if pp != cp:
                    d_max = float('inf')
                    break
                d_max = max(d_max, abs(pv - cv))
        else:
            d_max = float('inf')
        d_ok = d_max <= WL_D_TOL

        max_wl_diff = max(max_wl_diff, wl_max)
        max_d_diff = max(max_d_diff, d_max)

        status = "PASS" if (move_ok and cot_ok and wl_ok and d_ok) else "FAIL"

        if status == "FAIL":
            all_match = False

        fen_short = fen[:50]
        wl_str = f"wl_diff={wl_max:.4f}" if wl_max > 0 else "wl=exact"
        d_str = f"d_diff={d_max:.4f}" if d_max > 0 else "d=exact"
        print(f"[{i}] {status}  move={py['move']:6s}  "
              f"tokens={len(py['token_ids'])}  "
              f"{wl_str}  {d_str}  "
              f"cot={'exact' if cot_ok else 'DIFFER'}  "
              f"{fen_short}")

        if not cot_ok:
            # Show first divergence
            for j in range(min(len(py["token_ids"]), len(cpp["token_ids"]))):
                if py["token_ids"][j] != cpp["token_ids"][j]:
                    print(f"     First divergence at pos {j}: "
                          f"py={py['token_names'][j]} vs cpp={cpp['token_names'][j]}")
                    break
            if len(py["token_ids"]) != len(cpp["token_ids"]):
                print(f"     Length: py={len(py['token_ids'])} vs cpp={len(cpp['token_ids'])}")

    print(f"\n{'='*60}")
    if all_match:
        print(f"ALL {len(TEST_FENS)} FENS: 100% MATCH (tokens exact, moves exact, "
              f"max WL diff={max_wl_diff:.4f}, max D diff={max_d_diff:.4f})")
    else:
        print("MISMATCH DETECTED — see details above")
        sys.exit(1)

    if py_time > 0:
        print(f"\nSpeed: C++ {cpp_tok/cpp_time:.0f} tok/s vs Python {py_tok/py_time:.0f} tok/s "
              f"({py_time/cpp_time:.1f}x faster)")
    else:
        print(f"\nSpeed: C++ {cpp_tok/cpp_time:.0f} tok/s (Python results from cache)")


if __name__ == "__main__":
    main()

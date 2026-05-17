"""Stream-download ChessFENS shards in parallel, keep only (fen, best_move)
where best_move = policy_index[argmax(policy)], then delete the raw shard.

ChessFENS stores a 1858-float Leela policy per row (illegal = -1). Our
chessdecoder/dataloader/policy_index.py is exactly that Leela 1858 list, so
argmax -> policy_index[idx] -> UCI string is our encoder's `best_move`.
~175 MB/shard of policy floats collapse to a few MB of (fen, best_move).

Parallel across shards (download is I/O bound, argmax is CPU bound).
Resumable: shards whose output parquet already exists are skipped.
Disk-aware: stops submitting if /workspace free space drops below MIN_FREE_GB.
"""
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chessdecoder.dataloader.policy_index import policy_index

HF_TOKEN = os.environ["HF_TOKEN"]
N_SHARDS = 732
BASE_URL = "https://huggingface.co/datasets/Maxlegrec/ChessFENS/resolve/main"
OUT_DIR = "/workspace/ChessDecoder/parquet_files_chessfens"
TMP_DIR = "/tmp/chessfens_raw"
MIN_FREE_GB = 25
WORKERS = int(os.environ.get("CF_WORKERS", "48"))

POLICY = np.array(policy_index, dtype=object)


def free_gb(path="/workspace"):
    return shutil.disk_usage(path).free / 1e9


def convert_shard(idx: int) -> str:
    name = f"dataset-{idx:05d}.parquet"
    out_path = os.path.join(OUT_DIR, f"chessfens-{idx:05d}.parquet")
    if os.path.exists(out_path):
        return f"[{idx}] skip (exists)"

    raw_path = os.path.join(TMP_DIR, name)
    rc = subprocess.run(
        ["curl", "-sfL", "--retry", "4", "--retry-delay", "3",
         "-H", f"Authorization: Bearer {HF_TOKEN}",
         "-o", raw_path, f"{BASE_URL}/{name}"],
        capture_output=True,
    )
    if rc.returncode != 0:
        raise RuntimeError(f"download failed {name}: {rc.stderr.decode()[:200]}")

    pf = pq.ParquetFile(raw_path)
    fens_all, moves_all = [], []
    for g in range(pf.num_row_groups):
        tbl = pf.read_row_group(g, columns=["fen", "policy"])
        n = len(tbl)
        flat = tbl.column("policy").combine_chunks().values.to_numpy(
            zero_copy_only=False)
        pol = flat.reshape(n, flat.size // n)
        amax = pol.argmax(axis=1)
        fens_all.extend(tbl.column("fen").to_pylist())
        moves_all.extend(POLICY[amax].tolist())

    out_tbl = pa.table({
        "fen": pa.array(fens_all, type=pa.string()),
        "best_move": pa.array(moves_all, type=pa.string()),
    })
    tmp_out = out_path + ".tmp"
    pq.write_table(out_tbl, tmp_out, compression="zstd")
    os.replace(tmp_out, out_path)
    os.remove(raw_path)
    return f"[{idx}] ok rows={len(fens_all)} size={os.path.getsize(out_path)/1e6:.1f}MB"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else N_SHARDS

    todo = [i for i in range(start, end)
            if not os.path.exists(os.path.join(OUT_DIR, f"chessfens-{i:05d}.parquet"))]
    print(f"shards to do: {len(todo)} (of {end-start}) with {WORKERS} workers",
          flush=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(convert_shard, i): i for i in todo}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            elapsed = time.time() - t0
            eta = (len(todo) - done) / max(1e-9, done / elapsed) / 3600
            print(f"{res} | {done}/{len(todo)} free={free_gb():.0f}GB "
                  f"eta={eta:.1f}h", flush=True)
            if free_gb() < MIN_FREE_GB:
                print("ABORT: low disk; cancelling remaining", flush=True)
                for f in futs:
                    f.cancel()
                sys.exit(2)
    print(f"DONE {done} shards in {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()

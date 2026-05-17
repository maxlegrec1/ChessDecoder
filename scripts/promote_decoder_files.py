"""Move size-stable (fully scp'd) decoder parquets from the staging dir into
the training dir, so the encoder loader only ever globs complete files.

A file is "stable" when its size is unchanged across two polls AND it is not
the most-recently-modified file (the one scp is actively writing).
"""
import glob
import os
import time

SRC = "/workspace/ChessDecoder/parquet_files_decoder_tmp"
DST = "/workspace/ChessDecoder/parquet_files_decoder"
POLL = 45

os.makedirs(DST, exist_ok=True)
sizes = {}
while True:
    files = sorted(glob.glob(os.path.join(SRC, "*.parquet")))
    if files:
        newest = max(files, key=os.path.getmtime)
        for f in files:
            try:
                sz = os.path.getsize(f)
            except OSError:
                continue
            if f != newest and sizes.get(f) == sz and sz > 0:
                dst = os.path.join(DST, os.path.basename(f))
                try:
                    os.replace(f, dst)
                    print(f"promoted {os.path.basename(f)} ({sz/1e6:.0f}MB) "
                          f"total={len(os.listdir(DST))}", flush=True)
                except OSError as e:
                    print(f"skip {f}: {e}", flush=True)
            sizes[f] = sz
    # Stop once scp is done and staging is drained.
    if not files and not os.path.exists("/workspace/ChessDecoder/.scp_running"):
        print("staging empty and scp finished; exiting", flush=True)
        break
    time.sleep(POLL)

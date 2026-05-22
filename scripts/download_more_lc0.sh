#!/usr/bin/env bash
# Polite, chronological, contiguous lc0 test91 downloader+converter.
# Continues strictly AFTER the newest file already in parquet_files_decoder/,
# oldest-first, until ~BUDGET_GB of new parquet has been added.
#
#   Source : https://data.lczero.org/files/training_data/test91/  (full archive)
#   Convert: chessdecoder/dataloader/reconstitute_games.py <tar> -> <tar>.parquet
#   Politeness: 1 connection, curl retry/backoff on 429, sleep between files.
#   Footprint: 1 tar + 1 parquet at a time (tar deleted after convert).
set -u
cd /workspace/ChessDecoder
BASE="https://data.lczero.org/files/training_data/test91/"
OUT="parquet_files_decoder"
WORK="lc0_work"
BUDGET_GB="${1:-100}"
SLEEP_BETWEEN="${2:-3}"
mkdir -p "$WORK"
LOG=logs/download_more_lc0.log; mkdir -p logs

last=$(ls "$OUT" | grep -oE '[0-9]{8}-[0-9]{4}' | sort | tail -1)
echo "[$(date +%H:%M:%S)] resuming AFTER $last ; budget=${BUDGET_GB}GB" | tee -a "$LOG"

mapfile -t FILES < <(curl -s --max-time 60 "$BASE" \
  | grep -oE 'training-run2-test91-[0-9]{8}-[0-9]{4}\.tar' | sort -u \
  | awk -v last="$last" -F'test91-|\\.tar' '{ if ($2 > last) print }')
echo "[$(date +%H:%M:%S)] ${#FILES[@]} candidate files after $last" | tee -a "$LOG"

added_bytes=0
budget_bytes=$(( BUDGET_GB * 1024 * 1024 * 1024 ))
t0=$(date +%s); n=0
for f in "${FILES[@]}"; do
  name="${f%.tar}"
  pq="$OUT/${name}.parquet"
  if [ -f "$pq" ]; then echo "[skip] $name (exists)" | tee -a "$LOG"; continue; fi
  (( added_bytes >= budget_bytes )) && { echo "[$(date +%H:%M:%S)] budget reached" | tee -a "$LOG"; break; }

  tar="$WORK/$f"
  ds=$(date +%s)
  # polite download: 1 conn, resume, exponential-ish retry covering 429
  curl -sS --fail --location --retry 12 --retry-delay 8 --retry-all-errors \
       -C - -o "$tar" "${BASE}${f}"
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$tar" ]; then
    echo "[warn] download failed rc=$rc for $f (skipping)" | tee -a "$LOG"
    rm -f "$tar"; sleep "$SLEEP_BETWEEN"; continue
  fi
  de=$(date +%s)
  uv run python chessdecoder/dataloader/reconstitute_games.py "$tar" >/dev/null 2>&1
  cv=$(date +%s)
  src="${tar%.tar}.parquet"
  if [ -f "$src" ]; then
    sz=$(stat -c%s "$src"); mv "$src" "$pq"; added_bytes=$(( added_bytes + sz ))
  else
    echo "[warn] convert produced no parquet for $name" | tee -a "$LOG"; rm -f "$tar"; continue
  fi
  rm -f "$tar"
  n=$(( n + 1 ))
  el=$(( $(date +%s) - t0 ))
  rate=$(awk -v b=$added_bytes -v e=$el 'BEGIN{printf "%.3f", (e>0)?b/1073741824/(e/3600):0}')
  remain_gb=$(awk -v a=$added_bytes -v B=$budget_bytes 'BEGIN{printf "%.1f",(B-a)/1073741824}')
  eta_h=$(awk -v r=$rate -v rem=$remain_gb 'BEGIN{printf "%.1f",(r>0)?rem/r:0}')
  printf '[%s] #%d %s | dl=%ss conv=%ss pq=%dMB | cum=%.1fGB rate=%.2fGB/h ETA~%sh\n' \
    "$(date +%H:%M:%S)" "$n" "$name" "$((de-ds))" "$((cv-de))" "$((sz/1048576))" \
    "$(awk -v a=$added_bytes 'BEGIN{print a/1073741824}')" "$rate" "$eta_h" | tee -a "$LOG"
  sleep "$SLEEP_BETWEEN"
done
echo "[$(date +%H:%M:%S)] DONE: added $n files, $(awk -v a=$added_bytes 'BEGIN{printf "%.1f",a/1073741824}')GB" | tee -a "$LOG"

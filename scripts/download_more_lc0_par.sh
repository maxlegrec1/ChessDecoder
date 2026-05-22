#!/usr/bin/env bash
# Parallel chronological lc0 test91 downloader+converter.
#  - downloads SEQUENTIALLY (1 polite connection, 429 backoff)
#  - converts up to CONC files concurrently, each bounded to RECON_WORKERS
#    cores (CONC*RECON_WORKERS <= 224) so it never oversubscribes
#  - tar deleted right after its conversion; budget stop on added parquet GB
#  - logs per-file timing + system RAM each file
set -u
cd /workspace/ChessDecoder
BASE="https://data.lczero.org/files/training_data/test91/"
OUT="parquet_files_decoder"; WORK="lc0_work"
BUDGET_GB="${1:-100}"; CONC="${2:-5}"; export RECON_WORKERS="${3:-44}"
mkdir -p "$WORK" logs; SZF="$WORK/.sizes"; : > "$SZF"
LOG=logs/download_more_lc0.log

last=$(ls "$OUT" | grep -oE '[0-9]{8}-[0-9]{4}' | sort | tail -1)
echo "[$(date +%H:%M:%S)] PAR resume AFTER $last | budget=${BUDGET_GB}GB conc=$CONC recon_workers=$RECON_WORKERS" | tee -a "$LOG"
mapfile -t FILES < <(curl -s --max-time 60 "$BASE" \
  | grep -oE 'training-run2-test91-[0-9]{8}-[0-9]{4}\.tar' | sort -u \
  | awk -v last="$last" -F'test91-|\\.tar' '{ if ($2 > last) print }')
echo "[$(date +%H:%M:%S)] ${#FILES[@]} candidate files" | tee -a "$LOG"

budget_bytes=$(( BUDGET_GB * 1073741824 )); t0=$(date +%s); n=0
declare -a PIDS=()
reap(){ local p new=(); for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -0 "$p" 2>/dev/null && new+=("$p"); done; PIDS=(${new[@]+"${new[@]}"}); }

convert(){ # $1=tar $2=name $3=pq
  RECON_WORKERS="$RECON_WORKERS" nice -n 19 ionice -c3 \
    uv run python chessdecoder/dataloader/reconstitute_games.py "$1" >/dev/null 2>&1
  local src="${1%.tar}.parquet"
  if [ -f "$src" ]; then sz=$(stat -c%s "$src"); mv "$src" "$3"; echo "$sz" >> "$SZF"; fi
  rm -f "$1"
}

for f in "${FILES[@]}"; do
  name="${f%.tar}"; pq="$OUT/${name}.parquet"
  [ -f "$pq" ] && { echo "[skip] $name" >>"$LOG"; continue; }
  cum=$(awk '{s+=$1} END{print s+0}' "$SZF")
  (( cum >= budget_bytes )) && { echo "[$(date +%H:%M:%S)] budget reached (downloads)" | tee -a "$LOG"; break; }
  reap; while [ "${#PIDS[@]}" -ge "$CONC" ]; do sleep 3; reap; done
  tar="$WORK/$f"; ds=$(date +%s)
  curl -sS --fail -L --retry 12 --retry-delay 8 --retry-all-errors -C - -o "$tar" "${BASE}${f}" || \
    { echo "[warn] dl fail $f" | tee -a "$LOG"; rm -f "$tar"; sleep 5; continue; }
  de=$(date +%s)
  ( convert "$tar" "$name" "$pq" ) & PIDS+=("$!")
  n=$(( n+1 ))
  cum=$(awk '{s+=$1} END{print s+0}' "$SZF"); el=$(( $(date +%s)-t0 ))
  ram=$(free -g | awk '/^Mem:/{print $7"G avail/"$2"G"}')
  rate=$(awk -v c=$cum -v e=$el 'BEGIN{printf "%.2f",(e>0)?c/1073741824/(e/3600):0}')
  eta=$(awk -v c=$cum -v B=$budget_bytes -v r=$rate 'BEGIN{printf "%.1f",(r>0)?((B-c)/1073741824)/r:0}')
  printf '[%s] launched #%d %s dl=%ss inflight=%d | done=%.1fGB ram=%s rate=%sGB/h ETA~%sh\n' \
    "$(date +%H:%M:%S)" "$n" "$name" "$((de-ds))" "${#PIDS[@]}" \
    "$(awk -v c=$cum 'BEGIN{print c/1073741824}')" "$ram" "$rate" "$eta" | tee -a "$LOG"
  sleep 2
done
nrem=${#PIDS[@]}; echo "[$(date +%H:%M:%S)] downloads done, waiting on ${nrem} conversions..." | tee -a "$LOG"
wait
fin=$(awk '{s+=$1} END{printf "%.1f",(s+0)/1073741824}' "$SZF")
echo "[$(date +%H:%M:%S)] DONE: launched $n files, added ${fin}GB parquet" | tee -a "$LOG"

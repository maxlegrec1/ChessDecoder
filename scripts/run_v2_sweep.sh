#!/usr/bin/env bash
# 2^(4-1) resolution-IV fractional factorial (8 runs, generator D=A*B*C):
#   A=LR{3e-4,1e-3}  B=opt{adamw,muon}  C=wd{0,0.1}  D=grad_clip{1,10}
# Estimates all 4 main effects cleanly (2-factor interactions aliased).
# 10k steps, held-out val every 500, Stockfish 60 games every 3000.
# Sequential, full GPU, eff batch 2048, compiled + fast-attn.
set -u
cd /workspace/ChessDecoder
mkdir -p sweep_out

# lr optimizer wd grad_clip   (the 8 design points)
RUNS=(
  "3e-4 adamw 0   1"
  "3e-4 adamw 0.1 10"
  "3e-4 muon  0   10"
  "3e-4 muon  0.1 1"
  "1e-3 adamw 0   10"
  "1e-3 adamw 0.1 1"
  "1e-3 muon  0   1"
  "1e-3 muon  0.1 10"
)
STEPS=10000
VAL_EVERY=500
SF_EVERY=3000
SF_GAMES=60

pytag() { uv run python -c "print(f'lr{float('$1'):g}_$2_wd{float('$3'):g}_gc{float('$4'):g}')"; }

for r in "${RUNS[@]}"; do
  read -r lr opt wd gc <<< "$r"
  tag="$(pytag "$lr" "$opt" "$wd" "$gc")"
  csv="sweep_out/${tag}.csv"
  if [ -f "$csv" ] && tail -n1 "$csv" | cut -d, -f1 | grep -qx "$STEPS"; then
    echo "skip $tag (done)"; continue
  fi
  echo "=== run $tag ($(date +%H:%M:%S)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/v2_hp_sweep.py \
    --lr "$lr" --wd "$wd" --grad-clip "$gc" --optimizer "$opt" --seed 0 \
    --steps "$STEPS" --val-every "$VAL_EVERY" --micro-batch 2048 --accum 1 \
    --warmup 400 --val-samples 200000 \
    --sf-every "$SF_EVERY" --sf-games "$SF_GAMES" \
    > "sweep_out/${tag}.log" 2>&1
  echo "=== done $tag ($(date +%H:%M:%S)) ==="
done
echo "PHASE1_SWEEP_COMPLETE"

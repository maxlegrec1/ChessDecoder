#!/usr/bin/env bash
# Download the Lichess puzzle database (CC0, ~100MB zst → ~800MB csv, ~4.9M puzzles).
#
# Usage:  bash scripts/download_puzzles.sh [DEST_DIR]
#         DEST_DIR defaults to ./data
#
# Resulting file:  $DEST_DIR/lichess_db_puzzle.csv
# Columns: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

set -euo pipefail

DEST_DIR="${1:-data}"
URL="https://database.lichess.org/lichess_db_puzzle.csv.zst"
OUT="$DEST_DIR/lichess_db_puzzle.csv"
TMP="$OUT.zst"

mkdir -p "$DEST_DIR"

if [ -f "$OUT" ]; then
    echo "Already present: $OUT ($(wc -l < "$OUT") lines)"
    exit 0
fi

echo "Fetching $URL ..."
curl --fail --location --progress-bar "$URL" -o "$TMP"

if command -v zstd >/dev/null 2>&1; then
    echo "Decompressing via zstd ..."
    zstd -d --rm "$TMP" -o "$OUT"
else
    echo "zstd not found, decompressing via Python ..."
    uv run python -c "
import sys, zstandard
with open('$TMP','rb') as i, open('$OUT','wb') as o:
    zstandard.ZstdDecompressor().copy_stream(i, o)
"
    rm -f "$TMP"
fi

echo "Done: $OUT ($(wc -l < "$OUT") lines)"

#!/usr/bin/env bash
# Download Lc0 training data (test91) and convert to Parquet for pretraining.
#
# Usage:
#   ./scripts/download_and_convert_pretraining_data.sh [NUM_FILES] [OUTPUT_DIR]
#
# Arguments:
#   NUM_FILES   Number of tar files to download (default: 5)
#   OUTPUT_DIR  Directory for output parquets (default: parquets/)
#
# The tar files come from https://storage.lczero.org/files/training_data/test91/
# Each tar contains gzip-compressed V6 training records that reconstitute_games.py
# converts into Parquet files with columns: fen, played_move, best_move, game_id, etc.

set -euo pipefail

NUM_FILES="${1:-5}"
OUTPUT_DIR="${2:-parquets}"
TAR_DIR="lc0_tars"
BASE_URL="https://storage.lczero.org/files/training_data/test91/"

mkdir -p "$TAR_DIR" "$OUTPUT_DIR"

echo "=========================================="
echo " Lc0 Pretraining Data Downloader"
echo "=========================================="
echo "Downloading $NUM_FILES tar files from test91..."
echo "Tar cache:  $TAR_DIR/"
echo "Output dir: $OUTPUT_DIR/"
echo ""

# Fetch the file listing and pick the N most recent tar files
FILE_LIST=$(curl -s "$BASE_URL" \
    | grep -oP 'href="\Ktraining\.[^"]+\.tar(?=")' \
    | tail -n "$NUM_FILES")

if [ -z "$FILE_LIST" ]; then
    echo "ERROR: Could not fetch file listing from $BASE_URL"
    echo "Check your network connection and try again."
    exit 1
fi

echo "Files to download:"
echo "$FILE_LIST" | while read -r f; do echo "  - $f"; done
echo ""

# Download
for FILE in $FILE_LIST; do
    DEST="$TAR_DIR/$FILE"
    if [ -f "$DEST" ]; then
        echo "[skip] $FILE (already downloaded)"
    else
        echo "[download] $FILE ..."
        curl -# -o "$DEST" "${BASE_URL}${FILE}"
    fi
done

echo ""
echo "=========================================="
echo " Converting tar files to Parquet"
echo "=========================================="

for FILE in $FILE_LIST; do
    TAR="$TAR_DIR/$FILE"
    BASENAME=$(basename "$TAR" .tar)
    PARQUET="$OUTPUT_DIR/${BASENAME}.parquet"
    if [ -f "$PARQUET" ]; then
        echo "[skip] $PARQUET (already exists)"
    else
        echo "[convert] $TAR -> $PARQUET"
        uv run python reconstitute_games.py "$TAR"
        # reconstitute_games.py writes output next to the tar; move it
        SRC_PARQUET="${TAR%.tar}.parquet"
        if [ -f "$SRC_PARQUET" ]; then
            mv "$SRC_PARQUET" "$PARQUET"
        fi
    fi
done

echo ""
echo "Done! Parquet files are in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/*.parquet 2>/dev/null || echo "(no parquet files found)"

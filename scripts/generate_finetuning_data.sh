#!/usr/bin/env bash
# Generate finetuning data by running MCTS variations on pretraining parquets.
#
# Usage:
#   ./scripts/generate_finetuning_data.sh [ENGINE_PATH] [PARQUET_DIR] [OUTPUT_DIR]
#
# Arguments:
#   ENGINE_PATH  Path to TensorRT engine file (default: model_dynamic_leela.trt)
#   PARQUET_DIR  Directory with pretraining parquets (default: parquets/)
#   OUTPUT_DIR   Directory for enriched parquets (default: parquets_variations/)
#
# This script wraps scripts/generate_variations.py, which reads existing parquets,
# runs Leela MCTS on each position, and writes enriched parquets with variation data.
#
# Prerequisites:
#   - A TensorRT engine file (exported from a trained model)
#   - The C++ inference extension must be built (happens automatically with `uv sync`)
#   - Pretraining parquets must already exist (see download_and_convert_pretraining_data.sh)

set -euo pipefail

ENGINE_PATH="${1:-model_dynamic_leela.trt}"
PARQUET_DIR="${2:-parquets}"
OUTPUT_DIR="${3:-parquets_variations}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " Finetuning Data Generator"
echo "=========================================="
echo "Engine:     $ENGINE_PATH"
echo "Input dir:  $PARQUET_DIR/"
echo "Output dir: $OUTPUT_DIR/"
echo ""

if [ ! -f "$ENGINE_PATH" ]; then
    echo "ERROR: Engine file not found: $ENGINE_PATH"
    echo "You need a TensorRT engine file to run MCTS."
    exit 1
fi

PARQUET_COUNT=$(find "$PARQUET_DIR" -name "*.parquet" | wc -l)
if [ "$PARQUET_COUNT" -eq 0 ]; then
    echo "ERROR: No parquet files found in $PARQUET_DIR/"
    echo "Run download_and_convert_pretraining_data.sh first."
    exit 1
fi

echo "Found $PARQUET_COUNT parquet file(s) to process."
echo ""

uv run python scripts/generate_variations.py \
    --parquet-dir "$PARQUET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --simulations 600 \
    --max-variations 5 \
    --max-variation-depth 20 \
    --engine-path "$ENGINE_PATH" \
    --parallel-trees 128

echo ""
echo "Done! Enriched parquets are in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/*.parquet 2>/dev/null || echo "(no parquet files found)"

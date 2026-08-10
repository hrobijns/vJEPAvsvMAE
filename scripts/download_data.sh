#!/usr/bin/env bash
# Download the three experiment datasets (train + valid splits, ~525 GB total)
# Usage: bash scripts/download_data.sh /workspace/data
set -euo pipefail

BASE_PATH="${1:?usage: download_data.sh <base_path>}"

for dataset in active_matter shear_flow rayleigh_benard; do
    for split in train valid; do
        echo "=== downloading $dataset/$split ==="
        # --no-parallel: the parallel path needs curl >= 7.66, older than some
        # cluster login nodes provide
        uv run the-well-download --base-path "$BASE_PATH" --dataset "$dataset" --split "$split" --no-parallel
    done
done

echo "done. disk usage:"
du -sh "$BASE_PATH"/datasets/*

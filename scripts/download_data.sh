#!/usr/bin/env bash
# Download the three experiment datasets (train + valid splits by default,
# ~525 GB total; ~260 GB for train only).
# Usage: bash scripts/download_data.sh /workspace/data ["train valid"|"train"]
#
# The probing suite (scripts/analyze_*.py, rollout_*.py) only ever reads the
# train split (src/train.py does too — valid is unused end-to-end, see
# docs/OVERVIEW.md's "No held-out validation loss" caveat), so pass "train"
# to roughly halve the download if you're not planning to add a validation
# eval loop.
set -euo pipefail

BASE_PATH="${1:?usage: download_data.sh <base_path> [splits]}"
SPLITS="${2:-train valid}"

# --no-parallel is only needed on hosts with curl < 7.66 (some cluster login
# nodes); check `curl --version` and drop it below if this host has a newer one.
PARALLEL_FLAG="--no-parallel"

for dataset in active_matter shear_flow rayleigh_benard; do
    for split in $SPLITS; do
        echo "=== downloading $dataset/$split ==="
        uv run the-well-download --base-path "$BASE_PATH" --dataset "$dataset" --split "$split" $PARALLEL_FLAG
    done
done

echo "done. disk usage:"
du -sh "$BASE_PATH"/datasets/*

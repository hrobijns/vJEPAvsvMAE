#!/usr/bin/env bash
# Usage: bash scripts/download_data.sh DATA_ROOT DATASET ["train valid test"]
set -euo pipefail
base_path="${1:?provide a data root}"
dataset="${2:?provide rayleigh_benard, active_matter, or shear_flow}"
splits="${3:-train valid test}"
case "$dataset" in rayleigh_benard|active_matter|shear_flow) ;; *) echo "unsupported dataset: $dataset" >&2; exit 1;; esac
for split in $splits; do
    case "$split" in train|valid|test) ;; *) echo "unsupported split: $split" >&2; exit 1;; esac
    uv run --locked the-well-download --base-path "$base_path" --dataset "$dataset" --split "$split"
done

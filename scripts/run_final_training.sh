#!/bin/bash
# Final, frozen-encoder training: 3 fresh seeds (1, 2, 3) per (dataset,
# objective) pair, at the LR frozen by scripts/run_lr_minisweep.sh
# (configs/tuned_lr.json) -- run that first. Seeds 1/2/3 are independent of
# the seed-0 LR-tuning pilot runs, so no run is double-counted between LR
# selection and the final 3-seed estimate.
#
# Usage:
#   scripts/run_final_training.sh                    # all 6 pairs, sequential
#   scripts/run_final_training.sh active_matter jepa  # one pair (for running
#                                                      # pairs concurrently
#                                                      # across separate pods)
set -e
cd /workspace/vJEPAvsvMAE
source .venv/bin/activate
DATA_ROOT="${DATA_ROOT:-/workspace/data}"

LR_TABLE="configs/tuned_lr.json"
if [ ! -f "$LR_TABLE" ]; then
  echo "missing $LR_TABLE -- run scripts/run_lr_minisweep.sh first" >&2
  exit 1
fi

SEEDS=(1 2 3)
DATASETS=(active_matter shear_flow rayleigh_benard)
OBJECTIVES=(jepa mae)
if [ -n "$1" ] && [ -n "$2" ]; then
  DATASETS=("$1")
  OBJECTIVES=("$2")
fi

for ds in "${DATASETS[@]}"; do
  for obj in "${OBJECTIVES[@]}"; do
    lr=$(python3 -c "import json; print(json.load(open('$LR_TABLE'))['$ds']['$obj'])")
    for seed in "${SEEDS[@]}"; do
      echo "=== final training: $ds $obj seed=$seed lr=$lr ==="
      python -m src.train --config configs/${ds}_${obj}.yaml \
        --data-root "$DATA_ROOT" --seed $seed --lr $lr --no-wandb
    done
  done
done
echo "FINAL TRAINING DONE"

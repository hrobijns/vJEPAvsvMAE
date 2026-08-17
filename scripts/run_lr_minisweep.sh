#!/bin/bash
# LR mini-sweep: pilot runs on seed 0 ONLY, used to select each (dataset,
# objective) pair's learning rate by validation loss (never test-split or
# probing performance). The winning LR is frozen and recorded in
# configs/tuned_lr.json for scripts/run_final_training.sh's fresh seeds
# (1, 2, 3) to reuse -- these seed-0 pilot runs are NOT one of the three
# final seeds (see the plan's LR selection protocol note).
#
# Usage:
#   scripts/run_lr_minisweep.sh                    # all 6 pairs, sequential
#   scripts/run_lr_minisweep.sh active_matter jepa  # one pair (for running
#                                                    # pairs concurrently across
#                                                    # separate pods)
set -e
cd /workspace/vJEPAvsvMAE
source .venv/bin/activate
mkdir -p runs_lrsweep configs

SWEEP_STEPS=8000
DATA_ROOT="${DATA_ROOT:-/workspace/data}"

# Base LRs -- keep in sync with scripts/gen_configs.py's OBJECTIVES dict.
declare -A BASE_LR=( [jepa]=0.0001 [mae]=0.00005 )
MULTIPLIERS=(0.5 1 2)

DATASETS=(active_matter shear_flow rayleigh_benard)
OBJECTIVES=(jepa mae)
if [ -n "$1" ] && [ -n "$2" ]; then
  DATASETS=("$1")
  OBJECTIVES=("$2")
fi

for ds in "${DATASETS[@]}"; do
  for obj in "${OBJECTIVES[@]}"; do
    base=${BASE_LR[$obj]}
    candidates=()
    for mult in "${MULTIPLIERS[@]}"; do
      lr=$(python3 -c "print(${base} * ${mult})")
      out_root="runs_lrsweep/${ds}_${obj}/lr_${mult}x"
      run_dir="${out_root}/${ds}_${obj}_seed0"
      echo "=== LR mini-sweep: $ds $obj lr=$lr (${mult}x) ==="
      python -m src.train --config configs/${ds}_${obj}.yaml \
        --data-root "$DATA_ROOT" --seed 0 --steps $SWEEP_STEPS --lr $lr \
        --out "$out_root" --no-wandb
      candidates+=("${lr}:${run_dir}")
    done

    echo "=== picking LR for $ds $obj ==="
    python scripts/pick_lr.py --dataset "$ds" --objective "$obj" \
      --candidates "${candidates[@]}"
  done
done
echo "LR MINI-SWEEP DONE"

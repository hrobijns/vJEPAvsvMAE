#!/bin/bash
# One-time prerequisite for scripts/rollout_assessment.py: trains the post-hoc
# latent dynamics predictor + small decoder (src/objectives/rollout_heads.py)
# for all 3 datasets x 2 objectives, on top of the already-committed frozen
# encoders in checkpoints/. Re-run scripts/run_sweep.sh (or
# scripts/rollout_assessment.py directly) afterward.
set -e
cd /workspace/vJEPAvsvMAE
source .venv/bin/activate

for ds in active_matter shear_flow rayleigh_benard; do
  for obj in jepa mae; do
    echo "=== training rollout heads: $ds $obj ==="
    python scripts/train_rollout_heads.py \
      --config configs/rollout_heads_${ds}_${obj}.yaml --data-root /workspace/data
  done
done

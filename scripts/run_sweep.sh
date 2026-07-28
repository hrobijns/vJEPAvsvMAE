#!/bin/bash
set -e
cd /workspace/vJEPAvsvMAE
source .venv/bin/activate
mkdir -p sweep_results

for ds in active_matter shear_flow rayleigh_benard; do
  jepa_ckpt=runs/${ds}_jepa/encoder_100pct.pt
  mae_ckpt=runs/${ds}_mae/encoder_100pct.pt
  jepa_latest=runs/${ds}_jepa/latest.pt
  mae_latest=runs/${ds}_mae/latest.pt

  echo "=== $ds: pooled (broadened targets) ===" | tee -a sweep_results/progress.log
  python scripts/analyze_encoders.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets 3 --horizon 16 \
    --out sweep_results/${ds}_pooled.json > sweep_results/${ds}_pooled.log 2>&1

  echo "=== $ds: nonpooled (broadened targets, linear only) ===" | tee -a sweep_results/progress.log
  python scripts/analyze_encoders_local.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets 3 --horizon 16 --token-max-clips 64 --skip-mlp \
    --out sweep_results/${ds}_nonpooled.json > sweep_results/${ds}_nonpooled.log 2>&1

  echo "=== $ds: regime probing ===" | tee -a sweep_results/progress.log
  python scripts/analyze_regime.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets 3 \
    --out sweep_results/${ds}_regime.json > sweep_results/${ds}_regime.log 2>&1

  echo "=== $ds: rollout / sliding-window forecast probe ===" | tee -a sweep_results/progress.log
  python scripts/rollout_probe.py --jepa-ckpt $jepa_latest --mae-ckpt $mae_latest \
    --data-root /workspace/data --dataset $ds --n-offsets 3 --n-context-groups 3 \
    --out sweep_results/${ds}_rollout.json > sweep_results/${ds}_rollout.log 2>&1

  echo "=== $ds DONE ===" | tee -a sweep_results/progress.log
done
echo "ALL DONE" | tee -a sweep_results/progress.log

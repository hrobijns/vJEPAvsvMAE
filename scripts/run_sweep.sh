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

  echo "=== $ds: autoregressive latent rollout assessment ===" | tee -a sweep_results/progress.log
  python scripts/rollout_assessment.py --dataset $ds \
    --data-root /workspace/data --n-offsets 3 --n-steps 8 \
    --jepa-encoder-ckpt $jepa_ckpt --mae-encoder-ckpt $mae_ckpt --heads-dir runs \
    --out sweep_results/${ds}_rollout_assessment.json > sweep_results/${ds}_rollout_assessment.log 2>&1

  echo "=== $ds: representation stability under input noise ===" | tee -a sweep_results/progress.log
  python scripts/analyze_noise_robustness.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets 3 \
    --noise-stds 0 0.1 0.25 0.5 1.0 2.0 \
    --out sweep_results/${ds}_noise_robustness.json > sweep_results/${ds}_noise_robustness.log 2>&1

  echo "=== $ds DONE ===" | tee -a sweep_results/progress.log
done
echo "ALL DONE" | tee -a sweep_results/progress.log

#!/bin/bash
set -e
cd /workspace/vJEPAvsvMAE
source .venv/bin/activate
mkdir -p sweep_results

# shear_flow (256x512) and rayleigh_benard (512x128) clips are 2x/1.3x
# active_matter's raw size; build_dataset()'s contemporaneous_targets() call
# materializes several full-size intermediate tensors (Okubo-Weiss alone
# needs 4) on the ENTIRE stacked clips tensor at once, which OOM'd the
# container's ~175GB memory limit at n_offsets=3 on these larger datasets.
# n_offsets=1 keeps memory safely under budget; regime probing doesn't hit
# this path (no derived-quantity computation) so it keeps n_offsets=3.
declare -A CLIP_OFFSETS=( [shear_flow]=1 [rayleigh_benard]=1 )

for ds in shear_flow rayleigh_benard; do
  jepa_ckpt=runs/${ds}_jepa/encoder_100pct.pt
  mae_ckpt=runs/${ds}_mae/encoder_100pct.pt
  jepa_latest=runs/${ds}_jepa/latest.pt
  mae_latest=runs/${ds}_mae/latest.pt
  n_off=${CLIP_OFFSETS[$ds]}

  echo "=== $ds: pooled (broadened targets, n_offsets=$n_off) ===" | tee -a sweep_results/progress.log
  python scripts/analyze_encoders.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets $n_off --horizon 16 \
    --out sweep_results/${ds}_pooled.json > sweep_results/${ds}_pooled.log 2>&1

  echo "=== $ds: nonpooled (broadened targets, linear only, n_offsets=$n_off) ===" | tee -a sweep_results/progress.log
  python scripts/analyze_encoders_local.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets $n_off --horizon 16 --token-max-clips 64 --skip-mlp \
    --out sweep_results/${ds}_nonpooled.json > sweep_results/${ds}_nonpooled.log 2>&1

  echo "=== $ds: regime probing ===" | tee -a sweep_results/progress.log
  python scripts/analyze_regime.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets 3 \
    --out sweep_results/${ds}_regime.json > sweep_results/${ds}_regime.log 2>&1

  echo "=== $ds: rollout / sliding-window forecast probe (n_offsets=$n_off) ===" | tee -a sweep_results/progress.log
  python scripts/rollout_probe.py --jepa-ckpt $jepa_latest --mae-ckpt $mae_latest \
    --data-root /workspace/data --dataset $ds --n-offsets $n_off --n-context-groups 3 \
    --out sweep_results/${ds}_rollout.json > sweep_results/${ds}_rollout.log 2>&1

  echo "=== $ds DONE ===" | tee -a sweep_results/progress.log
done
echo "ALL DONE" | tee -a sweep_results/progress.log

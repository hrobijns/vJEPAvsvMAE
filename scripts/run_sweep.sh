#!/bin/bash
set -e
cd /workspace/vJEPAvsvMAE
source .venv/bin/activate
mkdir -p sweep_results

for ds in active_matter shear_flow rayleigh_benard; do
  jepa_ckpt=checkpoints/${ds}_jepa.pt
  mae_ckpt=checkpoints/${ds}_mae.pt

  # build_dataset() (used by analyze_encoders.py / analyze_encoders_local.py)
  # materializes n_offsets x n_trajectories clips AND several full-resolution
  # intermediate tensors (curl, gradients, laplacian, ...) all at once.
  # active_matter (175 traj) fits at n_offsets=3 (~12GB), but shear_flow
  # (896 traj) and rayleigh_benard (comparably large) can exceed a
  # constrained pod's memory even at n_offsets=1 — cap trajectory COUNT
  # directly via --max-traj for the two big datasets. Raise/drop these if
  # running on a pod with more/less headroom than this one's 29GB cgroup cap.
  n_offsets=3
  max_traj_flag=""
  if [ "$ds" != "active_matter" ]; then
    n_offsets=1
    max_traj_flag="--max-traj 300"
  fi

  echo "=== $ds: pooled (broadened targets, n_offsets=$n_offsets $max_traj_flag) ===" | tee -a sweep_results/progress.log
  python scripts/analyze_encoders.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets $n_offsets --horizon 16 $max_traj_flag \
    --out sweep_results/${ds}_pooled.json > sweep_results/${ds}_pooled.log 2>&1

  echo "=== $ds: nonpooled (broadened targets, linear + per-token MLP, n_offsets=$n_offsets $max_traj_flag) ===" | tee -a sweep_results/progress.log
  python scripts/analyze_encoders_local.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets $n_offsets --horizon 16 $max_traj_flag \
    --token-max-clips 64 --token-mlp-max-clips 16 \
    --out sweep_results/${ds}_nonpooled.json > sweep_results/${ds}_nonpooled.log 2>&1

  # analyze_regime.py and the rollout probes (rollout_probe.py /
  # rollout_assessment.py) intentionally skipped for this run, per request —
  # scope narrowed to the linear-probe suite (new PDE targets, per-token
  # linear + MLP, full noise layer x token sweep).

  echo "=== $ds: representation stability under input noise (pooled + token, full layer sweep) ===" | tee -a sweep_results/progress.log
  python scripts/analyze_noise_robustness.py --checkpoints $jepa_ckpt $mae_ckpt \
    --data-root /workspace/data --dataset $ds --n-offsets 3 \
    --noise-stds 0 0.1 0.25 0.5 1.0 2.0 --token-max-samples 16 \
    --out sweep_results/${ds}_noise_robustness.json > sweep_results/${ds}_noise_robustness.log 2>&1

  echo "=== $ds DONE ===" | tee -a sweep_results/progress.log
done
echo "ALL DONE" | tee -a sweep_results/progress.log

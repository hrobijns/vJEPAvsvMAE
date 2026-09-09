#!/usr/bin/env bash
# Supply account, partition, array range, and log path to sbatch; pass the
# prepared sweep directory first. Optional second argument 4 groups the three
# rates per objective into one job (submit array 0-3); default 12 runs one per job.
#SBATCH --job-name=well-lr-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
stride="${2:-12}"
[[ "$stride" == 4 || "$stride" == 12 ]] || { echo "group stride must be 4 or 12" >&2; exit 2; }
for (( index=SLURM_ARRAY_TASK_ID; index<12; index+=stride )); do
  .venv/bin/python -u -B scripts/lr_sweep.py run --output "${1:?provide sweep output}" \
    --index "$index" --cache-root "${WELL_CACHE_ROOT:-/tmp/well-training-cache-${UID}}"
done

#!/usr/bin/env bash
# Submit from the repository root with account, partition and output path.
# The single argument is a prepared YAML containing the run directory and seed.
#SBATCH --job-name=well-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --open-mode=append
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
config="${1:?provide a prepared run YAML}"
trainer_pid=""
stopping=0
wait_interrupted=0
request_stop() {
  stopping=1
  wait_interrupted=1
  if [[ -n "$trainer_pid" ]]; then
    kill -USR1 "$trainer_pid" 2>/dev/null || true
  fi
}
trap request_stop USR1

# Scratch may be private to this allocation, including after a requeue.
local_base=$(.venv/bin/python -u -B - "$config" "${WELL_CACHE_ROOT:-/tmp/well-training-cache-${UID}}" <<'PY'
import sys
from pathlib import Path
import yaml
from scripts.stage_training_cache import stage
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
data = cfg["data"]
if not data.get("memmap", False):
    raise ValueError("Slurm training requires a prepared full-trajectory memmap cache")
print(stage(data["base_path"], data["dataset_name"], sys.argv[2])["base_path"])
PY
)
if (( stopping )); then
  echo "Time-limit signal arrived during staging; no updates and no automatic requeue" >&2
  exit 1
fi
echo "Training allocation: job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) data=${local_base}"
.venv/bin/python -u -B -m src.train --config "$config" --data-root "$local_base" --no-wandb &
trainer_pid=$!
status=0
while true; do
  wait_interrupted=0
  wait "$trainer_pid" && status=0 || status=$?
  # A signal may interrupt wait just before the trainer exits. Wait again to
  # retrieve its exit status even if the process has already disappeared.
  (( wait_interrupted )) || break
done
if (( status == 75 && stopping )); then
  echo "Trainer saved forward progress; requeuing job ${SLURM_JOB_ID}"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
# Completion, unexpected failures and starts without progress never requeue here.
exit "$status"

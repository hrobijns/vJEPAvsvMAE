#!/usr/bin/env bash
#SBATCH --job-name=physical-probes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
study="$(realpath "${1:?provide the prepared study directory}")"
phase="${2:-run}"
case "$phase" in run|test) ;; *) echo "phase must be run or test" >&2; exit 1;; esac
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vjepa-matplotlib-$UID"
cd "$study/source"
exec .venv/bin/python scripts/probe_sweep.py "$phase" --output "$study" --task "${SLURM_ARRAY_TASK_ID:?use a Slurm job array}"

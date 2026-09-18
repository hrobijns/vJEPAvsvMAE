#!/usr/bin/env bash
#SBATCH --job-name=physical-probes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# One array task per frozen best-validation encoder, e.g. for a four-objective
# single-system study: sbatch --array=0-3 scripts/slurm_probe.sbatch STUDY run
# Test scoring refuses to start until every encoder has a sealed probe fit.
set -euo pipefail
study="$(realpath "${1:?provide the prepared study directory}")"
phase="${2:-run}"
case "$phase" in run|test) ;; *) echo "phase must be run or test" >&2; exit 1;; esac
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TMPDIR="$study/tmp"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$study/source"
exec "${PYTHON:-python}" scripts/probe_sweep.py "$phase" --output "$study" --task "${SLURM_ARRAY_TASK_ID:?use a Slurm job array}"

#!/bin/bash
# CSD3 (Wilkes3) training job. Usage:
#   sbatch csd3/slurm_train.sh configs/active_matter_jepa.yaml
# Re-submitting the same command resumes from runs/<name>/latest.pt, so a job
# killed at the walltime just needs another sbatch. Set your account below.
#SBATCH -J vjepa-vmae
#SBATCH -A CHANGEME-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

CONFIG="${1:?usage: sbatch csd3/slurm_train.sh <config.yaml>}"

module purge
module load rhel8/default-amp

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

# Compute nodes may not reach the internet: log offline, sync later from a
# login node with:  uv run wandb sync --sync-all
export WANDB_MODE="${WANDB_MODE:-offline}"

DATA_ROOT="${DATA_ROOT:-/rds/user/$USER/hpc-work/well_data}"

nvidia-smi -L
uv run python -m src.train --config "$CONFIG" --data-root "$DATA_ROOT"

#!/usr/bin/env bash
# Bootstrap a RunPod pod for training. Assumes:
#   - a PyTorch CUDA base image (e.g. runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04)
#   - a network volume mounted at /workspace (holds data + repo + runs)
# Usage: bash runpod/setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# uv for fast, reproducible installs
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync the locked environment (resolves linux wheels: torch>=2.4 + CUDA)
uv sync

# W&B: export WANDB_API_KEY before running, or `uv run wandb login`
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "NOTE: WANDB_API_KEY not set — either export it or run 'uv run wandb login'"
fi

# tmux + htop for long-running sessions
if command -v apt-get >/dev/null; then
    apt-get update -qq && apt-get install -y -qq tmux htop >/dev/null || true
fi

uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "setup complete. next: bash scripts/download_data.sh /workspace/data"

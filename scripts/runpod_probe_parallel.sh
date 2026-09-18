#!/usr/bin/env bash
# Run the four independent encoder fits across the visible GPUs, then score and report.
set -euo pipefail

REPO="${REPO:-/workspace/vJEPAvsvMAE}"
STUDY="${STUDY:-/workspace/rb_half_targets}"
PYTHON="${PYTHON:-python}"
GPU_COUNT="${GPU_COUNT:-2}"
DRIVER="$REPO/scripts/runpod_probe_study.sh"

[ "$GPU_COUNT" -ge 1 ] || { echo "GPU_COUNT must be positive" >&2; exit 2; }
[ -x "$DRIVER" ] || { echo "no probe driver at $DRIVER" >&2; exit 2; }
mkdir -p "$STUDY/logs"

pids=()
tasks=()
for task in 0 1 2 3; do
  gpu=$((task % GPU_COUNT))
  echo "launching encoder task $task on visible GPU $gpu" >&2
  CUDA_VISIBLE_DEVICES="$gpu" TASKS="$task" STUDY="$STUDY" PYTHON="$PYTHON" \
    "$DRIVER" fit >"$STUDY/logs/parallel_fit_${task}.log" 2>&1 &
  pids+=("$!")
  tasks+=("$task")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "encoder task ${tasks[$index]} failed; see $STUDY/logs/parallel_fit_${tasks[$index]}.log" >&2
    failed=1
  fi
done
[ "$failed" -eq 0 ] || exit 1

# One coordinator builds the shared test cache, scores every frozen fit, and renders.
STUDY="$STUDY" PYTHON="$PYTHON" "$DRIVER" test
STUDY="$STUDY" PYTHON="$PYTHON" "$DRIVER" report

#!/usr/bin/env bash
# Run the four independent encoder fits across the visible GPUs, then score and report.
set -euo pipefail

REPO="${REPO:-/workspace/vJEPAvsvMAE}"
STUDY="${STUDY:-/workspace/rb_half_targets}"
PYTHON="${PYTHON:-python}"
GPU_COUNT="${GPU_COUNT:-2}"
SOURCE_STUDY="${SOURCE_STUDY:-/workspace/rb_best_val}"
TARGETS="${TARGETS:-enstrophy convective_flux}"
DRIVER="$REPO/scripts/runpod_probe_study.sh"

[ "$GPU_COUNT" -ge 1 ] || { echo "GPU_COUNT must be positive" >&2; exit 2; }
[ -x "$DRIVER" ] || { echo "no probe driver at $DRIVER" >&2; exit 2; }

pids=()

# Preparation freezes the target subset and source commit. Reuse only sealed,
# immutable caches from the earlier study; features remain commit-bound and
# are extracted independently for each encoder below.
TARGETS="$TARGETS" STUDY="$STUDY" PYTHON="$PYTHON" "$DRIVER" prepare
if [ ! -e "$STUDY/cache" ] && [ -d "$SOURCE_STUDY/cache" ]; then
  cp -al "$SOURCE_STUDY/cache" "$STUDY/cache"
fi
STUDY="$STUDY" PYTHON="$PYTHON" "$DRIVER" cache
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

#!/usr/bin/env bash
# Noninteractive driver for the best-validation probe study on one Runpod pod.
#
# Verified pod layout: repository at /workspace/vJEPAvsvMAE, The Well data at
# /workspace/well, one NVIDIA L4 (23,034 MiB), and shared /workspace storage.
# A Rayleigh–Bénard fit writes about 60 GiB of feature shards per encoder and
# the four fits plus caches peak near 300 GiB before test-time pruning. Only
# ~6 GiB is free on the container root, so every study output, feature cache,
# and scratch directory lives under $STUDY on /workspace; the launcher refuses
# to write a study onto the root filesystem.
#
# Storage and concurrency are bounded explicitly: one GPU task runs at a time
# (the pod has one L4), a free-space preflight runs before every writing stage,
# and PRUNE_FEATURES=1 removes a checkpoint's reproducible feature artifacts
# once its sealed test scores exist.
#
# Usage, entirely environment-configured and never interactive:
#
#   STUDY=/workspace/rb_best_val scripts/runpod_probe_study.sh            # all
#   STUDY=/workspace/rb_best_val scripts/runpod_probe_study.sh fit
#   TASKS="2 3" STUDY=/workspace/rb_best_val scripts/runpod_probe_study.sh fit
#
# Stages: all (default) | prepare | cache | fit | test | report | status.
# Rerunning the same command after a crash, preemption, or pod restart resumes:
# every stage skips sealed artifacts and pending work is recomputed from
# `probe_sweep.py status`. The first failing task aborts with its log path and
# nothing already sealed is recomputed, overwritten, or deleted.
# Preparation freezes a detached source checkout, so $REPO must be committed
# and clean in src, scripts, and configs before the first invocation; later
# stages run that frozen checkout, never $REPO. TASKS restricts fit to an
# explicit subset for multi-GPU sharding; test runs from one coordinator after
# every fit completes so the shared test cache is built exactly once.
set -euo pipefail

REPO="${REPO:-/workspace/vJEPAvsvMAE}"
BASE="${BASE:-/workspace/well}"
STUDY="${STUDY:-/workspace/rb_best_val}"
HANDOFF="${HANDOFF:-$REPO/checkpoints/iclr2027/seed1}"
DATASET="${DATASET:-rayleigh_benard}"
TARGETS="${TARGETS:-}"
OBJECTIVES="${OBJECTIVES:-}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"
REPORT_MIN_FREE_GIB="${REPORT_MIN_FREE_GIB:-1}"
PRUNE_FEATURES="${PRUNE_FEATURES:-0}"
PYTHON="${PYTHON:-python}"
stage="${1:-all}"

case "$stage" in
  all | prepare | cache | fit | test | report | status) ;;
  *)
    echo "stage must be all, prepare, cache, fit, test, report or status" >&2
    exit 2
    ;;
esac

volume="$(dirname "$STUDY")"
[ -d "$volume" ] || { echo "study volume $volume does not exist" >&2; exit 2; }
[ -d "$BASE" ] || { echo "The Well data root $BASE does not exist" >&2; exit 2; }
command -v "$PYTHON" >/dev/null || { echo "no interpreter $PYTHON" >&2; exit 2; }
PYTHON="$(command -v "$PYTHON")"
[ -f "$HANDOFF/manifest.json" ] || { echo "no handoff at $HANDOFF" >&2; exit 2; }

if [ "$(df -P "$volume" | awk 'NR==2{print $1}')" = \
     "$(df -P / | awk 'NR==2{print $1}')" ] && [ "${ALLOW_ROOT_FS:-0}" != 1 ]; then
  echo "$STUDY is on the container root filesystem; use a /workspace path" >&2
  exit 2
fi

export TMPDIR="$volume/.probe_tmp"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export XDG_CACHE_HOME="$TMPDIR/cache"
export TORCH_HOME="$TMPDIR/torch"
export HF_HOME="$TMPDIR/huggingface"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

preflight() {
  local label="$1"
  local required="${2:-$MIN_FREE_GIB}"
  local free
  free="$(df -Pk "$volume" | awk 'NR==2{printf "%d", $4 / 1048576}')"
  echo "[$(date -u +%FT%TZ)] $label: ${free} GiB free on $volume" >&2
  if [ "$free" -lt "$required" ]; then
    echo "refusing to start $label: needs ${required} GiB free on $volume" >&2
    exit 1
  fi
}

# Import failures must surface here, not inside a multi-hour GPU task.
check_imports() {
  local modules="$1"
  "$PYTHON" -c "
import importlib, sys
import torch
if not torch.cuda.is_available():
    sys.exit('no CUDA device visible to ' + sys.executable)
try:
    torch.ones(1, device='cuda').item()
except RuntimeError as error:
    sys.exit(
        f'CUDA kernel preflight failed on {torch.cuda.get_device_name(0)}: {error}'
    )
for name in '$modules'.split():
    importlib.import_module(name)
print(f'{sys.executable}: torch {torch.__version__} on {torch.cuda.get_device_name(0)}')
" >&2 || exit 1
}

# Every stage after preparation runs the frozen source checkout, never $REPO.
sweep() {
  ( cd "$STUDY/source" && "$PYTHON" scripts/probe_sweep.py "$@" )
}

logged() {
  local log="$STUDY/logs/$1.log"
  shift
  echo "[$(date -u +%FT%TZ)] running: $* (log $log)" >&2
  if ! sweep "$@" >>"$log" 2>&1; then
    echo "failed: probe_sweep.py $*; see $log" >&2
    exit 1
  fi
  tail -n 2 "$log" >&2
}

pending() {
  sweep status --output "$STUDY" |
    "$PYTHON" -c "import json,sys; print(*json.load(sys.stdin)['$1'])"
}

tasks_for() {
  if [ -n "${TASKS:-}" ]; then
    echo "$TASKS"
  else
    pending "$1"
  fi
}

# Preparation is complete only with both the frozen study and its checkout.
prepared() {
  [ -f "$STUDY/study.json" ] && [ -f "$STUDY/source/scripts/probe_sweep.py" ]
}

# `all` plots hours after it starts, so its plotting dependency is checked now.
case "$stage" in
  status) ;;
  report | all) check_imports "h5py the_well matplotlib" ;;
  *) check_imports "h5py the_well" ;;
esac

target_args=()
if [ -n "$TARGETS" ]; then
  read -r -a target_args <<<"$TARGETS"
  target_args=(--targets "${target_args[@]}")
fi
objective_args=()
if [ -n "$OBJECTIVES" ]; then
  read -r -a objective_args <<<"$OBJECTIVES"
  objective_args=(--objectives "${objective_args[@]}")
fi
if [ "$stage" = prepare ] || [ "$stage" = all ]; then
  preflight prepare
  if prepared; then
    echo "study already prepared at $STUDY" >&2
  elif [ -e "$STUDY/study.json" ] || [ -e "$STUDY/source" ]; then
    echo "incomplete preparation at $STUDY: needs both study.json and" \
      "source/scripts/probe_sweep.py; remove the directory or restore the" \
      "frozen checkout before retrying" >&2
    exit 1
  else
    ( cd "$REPO" && "$PYTHON" scripts/probe_sweep.py prepare \
        --handoff "$HANDOFF" --base "$BASE" --output "$STUDY" \
        --dataset "$DATASET" "${target_args[@]}" "${objective_args[@]}" )
    prepared || { echo "preparation produced no usable study at $STUDY" >&2; exit 1; }
  fi
fi

prepared || {
  echo "no prepared study at $STUDY: needs study.json and" \
    "source/scripts/probe_sweep.py" >&2
  exit 1
}

if [ "$stage" = status ]; then
  sweep status --output "$STUDY"
  exit 0
fi

if [ "$stage" = cache ] || [ "$stage" = all ]; then
  for split in train valid; do
    preflight "cache $split"
    logged "cache_$split" cache --output "$STUDY" --dataset "$DATASET" --split "$split"
  done
fi

if [ "$stage" = fit ] || [ "$stage" = all ]; then
  fit_tasks="$(tasks_for pending_fits)"
  for task in $fit_tasks; do
    preflight "fit task $task"
    logged "fit_$task" run --output "$STUDY" --task "$task"
  done
fi

if [ "$stage" = test ] || [ "$stage" = all ]; then
  remaining="$(pending pending_fits)"
  if [ -n "$remaining" ]; then
    echo "test scoring is blocked: fits still pending for tasks $remaining" >&2
    exit 1
  fi
  preflight "cache test"
  logged cache_test cache --output "$STUDY" --dataset "$DATASET" --split test
  test_tasks="$(pending pending_tests)"
  for task in $test_tasks; do
    preflight "test task $task"
    logged "test_$task" test --output "$STUDY" --task "$task"
    if [ "$PRUNE_FEATURES" = 1 ]; then
      # Test scores are sealed; features are reproducible from cache + encoder.
      sweep status --output "$STUDY" |
        "$PYTHON" -c "
import json, shutil, sys
for row in json.load(sys.stdin)['encoders']:
    if row['test'] and any(row['features'].values()):
        shutil.rmtree(row['feature_dir'])
        print('pruned', row['feature_dir'])
" >&2
    fi
  done
fi

if [ "$stage" = report ] || [ "$stage" = all ]; then
  preflight report "$REPORT_MIN_FREE_GIB"
  logged report report --output "$STUDY"
fi

sweep status --output "$STUDY" | tee "$STUDY/logs/status.json"

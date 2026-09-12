# JEPA and masked autoencoding on The Well

A controlled comparison of frozen representations learned by matched video
encoders on Rayleigh–Bénard convection, active matter, and shear flow.

Four objectives separate latent versus pixel prediction from current versus
future targets. Every online encoder receives eight frames with 90% tube masking.

| Objective | Context | Prediction target |
|---|---|---|
| `jepa` | Masked frames 0–7 | EMA features at masked positions in 0–7 |
| `mae` | Masked frames 0–7 | Normalized pixels at masked positions in 0–7 |
| `jepa_future` | Masked frames 0–7 | EMA features at every position in 8–15 |
| `mae_future` | Masked frames 0–7 | Normalized pixels at every position in 8–15 |

Future objectives predict the adjacent clip jointly and use no reconstruction
loss on the context. JEPA uses a six-layer predictor and an EMA target encoder;
MAE uses a four-layer decoder and direct pixel targets. Both heads have width
192. The shared ViT encoder has width 384 and 12 blocks, with 2×16×16 patches.

This checkout contains one analysis workflow. Current configurations use the
**whole available trajectory** for training and evaluation (`frame_limit: null`).
Eight-frame training clips overlap with starting times spaced one frame apart.
Future examples pair each context with the immediately following eight frames.
The source determines trajectory length; there is no 200-frame cap. The RB data
checked here have 200 frames, shear flow is documented with 200, and active
matter with 81. The corrected workshop analysis remains the regression baseline,
with its 101-frame evaluation configuration under `configs/workshop/`.

## Setup and data

```bash
uv sync --locked
bash scripts/download_data.sh /path/to/data rayleigh_benard
```

The download wrapper takes one dataset and optional split names, defaulting to
`train valid test`. It downloads no data until invoked. Native data volumes are
substantial; select only the dataset/splits you need. Both `<base>/<dataset>`
and `<base>/datasets/<dataset>` layouts are accepted.

## Training

Checked-in YAML files are the configuration source of truth. The completed
8,000-step seed-0 learning-rate sweeps selected these defaults:

| System | JEPA | MAE | Future JEPA | Future MAE |
| --- | --- | --- | --- | --- |
| Rayleigh–Bénard | `5e-5` | `1e-4` | `2e-4` | `1e-4` |
| Active matter | `2e-4` | `1e-4` | `2e-4` | `1e-4` |
| Shear flow | `2e-4` | `1e-4` | `2e-4` | `1e-4` |

Each rate minimizes the recorded internal validation loss within its completed
pilot grid. All 36 pilots completed using full trajectories, and their
diagnostics were reviewed. Active matter and shear flow selected the largest
tested rates for every objective; rates above the grid were not tested. These
pilots do not establish downstream representation quality or sufficient final
training duration. The pilot recipe and subsequent analysis work are recorded under
[Next experiment stages](#next-experiment-stages).

```bash
uv run --locked python -m src.data.preprocess \
  --base /path/to/data --dataset rayleigh_benard --split train
uv run --locked python -m src.train \
  --config configs/rayleigh_benard_jepa.yaml --data-root /path/to/data \
  --seed 1 --out runs/full_trajectories --no-wandb
```

Repeat with `mae`, `jepa_future`, and `mae_future` configurations and desired
seeds. Replace the dataset name
with `active_matter` or `shear_flow` for the other systems. The trainer also
accepts `--lr`, `--steps`, `--mask-ratio`, and `--out` overrides.

Defaults remain eight-frame inputs, batch size 64, and 100,000 optimizer steps.
The trainer reports available clip counts and planned exposure at startup.
History rows and checkpoints include `data_exposure`: eligible training and
validation clips, planned clips and equivalent passes, and clips processed and
equivalent passes at the recorded step. Processed clips equal completed updates
times batch size; repetitions count, and validation clips do not. Equivalent
passes divide processed clips by the number of eligible training clips, so they
are not counts of unique examples or independently generated simulations.
For future objectives, one context/target pair counts as one input example;
the target does not double the processed-clip count. A trajectory of length `T`
has `T - 7` current windows or `T - 15` future pairs. These give 193 versus 185
examples at 200 frames, and 74 versus 66 at 81 frames. Each objective uses all
its eligible starts; current objectives retain the final eight starts that
cannot supply a future target.

Resume continues these counters from the saved optimizer step. At batch 64,
100,000 steps process 6.4 million clips: about 27.1 passes over RB's 236,425
full-trajectory training windows, compared with 55.6 over the workshop's 115,150.

Both the memmap and HDF5 backends reserve every eighth official-training
trajectory for pretraining validation. Official validation and test splits are
reserved for downstream analysis. Training caches retain the source trajectory
index, channels, normalization, full lengths, and content hashes. Old caches
must be rebuilt; they are not silently accepted or overwritten. Interrupted
preprocessing can resume when its source contract still matches.
Completed caches in the current format already contain full trajectories and
can be reused for current and future objectives without regeneration.

Checkpoints contain their configuration, input shape, and training identity.
`latest.pt` resumes only when source, temporal support, split, model,
optimization, seed, validation/image settings, and source-code hash agree.
The data location and loader worker count may change. Historical checkpoints remain loadable for
analysis but cannot automatically resume a new training run. Reported workshop
models are final 100,000-step endpoints; `encoder_best_val.pt` is an additional
training diagnostic, not the endpoint-selection rule for the comparison.
Encoder checkpoints at 25%, 50%, 75%, and 100% of the planned steps support
learning-curve analysis. The 100,000-step budget is a starting choice; assess
training sufficiency on validation data before changing the comparison budget.
Current and future objectives have distinct resume identities even though their
encoder geometry and learned parameter shapes agree. MAE future images, when
patch normalization is enabled, compare normalized predictions and targets;
they are not rescaled using future patch means or variances.

For training across Slurm allocations, prepare one run YAML with the desired
`seed`, unique `run_name`, shared `out_dir`, full-cache `data.base_path`, and
`data.num_workers: 4`. Keep this configuration and the source code fixed for the
run. Submit from the repository root after creating the log directory:

```bash
mkdir -p outputs/full_training/logs
sbatch --account YOUR_ACCOUNT --partition YOUR_GPU_PARTITION \
  --output=outputs/full_training/logs/%j.log \
  scripts/slurm_train.sbatch /path/to/prepared_run.yaml
```

This wrapper requests one GPU, four CPUs, 64 GiB RAM, and 12 hours. It stages
the cache locally on every allocation, including after a requeue, and disables
W&B; metrics remain in `history.jsonl` and the appended Slurm log. Thirty minutes
before the deadline, Slurm signals the wrapper, which asks the trainer to finish
its current update and any scheduled validation, save, and exit with code 75.
Only this planned stop after forward progress triggers automatic requeue.
Completion exits successfully; unexpected failures and starts without progress
require inspection and manual resubmission. A signal during cache staging also
stops without automatic requeue. Cluster-level preemption/requeue policy still
applies independently.

Continuation restores the optimizer, EMA target where applicable, learning-rate
and EMA schedule position, random states, and next shuffled batch. Each pass
uses its own seeded permutation and drops its incomplete final batch. Worker
prefetch cannot move the saved consumed position. Checkpoints are atomically
replaced at regular saves, every validation, milestones, and planned stops.
The checkpoint records the durable history length; resume removes a later log
tail and repairs applicable encoder exports before proceeding. Missing or short
history is an error. A completed run repairs exports and exits without training.
One trainer at a time may use a run directory. Older `latest.pt` files without
complete continuation state cannot resume under this code; analysis loading is
unchanged. CPU continuation is checked for exact equality; CUDA kernels remain
unconstrained, so bitwise equality across GPU machines is not promised.

## Frozen-encoder analysis

The workflow fits Ridge and MLP probes on official **training** data, selects
all probe settings and one encoder checkpoint per run on **validation**, then
scores the frozen choices on **test**. Inputs remain eight unmasked frames.
Global targets average over the clip; local targets average within sampled
2×16×16 patches. Target starts are explicit offsets **0, 8, 16, 40**, covering
frames 0–7, 8–15, 16–23, and 40–47 relative to the input start.

VRMSE is the default reported metric: RMSE divided by the target's population
standard deviation across evaluated examples, separately for every quantity,
horizon, and global/local setting. R² remains in result rows and can be plotted
with `--metric r2`. This adapts Walrus's variance normalization to scalar
physical quantities; it is not a direct comparison with its full-field scores.
See [the methods contract](docs/METHODS.md) for formulas and selection weights.

All 60 seed-1 encoder candidates are in
[the checkpoint handoff](checkpoints/iclr2027/seed1/README.md). Preparing a sweep
requires committed source changes. It creates a detached source checkout,
freezes the candidate roster, and skips repeated fitting for verified identical
encoder states with the same training configuration. It does not submit jobs.

```bash
STUDY=outputs/physical_probes/iclr2027_seed1
uv run --locked python scripts/probe_sweep.py prepare \
  --handoff checkpoints/iclr2027/seed1 --base /path/to/data --output "$STUDY"
# Use this frozen source for every subsequent stage.
PROBE_PYTHON="$STUDY/source/.venv/bin/python"
PROBE_RUNNER="$STUDY/source/scripts/probe_sweep.py"
for dataset in rayleigh_benard active_matter shear_flow; do
  for split in train valid; do
    "$PROBE_PYTHON" "$PROBE_RUNNER" cache --output "$STUDY" --dataset "$dataset" --split "$split"
  done
done
# The prepared seed-1 roster has 56 distinct candidates: indices 0 through 55.
sbatch --account YOUR_ACCOUNT --partition YOUR_GPU_PARTITION \
  --array=0-55%12 --output="$STUDY/logs/fit_%A_%a.log" \
  scripts/slurm_probe.sbatch "$STUDY" run
# After every candidate completes:
"$PROBE_PYTHON" "$PROBE_RUNNER" collect --output "$STUDY"
for dataset in rayleigh_benard active_matter shear_flow; do
  "$PROBE_PYTHON" "$PROBE_RUNNER" cache --output "$STUDY" --dataset "$dataset" --split test
done
sbatch --account YOUR_ACCOUNT --partition YOUR_GPU_PARTITION \
  --array=0-11%12 --output="$STUDY/logs/test_%A_%a.log" \
  scripts/slurm_probe.sbatch "$STUDY" test
# After all twelve selected checkpoints finish test scoring:
"$PROBE_PYTHON" "$PROBE_RUNNER" report --output "$STUDY"
```

Cache preparation can run independently for different systems/splits. The
launcher requests one GPU, four CPUs, 16 GiB RAM, and 12 hours per candidate.
It does not automatically requeue; completed immutable stages can be reused
when retrying an interrupted candidate. The expected aggregate MLP compute is
about 50 GPU-hours at the full stopping cap. Allow approximately 6–10 hours
with 8–12 L40S GPUs, excluding queueing and initial data preparation; this is
an extrapolation from timing checks, not a completed scientific sweep.

The underlying commands remain available for other checkpoint rosters:
`prepare-cache --split train|valid|test`, `extract-features --split ...`,
`fit-probes`, `select-checkpoints`, `score-probes --selection ...`, `aggregate`,
and `plot`. Feature directories contain separate immutable split artifacts;
`fit-probes` needs no test data. `score-probes` rejects unselected probe fits.
`extract-features --include-noise --split test` and `evaluate-noise` retain the
clean-fit noise workflow for later use; the initial sweep does not run it.

Outputs include separate Ridge/MLP and validation-selected results, global
depth curves, governing-parameter probes, regime/time and position controls,
combined encoder-plus-control Ridge probes, and future persistence baselines.
Checkpoint selection excludes those controls and governing parameters.
Aggregates average physical quantities inside each encoder seed before
calculating between-seed statistics. A single encoder seed has no between-run
standard deviation; MLP ensemble members are not extra encoder seeds.
Undefined metrics remain JSON `null`, and plots retain scores outside 0–1.

The workshop configuration preserves its 101-frame support and original targets
using offsets 0, 16, 40. The current probe-fitting protocol uses train/validation
splits and differs from the historical five-fold procedure described in the
submitted paper. Existing analysis artifacts must be rebuilt for the new
schema; old gaps and results are never silently reinterpreted. Historical
paper files and checkpoints are unchanged.

## Next experiment stages

1. **LR selection is complete for all three systems.** Each system's four
   objectives were swept with seed 0, 8,000 steps, and a 5,000-step warmup.
   JEPA candidates were `5e-5, 1e-4, 2e-4`; MAE candidates were
   `2.5e-5, 5e-5, 1e-4`, shared within each family. Rates were selected by
   minimum recorded internal validation loss during each completed pilot,
   with collapse diagnostics reviewed and candidate scores, selected rates,
   and run locations retained. Test and probe performance were not used.
2. **Seed 1 training is complete:** all twelve system/objective combinations
   reached 100,000 steps. Complete the validation-selected Ridge/MLP baseline
   and clean test comparison using the retained checkpoint candidates.
3. **Extend the analysis with collaborators:** attentive probing and noise
   experiments follow the initial physical-target and checkpoint handoff.
4. **Train seeds 2 and 3** to measure variation across independent training runs.
   Seed-0 pilots remain excluded from scientific results.

The following commands reproduce the completed LR-sweep workflow; they do not
need to be rerun for this probe study.

After preparing the full training cache, create the twelve independent pilots
and submit them as four GPU jobs. Each job runs one objective's three rates in
sequence, reusing its local cache. The four objectives can run concurrently,
subject to cluster resources.
Use the appropriate account and GPU partition for your cluster.

```bash
SWEEP=outputs/lr_sweeps/rayleigh_benard
uv run --locked python scripts/lr_sweep.py prepare \
  --dataset rayleigh_benard --base /path/to/full/data --output "$SWEEP" --workers 4
sbatch --account YOUR_ACCOUNT --partition YOUR_GPU_PARTITION \
  --array=0-3%4 --output="$SWEEP/logs/%A_%a.log" \
  scripts/slurm_lr_sweep.sbatch "$SWEEP" 4
# Run after all twelve pilots finish successfully.
uv run --locked python scripts/lr_sweep.py collect --output "$SWEEP"
```

The Slurm launcher copies the full normalized training cache to local disk and
reuses it for subsequent pilots in that job. `WELL_CACHE_ROOT` overrides the default
`/tmp/well-training-cache-$UID`; choose a path on your cluster's local SSD with
enough free space (RB's cache is about 137 GiB). Some clusters give each job a
private `/tmp`; cross-job reuse requires a local directory visible to both jobs.
Callers using the same directory share a copy lock, interrupted copies resume
while the scratch files remain available, and the training loader verifies the
full content hash.
This changes storage location only. Raw HDF5 files and probe splits are not copied.
The worker count is frozen during preparation; four is a measured RB setting,
so check throughput when changing machines or systems.
To run all twelve pilots as separate GPU jobs, submit `--array=0-11%12` and omit
the trailing `4`; this permits more concurrency but may require more copies.

Each run retains both the planned configuration and `runtime_config.yaml`, whose
only change is the local data path. Invocation and completion records retain both
hashes and the source identity. For training outside the sweep, the same staging
command prints the data path to pass to `src.train --data-root`:

```bash
LOCAL_DATA=$(uv run --locked python scripts/stage_training_cache.py \
  --base /path/to/full/data --dataset rayleigh_benard \
  --cache-root "/tmp/well-training-cache-$UID")
uv run --locked python -m src.train --config configs/rayleigh_benard_jepa.yaml \
  --data-root "$LOCAL_DATA"
```

The manifest freezes candidate configurations and source provenance. Each pilot
uses the whole internal validation split every 2,000 steps. The collector requires
all twelve completed runs and writes `selection.json` with every candidate's
validation history, diagnostic flags, chosen rate, and run location. Inspect the
diagnostics before adopting the rates; collection does not change training YAMLs
or start the final scientific runs. The same commands support the other systems.

## Methods and repository layout

[docs/METHODS.md](docs/METHODS.md) describes physical targets, numerical
operators, sampling, selection, metrics, and remaining experiment work.

- `src/data`: shared trajectory metadata, HDF5/memmap training, preprocessing.
- `src/models`, `src/objectives`, `src/train.py`: encoder, heads, and training.
- `src/physics`: system definitions and physical numerical operators.
- `src/evaluation`, `src/evaluate.py`: cache, extraction, probes, reporting, CLI.
- `configs`: explicit training and analysis configurations.
- `checkpoints`: historical workshop encoders and all ICLR seed-1 candidates.
- `tests`: analytic, protocol, integrity, training, and small workflow checks.

Historical code, results, plots, and the older checkpoint tier are recoverable
from Git history (checkpoint weights use Git LFS). They are no longer supported
entrypoints. The locally supplied paper remains ignored under `refs/`; it is
not a runtime or test dependency. The sibling checkout is also not a dependency.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --locked python -m unittest discover -s tests -q
uv run --locked python -m src.evaluate validate-checkpoints \
  checkpoints/neuripsworkshop/*.pt
```

The ordinary suite manufactures small Well HDF5 datasets and runs the same
pipeline for all three systems; it needs no dataset download or trained weights.
During consolidation, all twelve original local checkpoints were checked at
native shape before the old tier was removed: extracted reference features
matched exactly. Both pooled and token RB targets also matched exactly on two
real validation clips, beginning at frames 0 and 53. Ridge predictions agreed
to round-off. MLP normalization now preserves small physical variations and
scoring uses unrounded float64 targets; numerical differences from historical
MLP results are expected, while saved-probe replay must reproduce its own clean
endpoint.

A bounded real-RB workflow also completed cache preparation, native-size
feature extraction, probes, noise evaluation, aggregation, and plotting for
the retained JEPA and MAE seed-1 checkpoints. It used two regimes and ten
trajectories per split, all five physical targets and gaps 0/8/32, ten-step MLP
fits, and noise levels 0/.1 with one corruption draw. Separate small JEPA/HDF5
and MAE/memmap training jobs were interrupted after step 2 and restarted through
step 6 with two data-loader workers; optimizer counters, parameter updates,
checkpoint identities, and continuous training histories were checked.
These CPU checks are not a full scientific rerun or a test of GPU execution.

Future-objective validation passed all 24 unit tests, including four focused
tests of future-only losses, information boundaries, gradients/EMA, and pixel
diagnostics. Paired-window/backend and resume-identity checks extend existing
tests. A direct CPU comparison against the previous commit reproduced current
JEPA/MAE losses, metrics, post-update model/EMA weights, and MAE images exactly;
all six retained workshop encoder checkpoints still load.

All twelve dataset/objective combinations also completed 64-step training on
an NVIDIA L40S using real data subsets, native resolution, eight-frame inputs,
batch size 64, and bf16. JEPA variants used HDF5; MAE variants used memmap.
Both RB future objectives were interrupted at step 16 and resumed to step 64.
Checks covered finite nonzero encoder/head gradients, frozen JEPA teacher
gradients, milestone checkpoint reloads, continuous histories, and exposure
counts. All four encoders completed feature extraction, probes, noise evaluation,
aggregation, and plotting on every system, producing 21 PDFs. Analysis used seed
0, ten-step MLP fits, and noise levels 0/.1 with one corruption draw. Local
verification evidence is under ignored
`outputs/future_objectives_verification/`. These short jobs establish workflow
operation, not convergence.

The subsequent full-trajectory LR sweeps completed all 36 GPU pilots across
the three systems. The 24 active-matter and shear-flow pilots completed all
96 scheduled validations with finite losses and feature diagnostics, and no
automatic collapse flags. Full-data cache checks, checkpoint/configuration
identities, optimizer counters, data exposure, and selected minima were checked;
all eight loss/feature figures were inspected. Each of these 24 pilots reached
its recorded validation minimum at step 8,000. Local evidence is retained under
`outputs/lr_sweeps/active_matter/` and `outputs/lr_sweeps/shear_flow/`; RB evidence
is under `outputs/lr_sweeps/rayleigh_benard_local/`. Full scientific training
remains the next stage.

Continuation verification passed all 30 tests, including exact uninterrupted
versus resumed CPU training for all four objectives across shuffled-pass
boundaries, worker-count changes, and cache relocation. The checks also cover
history rollback, interrupted checkpoint writes, export recovery, completed
runs, incompatible resumes, concurrent writers, and Slurm exit routing. All six
workshop encoders still load. A native-resolution active-matter future-JEPA run
used bf16, batch 64, four workers, and seven real trajectories: Slurm signalled
it at step 8, requeued it once, and it resumed at step 9 from a different local
cache path before completing step 32. Saved CPU/CUDA RNG restoration, optimizer
counters, learning rates, validation baseline, history, and exports were checked.
Both allocations used the same L40S node; this does not establish bitwise
equivalence across GPU machines or full-budget convergence. The instrumented
entrypoint, allocation logs, report, and training plot are retained under ignored
`outputs/continuation_verification/`.

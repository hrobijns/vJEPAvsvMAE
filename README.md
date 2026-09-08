# JEPA and masked autoencoding on The Well

A controlled comparison of frozen representations learned by matched video
encoders on Rayleigh–Bénard convection, active matter, and shear flow.

Both current objectives predict masked content **within the same eight-frame
clip**: JEPA predicts EMA-encoder features; MAE reconstructs normalized pixels.
Neither currently trains on future clips. The encoder, tube masking, and input
pipeline are shared; the heads, losses, and learning rates differ.

This checkout contains one analysis workflow. Current configurations use the
**whole available trajectory** for training and evaluation (`frame_limit: null`).
Eight-frame training clips overlap with starting times spaced one frame apart.
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

Checked-in YAML files are the configuration source of truth. Existing learning
rates remain starting settings from the previous experiments; they have not
been retuned for future ICLR experiments.

```bash
uv run --locked python -m src.data.preprocess \
  --base /path/to/data --dataset rayleigh_benard --split train
uv run --locked python -m src.train \
  --config configs/rayleigh_benard_jepa.yaml --data-root /path/to/data \
  --seed 1 --out runs/full_trajectories --no-wandb
```

Repeat with the MAE configuration and desired seeds. Replace the dataset name
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
can be reused when removing a sampling cap.

Checkpoints contain their configuration, input shape, and training identity.
`latest.pt` resumes only when source, temporal support, split, model, and
optimization settings agree. Historical checkpoints remain loadable for
analysis but cannot automatically resume a new training run. Reported workshop
models are final 100,000-step endpoints; `encoder_best_val.pt` is an additional
training diagnostic, not the endpoint-selection rule for the comparison.
Encoder checkpoints at 25%, 50%, 75%, and 100% of the planned steps support
learning-curve analysis. The 100,000-step budget is a starting choice; assess
training sufficiency on validation data before changing the comparison budget.

## Frozen-encoder analysis

Use a fresh output root. Every completed stage is immutable and records its
inputs and code provenance. The same commands and result format serve all
three systems.

```bash
BASE=/path/to/data
OUT=outputs/rb_full_trajectories

for split in valid test; do
  uv run --locked python -m src.evaluate prepare-cache \
    --base "$BASE" --config configs/eval_rayleigh_benard.yaml \
    --split "$split" --cache-root "$OUT/cache"
done

for objective in jepa mae; do
  for seed in 1 2 3; do
    checkpoint="runs/full_trajectories/rayleigh_benard_${objective}_seed${seed}/encoder_100pct.pt"
    uv run --locked python -m src.evaluate extract-features \
      --checkpoint "$checkpoint" --cache-root "$OUT/cache" \
      --feature-root "$OUT/features"
    # Feature directory names include the checkpoint's content hash.
    for feature in "$OUT/features/${objective}_seed${seed}_"*; do
      uv run --locked python -m src.evaluate fit-probes \
        --feature-dir "$feature" --cache-root "$OUT/cache" \
        --output "$OUT/probes/${objective}_seed${seed}"
      uv run --locked python -m src.evaluate evaluate-noise \
        --feature-dir "$feature" --cache-root "$OUT/cache" \
        --probe-dir "$OUT/probes/${objective}_seed${seed}" \
        --output "$OUT/noise/${objective}_seed${seed}"
    done
  done
done

for kind in probes noise; do
  uv run --locked python -m src.evaluate aggregate "$OUT/$kind/"* \
    --kind "$kind" --objectives jepa mae --seeds 1 2 3 \
    --output "$OUT/${kind}_aggregate"
  uv run --locked python -m src.evaluate plot \
    --aggregate-dir "$OUT/${kind}_aggregate" --output "$OUT/${kind}_plots"
done
```

For workshop reproduction, use
`configs/workshop/eval_rayleigh_benard.yaml`, a fresh output root such as
`outputs/rb_workshop`, and replace the checkpoint assignment above with
`checkpoint="checkpoints/neuripsworkshop/rayleigh_benard_${objective}_seed${seed}.pt"`.
This uses the same analysis code with explicit frames 0–100. Full-trajectory
analysis of those historical weights is a separate experiment: it does not make
them models trained on full trajectories. New temporal protocols require new
analysis artifacts; existing completed outputs are not overwritten or mixed.

`fit-probes` produces the main selection, separate Ridge/MLP results, pooled
depth curves, regime decoding, regime/time and position controls, the combined
encoder-plus-control probe, and persistence baselines. `evaluate-noise`
loads the actual fitted clean probes; it does not refit them.

Aggregates retain individual checkpoint rows, per-target summaries, and
within-checkpoint target means followed by between-checkpoint mean/SD.
Single-seed smoke results are supported by specifying their actual roster
(e.g. `--seeds 0`); their between-seed SD is undefined. Probe ensemble members,
noise draws, targets, and deterministic controls are not counted as independent
encoder seeds. Undefined metrics are JSON `null`, with an explanatory status.

Plot outputs include comparison heatmaps, pooled Ridge/MLP depth curves, noise
curves with individual checkpoints, and `summary.tsv` containing every metric,
regime result, and control. Negative R² values remain visible; noise R² uses a
symmetric-log scale. Heatmap color spans R² 0–1; cell annotations retain actual
scores outside that range. No figure or result is copied into the paper.

For the other systems, select `configs/eval_active_matter.yaml` or
`configs/eval_shear_flow.yaml` and checkpoints trained on that dataset. Their
initial physical target is enstrophy; their governing parameters are also
probed. Richer target sets are deliberately deferred. The six retained RB
checkpoints cannot be substituted for models of another system.

## Methods and repository layout

[docs/METHODS.md](docs/METHODS.md) describes physical targets, numerical
operators, sampling, selection, metrics, and remaining experiment work.

- `src/data`: shared trajectory metadata, HDF5/memmap training, preprocessing.
- `src/models`, `src/objectives`, `src/train.py`: encoder, heads, and training.
- `src/physics`: system definitions and physical numerical operators.
- `src/evaluation`, `src/evaluate.py`: cache, extraction, probes, reporting, CLI.
- `configs`: explicit training and analysis configurations.
- `checkpoints/neuripsworkshop`: six reference RB encoders and provenance.
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

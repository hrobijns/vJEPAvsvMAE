# vJEPA vs vMAE on The Well

Controlled comparison of a latent-prediction objective (V-JEPA-style) against a
pixel-reconstruction objective (VideoMAE-style) with **identical encoders**, on
2D physics simulations from [The Well](https://polymathic-ai.org/the_well/).
End goal: test whether the JEPA objective encodes more physical variables in
its latent space, using training, frozen-encoder probing, and analysis outputs.

**Migration checkpoint:** the [current RB workflow](#current-rb-evaluation)
supersedes the legacy Rayleigh–Bénard target calculations and results.
Historical workshop/paper material remains for context, not as validation of
this workflow. See [known limitations and integration follow-up](#known-limitations-and-integration-follow-up)
before interpreting new outputs.

See [docs/OVERVIEW.md](docs/OVERVIEW.md) for a research-summary writeup
(motivation, architecture, headline results), and
[docs/LINEAR_PROBE.md](docs/LINEAR_PROBE.md) for a deep dive on the probing
pipeline. This README covers setup and reproduction.

## Historical results (legacy RB; superseded)

Numbers below are retained from [scripts/workshop_test_eval.py](scripts/workshop_test_eval.py) —
the historical held-out, multi-seed test-evaluation protocol (layer choice frozen
from train-CV *before* touching test data, probe fit once on train, reported
once on held-out test trajectories, mean±std over 3 independently-seeded
encoders). These are not corrected RB results and should not be cited as
current findings. [docs/LINEAR_PROBE.md](docs/LINEAR_PROBE.md) describes the
legacy method. Raw numbers in `sweep_results/*_workshop_test_eval.json`, figures in
[reports/figures/](reports/figures/).

**`rayleigh_benard` was the workshop's headline dataset** — the only one with a
complete 3-seed × {JEPA, MAE} run at the current (192-d predictor)
architecture. `active_matter`/`shear_flow` report on a single representative
encoder seed per objective; see "Pretrained encoders" below for why.

- **Contemporaneous physics** (pooled, present-time): the two objectives are
  close on most quantities — MAE fractionally ahead on the small-scale/
  high-frequency ones (`enstrophy` 0.996 vs 0.991, `okubo_weiss` 0.996 vs
  0.993, `convective_flux` 0.992 vs 0.985) — except `buoyancy_grad`, where
  JEPA leads (0.965 vs 0.942). Purely local/differential quantities
  (`pressure_grad_mag`) are near ceiling for both (≥0.997).
- **JEPA's advantage grows with forecast horizon.** At t+32, JEPA leads on
  every pooled quantity, often by a wide margin (`buoyancy_grad` 0.864 vs
  0.743, `okubo_weiss` 0.901 vs 0.835, `velocity_buoyancy_coherence` 0.790 vs
  0.678) — the clearest quantitative signal in the whole sweep. JEPA is the
  only objective explicitly trained to predict *in time*, not just reconstruct
  the present.
- **Noise robustness** (clean-fit protocol: probe fit once on clean features,
  no refitting, evaluated across a noise grid): JEPA is substantially more
  robust than MAE at low-to-moderate corruption on the differential
  quantities (e.g. `buoyancy_grad` Pearson r 0.91 vs 0.30 at σ=0.1), though
  this compresses at extreme noise (σ=1.0), where the two are closer or MAE
  edges ahead on a few quantities. Full table with both R² and Pearson r
  (R² alone goes arbitrarily negative under distribution shift, a real
  representational-fragility signature, not a bug) in LINEAR_PROBE.md.
- **Regime parameters** (Rayleigh, Prandtl) are decoded near-ceiling by both
  objectives (R² ≥ 0.99) — not where JEPA/MAE differ. Shuffled-control checks
  collapse to ~0 R², confirming no train/val leakage.
- **Depth**: JEPA needs deeper layers than MAE to reach peak decodability
  (pooled mean layer 7.5 vs 4.4 out of 0–12) — except `buoyancy_grad`, JEPA's
  clearest accuracy win, which peaks *shallower* for JEPA (5.3) than MAE (8.0).

## Pretrained encoders

`checkpoints/` has two subfolders:

- [`checkpoints/neuripsworkshop/`](checkpoints/neuripsworkshop/) — the 6
  `rayleigh_benard` checkpoints (3 JEPA seeds + 3 MAE seeds) behind this
  repo's historical results, current (192-d predictor) architecture. See its
  own README for exact per-file provenance and training config.
- [`checkpoints/old/`](checkpoints/old/) — the earlier single-seed set (one
  JEPA + one MAE per dataset, all 3 datasets), kept for quick-start/demo use
  on `active_matter`/`shear_flow`, which don't yet have a current-architecture
  3-seed run. See its own README for the predictor-width caveat on two of
  these files.

Each `.pt` bundles the trained weights **and** the exact config used to
produce them (architecture, LR, mask ratio, objective hyperparameters) — it's
self-contained, no separate config file needed.

**Load one:**

```python
from scripts.load_encoder import load_encoder

encoder, config, spec = load_encoder("checkpoints/neuripsworkshop/rayleigh_benard_jepa_seed1.pt")
# encoder: ViT-S in eval() mode. spec: ClipSpec(n_channels, n_frames, height, width)
# clip: (B, C, T, H, W) tensor, z-score normalized per The Well's own stats
features = encoder(clip)  # (B, n_tokens, 384) — no masking applied at inference
```

`uv run python scripts/load_encoder.py checkpoints/<tier>/<name>.pt` runs
this as a standalone sanity check (loads + random forward pass).

For Rayleigh–Bénard probing, use the [current RB workflow](#current-rb-evaluation).
The legacy [scripts/analyze_encoders.py](scripts/analyze_encoders.py) and
[scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py) remain
available for other datasets and exploratory work; their RB entrypoints
require explicit `--allow-legacy-rb` opt-in.

## Legacy probing suite

[scripts/workshop_test_eval.py](scripts/workshop_test_eval.py) is the
historical protocol — 4 probe families (contemporaneous, forecast-content,
noise robustness, regime), each with layer choice frozen from train-CV before
touching held-out test data, and mean±std reported over 3 encoder seeds
(rayleigh_benard only — see "Pretrained encoders" above). It's the source of
the archived numbers in this README and in
[docs/OVERVIEW.md](docs/OVERVIEW.md); full method and tables in
[docs/LINEAR_PROBE.md](docs/LINEAR_PROBE.md).

Its RB entrypoint also requires `--allow-legacy-rb`; opting in does not make
the legacy targets valid. Other datasets' implementations are unchanged by
this migration and have not received the RB-specific corrections.

Two supporting scripts do the earlier-stage, train-CV version of this analysis
(no held-out test split, single seed) — useful for exploring the
depth/layer-emergence landscape or a new dataset before committing to the
expensive held-out protocol:

| script | question |
|---|---|
| [scripts/analyze_encoders.py](scripts/analyze_encoders.py) | layer-wise, pooled: which layer best decodes each physical quantity? Also supports `--split valid` for a held-out generalization check. |
| [scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py) | same targets, per-token (non-pooled) plus a small MLP nonlinear readout — does spatial detail or nonlinearity recover signal pooling/linearity hides? |

Supporting script: [scripts/extract_regime_metadata.py](scripts/extract_regime_metadata.py)
recovers per-trajectory regime params (Reynolds/Schmidt, Rayleigh/Prandtl,
alpha/zeta) from Well filenames, feeding `workshop_test_eval.py`'s regime
family.

## Current RB evaluation

[scripts/rb_eval_v2.py](scripts/rb_eval_v2.py) is the RB entrypoint for this
checkpoint. It reads full official `valid`/`test` HDF5 trajectories directly,
not the legacy 101-frame memmaps. Each split must contain 175 trajectories
(35 regimes × 5 replicates), with 200 frames per trajectory. Targets use
physical units, periodic-x/Chebyshev-y derivatives, and physical quadrature;
encoder inputs use The Well's normalization. Probe selection uses validation
data, not test scores.

Run from the repository root. The commands below download data and perform
substantial computation; feature extraction uses CUDA when available. Use a
fresh output directory for each run: completed caches, features, and result
files are not overwritten. Keep manifests with their arrays and results.

```bash
uv sync --locked
BASE=/path/to/data
OUTPUT=sweep_results/rb_v2
mkdir -p "$OUTPUT"

uv run python scripts/rb_eval_v2.py validate-checkpoints \
  checkpoints/neuripsworkshop/rayleigh_benard_{jepa,mae}_seed{1,2,3}.pt \
  > "$OUTPUT/checkpoints.json"

for split in valid test; do
  uv run the-well-download --base-path "$BASE" --dataset rayleigh_benard --split "$split"
  uv run python scripts/rb_eval_v2.py prepare-cache \
    --base "$BASE" --split "$split" --cache-root "$OUTPUT/cache"
done

for objective in jepa mae; do
  for seed in 1 2 3; do
    checkpoint_id="${objective}_seed${seed}"
    uv run python scripts/rb_eval_v2.py extract-features \
      --checkpoint "checkpoints/neuripsworkshop/rayleigh_benard_${checkpoint_id}.pt" \
      --cache-root "$OUTPUT/cache" --feature-root "$OUTPUT/features"
    uv run python scripts/rb_eval_v2.py fit-selected-probes \
      --feature-dir "$OUTPUT/features/$checkpoint_id" --cache-root "$OUTPUT/cache" \
      --output "$OUTPUT/selected/$checkpoint_id.json"
  done
done

uv run python scripts/rb_eval_v2.py aggregate \
  "$OUTPUT"/selected/{jepa,mae}_seed{1,2,3}.json \
  --output "$OUTPUT/selected_aggregate.json"
uv run python scripts/rb_eval_v2.py persistence-baseline \
  --cache-root "$OUTPUT/cache" --output "$OUTPUT/persistence_baseline.json"
uv run python scripts/rb_eval_v2.py balanced-token-control \
  --cache-root "$OUTPUT/cache" --output "$OUTPUT/token_position_control.json"
```

The six endpoint checkpoints are tracked with Git LFS; if the `.pt` files
are only pointer files, fetch their payloads with `git lfs pull` first.
The general download wrapper defaults to `train valid`, so it does not
replace the explicit RB `test` download above. Memmap preprocessing is still
used for training and now preserves full trajectories and metadata; existing
incompatible memmaps must be moved aside and regenerated, not silently reused.

Selected probes cover pooled/token targets on `original_support` at gaps
0, 8, and 32, with validation-selected Ridge/MLP choices. Per-checkpoint JSON
contains row-level metrics and selections; the aggregate retains those rows,
source metadata, and seed summaries. Aggregate exactly one result family
from the same run across the six endpoints; do not mix protocols or caches.

For optional selected-probe noise and MLP-depth outputs, reuse the saved
selection and its features. Repeat for each checkpoint, changing `checkpoint_id`:

```bash
checkpoint_id=jepa_seed1
uv run python scripts/rb_eval_v2.py fit-selected-noise \
  --feature-dir "$OUTPUT/features/$checkpoint_id" \
  --selected-feature-dir "$OUTPUT/features/$checkpoint_id" \
  --cache-root "$OUTPUT/cache" --selected-result "$OUTPUT/selected/$checkpoint_id.json" \
  --output "$OUTPUT/selected_noise/$checkpoint_id.json"
uv run python scripts/rb_eval_v2.py fit-mlp-depth \
  --feature-dir "$OUTPUT/features/$checkpoint_id" --cache-root "$OUTPUT/cache" \
  --selected-result "$OUTPUT/selected/$checkpoint_id.json" \
  --output "$OUTPUT/mlp_depth/$checkpoint_id.json"
```

Aggregate each optional family separately using the same six-file pattern.
`fit-probes` is the separate base-protocol path: it accepts `--feature-dir`,
`--cache-root`, and `--output` like `fit-selected-probes`, but covers both
`original_support` and `developed` strata with a different probe/target grid.
Only aggregates from that base path are supported by
`plot --aggregate <base-aggregate.json> --output-dir <figures-directory>`;
it writes PDFs, a summary TSV, and a plot manifest. Do not pass selected-probe,
selected-noise, or MLP-depth aggregates to that plotter. Those workflows
currently provide JSON for downstream analysis. See each subcommand's
`--help` for options.

### Known limitations and integration follow-up

The migration preserves committed fixes, not a finished unified architecture.
Independent review found these inherited issues; fixes are deliberately
deferred to a separate integration planning pass:

- **Aggregation compatibility:** incompatible cache hashes/protocols can be
  combined without rejection. Add compatibility checks and regression tests.
- **Array integrity:** manifest self-hashes are checked, but hashes of the
  actual consumed feature/target files are not verified. Validate file contents
  against the recorded hashes rather than treating manifests alone as proof.
- **MLP replay precision:** selection and replay use different target precision
  and ensemble-averaging conventions. A near-constant synthetic target failed
  the clean-endpoint check. Align scoring and add low-variance regression cases.
- **Plot compatibility:** the plotter assumes base-protocol rows and fails on
  selected-probe aggregates. Define a consistent output contract during integration.
- **One supported workflow:** plan shared interfaces and dataset-specific
  mathematics across data preparation, training, extraction, probing, and
  analysis outputs. Consolidate legacy/`v2` paths only after parity/correctness
  checks; merely renaming files is not sufficient.

All 46 unit tests, analytic operator checks, six checkpoint validations, and
JEPA/MAE seed-1 CPU forward-pass smoke tests passed during migration. No full
scientific rerun was performed, and the review did not establish that actual
experiment results were affected. Passing those checks does not resolve the
known limitations. The source fork checkout remains intact; no broader
refactoring or deletion of existing upstream content is part of this checkpoint.

## Exploratory probing with legacy helpers

You don't need the full held-out pipeline to try an idea — `analyze_encoders.py`
exposes its building blocks (loading, features, ridge probe) as plain
functions. Minimal example, probing a made-up quantity against every layer of
a frozen encoder:

```python
import numpy as np
import torch
from scripts.analyze_encoders import (
    load_checkpoint_encoder, compute_layerwise_features_batched, ridge_r2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, spec = load_checkpoint_encoder(
    "checkpoints/neuripsworkshop/rayleigh_benard_jepa_seed1.pt", device)

# clips: (N, C, T, H, W) float tensor, z-scored per The Well's own stats,
# same channel order as `spec` — see src/data/well.py if loading raw Well
# data yourself, or read directly from a preprocessed memmap:
mm = np.load("/path/to/data/memmap/rayleigh_benard/train.npy", mmap_mode="r")
clips = torch.from_numpy(np.array(mm[0:8, :, 0:8])).float()  # 8 trajectories, first 8 frames

# your target: one scalar per clip, whatever physics you're curious about
my_target = clips[:, 0].mean(dim=(1, 2, 3))  # placeholder — replace with a real derived quantity

per_layer_feats = compute_layerwise_features_batched(encoder, clips)  # list of (N, D), one per layer
for layer_idx, feats in enumerate(per_layer_feats):
    print(f"layer {layer_idx}: R^2 = {ridge_r2(feats, my_target):.3f}")
```

For RB physical targets, use [scripts/rb_targets_v2.py](scripts/rb_targets_v2.py)
and its geometry-aware operators, not the legacy helpers below. The following
guidance describes the existing exploratory path for other datasets, not a
new validation of their target definitions.

For a target requiring finite-difference derivatives (gradients, curl,
divergence, Laplacian), reuse `curl2d`/`grad2d`/`divergence2d`/`laplacian2d`/
`tensor_div2d`/`okubo_weiss` from the same file rather than reimplementing
them — see `contemporaneous_targets()` for how each dataset's existing
targets are built from these, and
[docs/LINEAR_PROBE.md § Physics targets](docs/LINEAR_PROBE.md#physics-targets-derived-from-each-simulations-governing-pde)
for the derivation of each one (useful as a template for deriving a new
target from a dataset's governing PDE). To go from this kind of one-off
exploration to a properly held-out, multi-seed, citable number, add your
target to `contemporaneous_targets()` (or the relevant family in
`workshop_test_eval.py`) and it's automatically covered by the full protocol.

**How they were trained**: 100k steps, AdamW + cosine LR, tube masking at 0.9,
on a single A100/A40. Learning rate is tuned per (dataset, objective) via a
short mini-sweep on a pilot seed (`scripts/run_lr_minisweep.sh` +
`scripts/pick_lr.py`, selected on held-out validation loss) — see
[configs/tuned_lr.json](configs/tuned_lr.json) for the exact values used. See
git history / [scripts/gen_configs.py](scripts/gen_configs.py) for the exact
config that produced each checkpoint.

## Reproducing from scratch

```bash
uv sync
bash scripts/download_data.sh /path/to/data   # ~525 GB, hours — see RunPod workflow below
uv run python scripts/preprocess_memmap.py --base /path/to/data --dataset active_matter --split train
uv run python scripts/preprocess_memmap.py --base /path/to/data --dataset active_matter --split valid
uv run python -m src.train --config configs/active_matter_jepa.yaml --data-root /path/to/data
```

Repeat `preprocess_memmap.py` + `train.py` for `shear_flow` and `rayleigh_benard`
(configs already exist for both objectives × all 3 datasets). Each run takes
roughly 6–11 hours on a single A100; see the RunPod workflow below for
running on rented GPUs.

For the full 3-seed final training used for `rayleigh_benard`'s historical
results, `configs/tuned_lr.json` must exist first (already committed — it's
the output of `scripts/run_lr_minisweep.sh` + `scripts/pick_lr.py`), then:
`bash scripts/run_final_training.sh` runs all (dataset, objective, seed)
combinations reading LR from that file.

## Design

Both objectives share everything except the head:

| shared | ViT-S encoder (384d × 12), 2×16×16 tubelet patches, tube masking @ 0.9, T=8 clips at native resolution, per-channel z-score norm (Well stats), AdamW + cosine, same batch/steps |
|---|---|
| **MAE** | 4-layer decoder (192d) → MSE on masked patches (`norm_pix`) |
| **JEPA** | EMA target encoder (0.996→1.0) + 6-layer predictor (192d, narrower than the 384d encoder) → smooth-L1 on layer-normed target features at masked positions |

Datasets (per-dataset model pairs, 6 runs): `active_matter` (11ch, 256×256),
`shear_flow` (4ch, 256×512, incompressible NS), `rayleigh_benard` (4ch, 512×128).
`turbulent_radiative_layer_2D` is used only as a local smoke-test dataset.

## Quickstart

```bash
uv sync

# local smoke test (downloads ~700 MB, first file only)
uv run the-well-download --base-path ~/well_data --dataset turbulent_radiative_layer_2D --split train --first-only
uv run python -m src.train --config configs/debug_mae.yaml
uv run python -m src.train --config configs/debug_jepa.yaml

# real runs (on RunPod, see below)
uv run python -m src.train --config configs/active_matter_jepa.yaml
```

Checkpoints land in `runs/<run_name>/`: `latest.pt` (full resume state, saved
every 1k steps) and `encoder_{025,050,075,100}pct.pt` (encoder-only milestones
for the step-2 probing study).

## RunPod workflow

1. Create a **network volume** (≥700 GB) and a pod with 1× A100 80GB
   (PyTorch CUDA base image), volume mounted at `/workspace`.
2. Clone this repo into `/workspace`, then:
   ```bash
   export WANDB_API_KEY=...
   bash runpod/setup.sh
   bash scripts/download_data.sh /workspace/data   # ~525 GB, hours
   ```
3. Smoke test on GPU, then launch pairs smallest-first inside tmux:
   ```bash
   tmux new -s train
   uv run python -m src.train --config configs/active_matter_jepa.yaml
   uv run python -m src.train --config configs/active_matter_mae.yaml
   # then shear_flow pair, then rayleigh_benard pair
   ```
   Re-running the same command auto-resumes from `runs/<name>/latest.pt`
   (spot-interruption safe).

## Repo map

- [checkpoints/](checkpoints/) — pretrained encoders (see "Pretrained encoders" above)
- [src/data/well.py](src/data/well.py) — Well → (C,T,H,W) clip dataset, trajectory-disjoint train/val split
- [src/masking.py](src/masking.py) — shared tube masking
- [src/models/vit.py](src/models/vit.py) — shared ViT encoder
- [src/objectives/mae.py](src/objectives/mae.py), [src/objectives/jepa.py](src/objectives/jepa.py) — the two heads
- [src/train.py](src/train.py) — unified entrypoint, incl. held-out val loop + best-val checkpoint selection
- [scripts/gen_configs.py](scripts/gen_configs.py) — regenerates `configs/`
- [scripts/preprocess_memmap.py](scripts/preprocess_memmap.py) — Well HDF5 → fast fp16 memmap (needed before training/analysis)
- [scripts/download_data.sh](scripts/download_data.sh) — downloads the 3 datasets from The Well
- [scripts/load_encoder.py](scripts/load_encoder.py) — load a checkpoint + sanity-check forward pass
- [scripts/run_lr_minisweep.sh](scripts/run_lr_minisweep.sh), [scripts/pick_lr.py](scripts/pick_lr.py) — per-(dataset,objective) LR mini-sweep → `configs/tuned_lr.json`
- [scripts/run_final_training.sh](scripts/run_final_training.sh) — 3-seed final training for all (dataset, objective) pairs
- [scripts/rb_eval_v2.py](scripts/rb_eval_v2.py), [scripts/rb_pipeline_v2.py](scripts/rb_pipeline_v2.py) — current RB CLI and cache/extraction/probing/output implementation
- [scripts/rb_targets_v2.py](scripts/rb_targets_v2.py), [scripts/rb_derivatives.py](scripts/rb_derivatives.py), [scripts/rb_quadrature.py](scripts/rb_quadrature.py) — RB physical targets and geometry-aware operators
- [tests/](tests/) — preprocessing, configuration, target, checkpoint, and pipeline tests
- [scripts/workshop_test_eval.py](scripts/workshop_test_eval.py) — legacy probing pipeline; RB requires explicit opt-in
- [scripts/plot_workshop_figures.py](scripts/plot_workshop_figures.py) — renders the paper's figures from `workshop_test_eval.py`'s output
- [scripts/analyze_encoders.py](scripts/analyze_encoders.py), [scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py) — earlier-stage train-CV pooled/non-pooled probing (see "Probing suite" above)
- [scripts/extract_regime_metadata.py](scripts/extract_regime_metadata.py) — recovers per-trajectory regime params from Well filenames
- [scripts/extract_training_history.py](scripts/extract_training_history.py), [scripts/plot_training_curves.py](scripts/plot_training_curves.py) — training-loss curve extraction/plotting
- [docs/](docs/) — research-summary writeup and probing deep dive (architecture, variables, results)

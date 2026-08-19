# vJEPA vs vMAE on The Well

Controlled comparison of a latent-prediction objective (V-JEPA-style) against a
pixel-reconstruction objective (VideoMAE-style) with **identical encoders**, on
2D physics simulations from [The Well](https://polymathic-ai.org/the_well/).
End goal: test whether the JEPA objective encodes more physical variables in
its latent space (probing suite = step 2, separate from this repo's step-1
training infrastructure).

See [docs/OVERVIEW.md](docs/OVERVIEW.md) for a research-summary writeup
(motivation, architecture, headline results), and
[docs/LINEAR_PROBE.md](docs/LINEAR_PROBE.md) for a deep dive on the probing
pipeline. This README covers setup and reproduction.

## Results (summary)

Headline numbers below are from [scripts/workshop_test_eval.py](scripts/workshop_test_eval.py) —
the canonical held-out, multi-seed test-evaluation protocol (layer choice frozen
from train-CV *before* touching test data, probe fit once on train, reported
once on held-out test trajectories, mean±std over 3 independently-seeded
encoders). It's the protocol behind every number that should be cited as a
final result; see [docs/LINEAR_PROBE.md](docs/LINEAR_PROBE.md) for the full
method. Raw numbers in `sweep_results/*_workshop_test_eval.json`, figures in
[reports/figures/](reports/figures/).

**`rayleigh_benard` is the paper's headline dataset** — the only one with a
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
  repo's headline results, current (192-d predictor) architecture. See its
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

To probe what these encoders actually learned — layer-wise physics decodability,
pooled vs. per-token comparisons — see [scripts/analyze_encoders.py](scripts/analyze_encoders.py)
and [scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py).
Both need the corresponding dataset's memmap (see "Reproducing" below) or can
be pointed at your own tensors of the same shape.

## Probing suite (step 2)

[scripts/workshop_test_eval.py](scripts/workshop_test_eval.py) is the
canonical protocol — 4 probe families (contemporaneous, forecast-content,
noise robustness, regime), each with layer choice frozen from train-CV before
touching held-out test data, and mean±std reported over 3 encoder seeds
(rayleigh_benard only — see "Pretrained encoders" above). It's the source of
every headline number in this README and in
[docs/OVERVIEW.md](docs/OVERVIEW.md); full method and tables in
[docs/LINEAR_PROBE.md](docs/LINEAR_PROBE.md).

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

### Probing a new physical quantity

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

For the full 3-seed final training used for `rayleigh_benard`'s headline
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
- [scripts/workshop_test_eval.py](scripts/workshop_test_eval.py) — **canonical probing pipeline**: contemporaneous, forecast, noise-robustness, regime, held-out test split, multi-seed
- [scripts/plot_workshop_figures.py](scripts/plot_workshop_figures.py) — renders the paper's figures from `workshop_test_eval.py`'s output
- [scripts/analyze_encoders.py](scripts/analyze_encoders.py), [scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py) — earlier-stage train-CV pooled/non-pooled probing (see "Probing suite" above)
- [scripts/extract_regime_metadata.py](scripts/extract_regime_metadata.py) — recovers per-trajectory regime params from Well filenames
- [scripts/extract_training_history.py](scripts/extract_training_history.py), [scripts/plot_training_curves.py](scripts/plot_training_curves.py) — training-loss curve extraction/plotting
- [docs/](docs/) — research-summary writeup and probing deep dive (architecture, variables, results)

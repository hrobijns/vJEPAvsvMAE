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

Full sweep complete on all 3 datasets: pooled + per-token linear/MLP probing,
13-layer depth sweep, 6-level noise sweep (pooled + token), regime probing,
forecast/skill-score, and a held-out (`valid`-split) generalization check.
Full tables and discussion in [docs/LINEAR_PROBE.md](docs/LINEAR_PROBE.md);
raw numbers in `sweep_results/*.json`, figures in `reports/figures/`.

- **`rayleigh_benard` is where the two objectives genuinely diverge.** JEPA
  beats MAE on every axis there: clean pooled accuracy on coupled quantities
  (`convective_flux` 0.835 vs 0.576), depth (needs L6–L11 to peak vs. MAE's
  L0–L2 plateau, but reaches a higher peak), noise robustness, real
  nonlinear-only token-level information MAE lacks entirely, and forecast
  skill at longer horizons. Purely local quantities (gradients, pressure) are
  at ceiling for both, on every dataset.
- **`active_matter`/`shear_flow` show no clean-data gap** — sometimes MAE is
  fractionally ahead — but `active_matter` still shows a large noise-
  robustness split (MAE's R² goes negative under input noise; JEPA degrades
  gracefully), while `shear_flow` shows no split on any axis at all. Likely
  mechanism: `rayleigh_benard`'s buoyancy and `active_matter`'s active stress
  both feed back into the momentum equation (two-way coupling); `shear_flow`'s
  tracer is passively advected (one-way) — see LINEAR_PROBE.md's Discussion.
- **Regime parameters (Reynolds/Schmidt, Rayleigh/Prandtl, alpha/zeta) are
  decoded equally well by both objectives** — not where JEPA/MAE differ.
- **A held-out generalization check** (small `valid`-split slice, encoders
  never pretrained on it) reproduces and sharpens the noise-robustness
  finding — MAE's `active_matter` `nematic_order` R² is −15.4 at *zero* added
  noise on unseen trajectories. Sample size is small (4–5 trajectories per
  dataset), so treat as directional, not precise.
- **A proper held-out `test`-split evaluation** (layer choice frozen from
  train-CV *before* touching test data, probe fit once on train, evaluated
  once on 10/28/50 held-out trajectories) confirms every headline claim
  above — `rayleigh_benard`'s JEPA-ahead gap *widens* on test rather than
  shrinking, and `active_matter`'s MAE noise-collapse reproduces almost
  exactly (−0.78 at σ=2 vs. −1.02 on train). See LINEAR_PROBE.md's "Held-out
  test-split evaluation" section.

## Pretrained encoders

[`checkpoints/`](checkpoints/) has the final (100k-step) trained encoder for
all 6 runs — one JEPA and one MAE encoder per dataset:

| file | dataset | objective | params |
|---|---|---|---|
| `active_matter_jepa.pt` / `active_matter_mae.pt` | active_matter (11ch, 256×256) | JEPA / MAE | 23.5M |
| `shear_flow_jepa.pt` / `shear_flow_mae.pt` | shear_flow (4ch, 256×512) | JEPA / MAE | 23.5M |
| `rayleigh_benard_jepa.pt` / `rayleigh_benard_mae.pt` | rayleigh_benard (4ch, 512×128) | JEPA / MAE | 23.5M |

Each `.pt` bundles the trained weights **and** the exact config used to
produce them (architecture, LR, mask ratio, objective hyperparameters) — it's
self-contained, no separate config file needed.

**Load one:**

```python
from scripts.load_encoder import load_encoder

encoder, config, spec = load_encoder("checkpoints/active_matter_jepa.pt")
# encoder: ViT-S in eval() mode. spec: ClipSpec(n_channels, n_frames, height, width)
# clip: (B, C, T, H, W) tensor, z-score normalized per The Well's own stats
features = encoder(clip)  # (B, n_tokens, 384) — no masking applied at inference
```

`uv run python scripts/load_encoder.py checkpoints/<name>.pt` runs this as a
standalone sanity check (loads + random forward pass).

To probe what these encoders actually learned — layer-wise physics decodability,
pooled vs. per-token comparisons — see [scripts/analyze_encoders.py](scripts/analyze_encoders.py)
and [scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py).
Both need the corresponding dataset's memmap (see "Reproducing" below) or can
be pointed at your own tensors of the same shape.

## Probing suite (step 2)

Beyond contemporaneous physics decodability, six more probes ask whether the
representations differ in *what kind* of physics they capture:

| script | question | found |
|---|---|---|
| [scripts/analyze_encoders.py](scripts/analyze_encoders.py) | layer-wise, pooled: which layer best decodes each physical quantity (enstrophy, divergence, okubo-weiss, nematic order, ...)? Also supports `--split valid` for a held-out generalization check. | JEPA and MAE tie on `active_matter`/`shear_flow`; JEPA clearly ahead on `rayleigh_benard`'s coupled quantities. |
| [scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py) | same targets, per-token (non-pooled) plus a small MLP nonlinear readout — does spatial detail or nonlinearity recover signal pooling/linearity hides? | On `rayleigh_benard`, JEPA has real nonlinear-only local information (e.g. `convective_flux` MLP 0.90) that MAE lacks even nonlinearly (0.07). Elsewhere, MLP gains are symmetric between objectives. |
| [scripts/analyze_regime.py](scripts/analyze_regime.py) | does the pooled representation know the *regime* (Reynolds/Schmidt, Rayleigh/Prandtl, activity/alignment) — one value per trajectory, probed with train/val split by trajectory to avoid leakage | Both objectives decode regime near-perfectly (R² 0.77–0.999) — not where JEPA/MAE differ. |
| [scripts/forecast_content_probe.py](scripts/forecast_content_probe.py) | does a *fresh* linear/MLP probe on frozen present-time features forecast future physics, swept over multiple time gaps and noise levels, scored with a persistence-relative skill score? | On `rayleigh_benard`, JEPA's forecast-skill advantage *grows* with horizon; token-level flips `active_matter`'s ranking to JEPA despite MAE leading pooled. |
| [scripts/analyze_noise_robustness.py](scripts/analyze_noise_robustness.py) | injects Gaussian noise (several std levels) directly into the input physical variables before encoding, then ridge-probes (pooled and per-token, every layer) against the *clean* clip's physics — does decodability degrade gracefully or collapse abruptly? | MAE is the sharper decoder at zero noise but collapses far more sharply than JEPA once corrupted, on `active_matter`/`rayleigh_benard` (confirmed on held-out `valid`-split data too); `shear_flow` shows no split. |

Supporting scripts: [scripts/extract_regime_metadata.py](scripts/extract_regime_metadata.py)
(recovers per-trajectory regime params from Well filenames, needed by
`analyze_regime.py`). [scripts/run_sweep.sh](scripts/run_sweep.sh) runs the
full probing suite across all 3 datasets on a remote GPU box; results land in
`sweep_results/*.json`, summarized in
[reports/](reports/) (per-dataset PDF summaries + `probing_sweep_report.html`).

**How they were trained**: 100k steps, AdamW + cosine LR (JEPA 1e-4, MAE 5e-5 —
picked via a small LR sweep on active_matter, see [Design](#design) below),
tube masking at 0.9, on a single A100. See git history / [scripts/gen_configs.py](scripts/gen_configs.py)
for the exact config that produced each checkpoint.

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

- [checkpoints/](checkpoints/) — 6 pretrained encoders (see above)
- [src/data/well.py](src/data/well.py) — Well → (C,T,H,W) clip dataset
- [src/masking.py](src/masking.py) — shared tube masking
- [src/models/vit.py](src/models/vit.py) — shared ViT encoder
- [src/objectives/mae.py](src/objectives/mae.py), [src/objectives/jepa.py](src/objectives/jepa.py) — the two heads
- [src/train.py](src/train.py) — unified entrypoint
- [scripts/gen_configs.py](scripts/gen_configs.py) — regenerates `configs/`
- [scripts/preprocess_memmap.py](scripts/preprocess_memmap.py) — Well HDF5 → fast fp16 memmap (needed before training/analysis)
- [scripts/load_encoder.py](scripts/load_encoder.py) — load a checkpoint + sanity-check forward pass
- [scripts/linear_probe.py](scripts/linear_probe.py) — ridge-probe frozen features against physics targets
- [scripts/analyze_encoders.py](scripts/analyze_encoders.py), [scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py) — layer-wise pooled/non-pooled physics probing (the JEPA-vs-MAE comparison)
- [scripts/analyze_regime.py](scripts/analyze_regime.py), [scripts/extract_regime_metadata.py](scripts/extract_regime_metadata.py) — does pooled representation encode regime params?
- [scripts/forecast_content_probe.py](scripts/forecast_content_probe.py) — forecasting probe (see "Probing suite" above)
- [scripts/analyze_noise_robustness.py](scripts/analyze_noise_robustness.py) — Gaussian input-noise representation-stability probe (see "Probing suite" above)
- [scripts/run_sweep.sh](scripts/run_sweep.sh) — runs the full probing suite across all 3 datasets; outputs to `sweep_results/`, summarized in [reports/](reports/)
- [docs/](docs/) — research-summary writeup and per-probe deep dives (architecture, variables, results)

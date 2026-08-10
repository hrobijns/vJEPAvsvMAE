# vJEPA vs vMAE on The Well

Controlled comparison of a latent-prediction objective (V-JEPA-style) against a
pixel-reconstruction objective (VideoMAE-style) with **identical encoders**, on
2D physics simulations from [The Well](https://polymathic-ai.org/the_well/).
End goal: test whether the JEPA objective encodes more physical variables in
its latent space (probing suite = step 2, separate from this repo's step-1
training infrastructure).

See [docs/OVERVIEW.md](docs/OVERVIEW.md) for a research-summary writeup
(motivation, architecture, headline results), and
[docs/LINEAR_PROBE.md](docs/LINEAR_PROBE.md) / [docs/ROLLOUT_PROBE.md](docs/ROLLOUT_PROBE.md)
for a deep dive on each probing pipeline. This README covers setup and
reproduction.

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

| script | question |
|---|---|
| [scripts/analyze_encoders.py](scripts/analyze_encoders.py) | layer-wise, pooled: which layer best decodes each physical quantity (enstrophy, divergence, okubo-weiss, nematic order, ...)? |
| [scripts/analyze_encoders_local.py](scripts/analyze_encoders_local.py) | same targets, per-token (non-pooled) — does spatial detail help beyond pooling? |
| [scripts/analyze_regime.py](scripts/analyze_regime.py) | does the pooled representation know the *regime* (Reynolds/Schmidt, Rayleigh/Prandtl, activity/alignment) — one value per trajectory, probed with train/val split by trajectory to avoid leakage |
| [scripts/rollout_probe.py](scripts/rollout_probe.py) | using each model's own pretrained predictor/decoder under a causal (non-tube) mask, does it forecast a genuinely future window better than a persistence baseline? Single-shot only, not fed back — see the next row for that. |
| [scripts/rollout_assessment.py](scripts/rollout_assessment.py) | **genuine autoregressive rollout**: encoder → latent dynamics predictor → small decoder → decoded future window → re-encode → predict again, chained for many steps. Does the representation encode enough about the dynamics to forecast forward, and does that forecast stay usable as errors compound? Compares a fed-back chain against an oracle (always re-seeded with real context) and a persistence floor. Needs a one-time prerequisite: `scripts/train_rollout_heads.py` (see below). |
| [scripts/forecast_content_probe.py](scripts/forecast_content_probe.py) | sidesteps the predictor/decoder entirely — does a *fresh* ridge probe on frozen present-time features forecast future physics, swept over multiple time gaps? |
| [scripts/analyze_noise_robustness.py](scripts/analyze_noise_robustness.py) | injects Gaussian noise (several std levels) directly into the input physical variables before encoding, then ridge-probes against the *clean* clip's physics — does decodability degrade gracefully or collapse abruptly? Found: MAE is often the sharper decoder at zero noise but degrades far more sharply (occasionally collapsing outright) than JEPA once the input is corrupted, on 2 of 3 datasets. |

Supporting scripts: [scripts/extract_regime_metadata.py](scripts/extract_regime_metadata.py)
(recovers per-trajectory regime params from Well filenames, needed by
`analyze_regime.py`) and [scripts/load_predictor.py](scripts/load_predictor.py)
(loads a JEPA run's full predictor + EMA target encoder from `latest.pt`, needed
by `rollout_probe.py`). [scripts/run_sweep.sh](scripts/run_sweep.sh) runs the
full probing suite across all 3 datasets on a remote GPU box; results land in
`sweep_results/*.json`, summarized in
[reports/](reports/) (per-dataset PDF summaries + `probing_sweep_report.html`).

**`rollout_assessment.py`'s prerequisite**: unlike the other probes, it needs a
latent dynamics predictor and small pixel decoder trained *post-hoc* on top of
each frozen encoder — neither JEPA's original predictor (trained jointly with
its own encoder to fill in masked patches within one clip) nor MAE's decoder
(which fuses "predict" and "decode" into one call) fit the genuine
next-window-forecasting task this probe needs. Both objectives get the same
architecture and training recipe (see
[src/objectives/rollout_heads.py](src/objectives/rollout_heads.py)), so the
only difference between the two pipelines is which frozen encoder sits
underneath. Run once with
[scripts/run_rollout_heads_training.sh](scripts/run_rollout_heads_training.sh)
(6 dataset × objective combos, configs generated by
[scripts/gen_rollout_heads_configs.py](scripts/gen_rollout_heads_configs.py))
before `rollout_assessment.py` or `run_sweep.sh` will work.

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
| **JEPA** | EMA target encoder (0.996→1.0) + 6-layer predictor (384d) → smooth-L1 on layer-normed target features at masked positions |

Datasets (per-dataset model pairs, 6 runs): `active_matter` (11ch, 256×256),
`shear_flow` (4ch, 256×128, incompressible NS), `rayleigh_benard` (4ch, 512×128).
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
- [scripts/rollout_probe.py](scripts/rollout_probe.py), [scripts/load_predictor.py](scripts/load_predictor.py), [scripts/forecast_content_probe.py](scripts/forecast_content_probe.py) — forecasting probes (see "Probing suite" above)
- [src/objectives/rollout_heads.py](src/objectives/rollout_heads.py) — post-hoc latent dynamics predictor + small decoder trained on a frozen encoder, for genuine autoregressive rollout
- [scripts/train_rollout_heads.py](scripts/train_rollout_heads.py), [scripts/gen_rollout_heads_configs.py](scripts/gen_rollout_heads_configs.py), [scripts/run_rollout_heads_training.sh](scripts/run_rollout_heads_training.sh) — trains those heads (one-time prerequisite for `rollout_assessment.py`)
- [scripts/load_rollout_heads.py](scripts/load_rollout_heads.py), [scripts/rollout_assessment.py](scripts/rollout_assessment.py) — loads the trained heads and runs the autoregressive rollout assessment (see "Probing suite" above)
- [scripts/analyze_noise_robustness.py](scripts/analyze_noise_robustness.py) — Gaussian input-noise representation-stability probe (see "Probing suite" above)
- [scripts/run_sweep.sh](scripts/run_sweep.sh) — runs the full probing suite across all 3 datasets; outputs to `sweep_results/`, summarized in [reports/](reports/)
- [docs/](docs/) — research-summary writeup and per-probe deep dives (architecture, variables, results)

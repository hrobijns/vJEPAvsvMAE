# vJEPA vs vMAE on The Well — research summary

Companion to the top-level [README.md](../README.md), which covers setup and
reproduction. This is the results-facing writeup: what we're testing, how,
and what we've found so far.

## Question

Train two self-supervised video encoders — one with a **latent-prediction**
objective (V-JEPA-style: predict masked patches' *features*), one with a
**pixel-reconstruction** objective (VideoMAE-style: predict masked patches'
*pixels*) — with **identical ViT-S encoders**, on 2D physics simulations from
[The Well](https://polymathic-ai.org/the_well/). Then freeze the encoders and
probe what physics each one's latent space actually encodes. Does predicting
in latent space push the encoder toward representing more/different physical
structure than predicting in pixel space?

## Setup

Three datasets, one JEPA/MAE model pair each (6 runs total): `active_matter`
(11ch, 256×256, active nematics), `shear_flow` (4ch, 256×512, incompressible
NS + tracer), `rayleigh_benard` (4ch, 512×128, buoyancy-driven convection).

| shared | ViT-S encoder (384d × 12 blocks, 6 heads), 2×16×16 tubelet patches, tube masking @ 0.9, T=8 clips at native resolution, per-channel z-score norm, AdamW + cosine, 100k steps, same batch/steps |
|---|---|
| **MAE head** | 4-layer decoder (192d) → MSE on masked patches (`norm_pix`) |
| **JEPA head** | EMA target encoder (0.996→1.0) + 6-layer predictor (384d) → smooth-L1 on layer-normed target features at masked positions |

Full architecture/training detail in [README.md § Design](../README.md#design).

## Two probing pipelines

- **[Linear probe suite](LINEAR_PROBE.md)** — does the *frozen representation*
  linearly encode physics (contemporaneous quantities, trajectory-level
  regime parameters, future content, noise robustness)? No predictor or
  decoder involved — pure ridge regression on frozen features.
- **[Rollout probe suite](ROLLOUT_PROBE.md)** — can the model actually
  *forecast forward*, using its predictor/decoder to generate future
  latents/pixels, single-step and chained autoregressively?

## Headline findings

All numbers below are ridge-probe R² (5-fold CV); "best-layer" means the best
of the 13 checkpoint layers (12 blocks + final norm). Full tables in the two
probe docs; raw numbers in `sweep_results/*.json`.

**1. Contemporaneous physics — JEPA and MAE are close on `active_matter` and
`shear_flow`, but JEPA clearly ahead on `rayleigh_benard`.** Both objectives
decode "obvious" derived quantities well (enstrophy, order parameters,
pressure/buoyancy gradients: R² > 0.95 on `active_matter`/`shear_flow`); raw
signed vorticity and divergence are hard for both everywhere (near 0 or
negative R², likely a genuinely hard/high-frequency target rather than a
representational gap). On `rayleigh_benard`, JEPA is well ahead of MAE on
every dynamically-interesting quantity:

| quantity | JEPA best-R² | MAE best-R² |
|---|---|---|
| `velocity_buoyancy_coherence` | 0.705 | 0.033 |
| `convective_flux` | 0.870 | 0.588 |
| `okubo_weiss` | 0.799 | 0.466 |
| `enstrophy` | 0.774 | 0.467 |
| `future_enstrophy` | 0.628 | 0.431 |

**2. Regime parameters (Reynolds/Schmidt, Rayleigh/Prandtl, alpha/zeta) —
roughly matched, MAE slightly ahead in several cases.** Both objectives
decode the constant per-trajectory regime very well (R² 0.77–0.999 at
mid-layers), with shuffled-control R² near 0 confirming no train/val leakage.
This is not where JEPA/MAE differ.

**3. Autoregressive rollout — both objectives degrade similarly, and both
fall behind a naive persistence baseline once errors compound.** Across 8
autoregressive steps, `fed_back` rollout (using the model's own predicted
latents as input for the next step) drops well below the `oracle` baseline
(re-seeded with real context each step) for both objectives, and on
`active_matter`/`rayleigh_benard` it ends up *worse* than simply predicting
"nothing changes" (persistence). JEPA and MAE track each other closely here —
this is not a dimension where JEPA's advantage on `rayleigh_benard` shows up.

**4. Noise robustness — MAE is sharper on clean input, JEPA is more robust to
corruption.** At zero injected noise, MAE's final-layer R² is higher than
JEPA's on all three datasets (e.g. `active_matter` 0.562 vs 0.424). But under
heavy Gaussian corruption of the input (z-scored std 2.0), MAE collapses on
`active_matter` (R² → −2.17) and degrades sharply on `rayleigh_benard`
(0.708 → 0.486, vs JEPA's 0.695 → 0.628); the two are comparable on
`shear_flow`. So MAE's representation is sharper but more brittle; JEPA's is
blunter but degrades more gracefully.

## Status / caveats

- `reports/*.pdf` and `reports/probing_sweep_report.html` were generated
  before the rollout-assessment and noise-robustness probes existed
  (~11 days stale relative to those results) — treat this doc and
  `sweep_results/*.json` as the current source of truth, not the PDFs.
- LR values (JEPA 1e-4, MAE 5e-5) are the result of a completed LR sweep on
  `active_matter` (see `scripts/gen_configs.py` comment), not placeholders.

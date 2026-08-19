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
| **JEPA head** | EMA target encoder (0.996→1.0) + 6-layer predictor (192d, narrower than the 384d encoder) → smooth-L1 on layer-normed target features at masked positions |

Full architecture/training detail in [README.md § Design](../README.md#design).

### Training curves

All runs converge cleanly over the full 100k steps (JEPA's smooth-L1-on-latents
loss and MAE's pixel-MSE loss are on different scales and not directly
comparable). JEPA's loss on `shear_flow` and `rayleigh_benard` shows a
transient rise-then-recover in the first ~30k steps in the original
single-seed runs — worth a caveat/footnote if this ever gets plotted for the
paper, since we haven't dug into whether it's EMA-target warmup or something
else. For the 3-seed `rayleigh_benard` runs, each `runs/<run_name>/history.jsonl`
records train+val loss and feature-std diagnostics directly (`src/train.py`'s
held-out val loop, no wandb dependency) — this is the current source for
training curves; `scripts/plot_training_curves.py` can render it. (The
original single-seed runs' curves depended on now-removed `pod_logs/wandb/`
offline logs and are no longer reproducible from this repo alone — only the
qualitative observation above survives.)

## Probing pipeline

**[Linear probe suite](LINEAR_PROBE.md)** — does the *frozen representation*
linearly encode physics (contemporaneous quantities, trajectory-level regime
parameters, future content, noise robustness)? No predictor or decoder
involved — pure ridge regression on frozen features.

## Headline findings

`rayleigh_benard` numbers below are from the held-out, multi-seed test
evaluation (`scripts/workshop_test_eval.py`, 3 independently-trained encoder
seeds per objective — the paper's citable numbers). `active_matter`/
`shear_flow` numbers are single-seed. Full tables in
[LINEAR_PROBE.md](LINEAR_PROBE.md); raw numbers in
`sweep_results/*_workshop_test_eval.json`.

**1. Contemporaneous physics on `rayleigh_benard` — close, with a
quantity-specific split.** Pooled deltas are small (|Δ| ≤ 0.023) — MAE
fractionally ahead on small-scale/high-frequency quantities, JEPA ahead on
`buoyancy_grad`:

| quantity | JEPA mean±std | MAE mean±std |
|---|---|---|
| `buoyancy_grad` | 0.965±0.019 | 0.942±0.006 |
| `convective_flux` | 0.985±0.002 | 0.992±0.001 |
| `enstrophy` | 0.991±0.002 | 0.996±0.001 |
| `okubo_weiss` | 0.993±0.000 | 0.996±0.000 |
| `velocity_buoyancy_coherence` | 0.945±0.002 | 0.954±0.003 |

Purely local, near-differential quantities (`pressure_grad_mag`) are at
ceiling for both objectives (≥0.997) — a pixel-reconstruction objective
directly rewards keeping exactly this kind of information.

**2. JEPA's forecast advantage grows with horizon.** At t+32, JEPA leads MAE
on 6 of 7 pooled quantities (`buoyancy_grad` +0.121±0.012,
`velocity_buoyancy_coherence` +0.112±0.039), reversing the near-tie (MAE
fractionally ahead) at t+8. JEPA is the only objective explicitly trained to
predict *in time*, not just reconstruct the present.

**3. Regime parameters (Rayleigh, Prandtl) — decoded near-ceiling by both,
not where JEPA/MAE differ.** R² ≥ 0.99 for both objectives; shuffled-control
R² near 0 confirms no leakage.

**4. Noise robustness — JEPA is substantially more robust at low-to-moderate
corruption, though this compresses at extreme noise.** Regenerated this
session with an improved clean-fit protocol (probe fit once on clean
features, never recalibrated — the original matched-noise design was
measuring readout re-adaptation more than representation robustness). At
σ=0.1, JEPA's Pearson r on `buoyancy_grad` is 0.91 vs MAE's 0.30; the gap
narrows through σ=0.3–0.5 and at σ=1.0 the two are close or MAE edges ahead
on a few quantities (`enstrophy`, `okubo_weiss`). Full grid (both R² and
Pearson r, ridge and MLP) in [LINEAR_PROBE.md](LINEAR_PROBE.md).

**5. `rayleigh_benard` is the paper's headline dataset — the only one with a
complete 3-seed run at the current architecture.** `active_matter`/
`shear_flow` show a similar clean-data wash between objectives but their
current-architecture (192-d predictor) 3-seed JEPA retrain stopped
incomplete (recurring pod GPU-host fault); see "Status / caveats" below.

## Status / caveats

- Figures live in [reports/figures/](../reports/figures/),
  generated by `scripts/plot_workshop_figures.py` from
  `sweep_results/*_workshop_test_eval.json` — treat this doc,
  [LINEAR_PROBE.md](LINEAR_PROBE.md), and those JSON files as the current
  source of truth.
- LR is tuned **per (dataset, objective)**, not a single global value — see
  `configs/tuned_lr.json` (selected via a short mini-sweep on a pilot seed,
  on held-out validation loss; `scripts/run_lr_minisweep.sh` +
  `scripts/pick_lr.py`).
- **Held-out, multi-seed test evaluation: complete for `rayleigh_benard`
  (3 encoder seeds × {JEPA, MAE}), single-seed for `active_matter`/
  `shear_flow`.** `scripts/workshop_test_eval.py` selects each layer on a
  regime-balanced slice carved out of `train` (the shipped `valid` split was
  too small/regime-degenerate to use directly), then fits seeded MLPs on
  `train` and reports mean±std R² on `test` (10/28/50 trajectories for
  active_matter/shear_flow/rayleigh_benard), never touched until that final
  step. A broad sanity check (comparing every MLP number against a
  closed-form ridge fit at the same frozen layer) found the original fixed
  `weight_decay` let the MLP badly overfit a handful of hard, low-SNR
  targets; fixed by making `weight_decay` itself adaptively selected per
  layer (including ridge as a fair candidate in the same held-out
  comparison). Full protocol and numbers:
  [LINEAR_PROBE.md](LINEAR_PROBE.md#held-out-multi-seed-test-evaluation-the-reported-numbers).
  `active_matter`/`shear_flow`'s current-architecture (192-d predictor)
  3-seed JEPA retrain was in progress but stopped incomplete (recurring pod
  GPU-host fault); their committed checkpoints/numbers predate the
  384→192-d predictor-narrowing (MAE is unaffected — no predictor).
- **Held-out validation loss during pretraining: implemented.** `src/train.py`
  now trains against a trajectory-disjoint validation split carved out of
  `train` (the shipped `valid` split was too small/degenerate for this too),
  tracks best-val-loss checkpoint selection independently per objective (JEPA's
  latent loss and MAE's pixel loss are never compared to each other), and logs
  per-dimension feature-std collapse diagnostics for both objectives
  throughout training. Every milestone checkpoint
  (`encoder_{025,050,075,100}pct.pt` + `encoder_best_val.pt`) is retained per
  run, not deleted, so probe performance can be reconstructed at any step if
  needed. This closes the gap noted in earlier revisions of this doc.

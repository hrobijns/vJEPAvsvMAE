# Linear probe suite

Part of the [step-2 probing study](OVERVIEW.md). This suite asks one
question in several variants: **does the frozen encoder representation
linearly (or near-linearly) contain a given piece of physics** — no
predictor, no decoder, just ridge regression from frozen features to a
target. Contrast with the [rollout probe suite](ROLLOUT_PROBE.md), which
uses the predictor/decoder to actually generate forecasts.

## Method

Ridge regression (5-fold cross-validated), reported as R², from encoder
features to a physics target, computed **per transformer layer** (12 blocks +
final norm = 13 "layers" per checkpoint) so the analysis shows where in depth
each quantity becomes decodable. Two feature poolings are used depending on
the script: mean-pooled over all tokens ("pooled"), or per-token
("non-pooled" / "local").

## Scripts (form a suite, not independent tools)

Most later scripts import shared utilities (`ridge_r2`, `build_dataset`,
`contemporaneous_targets`, `compute_layerwise_features_batched`,
`load_checkpoint_encoder`) from `analyze_encoders.py`, and
(`local_target_maps`, `layerwise_token_features`) from
`analyze_encoders_local.py`.

| script | question | notes |
|---|---|---|
| [`scripts/linear_probe.py`](../scripts/linear_probe.py) | Basic pass/fail sanity check: pooled features vs. a near-circular energy target, vs. a pixel-mean baseline. | Simplest/oldest probe, saturates near R²=1 (target is close to circular) — a smoke test, not part of the main comparison. |
| [`scripts/analyze_encoders.py`](../scripts/analyze_encoders.py) | Main comparison: layer-wise, pooled, ridge-probe against **derived nonlinear physics targets** (dataset-specific — see below) plus `future_enstrophy` (a horizon-ahead target, contemporaneous features only, no predictor). | Also hosts shared utilities used across the whole suite. |
| [`scripts/analyze_encoders_local.py`](../scripts/analyze_encoders_local.py) | Follow-up: is pooling hiding something? (1) per-token linear probe with patch-aligned local targets (`avg_pool3d`), (2) small MLP nonlinear probe on the same pooled features as above. | `--skip-mlp` used in sweeps — MLP results were noisy/unreliable; lean sweeps use linear probes only. |
| [`scripts/analyze_regime.py`](../scripts/analyze_regime.py) | Does the pooled representation know the constant-per-trajectory physical regime (Reynolds/Schmidt, Rayleigh/Prandtl, alpha/zeta)? | One feature vector per trajectory, split by trajectory (never across) to avoid leakage; multi-order-of-magnitude params log10-transformed; includes a shuffled-control column per target as a leakage sanity check. Needs `scripts/extract_regime_metadata.py` sidecar files first. |
| [`scripts/forecast_content_probe.py`](../scripts/forecast_content_probe.py) | Trains a **fresh** ridge probe from present-time frozen features to a **separate future window's** physics, swept over multiple time gaps (`--gaps`). | No predictor/decoder at all — removes the "does the pretrained head generalize to new mask geometry" confound present in `rollout_probe.py`. |
| [`scripts/analyze_noise_robustness.py`](../scripts/analyze_noise_robustness.py) | Sweeps injected Gaussian noise (`--noise-stds`, in z-scored input units) into the input, ridge-probes final-layer pooled features against the **clean** clip's physics. | Does decodability degrade gracefully or collapse abruptly under corruption? |

Supporting: [`scripts/extract_regime_metadata.py`](../scripts/extract_regime_metadata.py)
reconstructs per-trajectory regime params from Well filenames (best-effort
regex parsing), needed by `analyze_regime.py`.

## Targets per dataset

| dataset | contemporaneous/derived targets | regime targets |
|---|---|---|
| `active_matter` | enstrophy, vorticity_signed, divergence, nematic_order, flow_align, order_grad_mag, strain_order_align, future_enstrophy | alpha, zeta |
| `shear_flow` | enstrophy, vorticity_signed, divergence, tracer_grad, advective_flux, strain_rate_mag, okubo_weiss, pressure_grad_mag, future_enstrophy | Reynolds, Schmidt |
| `rayleigh_benard` | enstrophy, vorticity_signed, divergence, buoyancy_grad, convective_flux, okubo_weiss, velocity_buoyancy_coherence, pressure_grad_mag, future_enstrophy | Rayleigh, Prandtl |

## Results

### Contemporaneous decodability (pooled), best-layer R², JEPA vs MAE

`active_matter` (n=525 clips):

| quantity | JEPA best (layer) | MAE best (layer) |
|---|---|---|
| enstrophy | 0.989 (L1) | 0.993 (L1) |
| vorticity_signed | −0.044 (L0) | −0.047 (L10) |
| divergence | −0.231 (L0) | −0.177 (L5) |
| nematic_order | 0.995 (L2) | 0.999 (L3) |
| flow_align | 0.444 (L0) | 0.421 (L1) |
| order_grad_mag | 0.995 (L1) | 0.997 (L2) |
| strain_order_align | 0.997 (L1) | 0.998 (L1) |
| future_enstrophy | 0.924 (L2) | 0.931 (L7) |

`shear_flow` (n=896 clips):

| quantity | JEPA best (layer) | MAE best (layer) |
|---|---|---|
| enstrophy | 0.997 (L2) | 0.996 (L3) |
| vorticity_signed | 0.109 (L8) | 0.078 (L1) |
| divergence | 0.626 (L5) | 0.619 (L7) |
| tracer_grad | 0.992 (L1) | 0.994 (L1) |
| advective_flux | 0.809 (L3) | 0.796 (L6) |
| strain_rate_mag | 1.000 (L5) | 1.000 (L3) |
| okubo_weiss | 0.999 (L2) | 0.999 (L3) |
| pressure_grad_mag | 1.000 (L2) | 1.000 (L3) |
| future_enstrophy | 0.985 (L2) | 0.982 (L4) |

`rayleigh_benard` (n=1400 clips) — the dataset where JEPA and MAE diverge
most:

| quantity | JEPA best (layer) | MAE best (layer) |
|---|---|---|
| enstrophy | 0.774 (L6) | 0.467 (L2) |
| buoyancy_grad | 1.000 (L7) | 1.000 (L2) |
| convective_flux | 0.870 (L6) | 0.588 (L2) |
| okubo_weiss | 0.799 (L6) | 0.466 (L2) |
| velocity_buoyancy_coherence | 0.705 (L12) | 0.033 (L0) |
| pressure_grad_mag | 1.000 (L2) | 1.000 (L1) |
| future_enstrophy | 0.628 (L6) | 0.431 (L3) |

(`vorticity_signed`/`divergence` are NaN for this dataset — target is
degenerate/ill-defined for `rayleigh_benard`'s flow, not a probing failure.)

Pattern: on `rayleigh_benard`, MAE's best layer is consistently early (L1–L2)
while JEPA's best layer is mid-to-late (L6–L12) — MAE's representation seems
to plateau early and not build up the deeper nonlinear structure JEPA does on
this dataset. On `active_matter`/`shear_flow` both objectives peak early
(L1–L3) and reach similar R², so this pattern is dataset-specific rather than
a general property of the objective.

### Regime-parameter decodability (pooled), best-layer R²

| dataset | param | JEPA best | MAE best | shuffled control (both) |
|---|---|---|---|---|
| active_matter | alpha | 0.997 | 0.998 | ≈ −0.4 to −0.5 |
| active_matter | zeta | 0.959 | 0.975 | ≈ −0.6 to −0.7 |
| shear_flow | Reynolds | 0.805 | 0.822 | ≈ −0.05 |
| shear_flow | Schmidt | 0.774 | 0.825 | ≈ −0.06 |
| rayleigh_benard | Rayleigh | 0.998 | 0.999 | ≈ −0.05 |
| rayleigh_benard | Prandtl | 0.993 | 0.993 | ≈ −0.06 |

Both objectives decode regime parameters well and are closely matched (MAE
marginally ahead in most cases); shuffled controls near 0 confirm the
train/val-by-trajectory split isn't leaking. Regime decodability is not where
JEPA and MAE differ.

### Noise robustness (pooled, final layer), mean R² across quantities

| dataset | noise std → | 0.0 | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| active_matter | JEPA | 0.424 | 0.438 | 0.436 | 0.417 | 0.287 | −0.112 |
| active_matter | MAE | 0.562 | 0.566 | 0.557 | 0.533 | −0.024 | **−2.165** |
| shear_flow | JEPA | 0.687 | 0.687 | 0.693 | 0.687 | 0.610 | 0.501 |
| shear_flow | MAE | 0.737 | 0.735 | 0.734 | 0.718 | 0.652 | 0.539 |
| rayleigh_benard | JEPA | 0.695 | 0.697 | 0.695 | 0.694 | 0.675 | 0.628 |
| rayleigh_benard | MAE | 0.708 | 0.708 | 0.703 | 0.689 | 0.638 | 0.486 |

MAE starts higher (sharper decoder at zero noise) on all three datasets, but
degrades faster once the input is corrupted — outright collapsing on
`active_matter` and dropping well below JEPA on `rayleigh_benard`; the two
are close on `shear_flow`. Noise std is in z-scored input units (std ≥ 1.0 is
heavy corruption relative to the unit-variance input).

Raw per-quantity numbers: `sweep_results/{dataset}_pooled.json`,
`sweep_results/{dataset}_nonpooled.json`, `sweep_results/{dataset}_regime.json`,
`sweep_results/{dataset}_noise_robustness.json`.

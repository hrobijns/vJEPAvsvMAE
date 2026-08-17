# Linear probe suite

Part of the [step-2 probing study](OVERVIEW.md). This suite asks one
question in several variants: **does the frozen encoder representation
linearly (or near-linearly) contain a given piece of physics** — no
predictor, no decoder, just ridge regression from frozen features to a
target.

## Method

Ridge regression (5-fold cross-validated) and a small MLP (2-layer, nonlinear
readout), both reported as R², from encoder features to a physics target,
computed **per transformer layer** (12 blocks + final norm = 13 "layers" per
checkpoint) so the analysis shows where in depth each quantity becomes
decodable. Two feature poolings are used depending on the script: mean-pooled
over all tokens ("pooled"), or per-token ("non-pooled" / "local"). Token-level
probing is available for every experiment in the linear-probe suite,
including noise robustness (capped to a sample subset — token tensors are
~n_tokens x larger per sample).

## Physics targets, derived from each simulation's governing PDE

Every target is tied to a specific term in the dataset's actual governing
PDE (sourced from [The Well](https://polymathic-ai.org/the_well/)'s own
per-dataset documentation), not chosen ad hoc:

**`active_matter`** — active nematics / Stokesian hydrodynamics of rod-like
active particles (Maddu et al. 2024, *J. Comput. Phys.* 504:112869). Channels:
concentration `c`, velocity `vx,vy`, order tensor `D`, strain-rate tensor `E`.
Domain `Lx=Ly=10`, 256×256, periodic BC.

| target | PDE term | why |
|---|---|---|
| `enstrophy` | `mean(curl(v)^2)` | turbulent-intensity diagnostic |
| `nematic_order` | `mean\|D\|` | nonlinear order-tensor invariant |
| `flow_align` | `mean(v·D)` | advection-alignment term in the Q-tensor evolution PDE |
| `order_grad_mag` | `mean\|∇D\|` | elastic/Frank-energy term |
| `strain_order_align` | `mean(E:D)` | co-rotational term (standard in Beris–Edwards nematodynamics) |
| `active_stress_div_mag` | `mean\|∇·D\|` | the active-stress term `alpha·∇·D` in the momentum equation — ties the regime parameter `alpha` to a concrete local field |
| `future_enstrophy` | enstrophy at a future frame | horizon-ahead target, contemporaneous features only, no predictor |

**`shear_flow`** — incompressible NS + passive tracer:
`∂u/∂t + ∇p − ν∆u = −u·∇u`, `∂s/∂t − D∆s = −u·∇s`, `ν=1/Re`, `D=ν/Sc`.
Domain `x∈[0,1]`×`y∈[-1,1]`, 256×512, periodic BC.

| target | PDE term | why |
|---|---|---|
| `enstrophy` | `mean(curl(v)^2)` | turbulent-intensity diagnostic |
| `tracer_grad` | `mean\|∇s\|^2` | derived differential quantity |
| `advective_flux` | `mean(u·∇s)` | advection term in the tracer PDE |
| `strain_rate_mag` | strain-rate tensor norm | velocity-gradient structure |
| `okubo_weiss` | strain² − vorticity² | strain- vs. vorticity-dominated regions |
| `pressure_grad_mag` | `mean\|∇p\|` | explicit PDE term |
| `tracer_laplacian` | `mean\|∆s\|` | the *diffusive* term in the tracer PDE (coefficient `D=ν/Sc`) — distinct from `tracer_grad` (2nd- vs. 1st-derivative, different frequency content) |
| `future_enstrophy` | enstrophy at a future frame | horizon-ahead target |

**`rayleigh_benard`** — Boussinesq convection: `∂b/∂t − κ∆b = −u·∇b`,
`∂u/∂t − ν∆u + ∇p − b·ê_y = −u·∇u`, `κ=(Ra·Pr)^{-1/2}`, `ν=(Ra/Pr)^{-1/2}`.
Domain `x∈[0,4]` (uniform) × `y∈[0,1]` (**Chebyshev nodes**, see caveat
below), 512×128, periodic in x, no-slip Dirichlet in y.

| target | PDE term | why |
|---|---|---|
| `enstrophy` | `mean(curl(v)^2)` | dissipation-relevant, real physical quantity |
| `buoyancy_grad` | `mean\|∇b\|^2` | relates to the diffusive term `κ∆b` |
| `convective_flux` | `mean(vy·b)` | the buoyancy-work term `b·ê_y` dotted with velocity — the textbook RB order parameter for convective heat transport |
| `okubo_weiss` | strain² − vorticity² | plumes vs. shear |
| `velocity_buoyancy_coherence` | normalized `convective_flux` | dimensionless coherence version |
| `pressure_grad_mag` | `mean\|∇p\|` | explicit PDE term |
| `buoyancy_laplacian` | `mean\|∆b\|` | the diffusive term whose coefficient is `κ` (hence `Ra, Pr`) — mirrors `shear_flow`'s `tracer_laplacian` |
| `future_enstrophy` | enstrophy at a future frame | horizon-ahead target |

**Dropped as pooled targets (all three datasets): `vorticity_signed`,
`divergence`.** On a periodic (or no-slip) domain, the *spatial mean* of
`curl(v)` and `div(v)` is exactly 0 by Stokes'/the divergence theorem,
regardless of the underlying physics — opposite boundaries cancel (periodic)
or velocity is pinned to 0 at the wall (no-slip). This is a degenerate target
with no signal to probe, which is exactly why earlier results showed
near-0/negative R² for these two everywhere, and literally NaN on
`rayleigh_benard` (zero-variance target). **`enstrophy` is not affected** —
it's the mean of vorticity *squared*, a genuine, non-degenerate
turbulent-intensity diagnostic (R² ~ 0.95–0.99 in practice). The *pointwise*
`vorticity_signed`/`divergence` fields are not degenerate (only their spatial
mean is), so both remain available as **token-level** targets.

**Caveat (documented, not fixed):** `rayleigh_benard`'s y-axis is sampled at
Chebyshev nodes (per The Well's own docs), not a uniform grid. All spatial
derivatives here (`buoyancy_grad`, `pressure_grad_mag`, `okubo_weiss`,
`buoyancy_laplacian`, and the token-level `vorticity_signed`/`divergence`)
are computed via uniform-spacing finite differences (`torch.roll` shift by
1 pixel), which is exact for `active_matter`/`shear_flow` (genuinely uniform
grids) but an approximation for `rayleigh_benard`'s y-direction — worst near
the top/bottom boundaries where Chebyshev nodes cluster. Scoped out of this
round; would need per-row y-coordinates and non-uniform finite-difference
weights to fix properly.

**Per-token calculation** (all targets, every dataset): compute the derived
field at full pixel resolution, then `F.avg_pool3d` over the exact
`(patch_t, patch_h, patch_w)` tubelet footprint (`local_target_maps()` in
`scripts/analyze_encoders_local.py`), so each token's target is the average
of that quantity over exactly the spacetime volume the token itself
summarizes.

## Scripts (form a suite, not independent tools)

Most later scripts import shared utilities (`ridge_r2`, `ridge_r2_grouped`,
`build_dataset`, `contemporaneous_targets`, `compute_layerwise_features_batched`,
`load_checkpoint_encoder`) from `analyze_encoders.py`, and
(`local_target_maps`, `layerwise_token_features`, `mlp_r2`) from
`analyze_encoders_local.py`.

| script | question | notes |
|---|---|---|
| [`scripts/linear_probe.py`](../scripts/linear_probe.py) | Basic pass/fail sanity check: pooled features vs. a near-circular energy target, vs. a pixel-mean baseline. | Simplest/oldest probe, saturates near R²=1 (target is close to circular) — a smoke test, not part of the main comparison. |
| [`scripts/analyze_encoders.py`](../scripts/analyze_encoders.py) | Main comparison: layer-wise, pooled, ridge-probe against **derived nonlinear physics targets** (dataset-specific — see above) plus `future_enstrophy`. | Also hosts shared utilities used across the whole suite. |
| [`scripts/analyze_encoders_local.py`](../scripts/analyze_encoders_local.py) | Follow-up: is pooling hiding something? (1) per-token linear probe with patch-aligned local targets, (2) small MLP nonlinear probe on **pooled** features, (3) small MLP nonlinear probe **per-token** (same features/targets as (1), swapping ridge for the MLP). | `--skip-mlp` skips both MLP variants. The pooled MLP was flagged noisy/unreliable in earlier runs — per [arXiv:2602.07050](https://arxiv.org/abs/2602.07050) ("Interpreting Physics in Video World Models"), attentive/per-token probes recover structure that pooled-then-nonlinear probes miss because mean-pooling destroys the local structure a small MLP needs; the per-token MLP tests whether that was a pooling artifact. |
| [`scripts/analyze_regime.py`](../scripts/analyze_regime.py) | Does the representation know the constant-per-trajectory physical regime (Reynolds/Schmidt, Rayleigh/Prandtl, alpha/zeta)? Pooled by default; `--token-level` additionally asks whether a *single token* already predicts the regime, or whether it only emerges from pooling. | One feature vector per trajectory (pooled) or grouped-by-trajectory CV via `ridge_r2_grouped` (token-level) — both avoid leakage, since a plain random split would let tokens/windows from the same trajectory land in both train and val folds sharing an (near-)identical target. Multi-order-of-magnitude params log10-transformed; shuffled-control column per target as a leakage sanity check. Needs `scripts/extract_regime_metadata.py` sidecar files first. |
| [`scripts/forecast_content_probe.py`](../scripts/forecast_content_probe.py) | Trains a **fresh** linear **and MLP** probe (pooled and per-token, every layer) from present-time frozen features to a **separate future window's** physics, swept over multiple time gaps (`--gaps`) and injected-noise levels (`--noise-stds`, context only — same convention as `analyze_noise_robustness.py`). | No predictor/decoder at all — avoids the "does the pretrained head generalize to new mask geometry" confound a rollout-style probe would carry. Full rigor stack (depth × linear-vs-MLP × noise), matching the rest of the suite. Each result also reports a `skill_*` field (see below) alongside the raw `probe_*`/`ceiling_*` R². |
| [`scripts/analyze_noise_robustness.py`](../scripts/analyze_noise_robustness.py) | Sweeps injected Gaussian noise (`--noise-stds`, in z-scored input units) into the input, ridge-probes pooled **and per-token** features — at **every layer**, not just the final one — against the **clean** clip's physics. | Does decodability degrade gracefully or collapse abruptly under corruption, and does that hold at every depth or only at the output — and is that pattern uniform across a token's whole field, or does pooling wash it out? Per-token variant capped via `--token-max-samples` (default 64); for those capped batches, pooled features are derived by mean-pooling the per-token ones rather than a second forward pass. |

Supporting: [`scripts/extract_regime_metadata.py`](../scripts/extract_regime_metadata.py)
reconstructs per-trajectory regime params from Well filenames (best-effort
regex parsing), needed by `analyze_regime.py`.

## Experiment design: depth × noise robustness

Two axes the physics-content results above don't yet make explicit:
**depth** (where in the encoder does a quantity become linearly readable?)
and **noise robustness** (how fragile is that readability, and at what
depth?). Both are directly informed by
[arXiv:2602.07050](https://arxiv.org/abs/2602.07050), which studies an
analogous question (where physics emerges across depth in V-JEPA-2/VideoMAE)
and finds a sharp "Physics Emergence Zone" transition partway through the
encoder. `scripts/plot_probing_suite.py` implements three core plots from
`sweep_results/*.json`:

1. **Emergence-depth plot** — R² vs. layer, one curve per quantity, JEPA vs.
   MAE overlaid, plus a summary scatter of each quantity's `emergence_layer`
   (first layer reaching 80% of that quantity's own peak R²). Formalizes the
   qualitative "MAE peaks early (L1–2), JEPA peaks mid-late (L6–12) on
   `rayleigh_benard`" observation (see [OVERVIEW.md](OVERVIEW.md)) into a
   citable per-quantity number.
2. **Noise × layer heatmap** — R² over the full (layer, noise-std) grid per
   quantity, from `analyze_noise_robustness.py`'s now-layer-swept output.
   Answers "is robustness a final-layer-only property, or does it hold at
   intermediate depths too?"
3. **Token linear-vs-MLP comparison** — bar comparison of per-token linear
   vs. per-token MLP R² at the final layer, per quantity. Tests whether any
   quantity is present nonlinearly-but-not-linearly at the token level.
4. **Forecast skill vs. quantity difficulty** — skill score (defined in the
   Results section below) vs. `R²_persistence` per quantity, one point per
   (quantity, gap), JEPA vs. MAE. Visualizes whether an objective's forecast
   advantage concentrates in genuinely hard-to-forecast quantities or in ones
   that were already easy for the naive baseline.

## Results

Full sweep, current target set, all three datasets — pooled linear, per-token
linear + MLP, full 13-layer depth sweep, full 6-level noise sweep (pooled and
token), regime probing, and forecast/skill-score. Raw numbers:
`sweep_results/{dataset}_pooled.json`, `{dataset}_nonpooled.json`,
`{dataset}_regime.json`, `{dataset}_noise_robustness.json`,
`{dataset}_forecast.json`. Figures: `reports/figures/*.png`, generated by
`scripts/plot_probing_suite.py`.

These are train-split cross-validated numbers — useful for characterizing
the full depth/noise/quantity landscape, but **not** the final reported
numbers: reporting `max` over 13 layers of train-CV curves as "best-layer
R²" is itself an unvalidated selection. The held-out, multi-seed numbers
that should be cited as final results are in [Held-out, multi-seed test
evaluation](#held-out-multi-seed-test-evaluation-the-reported-numbers)
below.

### Contemporaneous decodability (pooled), best-layer R², JEPA vs MAE

`active_matter`:

| quantity | JEPA best | MAE best |
|---|---|---|
| enstrophy | 0.989 | 0.993 |
| nematic_order | 0.995 | 0.999 |
| flow_align | 0.444 | 0.421 |
| order_grad_mag | 0.995 | 0.997 |
| strain_order_align | 0.997 | 0.998 |
| active_stress_div_mag | 0.994 | 0.997 |
| future_enstrophy | 0.924 | 0.931 |

`shear_flow`:

| quantity | JEPA best | MAE best |
|---|---|---|
| enstrophy | 0.993 | 0.992 |
| tracer_grad | 0.994 | 0.996 |
| advective_flux | 0.775 | 0.760 |
| strain_rate_mag | 1.000 | 1.000 |
| okubo_weiss | 0.998 | 0.998 |
| pressure_grad_mag | 0.999 | 1.000 |
| tracer_laplacian | 0.988 | 0.991 |
| future_enstrophy | 0.979 | 0.978 |

`rayleigh_benard` — the dataset where JEPA and MAE diverge most:

| quantity | JEPA best | MAE best | gap |
|---|---|---|---|
| enstrophy | 0.686 | 0.455 | +0.23 |
| buoyancy_grad | 1.000 | 1.000 | — |
| convective_flux | 0.835 | 0.576 | +0.26 |
| okubo_weiss | 0.722 | 0.460 | +0.26 |
| velocity_buoyancy_coherence | 0.583 | −0.049 | +0.63 |
| pressure_grad_mag | 1.000 | 1.000 | — |
| buoyancy_laplacian | 0.997 | 0.994 | +0.00 |
| future_enstrophy | 0.474 | 0.347 | +0.13 |

The split is quantity-specific, not blanket: purely local, near-differential
quantities (`buoyancy_grad`, `pressure_grad_mag`, `buoyancy_laplacian` — the
ones a pixel-reconstruction objective directly rewards keeping) are at
ceiling for both objectives on every dataset. The gap opens specifically on
`rayleigh_benard`'s genuinely *coupled* quantities — `convective_flux` (the
literal buoyancy-work term driving convection) and
`velocity_buoyancy_coherence`, where MAE is at or below zero R². On
`active_matter`/`shear_flow`, clean pooled accuracy is a wash between the two
objectives — sometimes MAE fractionally ahead.

`vorticity_signed`/`divergence` are dropped as *pooled* targets everywhere
(see above) but remain informative at token level — see below.

### Depth: where does physics become linearly readable?

`emergence_layer` = first of 13 layers (L0–L12) where a quantity's pooled
R² crosses 80% of its own peak. On `active_matter` and `shear_flow`, both
objectives converge almost immediately — emergence layer 0–3 for nearly every
quantity, no "building up" story on either dataset despite real differences
on other axes (noise, below). `rayleigh_benard` is qualitatively different:
MAE's peak accuracy on the coupled quantities is reached by L0–L2 and doesn't
improve with depth; JEPA needs L6–L11 to reach 80% of its own (higher) peak.
Figures: `reports/figures/{dataset}_emergence_curves.png` (full layer-wise
curves) and `{dataset}_emergence_summary.png` (per-quantity summary).
MAE's early plateau + JEPA's mid-to-late peak on `rayleigh_benard` formalizes
the qualitative observation in [OVERVIEW.md](OVERVIEW.md) into a citable,
quantity-level number.

### Token level: does a nonlinear readout find what linear can't?

Per-token linear vs. 2-layer-MLP readout, best layer. On `rayleigh_benard`,
the coupled quantities show a real, objective-specific nonlinear-only gap —
JEPA's MLP gain lands near ceiling, MAE's stays low even nonlinearly:

| quantity | JEPA lin | JEPA MLP | MAE lin | MAE MLP |
|---|---|---|---|---|
| convective_flux | 0.285 | 0.895 | 0.006 | 0.067 |
| velocity_buoyancy_coherence | 0.258 | 0.778 | 0.005 | 0.059 |
| divergence | 0.147 | 0.445 | 0.006 | 0.114 |
| vorticity_signed | 0.064 | 0.351 | −0.001 | −0.012 |
| buoyancy_grad | 0.982 | 1.000 | 0.990 | 1.000 |

On `active_matter`/`shear_flow`, MLP gains are present and roughly symmetric
between objectives on every quantity (e.g. `active_matter` `enstrophy`: JEPA
0.702→0.991, MAE 0.686→0.993) — a generic pooling/small-model effect, not an
objective-specific one. MAE's near-zero `convective_flux`/
`velocity_buoyancy_coherence` on `rayleigh_benard` stays near-zero even with
the nonlinear readout — the information isn't extractable from MAE's local
features, linearly or not. Figures:
`reports/figures/{dataset}_token_linear_vs_mlp.png`.

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
grouped-by-trajectory split isn't leaking. A global, low-frequency,
trajectory-constant scalar is trivial for either pretraining objective to
preserve — regime decodability is not where JEPA and MAE differ. Useful as a
negative control: the JEPA/MAE split elsewhere in this doc is specific to
locally-coupled, spatially-resolved dynamics, not "physics content" in
general.

### Noise robustness (pooled, final layer), mean R² across quantities

| dataset | noise std → | 0.0 | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| active_matter | JEPA | 0.786 | 0.789 | 0.794 | 0.790 | 0.729 | 0.398 |
| active_matter | MAE | 0.874 | 0.878 | 0.871 | 0.831 | 0.530 | **−1.024** |
| shear_flow | JEPA | 0.832 | 0.830 | 0.841 | 0.836 | 0.780 | 0.680 |
| shear_flow | MAE | 0.902 | 0.901 | 0.898 | 0.884 | 0.837 | 0.718 |
| rayleigh_benard | JEPA | 0.933 | 0.936 | 0.935 | 0.930 | 0.905 | 0.842 |
| rayleigh_benard | MAE | 0.943 | 0.945 | 0.941 | 0.921 | 0.849 | 0.646 |

Same pattern at the **token** level (mean R² across quantities, final
layer), but with a dataset-specific twist:

| dataset | noise std → | 0.0 | 0.25 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| active_matter | JEPA | 0.620 | 0.512 | 0.374 | 0.270 |
| active_matter | MAE | 0.745 | 0.635 | 0.256 | **0.039** |
| shear_flow | JEPA | 0.380 | 0.368 | 0.276 | 0.143 |
| shear_flow | MAE | **0.820** | 0.704 | 0.524 | **0.344** |
| rayleigh_benard | JEPA | **0.855** | 0.808 | 0.679 | **0.553** |
| rayleigh_benard | MAE | 0.697 | 0.610 | 0.419 | 0.291 |

MAE starts higher (sharper decoder) at zero noise on every dataset, both
granularities, but degrades faster once the input is corrupted. The
collapse is dataset-specific in magnitude: outright negative (pooled) on
`active_matter`, moderate on `rayleigh_benard`, mild on `shear_flow`.
`shear_flow` is the one dataset where MAE stays *ahead* of JEPA at every
noise level, both pooled and token — this dataset shows no JEPA robustness
advantage at all (see Discussion for why). `rayleigh_benard` is the reverse
at token level: JEPA leads at every noise level, consistent with its
contemporaneous and depth results above. Noise std is in z-scored input
units (std ≥ 1.0 is heavy corruption relative to the unit-variance input).
Figures: `reports/figures/{dataset}_noise_heatmap_*.png` (layer × noise grid
per quantity).

### Forecast skill score: predicting future physics, fairly

`forecast_content_probe.py` probes *future* physics from present-time frozen
features (gap frames ahead), scored with the persistence-relative skill
score (see below) rather than raw R², so quantities of very different
intrinsic predictability are comparable. Pooled, clean input, best layer,
skill (MLP), gap=8 vs. gap=32:

| dataset | example quantity | JEPA skill (gap=8 → 32) | MAE skill (gap=8 → 32) |
|---|---|---|---|
| active_matter | strain_order_align | 0.904 → 0.936 | 0.912 → 0.928 |
| shear_flow | okubo_weiss | 0.978 → 0.935 | 0.978 → 0.930 |
| rayleigh_benard | convective_flux | 0.900 → 0.861 | 0.916 → 0.661 |
| rayleigh_benard | enstrophy | 0.891 → 0.759 | 0.916 → 0.522 |

On `active_matter`/`shear_flow`, MAE is fractionally ahead or tied on clean
pooled forecasting, mirroring the contemporaneous-probe result — and, unlike
`rayleigh_benard`, the gap doesn't widen at the longer horizon. On
`rayleigh_benard`, JEPA's advantage **grows** with forecast horizon: roughly
tied at gap=8, clearly ahead at gap=32 (`convective_flux` 0.861 vs 0.661;
`enstrophy` 0.759 vs 0.522) — MAE's forecast skill decays faster as the
target moves further from the input than JEPA's does.

**Under noise** (linear probe, final layer, gap=8), the split from the
contemporaneous probe reappears and sharpens: on `active_matter`, MAE's skill
at σ=2 falls to **−6 to −9** across quantities (worse than persistence by an
order of magnitude) vs. JEPA's −1 to −2; on `rayleigh_benard`,
`buoyancy_laplacian` skill goes to −5.3 (MAE) vs. −1.2 (JEPA) at σ=2.

**Token-level** flips the `active_matter` ranking outright: JEPA beats MAE on
every quantity, linear and MLP alike (`divergence` MLP: 0.81 vs 0.69;
`vorticity_signed` MLP: 0.73 vs 0.65) — pooling was hiding a real JEPA
forecasting advantage the same way it hid the `flow_align` contemporaneous
signal. `shear_flow`'s token-level forecast results show no such reversal —
MAE stays ahead token-level too, consistent with every other axis on this
dataset. Figures: `reports/figures/{dataset}_forecast_skill_difficulty.png`.

#### A fair metric for comparing forecast quality across quantities

Raw R² isn't comparable *across* physics quantities that evolve at very
different rates: a near-static quantity looks "easy" (high R²) for any
method — including the naive "assume nothing changes" persistence baseline —
while a fast-changing or near-degenerate quantity looks "hard" (low/negative
R²) even for a good model. Naively averaging raw R² across quantities lets
one exploding or degenerate number dominate and compares unlike things.

`skill_score()` in `analyze_encoders.py` fixes this with a standard forecast-
verification technique (Murphy 1988):

```
skill = 1 - (1 - R²_probe) / (1 - R²_persistence)
```

0 → exactly as good as persistence. 1 → closes the entire remaining gap to a
perfect ceiling. Negative → worse than the naive baseline. It's algebraically
equivalent to `1 - MSE_probe/MSE_persistence`, so it needs no new computation
beyond R² values the suite already produces (NaN-guarded the same way
`ridge_r2_grouped` guards degenerate CV folds, for the case where
`R²_persistence` is itself ≈1 and there's no headroom to score against).

**Important nuance, confirmed by running this on real data before wiring it
into any live script:** skill score does not simply *shrink* outlier R²
values — it *recontextualizes* them. A quantity where persistence already
explains most of the variance (little headroom left) turns a bad raw R² into
an *even more* extreme skill score, since failing badly against an easy
baseline is a severe relative failure; a quantity where persistence itself
is already poor (a near-degenerate target, no real baseline to fail against)
turns the same bad raw R² into a comparatively tame skill score. This is the
mathematically correct interpretation, not a shrinkage artifact.

Wired into `forecast_content_probe.py` (`skill_linear`/`skill_mlp` per
target, per layer, per noise level, per gap).

Deferred (future work, not implemented): an effective-dimensionality probe
(iteratively orthogonalize probe directions until R² collapses to the
shuffled-control baseline) and a token-vs-pooled generalization-gap plot.

### Held-out, multi-seed test evaluation (the reported numbers)

This is the protocol behind every number that should be cited as a final
result. It supersedes the train-CV tables above for the three families it
covers (contemporaneous, forecast-content, noise robustness) — those tables
remain useful for the full depth/quantity landscape, but the numbers below
are the honest, held-out ones.

**Protocol** (`scripts/workshop_test_eval.py`):

1. Carve a regime-balanced selection set out of `train` itself: every 8th
   trajectory (interleaved, not a contiguous prefix/suffix — trajectories
   are laid out in contiguous per-source-file blocks, so this samples
   proportionally from every regime file in both halves). The Well's shipped
   `valid` split isn't used for this — it turned out too small and
   regime-degenerate to trust (4–5 trajectories per dataset, and for
   `shear_flow`/`rayleigh_benard` every one of them from a single regime
   file). The official `test` split is never touched until step 4.
2. At each of the 13 layers, fit candidate probes on the selection set: an
   MLP (`TinyMLP`: 128 hidden, dropout, early-stopped against an internal
   train sub-holdout) at three `weight_decay` values (1e-4/1e-2/1e-1, 3
   selection seeds each) *and* closed-form ridge regression. Score each
   candidate as `mean − 2·std` across its selection seeds (ridge has no
   seed variance, so its score is just its value) — a plain mean-only
   comparison let a handful of hard, low-signal-to-noise targets pick an
   MLP configuration that looked fine on this small selection set but was
   actually unstable (seed-to-seed test R² spread up to ~1.4) once
   evaluated on the real, untouched test split; the variance penalty and
   ridge-as-candidate together close that gap (see "Why MLP, and why did it
   need fixing" below).
3. Freeze the (layer, method) with the highest score, per (dataset,
   objective, quantity, feature-kind, time-offset).
4. If an MLP config won, refit 5 seeded MLPs on `train` at that
   (layer, weight_decay) and report mean ± std on `test`. If ridge won,
   report its single deterministic `test` value. Either way, a closed-form
   ridge fit at the same frozen layer is also always recorded
   (`ridge_test_r2`) as a transparent audit trail, whichever method won.

Three families, each independently following this protocol per quantity:
**contemporaneous** physics (t+0, pooled and per-token), **forecast-content**
physics at t+8 and t+32 (context and future windows never overlap, so
there's no temporal leakage between probe input and target — see
`forecast_content_probe.py`'s windowing), and **noise robustness** (input
corruption at each quantity's already-frozen contemporaneous layer, no
separate noise-layer search — this family always uses a plain MLP,
`weight_decay=1e-4`, since it reuses a layer already vetted by the
contemporaneous family and showed no instability in practice). Test-split
sizes: `active_matter` 10 trajectories, `shear_flow` 28, `rayleigh_benard`
50 (regime-balanced, Ra 1e6–1e10).

**Why MLP, and why did it need fixing.** The whole point of running an MLP
alongside the closed-form-optimal ridge baseline is to see where
nonlinearity helps — but that comparison is only meaningful if the MLP is
actually well-trained. A broad sanity check (comparing every reported MLP
number against a fresh ridge fit at the same frozen layer) found the
original single fixed `weight_decay=1e-4` let the MLP overfit badly on a
handful of low-SNR, long-horizon targets (test R² as bad as −0.66 where
ridge got +0.68, with wild seed-to-seed swings) while a naive fix — just
raising `weight_decay` everywhere — regressed dozens of already-good cells
(some `active_matter` token quantities fell from clearly beating ridge to
below it). The final protocol above (adaptive per-layer weight_decay,
ridge itself as a fair candidate, variance-penalized selection) resolves
this properly: across all 276 contemporaneous/forecast-content cells in the
final run, only 3 (1.1%) still land more than 0.05 below their own ridge
audit value, each a modest gap (≤0.17) on a token-level or near-null-signal
target, not a catastrophic one. Every reported number's `ridge_test_r2` is
saved alongside it in `sweep_results/*_workshop_test_eval.json` for
cross-checking.

#### Contemporaneous (t+0): near-tied

Pooled deltas are small across all three datasets (|Δ| mostly < 0.02, MAE
usually a hair ahead) — both objectives already recover the physics at
genuinely high absolute R² (0.93–1.00 for most quantities) at the moment
being probed, leaving little room for either to differentiate.

#### Forecast-content: the interesting result is dataset-specific, and lives at long horizon

| dataset | t+8 (pooled) | t+32 (pooled) |
|---|---|---|
| `active_matter` | ~tied to slightly MAE-ahead | ~tied — the real JEPA advantage is at **token** level, not pooled |
| `shear_flow` | small, mixed | **consistent, modest JEPA advantage on all 7 quantities** (Δ +0.001 to +0.110, seed std ≤ 0.05) |
| `rayleigh_benard` | ~tied | **clean, larger JEPA advantage on all 7 quantities** (Δ +0.008 to +0.195, seed std ≤ 0.05) |

`rayleigh_benard` t+32 is the cleanest single result in the whole sweep:
JEPA leads MAE on every pooled quantity, at good absolute R² for both (JEPA
0.80–0.97, MAE 0.63–0.96), with tight seed std (≤0.021) — a real,
low-variance, growing-with-horizon advantage, not a clipping or instability
artifact.

`shear_flow` t+32 shows the same pattern, at smaller magnitude: JEPA leads
MAE on all 7 pooled quantities (`advective_flux` +0.804 vs +0.720,
`tracer_grad` +0.754 vs +0.644, etc.), all with tight seed std (≤0.053).
Earlier runs of this analysis, before the MLP-training fix described above,
showed `advective_flux` and `tracer_grad` collapsing to catastrophic,
wildly seed-unstable negative R² here — that was **entirely a training
artifact** (an under-regularized MLP overfitting a hard, low-SNR target),
not a real property of either representation; ridge regression recovered
strong, stable signal for both quantities the whole time (+0.64–0.71), which
is exactly what first flagged the bug. At **token** level and the shorter
t+8 horizon, `shear_flow` shows a real, low-variance MAE advantage on
several quantities (`tracer_grad` Δ=−0.20, `advective_flux` Δ=−0.04,
`vorticity_signed` Δ=−0.08) — this pattern holds at both gaps and is
distinct from the now-fixed pooled t+32 instability.

#### Noise robustness (mean R² across quantities, pooled, at each quantity's own frozen layer)

| dataset | σ=0 | σ=0.5 | σ=1.0 | σ=2.0 |
|---|---|---|---|---|
| `active_matter` — JEPA | 0.886 | 0.845 | 0.839 | 0.761 |
| `active_matter` — MAE | 0.897 | 0.838 | 0.756 | **0.199** |
| `shear_flow` — JEPA | 0.965 | 0.951 | 0.800 | 0.749 |
| `shear_flow` — MAE | 0.949 | 0.937 | 0.863 | 0.741 |
| `rayleigh_benard` — JEPA | 0.968 | 0.966 | 0.938 | 0.884 |
| `rayleigh_benard` — MAE | 0.978 | 0.955 | 0.892 | 0.657 |

The headline noise-robustness finding replicates on held-out, multi-seed
evaluation: JEPA and MAE are near-tied at low noise, and MAE collapses
disproportionately as noise grows — sharply on `active_matter` (0.897→0.199
by σ=2, a near-total collapse) and `rayleigh_benard` (0.978→0.657), while
`shear_flow` stays close throughout, with MAE briefly ahead at σ=1.0 —
reported honestly rather than smoothed over, matching the overall pattern
(no consistent robustness gap on this dataset).

Raw numbers: `sweep_results/{dataset}_workshop_test_eval.json`. Figures:
`reports/figures/workshop/fig{2,3,4}_*`; per-cell layer choices in
`reports/figures/workshop/frozen_layers*.txt`.

#### Multiple encoder seeds

Everything above reports probe-seed variance (5 independently-initialized
MLP fits per cell) at a single pretrained encoder checkpoint per objective.
That doesn't say anything about run-to-run pretraining variance -- would a
different random init of the *encoder itself* give the same R²? Once
`scripts/run_final_training.sh` has produced 3 fresh encoder seeds per
(dataset, objective) pair, `workshop_test_eval.py` accepts all of them in
one `--checkpoints` list and probes each completely independently (its own
held-out layer/weight_decay/ridge selection, since a different encoder
seed's representation can genuinely peak at a different depth) before
combining per objective in the output JSON's `"aggregated"` section.

Combining is a **two-level nested decomposition**, not a naive pool of
every (encoder_seed × probe_seed) draw as if independent: the 5 probe-seed
draws within one encoder are correlated (same frozen weights), so pooling
all 15 values would overstate the effective sample size at the level that
actually matters — whether *retraining the objective* gives the same
answer. Instead: average away the probe-seed nuisance variance first (one
mean per encoder seed, already what `select_and_eval` produces), then treat
those 3 means as the real sample —

```
SE(grand mean)^2 = s_between^2 / n_seeds + s_within_avg^2 / (n_seeds * n_probe_seeds)
```

where `s_between` is the sample std (ddof=1) of the 3 per-encoder-seed
means and `s_within_avg` is the average of each seed's own probe-seed std
(`aggregate_by_objective()`/`_nested_stats()` in `workshop_test_eval.py`).
Both the nested SE and the simpler plain std-of-3-means are recorded
(`nested_se` and `encoder_seed_std`) — with only 2-3 encoder seeds neither
estimate is precise, so both are kept rather than presenting one as
settling the question. A single seed per objective (the original,
still-supported usage) is the degenerate n=1 case of the same formula:
`encoder_seed_std=0`, `nested_se = within_seed_std / sqrt(n_probe_seeds)`,
identical to what was reported before this extension.

`plot_workshop_figures.py` reads the `"aggregated"` section directly and
doesn't need to know how many encoder seeds went into it. Fig 2's
significance-fade now compares against the *combined* nested SE (probe-seed
and, if present, encoder-seed), a strictly more conservative bar than
probe-seed variance alone.

## Discussion: what kind of representation does each objective build?

The consistent shape across every section above — JEPA ahead specifically on
*coupled, nonlinear* quantities, specifically *under noise*, specifically
requiring *depth* to emerge, and specifically visible at *token* rather than
pooled granularity — is consistent with a standard story about the two
objectives' inductive bias, and lines up with existing literature on both
self-supervised video representation learning and this repo's own reference
paper.

**Why pixel reconstruction and latent prediction should differ this way.**
MAE's loss directly rewards retaining whatever information is needed to
reconstruct masked pixels, including high-frequency, purely local detail —
it has no incentive to discard input noise or build representations
invariant to nuisance variation, since doing so would only hurt
reconstruction fidelity. This is consistent with the original MAE paper's
own finding (He et al. 2022) that MAE features are strong after fine-tuning
but comparatively weak under *linear* probing relative to contrastive/
predictive objectives — pixel-reconstruction pretraining doesn't specifically
optimize for the kind of directly-linearly-accessible, noise-invariant
structure a frozen-encoder probe rewards. JEPA's latent-prediction loss, by
construction, only rewards features from which an EMA target's *own future
latent* is predictable — which selects against encoding whatever isn't
predictable (noise, fine texture) and selects for whatever is (the
underlying dynamical state). That's exactly the profile that would produce a
representation more robust to input corruption, and one that encodes
coupled dynamical quantities more deeply, at the cost of no particular
advantage on quantities any local reconstruction already gets for free. This
tracks LeCun's JEPA framing (energy-based/joint-embedding predictive
architectures as biased toward abstraction and predictability over pixel
fidelity by design) and the V-JEPA line of work (Assran et al., "predicting
in representation space" as the mechanism for discarding unpredictable
detail).

**Why the gap is quantity- and dataset-specific, not blanket.** [arXiv:2602.07050](https://arxiv.org/abs/2602.07050)
("Interpreting Physics in Video World Models") finds that per-token/nonlinear
probes recover physics signal that pooled, linear probes miss in video world
models — exactly the pattern in the token-level section above, concentrated
on `rayleigh_benard`. The same paper's "emergence zone" framing (physics
becomes readable partway through the network, not at the input or output)
matches the depth section's MAE-plateaus-early / JEPA-builds-with-depth
split, again specific to `rayleigh_benard`. The fact that this dataset shows
the largest, cleanest JEPA advantage, and `shear_flow` the smallest, is
itself informative: `rayleigh_benard`'s buoyancy and `active_matter`'s
active stress both feed back directly into the momentum equation (the field
being probed *drives* the velocity field that advects it — a genuine
feedback loop), while `shear_flow`'s tracer is passively advected with no
feedback onto velocity. The forecast-content JEPA advantage (see the
held-out section above) is present on all three datasets at t+32, but it's
roughly 2–3× larger on `rayleigh_benard` (Δ up to +0.195) than `shear_flow`
(Δ up to +0.110) — consistent with the idea that latent-prediction's
advantage scales with how locally decomposable the underlying dynamics are,
largest where there's a genuine feedback loop, present but smaller where
there isn't, rather than being a uniform property of the objective or an
on/off effect.

**Practical implication.** If the downstream use case is a frozen encoder
feeding a linear or lightly-nonlinear readout — the common pattern for
physics-informed monitoring, anomaly detection, or lightweight forecasting
heads — this suggests JEPA-style pretraining is the safer default
specifically when (a) the system's governing variables are genuinely
coupled/nonlinear rather than locally decomposable, and (b) the deployment
input is realistically noisy (sensor noise, missing data, distribution
shift from the clean simulation data pretraining saw). For simpler,
more locally-decomposable systems, or for genuinely clean input, the two
objectives are functionally comparable and MAE's simpler, single-stream
training loop (no EMA target, no predictor network) may be preferable on
engineering grounds alone.

# Linear probe suite

Part of the [step-2 probing study](OVERVIEW.md). This suite asks one
question in several variants: **does the frozen encoder representation
linearly (or near-linearly) contain a given piece of physics** — no
predictor, no decoder, just ridge regression from frozen features to a
target. Contrast with the [rollout probe suite](ROLLOUT_PROBE.md), which
uses the predictor/decoder to actually generate forecasts.

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
| [`scripts/forecast_content_probe.py`](../scripts/forecast_content_probe.py) | Trains a **fresh** linear **and MLP** probe (pooled and per-token, every layer) from present-time frozen features to a **separate future window's** physics, swept over multiple time gaps (`--gaps`) and injected-noise levels (`--noise-stds`, context only — same convention as `analyze_noise_robustness.py`). | No predictor/decoder at all — removes the "does the pretrained head generalize to new mask geometry" confound present in `rollout_probe.py`. Full rigor stack (depth × linear-vs-MLP × noise), matching the rest of the suite. Each result also reports a `skill_*` field (see below) alongside the raw `probe_*`/`ceiling_*` R². |
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
token), regime probing, forecast/skill-score, and a held-out (`valid`-split)
generalization check. Raw numbers: `sweep_results/{dataset}_pooled.json`,
`{dataset}_nonpooled.json`, `{dataset}_regime.json`,
`{dataset}_noise_robustness.json`, `{dataset}_forecast.json`, and the
`*_VALID.json` variants for the generalization check. Figures:
`reports/figures/*.png`, generated by `scripts/plot_probing_suite.py`.

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
R²) even for a good model. Averaging raw R² across quantities (as the
rollout tables in [ROLLOUT_PROBE.md](ROLLOUT_PROBE.md) do) lets one exploding
or degenerate number dominate and compares unlike things.

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
values — it *recontextualizes* them. `enstrophy`'s raw `fed_back` rollout R²
of −15.9 becomes a **more** extreme skill of **−134**, because persistence
already explains 87% of `enstrophy`'s variance (almost no headroom, so
failing that badly is a severe *relative* failure). Meanwhile
`vorticity_signed`'s raw R² of ≈−3 becomes a **tamer** skill of ≈−0.5,
because persistence is *also* bad there (a near-degenerate target) — there
was no real baseline to fail relative to. This is the mathematically correct
interpretation, not a shrinkage artifact.

Wired into `forecast_content_probe.py` (`skill_linear`/`skill_mlp` per
target, per layer, per noise level, per gap) and `rollout_assessment.py`
(`skill_latent`/`skill_physics` per target, per step, for both `fed_back` and
`oracle` against the `persistence` baseline already computed there).

Deferred (future work, not implemented): an effective-dimensionality probe
(iteratively orthogonalize probe directions until R² collapses to the
shuffled-control baseline) and a token-vs-pooled generalization-gap plot.

### Held-out generalization check (`valid` split)

Every result above is fit and cross-validated entirely within The Well's
`train` split — the same split both encoders pretrained on. To check whether
the findings hold on genuinely unseen simulation trajectories, we downloaded
a small slice of the `valid` split (4–5 trajectories per dataset, not the
full split — this is a spot-check, not a full re-validation) via `--split
valid` (now supported by `analyze_encoders.py`, `analyze_encoders_local.py`,
`analyze_noise_robustness.py`, and `forecast_content_probe.py`) and reran the
pooled, token, and noise-robustness probes against it.

**The noise-robustness finding replicates, and often sharpens, on data the
encoders never saw at all.** On `active_matter`, MAE's `nematic_order`
collapses to R² = **−15.4** at *zero* added noise on held-out trajectories
(vs. clean R²=0.999 on train); under σ=2 it reaches **−78.3**, while JEPA
stays comparatively contained (−16.7). On `rayleigh_benard`, MAE's
`buoyancy_grad` is already −2.73 on clean held-out data. This is the
strongest generalization evidence in the repo: MAE's fragility isn't an
artifact of synthetic noise injected on in-distribution data — it shows up
on simulation instances the encoder never pretrained on, at every
granularity checked.

**The clean-data pooled comparison has a real confound worth stating
plainly.** `rayleigh_benard`'s valid-split pooled R² is *higher* than train
for both objectives on several quantities (`enstrophy`: JEPA 0.69→0.98)
— not because generalization improved, but because train's number is
cross-validated across ~35 files spanning Rayleigh number from 1e6 to 1e10
(near-onset to strongly turbulent convection), while this valid check has
only 5 trajectories from one Ra=1e9/Pr=5 file. A probe scored within one
regime instance is an easier task than one cross-validated across the full
regime spread, which narrows the apparent JEPA/MAE gap without indicating
the underlying gap isn't real.

**Sample-size caveat**: 20–25 pooled samples and 4–5 trajectories per
dataset — individual quantity-level numbers here are noisy and shouldn't be
over-cited, but the noise-robustness gap's direction and magnitude are
large and consistent enough across datasets to be real signal. Raw numbers:
`sweep_results/{dataset}_{pooled,nonpooled,noise_robustness}_VALID.json`.

### Held-out test-split evaluation: freezing layer selection

The `valid`-split check above answers "does the pattern replicate on unseen
trajectories," but it still selects the best-performing layer via CV *within
that same split* — no different in kind from the train-CV tables earlier in
this doc, which report `max` over 13 layers' worth of CV curves as "best-
layer R²." That max-over-13 operation is itself a form of selection that
isn't validated against independent data, even though each individual
layer's R² is an honest CV estimate. `scripts/test_split_eval.py` fixes
this specifically: **the layer is chosen from the existing train-CV curves
and frozen** *before* any test data is touched; a probe (ridge and MLP) is
then fit once on `train` at that frozen layer and evaluated once on `test`
trajectories the encoder never pretrained on and the probe never used for
any selection (`ridge_fit_eval`/`mlp_fit_eval` in `analyze_encoders.py`/
`analyze_encoders_local.py`). No retraining — same frozen checkpoints as
everywhere else. Scoped to the three headline analyses (pooled
contemporaneous, token-level, noise robustness); regime/rollout/forecast
remain train-CV only.

Test-split sizes: `active_matter` 10 trajectories (all of `test`'s smaller
files), `shear_flow` 28, `rayleigh_benard` 50 (spanning Rayleigh number
1e6–1e10 evenly, matching train's regime spread — an initial 6-file sample
that happened to cluster at Ra=1e9 with no 1e10 coverage produced nonsense
negative R² for both objectives, a regime-coverage artifact rather than a
finding, caught and fixed before reporting).

**Pooled contemporaneous physics reproduces almost exactly:**

| dataset | quantity | JEPA train→test | MAE train→test |
|---|---|---|---|
| active_matter | enstrophy | 0.989 → 0.992 | 0.993 → 0.995 |
| shear_flow | advective_flux | 0.775 → 0.800 | 0.760 → 0.785 |
| rayleigh_benard | enstrophy | 0.686 → 0.618 | 0.455 → 0.228 |
| rayleigh_benard | convective_flux | 0.835 → 0.804 | 0.576 → 0.411 |
| rayleigh_benard | okubo_weiss | 0.722 → 0.656 | 0.460 → 0.221 |
| rayleigh_benard | velocity_buoyancy_coherence | 0.583 → 0.440 | −0.049 → −0.056 |

`rayleigh_benard`'s JEPA-ahead-on-coupled-quantities gap doesn't just
survive on held-out data, it **widens** relative to train (`enstrophy` gap
0.23→0.39, `okubo_weiss` 0.26→0.44, `convective_flux` 0.26→0.39) — the
opposite of what you'd expect from an artifact of train-CV selection bias.

**Noise robustness confirms the headline finding, at every noise level, on
genuinely held-out trajectories:**

| dataset | σ=0.0 | σ=0.5 | σ=1.0 | σ=2.0 |
|---|---|---|---|---|
| active_matter — JEPA | 0.784 | 0.751 | 0.770 | 0.621 |
| active_matter — MAE | 0.898 | 0.861 | 0.678 | **−0.784** |
| shear_flow — JEPA | 0.919 | 0.906 | 0.721 | 0.096 |
| shear_flow — MAE | 0.952 | 0.933 | 0.802 | 0.254 |
| rayleigh_benard — JEPA | 0.616 | 0.464 | 0.162 | −0.262 |
| rayleigh_benard — MAE | 0.542 | 0.331 | 0.194 | −0.354 |

`active_matter`'s collapse reproduces almost exactly (MAE −0.78 at σ=2 on
test vs. −1.02 on train, same order of magnitude). `shear_flow` confirms
MAE stays ahead at *every* noise level on held-out data too — genuinely no
robustness gap on this dataset. `rayleigh_benard` shows a new nuance not
visible in the train-CV table: JEPA leads at both ends (clean and heaviest
noise) but MAE is briefly ahead at σ=1.0 (0.194 vs 0.162) — reported
honestly rather than smoothed over; the overall pattern (JEPA more robust)
still holds at the extremes.

**Token-level is noisier at this sample size** (16 clips train/test — small
relative to the pooled evaluation's dozens of trajectories) and individual
numbers shouldn't be over-cited, but the qualitative pattern holds: on
`rayleigh_benard`, MAE's MLP goes clearly negative on the coupled quantities
(`convective_flux` −0.64, `enstrophy` −0.59) while JEPA stays positive on
some of the same quantities (`convective_flux` +0.34) — MAE actively fails
where JEPA doesn't, even if the exact magnitudes are noisy at n=16.

Raw numbers: `sweep_results/{dataset}_test_eval.json`.

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
split, again specific to `rayleigh_benard`. The fact that this dataset is
where JEPA pulls ahead, and `shear_flow` is where it doesn't, is itself
informative: `rayleigh_benard`'s buoyancy and `active_matter`'s active
stress both feed back directly into the momentum equation (the field being
probed *drives* the velocity field that advects it — a genuine feedback
loop), while `shear_flow`'s tracer is passively advected with no feedback
onto velocity. Noise-robustness and token-forecast gaps appear specifically
on the two systems with feedback and are essentially absent on the one
without it — consistent with the idea that latent-prediction's advantage is
concentrated where the underlying dynamics are least locally decomposable,
not a uniform property of the objective.

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
engineering grounds alone. Neither objective's advantage transfers to
multi-step autoregressive forecasting through a post-hoc-trained predictor
(see [ROLLOUT_PROBE.md](ROLLOUT_PROBE.md)) — representational quality, as
measured by every probe in this document, and forecasting competence through
a shallow downstream head are empirically not the same axis.

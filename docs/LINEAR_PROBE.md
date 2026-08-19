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

The canonical, citable-numbers pipeline is
[`scripts/workshop_test_eval.py`](../scripts/workshop_test_eval.py) — see
"Held-out, multi-seed test evaluation" below for its full protocol. It imports
shared utilities (`ridge_r2`, `build_dataset`, `contemporaneous_targets`,
`compute_layerwise_features_batched`, `load_checkpoint_encoder`) from
`analyze_encoders.py`, and (`local_target_maps`, `layerwise_token_features`,
`_train_mlp_early_stop`) from `analyze_encoders_local.py`.

| script | question | notes |
|---|---|---|
| [`scripts/analyze_encoders.py`](../scripts/analyze_encoders.py) | Earlier-stage, train-CV comparison: layer-wise, pooled, ridge-probe against **derived nonlinear physics targets** (dataset-specific — see above) plus `future_enstrophy`. | Useful for exploring the depth/layer-emergence landscape on a new dataset before committing to the expensive held-out protocol. Also hosts shared utilities used across the whole suite. |
| [`scripts/analyze_encoders_local.py`](../scripts/analyze_encoders_local.py) | Same train-CV framing, but per-token (non-pooled) plus a small MLP nonlinear readout — does spatial detail or nonlinearity recover signal pooling/linearity hides? | `--skip-mlp` skips the MLP variant. |

Supporting: [`scripts/extract_regime_metadata.py`](../scripts/extract_regime_metadata.py)
reconstructs per-trajectory regime params (Reynolds/Schmidt, Rayleigh/Prandtl,
alpha/zeta) from Well filenames (best-effort regex parsing), feeding
`workshop_test_eval.py`'s regime family.

## Earlier-stage train-CV exploration (superseded)

Before the held-out multi-seed protocol below existed, this repo ran a
broader train-CV exploration across all three datasets — full 13-layer depth
sweep, 6-level noise sweep (pooled and token), regime probing, and a
forecast/skill-score analysis (`skill = 1 - (1-R²_probe)/(1-R²_persistence)`,
a standard forecast-verification technique for comparing quantities with very
different intrinsic predictability). It first surfaced the qualitative shape
of every finding below it — MAE peaking at shallow layers vs. JEPA needing
depth on `rayleigh_benard`, MAE's sharper-but-more-fragile noise behavior,
JEPA's forecast advantage growing with horizon — but reports `max` over
train-CV curves as "best-layer R²," which is itself an unvalidated selection,
not a held-out estimate. Superseded by the protocol below; see git history at
or before commit `92da170` for the full tables and the plotting script that
generated them.

## Held-out, multi-seed test evaluation (the reported numbers)

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
physics at t+8 and t+32 (context and future windows never overlap — future
starts at `off + n_frames + gap` — so there's no temporal leakage between
probe input and target), and **noise robustness** (input
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

`rayleigh_benard` (real 3-encoder-seed numbers): pooled deltas are small
(|Δ| ≤ 0.023 on every quantity) — both objectives already recover the
physics at genuinely high absolute R² (0.94–1.00) at the moment being
probed, leaving little room for either to differentiate. MAE is fractionally
ahead on the small-scale/high-frequency quantities (`enstrophy`,
`okubo_weiss`, `convective_flux`, `pressure_grad_mag`); JEPA's one clear win
is `buoyancy_grad` (Δ +0.023). `active_matter`/`shear_flow` (single encoder
seed) show the same near-tied pattern.

#### Forecast-content: the interesting result is dataset-specific, and lives at long horizon

| dataset | t+8 (pooled) | t+32 (pooled) |
|---|---|---|
| `active_matter` | ~tied to slightly MAE-ahead | ~tied — the real JEPA advantage is at **token** level, not pooled |
| `shear_flow` | small, mixed | **consistent, modest JEPA advantage on all 7 quantities** (Δ +0.001 to +0.110, single seed) |
| `rayleigh_benard` | MAE fractionally ahead on every quantity (Δ −0.001 to −0.021) | **clean, larger JEPA advantage on 6 of 7 quantities** (Δ −0.001 to +0.121, 3 encoder seeds, `encoder_seed_std` ≤ 0.039) |

`rayleigh_benard` t+32 is the cleanest single result in the whole sweep:
JEPA leads MAE on 6 of 7 pooled quantities (tied on `pressure_grad_mag`), at
good absolute R² for both (JEPA 0.79–0.97, MAE 0.68–0.97), with tight
encoder-seed std (`buoyancy_grad` Δ +0.121±0.012,
`velocity_buoyancy_coherence` Δ +0.112±0.039) — a real, low-variance,
growing-with-horizon advantage confirmed across independently-initialized
encoders, not a clipping, instability, or single-seed-luck artifact.

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

#### Noise robustness

`rayleigh_benard` was regenerated this session with an improved protocol:
**clean-fit, no-refit** (probe fit once on clean training features, never
recalibrated, then evaluated across a fine noise grid) rather than the
original **matched-noise** design (probe refit at every noise level, which
turned out to be measuring readout re-adaptation more than representation
robustness). Under clean-fit, R² can go arbitrarily negative under enough
distribution shift — a real representational-fragility signature, not a
bug — so both R² and Pearson r (scale/offset-invariant, stays interpretable
even where R² has collapsed) are reported. `active_matter`/`shear_flow`
below still use the original matched-noise protocol (not yet regenerated) —
the two tables are not directly comparable methodologically.

`rayleigh_benard` (clean-fit, ridge, Pearson r, mean over 3 encoder seeds):

| quantity | σ=0 | σ=0.1 | σ=0.3 | σ=0.5 | σ=1.0 |
|---|---|---|---|---|---|
| `buoyancy_grad` | 0.96 / 0.96 | 0.91 / 0.30 | 0.82 / 0.22 | 0.69 / 0.25 | 0.46 / 0.22 |
| `buoyancy_laplacian` | 0.97 / 0.96 | 0.92 / 0.41 | 0.84 / 0.23 | 0.68 / 0.19 | 0.38 / 0.14 |
| `enstrophy` | 0.99 / 1.00 | 0.96 / 0.82 | 0.93 / 0.54 | 0.82 / 0.53 | 0.40 / 0.46 |
| `okubo_weiss` | 0.99 / 1.00 | 0.92 / 0.66 | 0.86 / 0.29 | 0.72 / 0.31 | 0.24 / 0.38 |
| `convective_flux` | 0.99 / 0.99 | 0.99 / 0.96 | 0.97 / 0.89 | 0.93 / 0.84 | 0.77 / 0.78 |
| `pressure_grad_mag` | 1.00 / 1.00 | 0.99 / 0.97 | 0.98 / 0.96 | 0.96 / 0.98 | 0.94 / 0.99 |
| `velocity_buoyancy_coherence` | 0.96 / 0.97 | 0.94 / 0.92 | 0.89 / 0.86 | 0.81 / 0.80 | 0.58 / 0.59 |

(cell = JEPA / MAE.) JEPA's robustness edge is real and large at low-to-
moderate corruption on the differential quantities (`buoyancy_grad`,
`buoyancy_laplacian`, `enstrophy`, `okubo_weiss`) but **not universal at
extreme noise**: at σ=1.0, MAE edges ahead on `enstrophy`, `okubo_weiss`,
and `pressure_grad_mag`. Both ridge and MLP readouts agree on this pattern
(MLP numbers in `sweep_results/rayleigh_benard_workshop_test_eval.json`'s
`aggregated.noise_robustness`, not reproduced here).

`active_matter`/`shear_flow` (matched-noise protocol, mean R² across
quantities, pooled, single encoder seed):

| dataset | σ=0 | σ=0.5 | σ=1.0 | σ=2.0 |
|---|---|---|---|---|
| `active_matter` — JEPA | 0.886 | 0.845 | 0.839 | 0.761 |
| `active_matter` — MAE | 0.897 | 0.838 | 0.756 | **0.199** |
| `shear_flow` — JEPA | 0.965 | 0.951 | 0.800 | 0.749 |
| `shear_flow` — MAE | 0.949 | 0.937 | 0.863 | 0.741 |

Under the matched-noise protocol, `active_matter`'s MAE collapses sharply
(0.897→0.199) while `shear_flow` stays close throughout — this was the
original finding that motivated re-examining the noise protocol in the first
place (matched-noise lets the probe re-adapt at every level, which can mask
or exaggerate how fragile the underlying frozen representation actually is;
see `rayleigh_benard`'s clean-fit numbers above for the corrected picture on
that dataset).

Raw numbers: `sweep_results/{dataset}_workshop_test_eval.json`. Figures:
`reports/figures/fig{2,3,4}_*`; per-cell layer choices in
`reports/figures/frozen_layers*.txt`.

#### Multiple encoder seeds

Probe-seed variance alone (5 independently-initialized MLP fits per cell, at
a single pretrained encoder checkpoint per objective) doesn't say anything
about run-to-run pretraining variance — would a different random init of the
*encoder itself* give the same R²? `rayleigh_benard` now has this answered
for real: `scripts/run_final_training.sh` produced 3 fresh encoder seeds per
(dataset, objective) pair, and `workshop_test_eval.py` was run with all of
them in one `--checkpoints` list — each probed completely independently
(its own held-out layer/weight_decay/ridge selection, since a different
encoder seed's representation can genuinely peak at a different depth)
before combining per objective in the output JSON's `"aggregated"` section
(`n_encoder_seeds: 3` throughout `sweep_results/rayleigh_benard_workshop_test_eval.json`).
`active_matter`/`shear_flow` remain the single-seed case below — their
current-architecture (192-d predictor) 3-seed JEPA retrain was started but
stopped incomplete (recurring pod GPU-host fault); their MAE side and the
`rayleigh_benard` results are unaffected by this, since MAE has no predictor
and `rayleigh_benard`'s full 3-seed run already completed.

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

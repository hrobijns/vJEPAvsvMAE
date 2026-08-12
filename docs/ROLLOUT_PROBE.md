# Rollout probe suite

Part of the [step-2 probing study](OVERVIEW.md). Where the
[linear probe suite](LINEAR_PROBE.md) asks "is physics linearly present in
the frozen representation," this suite asks "can the model actually forecast
forward" — it uses a predictor/decoder to generate future latents or pixels,
not just a ridge probe on present-time features.

## Two scripts, two questions

| script | question | uses |
|---|---|---|
| [`scripts/rollout_probe.py`](../scripts/rollout_probe.py) | Under a causal (non-tube) mask — context = first half of the clip, predict the second half — does the model's **own pretrained** predictor/decoder forecast better than persistence? Single-shot only, not chained. | Each encoder's original JEPA predictor or MAE decoder, loaded via `scripts/load_predictor.py`. |
| [`scripts/rollout_assessment.py`](../scripts/rollout_assessment.py) | **Genuine multi-step autoregressive rollout**: encoder → predict next window → decode to pixels → re-encode → predict again, chained for `n_steps`. Does forecast quality survive compounding error? | A **post-hoc-trained** predictor + decoder pair (see below), loaded via `scripts/load_rollout_heads.py`. |

`rollout_probe.py` is a cheaper, single-shot sanity check that reuses
whatever head each model already has. `rollout_assessment.py` is the more
faithful test of forecasting ability, but needs a prerequisite training step
because neither original head is suited to genuine next-window forecasting:
JEPA's predictor was trained jointly with its encoder to fill in masked
patches *within one clip*, not to predict a disjoint future window; MAE's
decoder fuses "predict" and "decode" into a single masked-patch-filling call.

## Rollout-heads architecture (prerequisite for `rollout_assessment.py`)

[`src/objectives/rollout_heads.py`](../src/objectives/rollout_heads.py)
defines `RolloutHeadModel`: a **frozen** pretrained encoder (JEPA's or MAE's,
whichever is under test) + a **freshly initialized** `JEPAPredictor` (384d,
depth 6) + a **freshly initialized** `MAEDecoder` (192d, depth 4) — the same
classes used in the original training objectives, reused verbatim, just
re-instantiated on top of a frozen encoder. Trained on disjoint consecutive
window pairs (`window_a` → `window_b`) with real features only — the decoder
never sees a predicted feature during training, so rollout-time predictor
error can't be masked by decoder compensation at train time. Mask-free: since
context/future are full fixed windows (not partial visibility), decoding uses
`decode_all()` ([`src/models/decoder.py`](../src/models/decoder.py)), a
bypass added specifically for this that decodes every position's own feature
directly with no mask-token machinery.

Training (one-time, both objectives, all 3 datasets):
```bash
uv run python scripts/gen_rollout_heads_configs.py   # regenerates configs/rollout_heads_*.yaml
bash scripts/run_rollout_heads_training.sh            # loops all 6 dataset x objective combos
```

## Rollout-assessment modes and baselines

At each step of the chain, `rollout_assessment.py` reports:
- **`fed_back`** — the real test: the model's own predicted/decoded output
  from step *t* becomes the input to step *t+1*. Errors compound.
- **`oracle`** — re-seeded with real context at every step, isolating
  one-step prediction error from compounding.
- **`persistence`** — naive baseline: assume nothing changes.
- **`ceiling`** (`ceiling_latent_r2`) — upper bound using the real future
  features directly, no prediction involved.

Metrics: `latent_r2` (JEPA — ridge R² of predicted latent vs. real future
physics), `physics_r2`/`physics_corr` (both objectives, decoded-to-pixels vs.
real future physics), `pixel_mse`/`pixel_rel_l2`, and `skill_latent`/
`skill_physics` — the persistence-relative skill score for `fed_back`/
`oracle` against the `persistence` baseline above (see
[LINEAR_PROBE.md](LINEAR_PROBE.md#a-fair-metric-for-comparing-forecast-quality-across-quantities)
for the metric and why raw R² isn't fair to average across quantities here —
directly relevant to this doc's own step-by-step tables below, which
currently report unweighted mean raw R²).

## Token-level extension

Both scripts now also report a per-token variant alongside the pooled one
described above (see [LINEAR_PROBE.md](LINEAR_PROBE.md) for why token-level
matters generally). Every window in both scripts is a full clip sharing the
encoder's native `(grid_t, grid_h, grid_w)` token grid — `rollout_probe.py`'s
disjoint context/future sub-windows are the one exception (JEPA's masked
future tokens correspond 1:1 by index to `local_target_maps()` computed on
just the real future sub-window, since `causal_temporal_mask` selects a
contiguous trailing block in the same time-major/h/w order as `tokenize()`;
MAE's reconstructed windows are already full pixel fields, so no special
handling is needed there either). `rollout_assessment.py`'s windows are all
full `n_frames` clips at every step, so predicted and target tokens
correspond 1:1 by index directly — simpler than `rollout_probe.py`'s case.
Both cap the token-level sample count (`--token-max-clips` /
`--token-max-seeds`) independent of the pooled sample count, since per-token
tensors are ~`n_tokens`× larger.

## Results

> **Pre-expansion snapshot**, pooled-only — predates the token-level
> extension above and the target-set changes in
> [LINEAR_PROBE.md](LINEAR_PROBE.md) (both scripts consume
> `contemporaneous_targets()`, which no longer includes pooled
> `vorticity_signed`/`divergence` and now includes
> `active_stress_div_mag`/`tracer_laplacian`/`buoyancy_laplacian`). Needs a
> fresh GPU sweep before these numbers are current.

### Single-shot (`rollout_probe.py`) — JEPA latent R², own predictor

`predictor` = model's own forecast, `ceiling` = real future features,
`persistence` = naive baseline. All three are close together on most
quantities — the pretrained predictor doesn't beat persistence by much,
though it's not far off the ceiling either:

| dataset | quantity | predictor | ceiling | persistence |
|---|---|---|---|---|
| active_matter | enstrophy | 0.937 | 0.950 | 0.946 |
| active_matter | nematic_order | 0.903 | 0.967 | 0.896 |
| shear_flow | okubo_weiss | 0.978 | 0.991 | 0.971 |
| rayleigh_benard | convective_flux | 0.657 | 0.789 | 0.661 |

MAE's single-shot `decoded_r2` (pixel decode → derived physics) is
unreliable in this single-shot, wrong-mask-geometry setting — often strongly
negative (e.g. `active_matter` `order_grad_mag` decoded_r2 = −5.77,
`rayleigh_benard` several quantities → −10³ to −10¹¹), while `decoded_corr`
(shape/correlation, ignoring scale) stays reasonable (0.7–0.98 on most
quantities). This is the motivation for `forecast_content_probe.py` (see
[linear probe suite](LINEAR_PROBE.md)) and for training dedicated rollout
heads rather than trusting the original MAE decoder out of distribution.

### Multi-step autoregressive (`rollout_assessment.py`), mean R² across
physics quantities, `fed_back` vs `oracle` vs `persistence`, steps 1→8:

**active_matter**
```
              step:  1    2    3    4    5    6    7    8
JEPA fed_back      0.52 0.43 0.36 0.32 0.31 0.29 0.29 0.31
JEPA oracle        0.52 0.48 0.46 0.49 0.43 0.43 0.45 0.45
JEPA persistence   0.50 0.41 0.40 0.32 0.30 0.37 0.36 0.37
MAE  fed_back      0.56 0.46 0.34 0.25 0.29 0.31 0.29 0.31
MAE  oracle        0.56 0.54 0.53 0.52 0.50 0.48 0.51 0.52
MAE  persistence   0.55 0.50 0.43 0.39 0.42 0.46 0.45 0.47
```

**shear_flow**
```
              step:  1    2    3    4    5    6    7    8
JEPA fed_back      0.63 0.61 0.59 0.55 0.57 0.56 0.56 0.60
JEPA oracle        0.63 0.64 0.64 0.61 0.65 0.65 0.65 0.69
JEPA persistence   0.64 0.63 0.61 0.57 0.60 0.60 0.60 0.63
MAE  fed_back      0.65 0.63 0.60 0.56 0.59 0.58 0.58 0.61
MAE  oracle        0.65 0.66 0.65 0.63 0.67 0.68 0.68 0.72
MAE  persistence   0.66 0.65 0.63 0.59 0.62 0.62 0.62 0.66
```

**rayleigh_benard**
```
              step:  1    2    3    4    5    6    7    8
JEPA fed_back      0.69 0.68 0.60 0.52 0.45 0.38 0.39 0.40
JEPA oracle        0.69 0.70 0.69 0.69 0.68 0.67 0.67 0.67
JEPA persistence   0.68 0.68 0.66 0.64 0.60 0.56 0.52 0.52
MAE  fed_back      0.71 0.70 0.65 0.55 0.48 0.41 0.40 0.41
MAE  oracle        0.71 0.71 0.71 0.70 0.69 0.69 0.68 0.68
MAE  persistence   0.71 0.70 0.66 0.56 0.49 0.44 0.44 0.45
```

Takeaways:
- `fed_back` consistently falls well below `oracle` for both objectives on
  every dataset — errors compound as expected from an autoregressive chain.
- On `active_matter` and `rayleigh_benard`, `fed_back` drops *below*
  `persistence` by mid-chain — i.e. the autoregressive forecast becomes worse
  than assuming nothing changes. On `shear_flow` `fed_back` stays close to
  (slightly under) `persistence` throughout.
- JEPA and MAE track each other closely at every step on every dataset — this
  is not a dimension where JEPA's `rayleigh_benard` advantage (see
  [OVERVIEW.md](OVERVIEW.md)) shows up. Whatever extra physics JEPA encodes
  contemporaneously doesn't translate into better multi-step forecasting
  through these post-hoc heads.

Raw per-quantity numbers: `sweep_results/{dataset}_rollout.json` (single-shot)
and `sweep_results/{dataset}_rollout_assessment.json` (multi-step).

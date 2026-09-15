# Analysis plan for the three-system JEPA–vMAE study

This is the working plan for choosing checkpoints and then analysing the selected representations. It is deliberately split into two stages so that official-test results cannot influence checkpoint selection.

## Decisions fixed before analysis

- We compare four independently trained objectives for each physical system: `jepa`, `mae`, `jepa_future`, and `mae_future`.
- We will select **one encoder checkpoint per system, objective, and training seed** using downstream validation performance.
- The checkpoint candidates are the common milestones **25k, 50k, 75k, and 100k**. We will not include `encoder_best_val.pt`, because its step is objective-dependent and JEPA's EMA target makes pretraining loss difficult to compare over time.
- We will not run a separate fixed-100k scientific comparison. Equal steps would match updates and sampled examples, but not convergence or exact compute under different objectives and learning rates.
- Dropping the fixed-100k comparison was the explicit project decision. Excluding a separate `best_val` sensitivity analysis is an additional simplification in this plan: its asymmetric, objective-dependent candidate steps would reintroduce the moving-loss problem. Revisit that choice only by amending this plan before official-test access.
- Within each independently frozen system study, official test data remains unopened until that system's checkpoint and probe choices are frozen in a manifest. Results from an earlier system must not be used to revise the declared policy for later systems.
- The present extension has only encoder seed 1. Results are descriptive for these trained encoders; probe initializations are not additional encoder seeds.

## Terminology: three kinds of question

The word “parameter” can hide important differences, so this plan separates:

1. **Governing regime parameters:** constants for an entire trajectory, such as Rayleigh number or Reynolds number. These are probed only from trajectory-level pooled representations and do not have local or future versions.
2. **Pooled physical quantities:** a state-derived physical field is averaged over an eight-frame target clip and physical space. The probe predicts one value per sampled clip.
3. **Local/token physical quantities:** the same field is averaged within a 2×16×16 spacetime patch. The probe predicts the value associated with an encoder token.

The 2×2 model design concerns the **pretraining target**—pixel versus latent and current versus adjacent-future. Probe target offsets are a separate evaluation axis applied to all four objectives.

# Stage 1 — choose the encoder checkpoints

## Goal

Produce a frozen table containing one selected encoder for every run in the
study scope. The complete seed-1 study has 12 runs: three systems × four
objectives. A dataset-scoped study has four runs and may be completed
independently, allowing Rayleigh–Bénard to be analysed before the other systems.

When seeds 2 and 3 are trained, repeat exactly the same procedure independently
for each seed.

## Candidate roster

For every `(system, objective, seed)` use:

- `encoder_025pct.pt` — 25,000 updates;
- `encoder_050pct.pt` — 50,000 updates;
- `encoder_075pct.pt` — 75,000 updates;
- `encoder_100pct.pt` — 100,000 updates.

The complete seed-1 study has 48 candidate encoders. A dataset-scoped study has
16. Candidate paths, steps, hashes, input geometry, and exposure must match
`checkpoints/iclr2027/seed1/index.json` and `manifest.json`.

`scripts/probe_sweep.py prepare` validates the complete declared handoff before
applying an optional `--dataset` scope. It freezes exactly four milestones per
included run, requires all four objectives, and rejects missing, duplicated,
unexpected, mis-timed, or metadata-inconsistent candidates. Excluded `best_val`
files never absorb an incomplete group. The candidate policy and scoped roster
are stored in `study.json`; every later command re-checks both before probing.

## Data boundary

For every candidate:

- extract features from the same predefined official-training and official-validation examples;
- fit probes on official training only;
- select probe settings and the encoder checkpoint on official validation only;
- do not prepare or inspect official-test results during this stage.

All candidates must use identical samples, target definitions, feature normalization rules, probe search spaces, and failure handling.

## Tasks used to select a checkpoint

Checkpoint selection uses the state-derived physical quantities listed under Stage 2, not the governing regime parameters. For every physical quantity, evaluate:

- pooled representation;
- local/token representation;
- target offsets 0, 8, 16, and 40.

For each task cell, search all 12 block outputs and the final encoder norm. Fit both:

- Ridge, with the predefined regularization grid;
- a one-hidden-layer MLP, with the predefined three initializations and validation-selected stopping state.

Select the layer, probe family, and probe settings for that task using validation VRMSE. Exact Ridge/MLP ties prefer Ridge.

## Checkpoint score

Let `H_h` be the mean validation VRMSE at target offset `h`, averaged equally over:

- every physical quantity for that system; and
- pooled and local/token representations.

The checkpoint score is:

```text
S = 0.5 H_0 + (H_8 + H_16 + H_40) / 6
```

This gives 50% total weight to the present target and 50% total weight to the three future targets. Lower is better.

A candidate must have a complete finite task roster. We will not silently omit a difficult or undefined quantity and average the remainder. If no candidate is eligible, selection stops for that run. Exact checkpoint-score ties prefer the earlier step and then the checkpoint SHA-256.

## Why we aggregate rather than choose one variable

Selecting with one variable would produce a checkpoint specialised to that downstream task. For example, selecting Rayleigh–Bénard checkpoints using only enstrophy would not justify calling the winner the best general physical representation.

The balanced score instead asks which checkpoint makes the complete predefined set of physical quantities most recoverable. Regime parameters and controls remain diagnostics and cannot dominate selection.

## Stage 1 output and stopping condition

Before Stage 2 begins, save and inspect a selection manifest containing, for every run:

- system, objective, and encoder seed;
- selected checkpoint path, step, and hash;
- scores for every candidate and horizon;
- chosen probe family/layer/settings for every task cell;
- candidate failures or undefined metrics;
- exact data, code, cache, feature, and probe-fit identities.

Stage 1 is complete only when every run group in the frozen study scope—four for a single system or 12 for the complete study—has an eligible selected checkpoint and the manifest is frozen. Do not change the scoring rule after this point.

# Stage 2 — analyse the selected representations

## Goal

Use only the Stage-1-selected encoders in the frozen study scope to determine how:

- latent versus pixel pretraining targets affect physical representations;
- current versus adjacent-future pretraining targets affect physical representations;
- those effects vary by physical system, spatial scale, target horizon, and encoder depth.

Official test is used once for the frozen checkpoints and frozen probes. Validation scores are selection results; test scores are the scientific performance results.

## Probe inventory by system and level

### Rayleigh–Bénard convection

This reproduces and extends the workshop paper's target roster.

**Governing regime level — trajectory-level pooled features only**

- `log10_Rayleigh`: strength of buoyant forcing relative to diffusion.
- `log10_Prandtl`: ratio of momentum diffusivity to thermal diffusivity.

These labels are constant along a trajectory. They are evaluated at target offset 0 and are not included in checkpoint selection.

**Pooled physical level — one target per eight-frame clip**

- Enstrophy, `⟨ω²⟩`: rotational activity.
- Buoyancy-gradient energy, `⟨|∇b|²⟩`: scalar gradients.
- Convective flux, `⟨u_y b⟩`: vertical buoyancy transport.
- Pressure-gradient magnitude, `⟨|∇p|⟩`: pressure-force scale.
- Buoyancy-Laplacian magnitude, `⟨|∇²b|⟩`: diffusive curvature.

**Local/token physical level — one target per sampled patch**

- Local enstrophy.
- Local buoyancy-gradient energy.
- Local convective flux.
- Local pressure-gradient magnitude.
- Local buoyancy-Laplacian magnitude.

The workshop paper probed these same five physical quantities at pooled and token scales. Its labels used the number of frames between the end of the context and the start of the target: paper `t+0`, `t+8`, and `t+32` correspond to the new target-start-minus-context-start offsets **0, 16, and 40**. New offset **8** is the immediately adjacent clip and has no workshop-paper counterpart. All new results and filenames must use the explicit offset convention.

### Active matter

**Governing regime level — trajectory-level pooled features only**

- `alpha`: activity/dipole-strength parameter, kept in its stored unlogged form.
- `zeta`: alignment parameter, kept in its stored unlogged form.

These labels are constant along a trajectory, evaluated at target offset 0, and excluded from checkpoint selection.

**Pooled physical level — one target per eight-frame clip**

- Kinetic energy, `⟨(u_x² + u_y²)/2⟩`.
- Enstrophy, `⟨ω²⟩`.
- Nematic order, the averaged local magnitude of orientational order.
- Nematic-gradient energy, the averaged squared spatial gradient of the normalized nematic tensor.

**Local/token physical level — one target per sampled patch**

- Local kinetic energy.
- Local enstrophy.
- Local nematic order.
- Local nematic-gradient energy.

“Nematic-gradient energy” is a descriptive stored name. It does not include an elastic coefficient and is not claimed to be a complete physical elastic energy.

### Shear flow

**Governing regime level — trajectory-level pooled features only**

- `log10_Reynolds`: inertial-to-viscous regime parameter.
- `log10_Schmidt`: momentum-to-tracer-diffusivity ratio.

These labels are constant along a trajectory, evaluated at target offset 0, and excluded from checkpoint selection.

**Pooled physical level — one target per eight-frame clip**

- Cross-stream kinetic energy, `⟨u_y²/2⟩`.
- Enstrophy, `⟨ω²⟩`.
- Tracer variance, using the spatial mean subtracted independently at each frame.
- Tracer-gradient energy, `⟨|∇s|²⟩`.

**Local/token physical level — one target per sampled patch**

- Local cross-stream kinetic energy.
- Local enstrophy.
- Local contribution to tracer variance about the full-frame spatial mean.
- Local tracer-gradient energy.

The local tracer-variance target does not subtract a separate mean inside each patch.

## Temporal and spatial sampling

For every selected encoder and every state-derived physical quantity, evaluate target-start offsets:

- 0: the same eight-frame interval as the input context; this is paper `t+0`.
- 8: the immediately adjacent eight-frame target; this is new and has no paper counterpart.
- 16: the target beginning eight frames after the context ends; this is paper `t+8`.
- 40: the target beginning 32 frames after the context ends; this is paper `t+32`.

These are relative offsets, not one absolute timestep in the dataset.

Current full-trajectory sampling uses:

- three equally spaced pooled contexts per trajectory;
- one local context per trajectory, cycling deterministically through five possible starts;
- 64 deterministic token positions for each local context;
- identical contexts and token positions across objectives and checkpoints.

Thus “global” does not mean using every possible timestep. It means predicting a clip-level spatial/temporal average from predefined contexts distributed across the full available trajectory.

## Probe and metric outputs

For every selected encoder, report:

- Ridge and MLP results separately;
- the validation-selected probe family and encoder layer;
- validation and test VRMSE as the primary normalized error;
- R², Pearson correlation, and MSE as supporting metrics;
- performance across all 12 block outputs and the final norm for accessibility/depth analysis;
- pooled and local/token results separately;
- each target offset separately before any summary average.

VRMSE is calculated separately for each quantity/scale/horizon before averaging. Values that are undefined remain undefined; they are not silently dropped.

## Model comparisons

At every system, physical quantity, scale, horizon, and selected layer/probe setting, organise results as the 2×2 pretraining design:

| | Current pretraining target | Adjacent-future pretraining target |
|---|---|---|
| Pixel space | `mae` | `mae_future` |
| Latent space | `jepa` | `jepa_future` |

Report:

1. `jepa` versus `mae`: latent versus pixel prediction for the current clip.
2. `jepa_future` versus `mae_future`: latent versus pixel prediction for the future clip.
3. `jepa_future` versus `jepa`: future versus current pretraining in latent space.
4. `mae_future` versus `mae`: future versus current pretraining in pixel space.
5. The interaction: whether future-target pretraining changes representation quality differently for latent and pixel objectives.

Downstream target horizon remains a separate axis. In particular, test whether future-target pretraining helps future-horizon probes more than present-horizon probes.

Because the heads, targets, loss geometry, EMA use, and selected LRs differ, phrase conclusions as comparisons of the **implemented latent-target and pixel-target recipes**, not proof that target space alone caused an effect.

## Robustness and controls after the core clean analysis

After the core clean results are complete:

- report regime/time and position-only controls;
- report encoder-plus-control pooled Ridge probes;
- compare future probes with the physical persistence baseline;
- reproduce the workshop-style input-noise analysis using the already selected clean probes, with normalized-channel noise `σ = 0, 0.05, 0.1, 0.2, 0.5, 1` and three fixed paired draws.

Noise does not trigger new checkpoint, layer, family, or probe selection. As in the workshop paper, the primary noise view is pooled, present-horizon physical probing; broader noise results may be reported separately if run.

## Stage 2 output and stopping condition

Stage 2 is complete when:

- every selected checkpoint has a matching official-test feature artifact;
- every planned regime, pooled, and local/token target has a complete result or an explicit undefined/failure status;
- all four objectives are present for every system;
- checkpoint selection provenance is attached to every test result;
- tables and plots show individual seed-1 results without a fabricated between-seed error bar;
- no test result has been used to revise checkpoint or probe choices.

The next scientific expansion is to train encoder seeds 2 and 3 under the same training and selection protocol, then report between-seed variation.

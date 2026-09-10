# Training and analysis contract

The implementation retains the corrected workshop analysis while separating
system-specific physics from sampling, probes, and reporting. Four pretraining
objectives share this analysis; attentive probes remain a later research change.

## Pretraining objectives

Every online encoder takes an eight-frame context with 90% random tube masking:
the same spatial positions are hidden throughout the clip. The encoder is a
12-block, width-384 ViT with 2×16×16 patches. Both JEPA predictors use six blocks;
both MAE decoders use four. All heads have width 192 and six attention heads.
These retain the existing family-specific head capacities; this comparison does
not claim identical compute or parameter counts across families.

`jepa` predicts EMA-encoder features at masked positions in the current clip;
`mae` predicts normalized pixels at those same masked positions. `jepa_future`
and `mae_future` instead predict every patch of the immediately following
eight-frame clip. For context frames 0–7, the future target is 8–15. There is
no additional gap, no hidden-context loss, and no autoregressive rollout.

The future heads jointly process visible context features and one query per
future patch. Their fixed positions distinguish temporal patch indices 0–3
in the context from 4–7 in the future. Each encoder invocation still processes
one eight-frame clip, using its own positions 0–3. Future values never enter
the online encoder or prediction head.

Both JEPA variants use smooth-L1 loss on target features after per-token layer
normalization. The target encoder processes the full target clip with stopped
gradients and follows the online encoder by EMA, with cosine momentum from
0.996 toward 1 over the training budget. Predictor parameters use gradients,
not EMA. Both MAE variants use per-patch normalized pixel MSE (patch variance
uses PyTorch's sample-variance convention; epsilon 1e-6) and have no EMA encoder.
Future MAE diagnostic images remain in normalized patch units and do not use
future patch statistics to rescale predictions into physical fields.

The adjacent future-only target follows the task structure of
[Qu et al.](https://arxiv.org/abs/2603.13227). Our eight-frame masked inputs and
EMA-based JEPA retain our controlled comparison's recipe; Qu et al. use
16-frame unmasked contexts and VICReg without an EMA teacher. This is an
analogous task, not a reproduction of their architecture or training recipe.

## Physical inputs and targets

Targets are computed from raw physical fields in float64. A separate,
channel-normalized fp16 copy supplies encoder inputs, using The Well's stored
z-score statistics and channel order. Token targets follow the encoder's
(time, first spatial axis, second spatial axis) patch order. Pooled features
remain an arithmetic mean over tokens; physical target means use volume
weights. These are intentionally different operations on the RB grid.

| System | Geometry | Physical targets | Governing parameters |
|---|---|---|---|
| Rayleigh–Bénard | Periodic x on [0,4), Chebyshev–Gauss y on [0,1] | Enstrophy; buoyancy-gradient energy; convective flux; pressure-gradient magnitude; buoyancy-Laplacian magnitude | log10 Rayleigh, log10 Prandtl |
| Active matter | Periodic x/y, lengths 10 and 10 | Enstrophy | Alpha, zeta (unlogged) |
| Shear flow | Periodic x/y, lengths 1 and 2 | Enstrophy | log10 Reynolds, log10 Schmidt |

Enstrophy here means `ω²`, with `ω = ∂x u_y − ∂y u_x` (no factor of one half).
RB's remaining fields are `|∇b|²`, `u_y b`, `|∇p|`, and `|∇²b|`. Compute the
pointwise quantity first, then average across the clip's physical volume or
inside each token patch. Do not average the input fields before differentiating.

RB uses Fourier differentiation in x, barycentric Chebyshev differentiation
in y, and Fejér quadrature. The raw coordinate arrays in the workshop data
misrepresent the grid; the explicit override preserves the geometry established
from the conductive profile and physical velocity divergence. Other systems
use Fourier derivatives on both uniform periodic axes. Their initial target
set excludes the old unvalidated normalized-field formulas.

Shear-flow exports label both axes from 0 to 1, including endpoints. Ingestion
accepts these labels, while derivatives use physical lengths 1 and 2 and
periodic spacing `length / number_of_samples`, as specified by the
[simulation generator](https://github.com/RudyMorel/the-well-rbc-sf/blob/8a45cb3803dd2d50dc2c80a08b796976d0ccbba5/src/generate_sf.py).

The system definitions follow [The Well's RB documentation](https://polymathic-ai.org/the_well/datasets/rayleigh_benard/),
[active-matter documentation](https://polymathic-ai.org/the_well/datasets/active_matter/),
and [shear-flow documentation](https://polymathic-ai.org/the_well/datasets/shear_flow/).
Regime values come from HDF5 scalars/attributes; filenames are not parsed as
scientific metadata. Channel order and geometry are validated at ingestion.

## Sampling and selection

Current training and evaluation configurations use each trajectory's full
available length (`frame_limit: null`). Starts are spaced one frame apart.
Current objectives use all `T - 7` eight-frame windows in a trajectory of length
`T`; future objectives use all `T - 15` adjacent context/target pairs. This gives
193 versus 185 examples for 200 frames, or 74 versus 66 for 81 frames. Current
objectives keep their extra final starts. Pairs never cross trajectory boundaries,
and a trajectory shorter than the requested window is rejected. The source determines
the length: there is no fixed 200-frame cap. Training retains the internal
trajectory split, batch size 64, and the 100,000-step starting budget.

Logs and checkpoints record `data_exposure`: eligible training/validation clip
counts, planned clips and equivalent passes, and actual clips processed and
equivalent passes. Actual clips are completed optimizer updates times batch
size, including repetitions and excluding validation. Equivalent passes divide
by eligible training clips; the last incomplete training batch is dropped.
These counters continue from the checkpoint step on resume and do not measure
distinct clips or independent physical observations.
A future context/target pair counts once in these counters; its target does
not double processed exposure. Dataset caches retain full trajectories and
serve both tasks without a format change. Objective names bind resume identity
to the target policy while keeping all encoder checkpoint shapes at eight frames.

The workshop regression configuration explicitly caps support at 101 frames.
Both protocols use the following unchanged evaluation rules. An eight-frame
context starting at `s` has its contemporary target at `s`; future targets
start at `s + 8 + gap`. Gaps 8 and 32 count intervening frames, not physical
time units or context-start offsets. Every requested target must fit inside
the configured support.

Three pooled contexts are equally spaced across the eligible start interval.
For a 200-frame trajectory these starts are 0, 76, 152. Token contexts cycle
through five temporal quantiles according to stable trajectory order:
0, 38, 76, 114, 152. The longest future target then ends at frame 199.
With the workshop's 101-frame cap, pooled starts remain 0, 26, 53 and token
starts remain 0, 13, 26, 40, 53; the longest target ends at frame 100.
Each token context retains 64 deterministic, uniformly sampled positions.
All compared checkpoints see identical samples, positions, and corruption draws.
Active matter's 81 frames yield pooled starts 0, 16, 33; removing the former
101-frame cap does not change its sampling. No padding or wrapping is used.

The official validation split supplies five trajectory-grouped folds. Replicate
assignment rotates across regimes, reproducing RB's one-run-per-regime folds
while balancing token sampling times. Fewer replicates do not imply five runs
per regime: trajectories are assigned cyclically, and an empty fold causes an
error. All clips/tokens of a trajectory stay together. This is not an evaluation
of unseen physical regimes.

Ridge searches every block output and the final norm, with penalties
`1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100`. Fold training data determine feature
standardization and target centering. The solve is
`(XᵀX + α n I) w = Xᵀ(y − mean(y))`.

Physical MLP probes independently select their encoder layer. Each is a
128-unit ReLU MLP with dropout 0.1, full-batch Adam, LR 0.01 and weight decay
0.0001. Five seeded fits monitor distinct validation folds to select 150–2000
training steps, checked every 20 steps with patience 100. Each chosen fit is
then retrained on all validation data. Predictions from the five refits are
averaged in float64. Pooled depth analysis refits every layer using its own
validation-selected duration. The workshop's additional regime MLP check uses
the Ridge-selected layer; it does not introduce a separate analysis pipeline.

The physical comparison selects Ridge or MLP by validation CV R²; exact ties
prefer Ridge. Test labels enter scoring after selection. Constant-target or
otherwise undefined validation scores cannot select a winning probe. Fold
averages use the finite fold scores; wholly undefined cells remain explicit.

Noise evaluation reuses saved clean fits, layers, and statistics at sigmas
0, .05, .1, .2, .5, 1, with three deterministic paired corruption draws. Targets
stay clean. Sigma zero uses the original clean features and must reproduce
the saved clean scores, including near-constant targets.

## Controls, metrics, and evidence

Pooled nuisance controls contain a quadratic basis of transformed regime
parameters and normalized trajectory age. Token controls use a quadratic basis
of token coordinates. A combined pooled Ridge probe appends nuisance variables
to encoder features. Persistence predicts a future target by copying the
current physical target. Controls are stored once per aggregate scientific cell.

Report R², Pearson r, MSE, and log10 MSE in physical target units, with regime
metrics on the specified parameter transforms. R² uses the test target mean in
its denominator and may be arbitrarily negative. Undefined metrics are `null`;
summary code does not silently reduce the seed count to omit them. Target means
are computed inside each checkpoint before between-checkpoint statistics.
These descriptive seed summaries do not claim formal statistical significance.

Every completed cache, feature set, probe fit, result aggregate, and plot set
has a manifest. Consumers verify the self-hash and the content hash of each
file they actually read. Features bind to exact cache manifests; saved probes
bind to their clean features; aggregates require matching protocols, cache
identities, analysis code, checkpoint geometry/training protocol, and the
explicit expected roster. New schema artifacts must be regenerated from data;
there is no converter that treats old result JSON as current evidence.

## Required next experiment stages

1. **LR selection is complete for all three systems.** All four objectives were
   swept using full trajectories, seed 0, and 8,000 optimizer steps per candidate,
   with a 5,000-step warmup and cosine schedule over the 8,000-step pilot budget.
   Both JEPA variants used candidates `5e-5, 1e-4, 2e-4`; both MAE variants used
   `2.5e-5, 5e-5, 1e-4`: 12 completed pilots per system, 36 across all three.
   Each dataset/objective's rate was selected by its minimum recorded internal
   pretraining validation loss across the completed pilot. Test and probe
   performance were not used. Collapse diagnostics were reviewed, and all
   candidate losses, selected rates, and run locations were retained. RB selected
   JEPA `5e-5`, MAE `1e-4`, future JEPA `2e-4`, and future MAE `1e-4`. Active matter
   and shear flow both selected `2e-4` for both JEPA variants and `1e-4` for both
   MAE variants. All training YAML defaults match these selections. Every
   active-matter and shear-flow pilot reached its recorded minimum at step 8,000;
   their selected rates are the largest tested. These results establish choices
   within the agreed grids, with final training duration and downstream quality
   still to be assessed.
2. **Train fresh scientific seeds 1, 2, and 3** using the selected rates and the
   100,000-step starting budget. Seed-0 pilots are excluded from final results.
   Compare final endpoints; best-validation checkpoints remain diagnostics.
   Assess training sufficiency through validation losses and intermediate
   checkpoints before deciding whether the shared budget needs to increase.
3. **Extend the analysis.** Add physical-quantity probes for the adjacent clip
   (8–15 for context 0–7), alongside the existing current and more distant
   targets (0–7, 16–23, 40–47). Existing `gap=0` denotes the current clip, so this
   extension must represent the adjacent target explicitly without changing the
   meaning of workshop results. Attentive probes and richer non-RB physical
   quantities are separate research changes.

The implementation checks and short GPU runs do not replace these stages or
establish model convergence. Historical learning rates and workshop differences
are not new ICLR results.

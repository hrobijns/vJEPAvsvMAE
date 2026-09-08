# Analysis contract

The implementation retains the corrected workshop analysis while separating
system-specific physics from sampling, probes, and reporting. It does not
implement future-clip pretraining or attentive probes.

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
available length (`frame_limit: null`). All overlapping eight-frame training
clips are eligible, with starts spaced one frame apart. The source determines
the length: there is no fixed 200-frame cap. Training retains the internal
trajectory split, batch size 64, and the 100,000-step starting budget.

Logs and checkpoints record `data_exposure`: eligible training/validation clip
counts, planned clips and equivalent passes, and actual clips processed and
equivalent passes. Actual clips are completed optimizer updates times batch
size, including repetitions and excluding validation. Equivalent passes divide
by eligible training clips; the last incomplete training batch is dropped.
These counters continue from the checkpoint step on resume and do not measure
distinct clips or independent physical observations.

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

## Next experiment phase

Add latent/pixel × same-clip/future-clip objectives, validate the four-way comparison
on RB, and scale to all three systems. Choose richer non-RB physical quantities
and add attentive probing as separate research changes. Preserve matched data,
context and masking policies, and record optimization/data-exposure budgets.
Old checkpoint LRs and the workshop's observed differences are not new ICLR
results.

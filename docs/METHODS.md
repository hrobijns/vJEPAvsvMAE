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
| Active matter | Periodic x/y, lengths 10 and 10 | Kinetic energy; enstrophy; nematic order; squared nematic gradient | Alpha, zeta (unlogged) |
| Shear flow | Periodic x/y, lengths 1 and 2 | Cross-stream kinetic energy; enstrophy; tracer variance; squared tracer gradient | log10 Reynolds, log10 Schmidt |

Enstrophy here means `ω²`, with `ω = ∂x u_y − ∂y u_x` (no factor of one half).
RB's remaining fields are `|∇b|²`, `u_y b`, `|∇p|`, and `|∇²b|`. Compute the
pointwise quantity first, then average across the clip's physical volume or
inside each token patch. Do not average the input fields before differentiating.

RB uses Fourier differentiation in x, barycentric Chebyshev differentiation
in y, and Fejér quadrature. The raw coordinate arrays in the workshop data
misrepresent the grid; the explicit override preserves the geometry established
from the conductive profile and physical velocity divergence. Other systems
use Fourier derivatives on both uniform periodic axes. Targets are always derived from raw fields, not normalized encoder inputs.

Active-matter kinetic energy is `(u_x² + u_y²)/2`. Let `c` be concentration
and `D` the symmetric second orientation moment, whose trace is `c`. Define
`Q = D/c - I/2`; concentration must be positive. The pointwise nematic order
is `sqrt(2 sum_ij Q_ij²)`. The spatial variation target is
`sum_ijk (∂k Q_ij)²`, including changes in both strength and orientation. It
is stored as `nematic_gradient_energy`, but does not include an elastic
coefficient and is not claimed to be a complete physical elastic energy.
These definitions follow the orientation-moment convention in
[Maddu, Weady, and Shelley](https://arxiv.org/abs/2308.06675). Compute the
pointwise order before spatial averaging, so differently oriented ordered
regions do not cancel into an apparently disordered global target.

Shear-flow cross-stream kinetic energy is `u_y²/2`. For tracer `s`, subtract
the spatial mean independently at each frame to obtain the variance field
`(s - mean_xy(s))²`. Its global reduction is the time average of spatial
variance; its local reduction is a patch's contribution to that variance,
not variance about the patch's own mean. The gradient target is `|∇s|²`;
it is not multiplied by diffusivity. Both targets characterize tracer
inhomogeneity and mixing without explicitly scaling by a regime parameter.

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

New runs shuffle complete passes with a permutation determined by the run seed
and pass number. The next batch is derived from completed optimizer updates,
not the loader's prefetched position. Separate loader generators keep worker
creation from consuming the masking random stream. Continuation checkpoints
retain Python, NumPy, Torch CPU/CUDA random states, model and optimizer state
(including JEPA's EMA target), best validation values, and the first feature
standard deviations used for collapse warnings. Schedules use the original
total budget and the restored global update number; an allocation boundary
does not restart warmup or EMA scheduling. This preserves the shuffled-pass
sampling distribution, but new seeded permutations do not reproduce the older
trainer's batch order.

The continuation identity includes source code, data and sampling, objective,
architecture, optimizer, seed, precision, and validation/image settings. It
allows cache relocation and worker-count changes. Atomic latest checkpoints
precede derived encoder exports and record the flushed history byte position.
Resume discards uncheckpointed history and recreates applicable interrupted
exports; incomplete legacy state is rejected. The Slurm wrapper requests a
stop 30 minutes before its 12-hour allocation ends, finishes the current update
and scheduled validation, and requeues only after a planned stop with progress.
It restages local data for each allocation and keeps metrics locally with W&B
disabled. Unexpected failures require inspection before manual resubmission.
CPU comparisons test exact continuation; production CUDA kernels retain their
existing nondeterministic behavior.

The workshop sampling configuration explicitly caps support at 101 frames.
New configurations use `target_offsets = [0, 8, 16, 40]`: target-start minus
context-start, measured in saved frames. For context 0–7 the targets are
0–7, 8–15, 16–23, and 40–47. The workshop configuration uses [0, 16, 40],
which preserves the original physical horizons formerly labeled gaps 0, 8,
and 32. Legacy gap configurations are rejected rather than reinterpreted.
Every requested target must fit inside the available trajectory support.

Three global contexts are equally spaced across eligible starts. For 200
frames these are 0, 76, 152; local contexts cycle through 0, 38, 76, 114, 152
according to trajectory order. The workshop's starts remain 0, 26, 53 globally
and 0, 13, 26, 40, 53 locally. Active matter's 81 frames give global starts
0, 16, 33. Each local context retains 64 deterministic uniformly sampled
positions. Checkpoints use identical samples and positions; no padding or
wrapping is used. Encoders see only the full unmasked input clip, never the
future target frames. Features include all 12 block outputs and the final norm.

Probes fit on official training trajectories. Official validation selects
probe parameters, layers, families, and encoder checkpoints; official test
is used only after those choices are recorded. This replaces the workshop's
five-fold fitting inside the official validation split. It is not an
assessment of unseen governing regimes. Training determines all fitted
feature standardization and target normalization statistics.

Ridge searches every encoder output and penalties
`1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100`. Targets are centered and the
weights solve `(XᵀX + α n I) w = Xᵀ(y − mean(y))`.
MLPs independently search every output: 128 ReLU units, dropout 0.1,
full-batch Adam, LR 0.01, weight decay 0.0001. Three fixed initializations
(0, 1, 2) each select a stopping state between 150 and 2,000 updates,
checking validation MSE every 20 updates with patience 100. Retain those
states and average predictions in float64 on both validation and test;
there is no post-selection refitting. Hyperparameters are fixed apart from
stopping duration and layer; there is no additional MLP LR sweep.

Each quantity/horizon/global-local cell selects Ridge or MLP by minimum
validation VRMSE after each family's layer/settings search. Exact family
ties prefer Ridge. Governing-parameter probes remain separate diagnostics.
Both Ridge and MLP report test scores at every encoder output for global and
local targets. Selected-family summaries retain the chosen score, family, and
layer; full depth curves remain in the separate Ridge and MLP results.

One encoder checkpoint is selected for each dataset/objective/training seed.
Candidates are the 25k, 50k, 75k, and 100k milestones plus the minimum
pretraining-validation-loss checkpoint. Verified identical encoder states
with the same configuration are fit once. Average quantities and global/local
settings equally within each horizon; assign 50% weight to the current
horizon and 50% to the equal mean over the three future horizons. The lowest
balanced validation VRMSE wins; exact checkpoint ties prefer earlier steps,
then checkpoint hash. Controls and governing parameters do not enter this
score. Missing task cells are errors, and undefined values are exposed rather
than silently dropping quantities or changing weights. A candidate without a
complete finite score is ineligible; if none are eligible, selection stops.
The manifest freezes all choices before test scoring, which requires a
matching selected probe-fit artifact.

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

Compute errors in the physical target units, with governing parameters using
the specified transforms. The primary metric is VRMSE: `sqrt(mean((prediction-target)²) / var(target))`,
with population variance (`ddof=0`) across all evaluated examples for that
quantity, horizon, and representation. Global examples are clip averages;
local examples are sampled patch averages. No dimensional epsilon is added;
zero target variance is explicitly undefined. For nonconstant targets this
is `sqrt(1-R²)`. Compute scores before averaging across quantities or encoder
seeds; converting an averaged R² would produce a different result.

This follows the variance-normalization principle of
[Walrus, Appendix F.1.1](https://arxiv.org/abs/2511.15684), whose field metric
uses spatial variance within each target field and a denominator epsilon.
Our scalar-target results are not directly comparable with its published
full-field scores. A direct comparison requires the same targets and sampling.
R², Pearson correlation, MSE, and log MSE remain available internally.
Undefined metrics are `null`; summary code does not silently reduce the seed
count to omit them. A single encoder seed has no between-run standard
deviation. Probe initializations and sampled positions are not independent
encoder runs. These descriptive summaries do not claim statistical significance.

Every completed cache, feature set, probe fit, result aggregate, and plot set
has a manifest. Consumers verify the self-hash and the content hash of each
file they actually read. Features bind to exact cache manifests; saved probes
bind to their clean features; aggregates require matching protocols, cache
identities, analysis code, checkpoint geometry/training protocol, and the
explicit expected roster. Selected checkpoint steps may differ only under the
same recorded selection policy, while configured training budgets still match.
Actual steps remain in the report provenance. New schema artifacts must be
regenerated from data; there is no converter that treats old result JSON as current evidence.

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
2. **Complete the seed-1 physical probe comparison.** All twelve training runs
   reached 100,000 steps. Use the retained candidates, full-trajectory sampling,
   richer physical targets, and validation-selected Ridge/MLP protocol above.
   Clean test scoring follows the recorded selections; smoke results are not
   scientific performance measurements.
3. **Extend the collaborator analysis** with attentive probing and noise
   experiments. The full candidate handoff allows checkpoint selection to be
   reassessed for a later, explicitly defined evaluation protocol.
4. **Train scientific seeds 2 and 3** with the selected training rates to measure
   between-run variation. The initial twelve encoders cover only seed 1.

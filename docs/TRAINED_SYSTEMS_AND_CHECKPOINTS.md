# Trained systems, data, and checkpoint choice

This note is a handover for the extended JEPA–vMAE analysis. It records what was trained, which data entered training, what is currently available under `checkpoints/`, and how we should choose encoder checkpoints without looking at the test results.

## Short answer

- The extension contains **12 completed 100,000-step runs**: three physical systems × four objectives, currently **only for training seed 1**.
- Training used the **entire temporal extent of every trajectory in The Well's official training split**, rather than the workshop paper's frames 0–100 only.
- It did **not** use every trajectory in the complete published dataset for gradient updates. Every eighth official-training trajectory was reserved for internal pretraining validation, while The Well's official validation and test splits were kept out of pretraining.
- Each run retained the 25k, 50k, 75k, and 100k encoder plus the encoder at minimum internal pretraining-validation loss. Some files are duplicates because the minimum occurred at 100k. There are 60 files holding 56 distinct encoder states, of which the 48 milestone files (12 runs × 4) are the checkpoint-selection candidates.
- We should **not** choose JEPA and vMAE checkpoints by comparing their pretraining losses or require a separate fixed-step comparison. The analysis will select one checkpoint per run from the common 25k/50k/75k/100k grid using downstream validation only, then use that frozen checkpoint for the scientific analysis.

## What was trained

### Physical systems

| System | The Well rollout | Input fields | Governing parameters used as diagnostics | Physical quantities used for representation probing |
|---|---|---|---|---|
| Rayleigh–Bénard convection | 200 frames, 512×128; four channels | buoyancy, pressure, horizontal and vertical velocity | log10 Rayleigh, log10 Prandtl | enstrophy, buoyancy-gradient energy, convective flux, pressure-gradient magnitude, buoyancy-Laplacian magnitude |
| Active matter | 81 frames, 256×256; eleven channels | concentration, velocity, orientation tensor, strain-rate tensor | alpha, zeta | kinetic energy, enstrophy, nematic order, squared nematic gradient |
| Shear flow | 200 frames, 256×512; four channels | tracer, pressure, horizontal and vertical velocity | log10 Reynolds, log10 Schmidt | cross-stream kinetic energy, enstrophy, tracer variance, squared tracer gradient |

The geometries and target definitions are not interchangeable. Rayleigh–Bénard uses a periodic horizontal direction and a Chebyshev wall-normal grid; active matter and shear flow use two periodic directions. Physical targets are calculated from raw float64 fields. A separate channel-wise z-normalized fp16 copy is passed to the encoder.

Primary dataset descriptions: [Rayleigh–Bénard](https://polymathic-ai.org/the_well/datasets/rayleigh_benard/), [active matter](https://polymathic-ai.org/the_well/datasets/active_matter/), and [shear flow](https://polymathic-ai.org/the_well/datasets/shear_flow/). Exact target formulas and numerical operators are in [`METHODS.md`](METHODS.md).

### Four objectives per system

| Objective | Encoder input | Training target | Objective-specific head |
|---|---|---|---|
| `jepa` | masked frames 0–7 | EMA target-encoder features at masked positions in frames 0–7 | six-block predictor |
| `mae` | masked frames 0–7 | normalized pixels at masked positions in frames 0–7 | four-block decoder |
| `jepa_future` | masked frames 0–7 | EMA target-encoder features at every position in the adjacent frames 8–15 | six-block future predictor |
| `mae_future` | masked frames 0–7 | normalized pixels at every position in the adjacent frames 8–15 | four-block future decoder |

This 2×2 design concerns the **pretraining target**: pixel versus latent space, and current versus adjacent-future clips. Downstream probe horizon is a separate evaluation axis applied to all four models. The reduced workshop-aligned study uses `target_offsets` 0, 16, and 40.

All online encoders are matched: a 12-block ViT, width 384, six attention heads, 2×16×16 spacetime patches, and 90% tube masking. The heads have width 192 and six attention heads. The predictor/decoder depths intentionally remain family-specific, so this is not a claim of identical total parameter count or compute. The frozen **online encoder** is the representation used for probing; the objective-specific head is discarded.

The spatial resolutions happen to give 1,024 tokens per eight-frame Rayleigh–Bénard or active-matter clip and 2,048 per shear-flow clip.

### Optimization and learning rates

Every final run used AdamW, batch size 64, weight decay 0.05, betas (0.9, 0.95), gradient clipping at 1.0, bfloat16, 5,000 warm-up updates, cosine decay to `1e-6`, and a nominal budget of 100,000 optimizer updates.

Learning rates were selected separately within each dataset/objective using completed 8,000-step seed-0 pilots. Selection used minimum internal pretraining-validation loss **within that objective's LR grid**, with collapse diagnostics reviewed. Every LR candidate used the same 8k pilot schedule, which makes the within-objective comparison controlled, but JEPA's target still moved during each pilot and a minimum could occur at a different schedule position. The pilot cosine schedule also ended at 8k, whereas the final-run schedule ends at 100k. The pilots were not counted as scientific encoder seeds.

| System | `jepa` | `mae` | `jepa_future` | `mae_future` |
|---|---:|---:|---:|---:|
| Rayleigh–Bénard | `5e-5` | `1e-4` | `2e-4` | `1e-4` |
| Active matter | `2e-4` | `1e-4` | `2e-4` | `1e-4` |
| Shear flow | `2e-4` | `1e-4` | `2e-4` | `1e-4` |

Using different learning rates is methodologically reasonable: smooth-L1 error in a changing latent target space and MSE in normalized pixel space have different scales and optimization geometry. Objective-specific tuning is therefore part of giving both models a fair optimization opportunity, not an unwanted confound in the controlled objective comparison. We keep the LR-selection procedure fixed, rather than forcing the same numerical LR. We must not claim that one model has a “lower loss” than another across objective families; the values are not on a common scale.

## Did training use all the data?

### What “all” means here

The answer is **all eligible windows from the official training split form the sampling support, but not all published trajectories are used for gradient updates**.

1. The current configs set `frame_limit: null`, so they retain each trajectory's full available length. This differs from the workshop models, which used only frames 0–100 inclusive.
2. Within The Well's official training split, every eighth trajectory is held out for internal pretraining validation. The remaining seven eighths supply optimizer updates. This split is trajectory-disjoint, so clips from one trajectory cannot leak between pretraining fit and pretraining validation.
3. Every valid start is eligible, at a stride of one saved frame. A length-`T` trajectory provides `T - 7` current-objective clips or `T - 15` adjacent context/future pairs.
4. The Well's official validation and test trajectories are not used for self-supervised pretraining. In the downstream workflow, probes are fit on the official training split, model/probe choices are made on official validation, and official test is opened only after those choices are frozen.
5. “Eligible” does not mean each clip is seen once. At 100,000 updates and batch size 64, each run processes 6.4 million sampled clips, repeating the finite support over many shuffled passes.

### Actual pretraining support recorded in the seed-1 checkpoints

| System | Published trajectories | Official train | Fit / internal val trajectories | Current fit / val clips | Future fit / val pairs | Planned passes at 100k: current / future |
|---|---:|---:|---:|---:|---:|---:|
| Rayleigh–Bénard | 1,750 | 1,400 | 1,225 / 175 | 236,425 / 33,775 | 226,625 / 32,375 | 27.07 / 28.24 |
| Active matter | 225 | 175 | 153 / 22 | 11,322 / 1,628 | 10,098 / 1,452 | 565.27 / 633.79 |
| Shear flow | 1,120 | 896 | 784 / 112 | 151,312 / 21,616 | 145,040 / 20,720 | 42.30 / 44.13 |

The clip counts come directly from `data_exposure` in `checkpoints/iclr2027/seed1/index.json`. The trajectory counts follow exactly from those clip counts and the recorded rollout lengths. “Planned passes” is `6,400,000 / eligible fit clips`; it is an exposure measure, not a count of independent simulations.
The same derivation gives the official split sizes. Dividing current-objective clip counts by 193 windows for a 200-frame rollout, or 74 for an 81-frame rollout, recovers the fit and held-out trajectory counts above. Adding those gives official-train sizes of 1,400, 175, and 896. With The Well's 80/10/10 splits, official validation/test contain 175/175 Rayleigh–Bénard, 25/25 active-matter, and 112/112 shear-flow trajectories.

A caveat for interpreting cross-system results: the same 100,000 updates correspond to very different numbers of passes, especially for active matter. Equal optimizer steps hold the optimization budget fixed but do not equalize unique trajectories, tokens, wall-clock compute, or dataset passes.

## Which checkpoints exist?

### Extended study

`checkpoints/iclr2027/seed1/` contains one directory for each system/objective pair. Each run has:

- `encoder_025pct.pt` — 25,000 updates;
- `encoder_050pct.pt` — 50,000 updates;
- `encoder_075pct.pt` — 75,000 updates;
- `encoder_100pct.pt` — 100,000 updates;
- `encoder_best_val.pt` — the update with the lowest internal pretraining-validation loss, evaluated every 2,000 updates.

The actual `best_val` steps are:

| System | `jepa` | `mae` | `jepa_future` | `mae_future` |
|---|---:|---:|---:|---:|
| Rayleigh–Bénard | 100k | 100k | 98k | 100k |
| Active matter | 16k | 96k | 20k | 18k |
| Shear flow | 4k | 98k | 4k | 100k |

When `best_val` is 100k, that file and `encoder_100pct.pt` represent the same encoder state: 60 paths hold 56 distinct encoder states. Checkpoint selection ignores `best_val` entirely, so the Stage-1 roster is the 48 milestone candidates (12 runs × 4 milestones).

These twelve extended runs are all **seed 1**. Probe initializations, noise draws, sampled token positions, checkpoints from different training steps, and the JEPA online/target encoders are not additional independent training seeds. A seed-1 comparison is descriptive and cannot provide an empirical between-seed standard deviation or support a statistical-significance claim.

### Workshop baseline

`checkpoints/neuripsworkshop/` contains six final 100k encoders: Rayleigh–Bénard `jepa` and `mae`, each for seeds 1, 2, and 3. These are the workshop-paper models trained on frames 0–100. They are useful as the historical baseline, but should not be pooled with the new full-trajectory seed-1 runs as if they were replicate seeds under one protocol.

## The JEPA EMA: what it is and why it matters

There is no EMA in the predictor. The three JEPA components are:

1. **Online encoder:** updated by AdamW through backpropagation. This is the encoder we freeze and probe.
2. **Predictor:** also updated by AdamW through backpropagation. It maps visible online-encoder features to target positions and is discarded after pretraining.
3. **Target encoder:** receives no gradients. After each optimizer update its weights move toward the online encoder: `target ← m × target + (1-m) × online`. Momentum `m` follows a cosine schedule from 0.996 toward 1.0, so the target changes increasingly slowly late in training.

MAE's pixel target is fixed, so MAE validation loss at two checkpoints measures reconstruction against the same kind of target. JEPA's target is the EMA encoder's own feature representation, which changes after every update. A JEPA loss at 4k and one at 100k are therefore errors against different targets; they are not a clean within-run learning curve for frozen-encoder quality.

Early in training the target encoder is still close to its initialization, and its layer-normalized features may be relatively easy for the predictor to match. The target can later encode richer structure and become harder to predict, allowing validation loss to rise even while the online representation improves. This is particularly relevant for shear-flow `jepa` and `jepa_future`: their recorded minima are at 4k, still inside the 5k learning-rate warm-up. Conversely, active-matter `mae_future` reaches its minimum at 18k after 1,152,000 processed pairs—about 114 passes over its small fit support—so ordinary overfitting is a separate plausible concern there. Rayleigh–Bénard `jepa`, `mae`, and `mae_future` have `best_val` at 100k, so those files duplicate their 100% encoder states.

JEPA loss can also improve if features partly collapse and become easy to predict. This is why `src/train.py` tracks target, context, and predictor feature standard deviations and warns when they fall below 10% of their first validation value. Before treating the early 4k, 16k, or 20k JEPA minima as serious candidates, inspect the original `history.jsonl` collapse diagnostics; those histories are not included in the encoder-only handoff. The minimum stays in the handoff so it is available for a later, explicitly defined protocol, but it is outside the Stage-1 candidate policy and receives no privileged status.

## Recommended checkpoint-selection protocol

**Astra consultation and project decision.** Astra was consulted through the session's `/slow` model and distinguished fixed-update from task-selected comparisons. We are choosing a single task-selected analysis because the study's estimand is representation quality under separately tuned objectives; equal steps would not equalize convergence or exact compute. This choice is recorded before test inspection.

### One checkpoint-selection policy

For each `(system, objective, training seed)`:

1. Use the common candidate grid: **25k, 50k, 75k, and 100k**. Do not include `best_val`: it gives objectives different candidate steps and JEPA's moving EMA target makes its loss minimum especially difficult to compare over training.
2. Extract frozen online-encoder features from the same predefined official-train and official-validation clips for every candidate. Never use official test during selection.
3. Fit Ridge and single-seed MLP probes on official training trajectories at transformer layer 4, chosen prospectively from the workshop paper's general layer-3–4 peak. Select probe family, hyperparameters, and stopping state separately for each physical task using official validation only and the same search space for every encoder.
4. Give every physical quantity and the pooled/local settings equal weight within a horizon. Let `H_h` be that mean validation VRMSE at target offset `h`. Minimize `S = 0.5 H_0 + 0.25(H_16 + H_40)`: 50% present and 50% shared equally among the two future horizons.
5. Governing-parameter diagnostics, nuisance controls, persistence baselines, pretraining loss, and test performance do not enter this checkpoint score.
6. Require every planned task cell to have a finite score. If any are missing, the candidate is ineligible rather than benefiting from a smaller average.
7. Break an exact checkpoint-score tie by earlier optimizer step, then checkpoint SHA-256; exact probe-family ties prefer Ridge.
8. Freeze one selected checkpoint per system/objective/seed in a manifest before extracting or scoring official-test features.

`scripts/probe_sweep.py prepare` implements this candidate policy directly: it freezes only the 25k/50k/75k/100k milestones, requires every dataset/seed in the handoff to supply all four objectives, refuses a run group whose four milestones are missing, duplicated, mislabelled, at the wrong fraction of the configured budget, or inconsistent in run metadata, and records the candidate policy in `study.json` so `load`, `run`, `collect`, `test`, and `report` reject a study prepared under any other roster. `best_val` files stay in the handoff but never enter the sweep, and a run group that declares only excluded files is reported as an incomplete milestone roster rather than silently dropped. For a different roster, call `src.evaluate select-checkpoints` with the intended probe-fit paths and pass the resulting selection artifact to `score-probes`. Scaled from the earlier 56-candidate estimate, the 48-candidate sweep is roughly 45 GPU-hours of MLP fitting, or about 5–9 hours on 8–12 L40S GPUs, excluding queueing and initial data preparation; caches and extracted features are reusable.

Candidate identities and steps are grounded in `checkpoints/iclr2027/seed1/index.json` and `manifest.json`. `src/train.py::export_encoders` confirms that the exported representation is the online encoder, `src/objectives/jepa.py::post_step` contains the EMA update, and `src/evaluation/selection.py` implements the balanced score and failure/tie rules.

### Why selected steps may differ

Different JEPA and vMAE checkpoints may win. That is intentional: selection uses a common downstream representation metric while allowing separately tuned objectives to progress at different rates. Report the selected step and `data_exposure` for every run. Selecting an early checkpoint after searching a completed 100k run does not make that model a cheap early-stopped run; the full search budget remains part of the method.

Do not choose different total training budgets merely because the numerical losses occupy different ranges. Loss scale supports objective-specific LR tuning; it does not demonstrate that one objective needs more updates. A longer-budget experiment would need a prospective common extension rule and newly defined LR/EMA schedules.

### Seed policy

The immediate seed-1 analysis must be labeled exploratory/descriptive. It supports reproducible measurements and 2×2 contrasts for these particular encoders, but not an expected ranking over training randomness, a between-run standard deviation, or a statistical-significance claim about objective families.

For a paper-level comparison, train seeds 2 and 3 under the same configs and candidate schedule. Apply the same frozen checkpoint-selection policy without consulting test results, calculate each scientific summary within an encoder seed, then report the mean and between-seed standard deviation. Probe initializations estimate probe variability, not encoder-training variability.

## Analysis handover checklist

- Use `checkpoints/iclr2027/seed1/index.json` as the authoritative candidate roster; paths alone do not establish distinct states.
- Fetch Git LFS payloads only for the systems being processed.
- Keep system, objective, seed, checkpoint step, checkpoint hash, training protocol, and data exposure in every artifact manifest.
- Use full-trajectory evaluation configs (`configs/eval_*.yaml`) for the extension. Do not accidentally use `configs/workshop/eval_rayleigh_benard.yaml`, which deliberately retains the paper's 101-frame support.
- Fit on official train; select checkpoints/probes on official validation; score official test only after the selection manifest is frozen.
- Never compare JEPA loss numerically with vMAE loss.
- Never count MLP initializations or selected checkpoints as encoder seeds.
- Report that the current extension has one training seed and therefore no between-run uncertainty estimate.

# Rayleigh–Bénard Stage 1 + Stage 2 findings

## Study

The study evaluated the four seed-1 objectives (`jepa`, `jepa_future`, `mae`,
and `mae_future`) on the official Rayleigh–Bénard train/validation/test splits.
Stage 1 compared the 25%, 50%, 75%, and 100% training milestones plus each
run's minimum-pretraining-validation-loss checkpoint at prospectively fixed
transformer block 4, then selected one encoder state per objective using
validation VRMSE. After freezing those four checkpoints, the final analysis fit
Ridge and MLP probes at all 12 transformer blocks and the final normalization
output. Validation independently selected each family's output and
hyperparameters in every cell, then selected the family for the main summary.
Only those frozen choices were evaluated on test.

Physics probes covered pooled and token representations, offsets 0, 16, and
40, and five targets. Results below average the five target-specific scores
within each representation/offset cell. Lower VRMSE and higher R² are better.

## Frozen checkpoint selections

| Objective | Selected step | Block-4 checkpoint-selection VRMSE |
|---|---:|---:|
| JEPA | 25,000 | 0.2671 |
| Future JEPA | 50,000 | 0.2890 |
| MAE | 50,000 | 0.2639 |
| Future MAE | 100,000 | 0.2694 |

These scores are from the prospective block-4 candidate comparison. The
full-depth selection artifact binds the four already-frozen winners to their
new probe fits; its all-output validation scores are not a second checkpoint
search.

Adding the four `best_val` labels introduced only one new encoder state: Future JEPA at step 98,000. The JEPA, MAE, and Future MAE `best_val` states were tensor-identical to their 100,000-step milestone states. Future JEPA's 98,000-step state scored 0.2940 validation VRMSE versus 0.2890 for its selected 50,000-step state. Consequently, all four selections and all Stage 2 test results remained unchanged.

## Test physics results

Each entry averages the five target-specific test scores. The encoder output
and probe family were selected independently in each target cell using
validation only.

| Representation | Offset | JEPA VRMSE / R² | Future JEPA | MAE | Future MAE |
|---|---:|---:|---:|---:|---:|
| Pooled | 0 | 0.0966 / 0.9898 | 0.0857 / 0.9912 | **0.0840 / 0.9917** | 0.0907 / 0.9898 |
| Pooled | 16 | 0.1519 / 0.9748 | **0.1179 / 0.9838** | 0.1484 / 0.9757 | 0.1434 / 0.9776 |
| Pooled | 40 | 0.2908 / 0.9002 | **0.1887 / 0.9599** | 0.3961 / 0.7966 | 0.3939 / 0.7983 |
| Token | 0 | **0.1852 / 0.9579** | 0.1900 / 0.9566 | 0.1853 / 0.9594 | 0.1867 / 0.9584 |
| Token | 16 | 0.5462 / 0.6773 | **0.5079 / 0.7145** | 0.5402 / 0.6902 | 0.5402 / 0.6876 |
| Token | 40 | 0.5475 / 0.6645 | **0.5400 / 0.6688** | 0.5521 / 0.6569 | 0.5512 / 0.6603 |

## Findings

1. **Future JEPA is strongest after validation selects encoder depth.** It has
the lowest mean VRMSE in four of six representation/horizon cells and 18 of 30
target-level cells. Its equal-cell mean is 0.2717, 10.3% below JEPA (0.3030)
and 14.5% below both MAE variants (0.3177).

2. **The gain is largest for pooled future prediction.** Future JEPA reaches
0.1179 at offset 16 and 0.1887 at offset 40. Its offset-40 error is 35.1% below
JEPA and 52.4% below MAE, with mean target-specific R² of 0.9599. It wins all
five pooled offset-40 targets.

3. **Full-depth analysis changes the token conclusion.** Future JEPA now has
the lowest token mean at offsets 16 and 40. The offset-16 improvement over JEPA
is 7.0%; the offset-40 improvement is smaller at 1.4%. Immediate token
performance is effectively tied between JEPA (0.1852) and MAE (0.1853).

4. **Future JEPA's useful information is concentrated late.** Validation
selected block 12 or final normalization for 22 of its 30 cells. By contrast,
JEPA most often selected blocks 4–6, MAE favored early-to-middle outputs, and
Future MAE selected block 1 for four of five immediate token targets. Relative
to fixed block 4, output selection lowers equal-cell test VRMSE by 18.2% for
Future JEPA, 4.2% for JEPA, 2.2% for Future MAE, and 0.5% for MAE.

5. **Every selected cell uses the nonlinear probe.** Validation selected MLP
in all 120 objective/cell combinations after each family independently searched
depth. Ridge remains reported as a controlled linear diagnostic, not the main
encoder ranking.

6. **Every encoder beats persistence on average at both future horizons.**
Persistence VRMSE is 0.4874 and 0.8121 for pooled offsets 16 and 40, versus
0.1179 and 0.1887 for the best encoder. Token persistence is 0.7528 and 0.7884,
versus Future JEPA's 0.5079 and 0.5400. Offset 0 is intentionally N/A because
copying the current target would be the identity.

7. **Regime/time metadata remains an important pooled long-horizon control.**
Ridge plus controls gives offset-40 pooled VRMSE 0.1832–0.2224. This is far
below encoder-only selected probes for JEPA and both MAE variants, but only
slightly below Future JEPA's 0.1887. Future JEPA therefore nearly closes the
metadata-control gap at the long pooled horizon.

8. **All representations encode the simulation regime strongly.** Across both
regime targets and both probe families, every objective has test R² above
0.9967.

## Interpretation and limits

The practical result is a depth-dependent advantage for future-conditioned
latent prediction: use Future JEPA for future readout, especially pooled
long-horizon prediction. Current pooled readout still slightly favors MAE, while
current token readout is effectively tied. The depth result is substantive,
not cosmetic: fixed block 4 obscured most of Future JEPA's advantage because
its validation-selected outputs are usually block 12 or final normalization.

This is a single checkpoint seed and single probe initialization, so there is
no between-seed uncertainty estimate. The workshop labels `t+8` and `t+32`
denote 8- and 32-frame gaps after an eight-frame context; they are the same
physical target windows as this study's target-start offsets 16 and 40. Direct
numerical comparison still requires care because the studies used different
checkpoint, probe-fitting, ensembling, and seed-selection protocols.

## Artifacts

- Checkpoint-selection study: `reports/rayleigh_benard_stage1_stage2/selection/selections.json`
- Full-depth frozen selections: `reports/rayleigh_benard_full_depth/selection/selections.json`
- Full-depth aggregate: `reports/rayleigh_benard_full_depth/reports/rayleigh_benard/aggregate/summary.json`
- Flat table: `reports/rayleigh_benard_full_depth/reports/rayleigh_benard/plots/summary.tsv`
- Physics, persistence, and depth plots: `reports/rayleigh_benard_full_depth/reports/rayleigh_benard/plots/*.pdf`
- Workshop Figure 1-style comparison: `reports/rayleigh_benard_full_depth/reports/rayleigh_benard/plots/workshop_figure1_comparison.pdf`

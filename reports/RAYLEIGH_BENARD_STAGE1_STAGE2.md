# Rayleigh–Bénard Stage 1 + Stage 2 findings

## Study

The study evaluated the four seed-1 objectives (`jepa`, `jepa_future`, `mae`, and `mae_future`) on the official Rayleigh–Bénard train/validation/test splits. Transformer layer 4 was fixed prospectively for every probe (stored as zero-indexed layer `3`). Stage 1 compared the 25%, 50%, 75%, and 100% training milestones plus each run's minimum-pretraining-validation-loss checkpoint, then selected one encoder state per objective using validation VRMSE. Stage 2 evaluated only those frozen selections on the test split.

Physics probes covered pooled and token representations, offsets 0, 16, and 40, and five targets. Results below average the five target-specific scores within each representation/offset cell. Lower VRMSE and higher R² are better.

## Frozen checkpoint selections

| Objective | Selected step | Validation VRMSE |
|---|---:|---:|
| JEPA | 25,000 | 0.2671 |
| Future JEPA | 50,000 | 0.2890 |
| MAE | 50,000 | 0.2639 |
| Future MAE | 100,000 | 0.2694 |

Adding the four `best_val` labels introduced only one new encoder state: Future JEPA at step 98,000. The JEPA, MAE, and Future MAE `best_val` states were tensor-identical to their 100,000-step milestone states. Future JEPA's 98,000-step state scored 0.2940 validation VRMSE versus 0.2890 for its selected 50,000-step state. Consequently, all four selections and all Stage 2 test results remained unchanged.

## Test physics results

| Representation | Offset | JEPA VRMSE / R² | Future JEPA | MAE | Future MAE |
|---|---:|---:|---:|---:|---:|
| Pooled | 0 | 0.1078 / 0.9875 | 0.1514 / 0.9753 | **0.0855 / 0.9915** | 0.0924 / 0.9898 |
| Pooled | 16 | 0.1617 / 0.9719 | 0.1764 / 0.9664 | **0.1493 / 0.9755** | 0.1501 / 0.9752 |
| Pooled | 40 | 0.3214 / 0.8763 | **0.2780 / 0.9129** | 0.3972 / 0.7959 | 0.3970 / 0.7963 |
| Token | 0 | 0.1974 / 0.9505 | 0.2417 / 0.9256 | **0.1840 / 0.9600** | 0.2122 / 0.9471 |
| Token | 16 | 0.5542 / 0.6716 | 0.5613 / 0.6632 | **0.5460 / 0.6840** | 0.5463 / 0.6815 |
| Token | 40 | 0.5562 / 0.6550 | 0.5800 / 0.6303 | 0.5533 / 0.6563 | **0.5518 / 0.6585** |

## Findings

1. **MAE is strongest at short and medium horizons.** It has the lowest mean VRMSE in four of the six representation/offset cells: pooled 0 and 16, and token 0 and 16. Against JEPA, its VRMSE is 20.7% lower for pooled offset 0, 7.7% lower for pooled offset 16, and 6.8% lower for token offset 0.

2. **Future JEPA is the clear pooled long-horizon winner.** At offset 40 its pooled VRMSE is 0.2780, 13.5% below JEPA and 30.0% below MAE; its mean R² rises to 0.9129 versus 0.8763 for JEPA and 0.7959 for MAE. This is the main positive result for future-conditioned latent prediction.

3. **The future objective does not improve token prediction.** Future JEPA trails JEPA at all three token offsets. Future MAE is nearly tied with MAE at offsets 16 and 40, but is worse at offset 0. The long-horizon benefit is specific to Future JEPA's pooled representation in this run.

4. **No objective dominates every target.** Across the 30 target/representation/offset cells, the lowest VRMSE belongs to MAE in 12, Future MAE in 9, Future JEPA in 6, and JEPA in 3. An equal-cell average gives JEPA the lowest overall VRMSE (0.3165), narrowly ahead of MAE (0.3192), but that scalar hides the much more useful horizon-specific pattern above.

5. **Nonlinearity matters.** Validation selected the MLP in 118 of 120 objective/cell combinations; Ridge was selected only twice. Report encoder quality using the selected-probe results rather than Ridge alone.

6. **Long-horizon pooled results are control-sensitive.** Adding regime/time controls to Ridge lowers offset-40 pooled mean VRMSE to 0.218–0.237 for every objective, below the encoder-only selected probes (0.278–0.397). The encoder comparison remains valid under the frozen protocol, but this gap shows that known regime/time metadata carries substantial complementary long-horizon information.

7. **All representations encode the simulation regime strongly.** For `log10_Prandtl` and `log10_Rayleigh`, every objective/probe combination has test R² above 0.9939. MAE variants produce the lowest regime VRMSEs; the best individual result is MAE Ridge on `log10_Rayleigh` (VRMSE 0.0234, R² 0.99945).

## Interpretation and limits

The practical conclusion is not that JEPA or MAE wins universally. Use MAE for immediate-state and medium-horizon readout; Future JEPA is the best candidate when the downstream requirement is pooled long-horizon prediction. Token-level future prediction remains weak relative to pooled prediction for every objective.

This is a single checkpoint seed and a single fixed transformer layer, so there is no between-seed uncertainty estimate and no post-hoc depth search. The workshop labels `t+8` and `t+32` denote 8- and 32-frame gaps after an eight-frame context; they are the same physical target windows as this study's target-start offsets 16 and 40. Direct numerical comparison still requires care because the studies used different checkpoint, layer, probe-fitting, ensembling, and seed-selection protocols.

## Artifacts

- Frozen selections: `reports/rayleigh_benard_stage1_stage2/selection/selections.json`
- Aggregate results: `reports/rayleigh_benard_stage1_stage2/reports/rayleigh_benard/aggregate/summary.json`
- Flat table: `reports/rayleigh_benard_stage1_stage2/reports/rayleigh_benard/plots/summary.tsv`
- Plots: `reports/rayleigh_benard_stage1_stage2/reports/rayleigh_benard/plots/*.pdf`
- Workshop Figure 1-style comparison: `reports/rayleigh_benard_stage1_stage2/reports/rayleigh_benard/plots/workshop_figure1_comparison.pdf`

| Target | JEPA $R^2$ | MAE $R^2$ | JEPA MSE | MAE MSE |
|---|---|---|---|---|
| Rayleigh number | 0.998 ± 0.000 | 0.999 ± 0.000 | 0.0034 ± 0.0002 | 0.0028 ± 0.0004 |
| Prandtl number | 0.992 ± 0.001 | 0.995 ± 0.001 | 0.0038 ± 0.0006 | 0.0024 ± 0.0004 |
| Rayleigh (shuffled control) | -0.037 ± 0.011 | -0.101 ± 0.029 | 2.07 ± 0.02 | 2.20 ± 0.06 |
| Prandtl (shuffled control) | -0.039 ± 0.010 | -0.059 ± 0.021 | 0.469 ± 0.005 | 0.478 ± 0.009 |

MSE is in squared log₁₀ units (targets are log₁₀Ra, log₁₀Pr), derived exactly from R² via MSE = (1 − R²)·Var(y_test), with Var(log₁₀Ra) = 2.0 and Var(log₁₀Pr) = 0.451 on the regime-balanced test split (35 combos × 5 ICs). RMSE ≈ 0.05–0.06 dex for the real targets, i.e. Ra/Pr recovered to within a factor of ~1.1–1.15.

## Qu-comparable MSE (normalized targets, Qu et al. convention)

Qu et al. compute MSE on standardized targets: Ra as (log₁₀Ra − 8.0)/1.41, Pr as **raw** (Pr − 2.69)/3.38. Those constants are exactly the balanced-grid population stats, so for Rayleigh the conversion is exact and reduces to MSE_Qu = 1 − R². Prandtl differs (Qu standardize raw Pr, our probe predicts log₁₀Pr): converting the aggregate is only possible under an assumption (per-sample log₁₀ errors ≈ homoscedastic Gaussian), so the Pr and combined values below are approximations — marked ≈.

| Target (Qu-normalized) | JEPA MSE | MAE MSE |
|---|---|---|
| Rayleigh (exact) | 0.0017 | 0.0014 |
| Prandtl (≈) | 0.034 | 0.021 |
| RB combined, ½(Ra + Pr) (≈) | 0.018 | 0.011 |

Qu et al. report **0.13 (JEPA)** and **0.18 (VideoMAE)** for the combined RB regime task — roughly an order of magnitude above our numbers even with the approximation caveat. Caveats before quoting: (1) the Pr/combined values assume homoscedastic Gaussian log-errors (raw-space MSE is dominated by the Pr = 10 trajectories, so heteroscedasticity would move it); an exact number needs per-example probe predictions — ask Hugo to re-run the regime family with predictions dumped (small change in `regime_family()`); (2) protocol differs from Qu's (frozen encoder + selected probe here vs their pipeline), so present as "same task and metric, different readout protocol".

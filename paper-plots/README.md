# paper-plots

Redesigned result figures for the workshop paper (Rayleigh–Bénard only).
All scripts read `sweep_results/rayleigh_benard_workshop_test_eval.json` —
no recomputation, same data as `reports/figures/`. Regenerate everything with:

```bash
uv run python paper-plots/plot_fig2_probe_heatmaps.py
uv run python paper-plots/plot_fig3_depth.py          # --gap 32 (default) or 8, 0
uv run python paper-plots/plot_fig4_noise.py
uv run python paper-plots/make_regime_table.py
```

Shared visual language (`style.py`): Ray-JEPA blue `#3b6fd4`, Ray-vMAE
orange `#d4703b`, serif, no top/right spines, SEM bands across the 3 encoder
seeds, PNG+PDF output. "Ray-JEPA / Ray-vMAE" everywhere — no Model A/B.

## Figure 2 — WHAT is represented? (`fig2_probe_heatmaps`)

One merged grid with a single shared row-label column: rows = physical
quantities (local/differential, then coupled/integrated, then token-only
divergence and signed vorticity at the bottom); two column blocks, Pooled
and Token, each spanning t+0 / t+8 / t+32. Token-only rows show an em-dash
in the Pooled block (no pooled target exists by construction). Cell
**shading** = ΔR² (Ray-JEPA − Ray-vMAE, clipped to ±0.2 for color only);
cell **text** = the two absolute scores "Ray-JEPA / Ray-vMAE". One-line key
above the grid; vertical group labels (local/differential, coupled/
integrated, token-only) along the right edge; horizontal colorbar centered
below the grid.

- Grey cells: both objectives below R² = 0.15 — the delta between two
  near-noise numbers isn't a meaningful comparison (say this in the caption).
- Faded shading: |Δ| smaller than the combined nested SE of the two means,
  i.e. not distinguishable from seed noise.
- Numbers are across-encoder-seed means at each cell's own held-out-valid
  frozen layer, evaluated once on the test split (same protocol as before).

## Figure 3 — WHERE does the difference emerge? (`fig3_depth_gap32`)

Main panel: mean R² over the 7 pooled quantities vs encoder layer at t+32,
Ray-JEPA vs Ray-vMAE, ±SEM across encoder seeds. Small aligned strip below: ΔR² vs
layer with the combined-in-quadrature SEM band — makes "early layers ≈
similar, deeper layers → growing JEPA advantage" explicit instead of asking
the reader to judge the gap between curves. Curves are the held-out-valid
layer sweeps (full curves, no argmax selection bias).

Appendix: `fig3_appendix_depth_per_quantity_gap32` — same plot per quantity
(replaces the standalone buoyancy-gradient panel in the old fig3).

## Figure 4 — HOW ROBUST is it? (`fig4_noise`)

(a) mean ridge Pearson r over quantities vs injected noise σ (absolute);
(b) degradation from clean, r(σ) − r(0), each seed baselined on its own
clean value. Linear x-axis (the old symlog visually exaggerated the
low-noise region). Pearson r, not R²: under the clean-fit protocol R² goes
arbitrarily negative at high noise (seed means < −100 in the raw JSON) and
is incomparable there; caption should state this. Ridge probe, so the only
variance shown is encoder-seed variance; frozen layers fixed from the t+0
selection.

Appendix: `fig4_appendix_noise_per_quantity` — per-quantity curves showing
the texture the mean hides (edge is large on differential quantities,
compresses/crosses on enstrophy and Okubo–Weiss at extreme noise).

## Regime table (`regime_table.md` / `.tex`)

Rayleigh/Prandtl regression R² (mean ± nested SE) plus shuffled-label
controls, as a table rather than a figure — conceptually separate from the
derived-physics probing. Qu et al.'s JEPA/VideoMAE numbers are not in our
JSON; add their reported values as extra columns from their paper.

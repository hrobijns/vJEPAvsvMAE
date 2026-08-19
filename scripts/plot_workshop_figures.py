"""Publication figures 2-4 for the NeurIPS workshop paper.
Figure 1 (architecture diagram) is made separately.

This is the "formal sweep" version: all three figures now read exclusively
from `sweep_results/{dataset}_workshop_test_eval.json`'s "aggregated"
section (produced by scripts/workshop_test_eval.py's aggregate_by_objective()),
which fixes the rigor gap the original version of this file had -- picking
each cell's "best layer" via argmax over the SAME train-split data the R^2
was then read from. Here the layer is selected on a held-out slice (carved
out of train, not the Well `valid` split -- see workshop_test_eval.py's
docstring for why) and every reported R^2 is a single held-out TEST-split
evaluation. Linear-probe results are dropped entirely from these figures per
the formal-sweep methodology (MLP only throughout, ridge kept only as an
audit trail in the raw JSON, not plotted).

Supports 1-or-more encoder seeds per objective transparently: the
"aggregated" section already combines however many seeds were probed via a
nested (encoder-seed x probe-seed) variance decomposition -- this script
just reads `mean`/`nested_se` and doesn't need to know how many seeds went
into them. With a single seed per objective this is identical to the
original per-checkpoint numbers.

  Fig 2: variable x time-into-future heatmap, one grid per dataset. Rows =
    physical quantities (local/differential, then coupled/integrated, then
    token-only quantities with no pooled target), columns = {t+0, t+8, t+32}
    x {pooled, token}. Cell = mean_R^2(JEPA) - mean_R^2(MAE) (each side
    itself the across-encoder-seed mean where >1 seed was probed), at that
    (quantity, objective, pooled/token, time offset)'s own held-out-valid-
    selected frozen layer(s) -- t+0, gap=8, and gap=32 each get their own
    layer selection (contemporaneous-optimal, gap=8-optimal, and
    gap=32-optimal depth genuinely differ). Cells whose delta is smaller
    than the combined nested SE of the two means (i.e. not distinguishable
    from seed noise) are rendered at reduced opacity.
  Fig 3: R^2 vs. encoder depth, averaged over variables, MLP probe, pooled,
    one panel per dataset -- uses the held-out-valid R^2 curve computed as a
    byproduct of Fig 2's layer selection (genuinely held-out, no argmax
    selection bias -- this is a full curve, not a max, so no re-selection
    bias here either way; averaged across encoder seeds when >1 was probed).
  Fig 4: R^2 vs. injected noise std, averaged over variables, MLP probe
    (the linear-only gap from the original version is now closed -- see
    workshop_test_eval.py's noise-robustness family), pooled, one panel per
    dataset. Each quantity's layer is frozen from its own contemporaneous
    (t+0) selection and reused across all noise levels (no separate
    noise-layer search).

A frozen_layers.txt summary is written alongside the figures listing every
(dataset, quantity, objective) -> layer choice(s) (one per encoder seed
probed), for the paper's methods text.

Usage:
    uv run python scripts/plot_workshop_figures.py \
        --sweep-dir sweep_results --out-dir reports/figures \
        --datasets active_matter shear_flow rayleigh_benard
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def layer_labels(n_layers: int) -> list[str]:
    return [f"L{i}" for i in range(n_layers - 1)] + ["norm"]


OBJECTIVES = ("jepa", "mae")
COLORS = {"jepa": "#3b6fd4", "mae": "#d4703b"}
GAPS = [8, 32]

# Hand-curated row ordering for Figure 2 (local/differential quantities
# first, then coupled/integrated ones) -- not derived from any config, see
# LINEAR_PROBE.md's prose characterization. Review before finalizing.
ROW_GROUPS = {
    "active_matter": {
        "local": ["order_grad_mag", "active_stress_div_mag"],
        "coupled": ["nematic_order", "flow_align", "strain_order_align", "enstrophy"],
    },
    "shear_flow": {
        "local": ["tracer_grad", "pressure_grad_mag", "tracer_laplacian", "strain_rate_mag"],
        "coupled": ["advective_flux", "okubo_weiss", "enstrophy"],
    },
    "rayleigh_benard": {
        "local": ["buoyancy_grad", "pressure_grad_mag", "buoyancy_laplacian"],
        "coupled": ["convective_flux", "velocity_buoyancy_coherence", "okubo_weiss", "enstrophy"],
    },
}

# Token-only quantities: no pooled counterpart exists (spatial mean is ~0 by
# construction -- divergence integrates to ~0 by continuity, vorticity by
# symmetry), so the P column is structurally N/A, not just missing data.
TOKEN_ONLY_ROWS = ["divergence", "vorticity_signed"]

# Display names for figure text -- config/JSON keys stay snake_case, plots
# get human-readable labels.
DATASET_LABELS = {
    "active_matter": "Active Matter",
    "shear_flow": "Shear Flow",
    "rayleigh_benard": "Rayleigh–Bénard",
}

QUANTITY_LABELS = {
    "order_grad_mag": "Order Grad. Mag.",
    "active_stress_div_mag": "Active Stress Div. Mag.",
    "nematic_order": "Nematic Order",
    "flow_align": "Flow Alignment",
    "strain_order_align": "Strain–Order Align.",
    "enstrophy": "Enstrophy",
    "tracer_grad": "Tracer Grad.",
    "pressure_grad_mag": "Pressure Grad. Mag.",
    "tracer_laplacian": "Tracer Laplacian",
    "strain_rate_mag": "Strain Rate Mag.",
    "advective_flux": "Advective Flux",
    "okubo_weiss": "Okubo–Weiss",
    "buoyancy_grad": "Buoyancy Grad.",
    "buoyancy_laplacian": "Buoyancy Laplacian",
    "convective_flux": "Convective Flux",
    "velocity_buoyancy_coherence": "Velocity–Buoyancy Coh.",
    "divergence": "Divergence",
    "vorticity_signed": "Vorticity (Signed)",
}


def _nice_dataset(name: str) -> str:
    return DATASET_LABELS.get(name, name.replace("_", " ").title())


def _nice_quantity(name: str) -> str:
    return QUANTITY_LABELS.get(name, name.replace("_", " ").title())


def _entry(agg: dict, q: str, obj: str) -> dict | None:
    return agg.get(q, {}).get(obj)


def _delta_and_se(agg: dict, q: str):
    """Returns (delta, combined_se) for a (quantity) cell from an aggregated
    {quantity: {"jepa"/"mae": _nested_stats(...)}} dict, or (nan, nan) if
    either side is missing. combined_se = sqrt(nested_se_jepa^2 +
    nested_se_mae^2) -- nested_se already folds in both probe-seed and
    (if >1) encoder-seed variance, so a cell is flagged only when the delta
    doesn't clear the FULL uncertainty budget, not just probe-seed noise."""
    ej, em = _entry(agg, q, "jepa"), _entry(agg, q, "mae")
    if ej is None or em is None:
        return np.nan, np.nan
    delta = ej["mean"] - em["mean"]
    se = np.sqrt(ej["nested_se"] ** 2 + em["nested_se"] ** 2)
    return delta, se


def _write_frozen_layers(out_dir: Path, records: list[tuple], filename: str = "frozen_layers.txt"):
    """records: (dataset, quantity, objective, feature_kind, layers) rows --
    `layers` is a list, one entry per encoder seed probed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{'dataset':<18}{'quantity':<30}{'objective':<6}{'feature':<11}layers", "-" * 80]
    for dataset, q, obj, kind, layers in records:
        layers_str = ",".join(str(l) for l in layers)
        lines.append(f"{dataset:<18}{q:<30}{obj:<6}{kind:<11}{layers_str}")
    (out_dir / filename).write_text("\n".join(lines) + "\n")
    print(f"[frozen-layers] wrote {filename} ({len(records)} entries)")


def plot_variable_time_heatmap(sweep_dir: Path, out_dir: Path, datasets: list[str]):
    panels = []  # (dataset, pooled_rows, n_local_p, pooled_grid, pooled_se, token_rows, n_local_t, n_local_coupled_t, token_grid, token_se)
    frozen_records = []
    for dataset in datasets:
        data = _load(sweep_dir / f"{dataset}_workshop_test_eval.json")
        if data is None or "aggregated" not in data:
            print(f"[var-time-heatmap] skip {dataset}: missing {dataset}_workshop_test_eval.json "
                  f"or it predates the 'aggregated' section")
            continue
        agg = data["aggregated"]
        pooled_c = agg["contemporaneous"]["pooled"]
        token_c = agg["contemporaneous"]["token"]

        local_rows, coupled_rows = ROW_GROUPS[dataset]["local"], ROW_GROUPS[dataset]["coupled"]
        pooled_rows = local_rows + coupled_rows
        token_rows = local_rows + coupled_rows + TOKEN_ONLY_ROWS
        n_local_p = len(local_rows)
        n_local_t, n_local_coupled_t = len(local_rows), len(local_rows) + len(coupled_rows)

        forecast_pooled = {gap: agg["forecast"][str(gap)]["pooled"] for gap in GAPS}
        forecast_token = {gap: agg["forecast"][str(gap)]["token"] for gap in GAPS}

        for q in pooled_rows:
            for obj in OBJECTIVES:
                e = _entry(pooled_c, q, obj)
                if e is not None:
                    frozen_records.append((dataset, q, obj, "pooled-t0", e["frozen_layers"]))
                for gap in GAPS:
                    e = _entry(forecast_pooled[gap], q, obj)
                    if e is not None:
                        frozen_records.append((dataset, q, obj, f"pooled-fc{gap}", e["frozen_layers"]))
        for q in token_rows:
            for obj in OBJECTIVES:
                e = _entry(token_c, q, obj)
                if e is not None:
                    frozen_records.append((dataset, q, obj, "token-t0", e["frozen_layers"]))
                for gap in GAPS:
                    e = _entry(forecast_token[gap], q, obj)
                    if e is not None:
                        frozen_records.append((dataset, q, obj, f"token-fc{gap}", e["frozen_layers"]))

        pooled_grid = np.full((len(pooled_rows), 3), np.nan)
        pooled_se = np.full((len(pooled_rows), 3), np.nan)
        for i, q in enumerate(pooled_rows):
            pooled_grid[i, 0], pooled_se[i, 0] = _delta_and_se(pooled_c, q)
            for gi, gap in enumerate(GAPS):
                pooled_grid[i, 1 + gi], pooled_se[i, 1 + gi] = _delta_and_se(forecast_pooled[gap], q)

        token_grid = np.full((len(token_rows), 3), np.nan)
        token_se = np.full((len(token_rows), 3), np.nan)
        for i, q in enumerate(token_rows):
            token_grid[i, 0], token_se[i, 0] = _delta_and_se(token_c, q)
            for gi, gap in enumerate(GAPS):
                token_grid[i, 1 + gi], token_se[i, 1 + gi] = _delta_and_se(forecast_token[gap], q)

        panels.append((dataset, pooled_rows, n_local_p, pooled_grid, pooled_se,
                        token_rows, n_local_t, n_local_coupled_t, token_grid, token_se))

    _write_frozen_layers(out_dir, frozen_records)

    if not panels:
        print("[var-time-heatmap] no panels to plot")
        return

    # Color is clipped to +/-CLIP: a handful of (quantity, gap, layer) cells
    # have pathological MLP-probe R^2 (hard target at long horizons, small
    # held-out sample) -- letting those set vmax would wash out every other
    # cell to near-white. The true value is still shown as text; only the
    # color is capped.
    CLIP = 1.0
    n_ds = len(panels)
    max_p_rows = max(len(p[1]) for p in panels)
    max_t_rows = max(len(p[5]) for p in panels)
    fig, axes = plt.subplots(
        2, n_ds, figsize=(3.8 * n_ds, 0.36 * (max_p_rows + max_t_rows) + 2.0),
        gridspec_kw={"height_ratios": [max_p_rows, max_t_rows]},
    )
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    def draw(ax, rows, grid, se, dividers, show_title, dataset):
        im = ax.imshow(np.clip(grid, -CLIP, CLIP), aspect="auto", cmap="RdBu", vmin=-CLIP, vmax=CLIP)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                clipped = abs(v) > CLIP
                label = f"{v:+.1f}*" if clipped else f"{v:+.2f}"
                txt_color = "white" if (clipped or abs(v) > 0.6 * CLIP) else "black"
                not_sig = (not np.isnan(se[i, j])) and abs(v) < se[i, j]
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=txt_color,
                         alpha=0.4 if not_sig else 1.0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["t+0", "t+8", "t+32"], fontsize=10, fontweight="bold")
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([_nice_quantity(r) for r in rows], fontsize=10)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        for boundary in dividers:
            if 0 < boundary < len(rows):
                ax.axhline(boundary - 0.5, color="0.3", lw=1.0)
        if show_title:
            ax.set_title(_nice_dataset(dataset), fontsize=14, pad=12)
        return im

    for col, (dataset, pooled_rows, n_local_p, pooled_grid, pooled_se,
              token_rows, n_local_t, n_local_coupled_t, token_grid, token_se) in enumerate(panels):
        draw(axes[0][col], pooled_rows, pooled_grid, pooled_se, (n_local_p,), True, dataset)
        im = draw(axes[1][col], token_rows, token_grid, token_se, (n_local_t, n_local_coupled_t), False, dataset)

    fig.subplots_adjust(left=0.18, right=0.86, top=0.94, bottom=0.03, wspace=1.4, hspace=0.15)

    pooled_pos, token_pos = axes[0][0].get_position(), axes[1][0].get_position()
    fig.text(0.02, 0.5 * (pooled_pos.y0 + pooled_pos.y1), "POOLED", rotation=90, va="center", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.5 * (token_pos.y0 + token_pos.y1), "TOKEN", rotation=90, va="center", fontsize=13, fontweight="bold")

    cbar_ax = fig.add_axes((0.89, 0.15, 0.018, 0.5))
    cbar = fig.colorbar(im, cax=cbar_ax, label=r"$\Delta R^2$ (JEPA $-$ MAE)")
    cbar.set_label(r"$\Delta R^2$ (JEPA $-$ MAE)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2_variable_time_heatmap.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_variable_time_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[var-time-heatmap] wrote fig2_variable_time_heatmap.{{pdf,png}} ({n_ds} datasets x 2 rows)")


GREY = "#c8c8c8"
LOW_SIGNAL_THRESHOLD = 0.15  # both R^2 below this -> neither objective actually decodes it
HEATMAP_CLIP = 0.2  # color-scale range for delta R^2 -- data is small-magnitude, unlike fig2's +/-1.0


def plot_rb_detailed_heatmap(sweep_dir: Path, out_dir: Path, dataset: str = "rayleigh_benard"):
    """RB-only variant of the variable x time heatmap: each cell shows model
    A's R^2 (top), the A-B delta in large bold text (middle), and model B's
    R^2 (bottom), instead of the delta-only + significance-fade design in
    plot_variable_time_heatmap. Cells where BOTH objectives score below
    LOW_SIGNAL_THRESHOLD are greyed out -- neither objective actually decodes
    that (quantity, gap), so the delta between two near-noise numbers isn't a
    meaningful comparison (explained in the figure caption, not on-figure).
    Pooled and token panels sit side by side; cells are narrow and tall,
    sized to just fit the 3-line text stack. Legend is deliberately minimal:
    a colorbar (concise title only) and one example cell with generic
    "Model A/B" labels -- no on-figure prose."""
    data = _load(sweep_dir / f"{dataset}_workshop_test_eval.json")
    if data is None or "aggregated" not in data:
        print(f"[rb-detailed-heatmap] skip: missing {dataset}_workshop_test_eval.json "
              f"or it predates the 'aggregated' section")
        return
    agg = data["aggregated"]
    pooled_c = agg["contemporaneous"]["pooled"]
    token_c = agg["contemporaneous"]["token"]
    forecast_pooled = {gap: agg["forecast"][str(gap)]["pooled"] for gap in GAPS}
    forecast_token = {gap: agg["forecast"][str(gap)]["token"] for gap in GAPS}

    local_rows, coupled_rows = ROW_GROUPS[dataset]["local"], ROW_GROUPS[dataset]["coupled"]
    pooled_rows = local_rows + coupled_rows
    token_rows = local_rows + coupled_rows + TOKEN_ONLY_ROWS
    n_local_p = len(local_rows)
    n_local_t, n_local_coupled_t = len(local_rows), len(local_rows) + len(coupled_rows)

    def build_grids(rows, contemp_source, forecast_source):
        jepa_grid = np.full((len(rows), 3), np.nan)
        mae_grid = np.full((len(rows), 3), np.nan)
        for i, q in enumerate(rows):
            ej, em = _entry(contemp_source, q, "jepa"), _entry(contemp_source, q, "mae")
            if ej is not None:
                jepa_grid[i, 0] = ej["mean"]
            if em is not None:
                mae_grid[i, 0] = em["mean"]
            for gi, gap in enumerate(GAPS):
                ej, em = _entry(forecast_source[gap], q, "jepa"), _entry(forecast_source[gap], q, "mae")
                if ej is not None:
                    jepa_grid[i, 1 + gi] = ej["mean"]
                if em is not None:
                    mae_grid[i, 1 + gi] = em["mean"]
        return jepa_grid, mae_grid

    pooled_jepa, pooled_mae = build_grids(pooled_rows, pooled_c, forecast_pooled)
    token_jepa, token_mae = build_grids(token_rows, token_c, forecast_token)

    def draw(ax, rows, jepa_grid, mae_grid, dividers, show_title):
        delta_grid = jepa_grid - mae_grid
        both_low = (jepa_grid < LOW_SIGNAL_THRESHOLD) & (mae_grid < LOW_SIGNAL_THRESHOLD)
        display = np.clip(delta_grid, -HEATMAP_CLIP, HEATMAP_CLIP)
        ax.imshow(display, aspect="auto", cmap="RdBu", vmin=-HEATMAP_CLIP, vmax=HEATMAP_CLIP)
        for i in range(delta_grid.shape[0]):
            for j in range(delta_grid.shape[1]):
                j_r2, m_r2, d = jepa_grid[i, j], mae_grid[i, j], delta_grid[i, j]
                if np.isnan(d):
                    continue
                if both_low[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=GREY, edgecolor="none", zorder=1))
                    txt_color = "0.35"
                else:
                    txt_color = "white" if abs(d) > 0.6 * HEATMAP_CLIP else "black"
                clipped = abs(d) > HEATMAP_CLIP
                d_label = f"{d:+.2f}" + ("*" if clipped else "")
                ax.text(j, i - 0.29, f"{j_r2:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color, zorder=2)
                ax.text(j, i, d_label, ha="center", va="center", fontsize=10.5, fontweight="bold", color=txt_color, zorder=2)
                ax.text(j, i + 0.29, f"{m_r2:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color, zorder=2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["t+0", "t+8", "t+32"], fontsize=10, fontweight="bold")
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([_nice_quantity(r) for r in rows], fontsize=10)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        for boundary in dividers:
            if 0 < boundary < len(rows):
                ax.axhline(boundary - 0.5, color="0.3", lw=1.0)
        if show_title:
            ax.set_title(_nice_dataset(dataset), fontsize=13, pad=10)

    # Explicit inch-based layout so cells stay narrow+tall (just fitting the
    # 3-line text stack) regardless of matplotlib's default axes sizing, and
    # so the pooled/token panels (different row counts) sit side by side
    # top-aligned with visually matched cell size.
    CELL_W_IN, CELL_H_IN = 0.62, 0.60
    LABEL_W_IN, PANEL_GAP_IN = 1.75, 0.55
    LEFT_MARGIN_IN, RIGHT_MARGIN_IN = 0.1, 0.1
    TOP_MARGIN_IN, LEGEND_H_IN, BOTTOM_MARGIN_IN = 0.85, 1.15, 0.1

    pooled_w_in = LABEL_W_IN + 3 * CELL_W_IN
    token_w_in = LABEL_W_IN + 3 * CELL_W_IN
    pooled_h_in = len(pooled_rows) * CELL_H_IN
    token_h_in = len(token_rows) * CELL_H_IN
    grid_h_in = max(pooled_h_in, token_h_in)

    fig_w_in = LEFT_MARGIN_IN + pooled_w_in + PANEL_GAP_IN + token_w_in + RIGHT_MARGIN_IN
    fig_h_in = TOP_MARGIN_IN + grid_h_in + LEGEND_H_IN + BOTTOM_MARGIN_IN
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    grid_w_in = 3 * CELL_W_IN
    top_y_in = fig_h_in - TOP_MARGIN_IN
    pooled_grid_left_in = LEFT_MARGIN_IN + LABEL_W_IN
    ax_pooled = fig.add_axes((pooled_grid_left_in / fig_w_in, (top_y_in - pooled_h_in) / fig_h_in,
                               grid_w_in / fig_w_in, pooled_h_in / fig_h_in))
    token_grid_left_in = LEFT_MARGIN_IN + pooled_w_in + PANEL_GAP_IN + LABEL_W_IN
    ax_token = fig.add_axes((token_grid_left_in / fig_w_in, (top_y_in - token_h_in) / fig_h_in,
                              grid_w_in / fig_w_in, token_h_in / fig_h_in))

    draw(ax_pooled, pooled_rows, pooled_jepa, pooled_mae, (n_local_p,), False)
    draw(ax_token, token_rows, token_jepa, token_mae, (n_local_t, n_local_coupled_t), False)
    ax_pooled.text(0, 1.0 + 0.5 / max(len(pooled_rows), 1), "POOLED", transform=ax_pooled.transAxes,
                    fontsize=10, fontweight="bold", ha="left", va="bottom")
    ax_token.text(0, 1.0 + 0.5 / max(len(token_rows), 1), "TOKEN", transform=ax_token.transAxes,
                   fontsize=10, fontweight="bold", ha="left", va="bottom")
    fig.text(0.5, 1 - 0.06, _nice_dataset(dataset), fontsize=14, ha="center", va="top")

    # Legend: colorbar (concise title, normalized to +/-HEATMAP_CLIP) + one
    # example cell with generic Model A/B labels -- no other on-figure text.
    legend_bottom_in = BOTTOM_MARGIN_IN
    cbar_w_in, cbar_h_in = 1.9, 0.22
    cbar_left_in = LEFT_MARGIN_IN + 0.1
    cbar_ax = fig.add_axes((cbar_left_in / fig_w_in, (legend_bottom_in + 0.35) / fig_h_in,
                             cbar_w_in / fig_w_in, cbar_h_in / fig_h_in))
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(vmin=-HEATMAP_CLIP, vmax=HEATMAP_CLIP))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$\Delta R^2$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    key_w_in, key_h_in = 0.9, 0.9
    key_left_in = cbar_left_in + cbar_w_in + 0.9
    key_ax = fig.add_axes((key_left_in / fig_w_in, legend_bottom_in / fig_h_in,
                            key_w_in / fig_w_in, key_h_in / fig_h_in))
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.axis("off")
    key_ax.imshow([[0.06]], extent=(0, 1, 0, 1), aspect="auto", cmap="RdBu",
                   vmin=-HEATMAP_CLIP, vmax=HEATMAP_CLIP, zorder=1)
    key_ax.text(0.5, 0.78, "Model A $R^2$", ha="center", va="center", fontsize=7, zorder=2)
    key_ax.text(0.5, 0.5, r"$\Delta R^2$", ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=2)
    key_ax.text(0.5, 0.22, "Model B $R^2$", ha="center", va="center", fontsize=7, zorder=2)
    key_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="0.3", lw=0.8, zorder=3, transform=key_ax.transAxes))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2_rb_detailed_heatmap.pdf", dpi=200)
    fig.savefig(out_dir / "fig2_rb_detailed_heatmap.png", dpi=200)
    plt.close(fig)
    print("[rb-detailed-heatmap] wrote fig2_rb_detailed_heatmap.{pdf,png}")


def plot_rb_depth_story(sweep_dir: Path, out_dir: Path, dataset: str = "rayleigh_benard",
                          gap: int = 32, highlight_quantity: str = "buoyancy_grad"):
    """Two-panel JEPA-advantage-vs-depth story for the forecast (t+{gap})
    family, where the biggest, most consistent JEPA lead lives (see
    LINEAR_PROBE.md's forecast-content section). Left: mean R^2-vs-layer
    curve averaged across every pooled forecast quantity -- JEPA leads at
    every layer. Right: `highlight_quantity` alone (default buoyancy_grad,
    chosen by inspecting every quantity's delta-vs-layer curve for the
    cleanest near-monotonic growth in JEPA's advantage with depth, not just
    the largest peak) -- the same pattern, much more pronounced: JEPA keeps
    climbing through mid-depth while MAE plateaus early."""
    data = _load(sweep_dir / f"{dataset}_workshop_test_eval.json")
    if data is None or "aggregated" not in data:
        print(f"[rb-depth-story] skip: missing {dataset}_workshop_test_eval.json "
              f"or it predates the 'aggregated' section")
        return
    fc = data["aggregated"]["forecast"][str(gap)]["pooled"]
    qs = [q for q in fc if "jepa" in fc[q] and "mae" in fc[q]]
    n_layers = len(fc[qs[0]]["jepa"]["mean_valid_r2_curve"])
    x = np.arange(n_layers)

    jepa_all = np.array([fc[q]["jepa"]["mean_valid_r2_curve"] for q in qs])
    mae_all = np.array([fc[q]["mae"]["mean_valid_r2_curve"] for q in qs])
    jepa_avg, mae_avg = jepa_all.mean(0), mae_all.mean(0)

    jepa_hl = np.array(fc[highlight_quantity]["jepa"]["mean_valid_r2_curve"])
    mae_hl = np.array(fc[highlight_quantity]["mae"]["mean_valid_r2_curve"])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharex=True)

    def panel(ax, jepa_y, mae_y, title):
        ax.fill_between(x, jepa_y, mae_y, where=jepa_y >= mae_y, color=COLORS["jepa"], alpha=0.12, zorder=0)
        ax.plot(x, jepa_y, color=COLORS["jepa"], lw=2.6, marker="o", ms=5, label="JEPA", zorder=2)
        ax.plot(x, mae_y, color=COLORS["mae"], lw=2.6, marker="o", ms=5, label="MAE", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels(n_layers), fontsize=8.5, rotation=90)
        ax.set_xlabel("Encoder layer", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color="0.9", lw=0.6, zorder=-1)

    panel(axes[0], jepa_avg, mae_avg, f"All {len(qs)} quantities (mean), t+{gap}")
    panel(axes[1], jepa_hl, mae_hl, f"{_nice_quantity(highlight_quantity)}, t+{gap}")
    axes[0].set_ylabel(r"$R^2$", fontsize=12)
    axes[0].legend(fontsize=10, loc="lower right")
    fig.suptitle(f"{_nice_dataset(dataset)}: JEPA's forecast advantage vs. encoder depth", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig3_rb_depth_story.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_rb_depth_story.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[rb-depth-story] wrote fig3_rb_depth_story.{pdf,png}")


def plot_rb_noise_story(sweep_dir: Path, out_dir: Path, dataset: str = "rayleigh_benard"):
    """Small-multiples noise-robustness plot, RB only: one subplot per
    quantity (JEPA vs MAE, Pearson r vs. noise std, +/- encoder-seed std
    band), rather than a single averaged curve -- the per-quantity texture
    is the actual finding here (JEPA's robustness edge is large on the
    differential quantities at low-to-moderate noise, but compresses or
    reverses at extreme noise on a few others; averaging would wash that
    out). Uses Pearson r, not R^2 -- R^2 goes arbitrarily negative under the
    clean-fit protocol's distribution shift and isn't comparable once deeply
    negative, while Pearson r stays interpretable throughout."""
    data = _load(sweep_dir / f"{dataset}_workshop_test_eval.json")
    if data is None or "aggregated" not in data or "noise_robustness" not in data["aggregated"]:
        print(f"[rb-noise-story] skip: missing {dataset}_workshop_test_eval.json "
              f"or its 'aggregated.noise_robustness' section")
        return
    nr = data["aggregated"]["noise_robustness"]
    noise_stds = data["noise_stds"]
    qs = sorted(q for q in nr if "jepa" in nr[q] and "mae" in nr[q])

    n_cols = 4
    n_rows = -(-len(qs) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.7 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, q in zip(axes, qs):
        for obj in OBJECTIVES:
            mean = np.array([nr[q][obj][str(s)]["ridge_pearson_r_mean"] for s in noise_stds])
            std = np.array([nr[q][obj][str(s)]["ridge_pearson_r_std"] for s in noise_stds])
            ax.plot(noise_stds, mean, color=COLORS[obj], lw=2.2, marker="o", ms=4, label=obj.upper(), zorder=2)
            ax.fill_between(noise_stds, mean - std, mean + std, color=COLORS[obj], alpha=0.15, zorder=1)
        ax.set_title(_nice_quantity(q), fontsize=11)
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xticks(noise_stds)
        ax.set_xticklabels([str(s) for s in noise_stds], fontsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
    for ax in axes[len(qs):]:
        ax.axis("off")

    axes[0].set_ylim(-0.05, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, loc="lower right",
               bbox_to_anchor=(0.98, 0.02) if len(qs) < len(axes) else (0.5, -0.02),
               ncol=1 if len(qs) < len(axes) else 2)
    for row in range(n_rows):
        axes[row * n_cols].set_ylabel("Pearson $r$", fontsize=10)
    fig.text(0.5, 0.0, r"Input noise $\sigma$ (z-scored units)", ha="center", fontsize=11)
    fig.suptitle(f"{_nice_dataset(dataset)}: noise robustness, clean-fit protocol (ridge, mean $\\pm$ encoder-seed std)", fontsize=12.5)
    fig.tight_layout(rect=(0.01, 0.03, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig4_rb_noise_story.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_rb_noise_story.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[rb-noise-story] wrote fig4_rb_noise_story.{pdf,png}")


def _depth_or_noise_panel(ax, x_vals, jepa_curves, mae_curves, xlabel, xscale=None):
    """jepa_curves/mae_curves: dict[quantity] -> list[R^2] aligned to x_vals.
    Plots only the across-variable mean curve for each objective (no
    per-quantity background lines)."""
    jepa_mean = np.nanmean(np.array(list(jepa_curves.values())), axis=0)
    mae_mean = np.nanmean(np.array(list(mae_curves.values())), axis=0)
    ax.plot(x_vals, jepa_mean, color=COLORS["jepa"], lw=3.0, marker="o", ms=6, label="JEPA")
    ax.plot(x_vals, mae_mean, color=COLORS["mae"], lw=3.0, marker="o", ms=6, label="MAE")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    if xscale:
        ax.set_xscale(xscale)


def plot_depth_curves(sweep_dir: Path, out_dir: Path, datasets: list[str]):
    panels = []
    for dataset in datasets:
        data = _load(sweep_dir / f"{dataset}_workshop_test_eval.json")
        if data is None or "aggregated" not in data:
            print(f"[depth-curves] skip {dataset}: no {dataset}_workshop_test_eval.json "
                  f"or it predates the 'aggregated' section")
            continue
        pooled_c = data["aggregated"]["contemporaneous"]["pooled"]
        quantities = sorted(q for q in pooled_c if "jepa" in pooled_c[q] and "mae" in pooled_c[q])
        if not quantities:
            print(f"[depth-curves] skip {dataset}: missing jepa/mae entries")
            continue
        n_layers = len(pooled_c[quantities[0]]["jepa"]["mean_valid_r2_curve"])
        jepa_curves = {q: pooled_c[q]["jepa"]["mean_valid_r2_curve"] for q in quantities}
        mae_curves = {q: pooled_c[q]["mae"]["mean_valid_r2_curve"] for q in quantities}
        panels.append((dataset, n_layers, jepa_curves, mae_curves))

    if not panels:
        print("[depth-curves] no panels to plot")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (dataset, n_layers, jepa_curves, mae_curves) in zip(axes, panels):
        labels = layer_labels(n_layers)
        _depth_or_noise_panel(ax, range(n_layers), jepa_curves, mae_curves, "Encoder depth")
        ax.set_xticks(range(0, n_layers, max(1, n_layers // 6)))
        ax.set_xticklabels([labels[j] for j in range(0, n_layers, max(1, n_layers // 6))], fontsize=11)
        ax.set_title(_nice_dataset(dataset), fontsize=15)
    axes[0].set_ylabel(r"$R^2$ (mean over variables)", fontsize=14)
    axes[0].set_ylim(-0.05, 1.02)
    axes[0].legend(fontsize=12, loc="lower right")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig3_depth_curves.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_depth_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[depth-curves] wrote fig3_depth_curves.{{pdf,png}} ({len(panels)} panels)")


def plot_noise_curves(sweep_dir: Path, out_dir: Path, datasets: list[str]):
    panels = []
    frozen_records = []
    for dataset in datasets:
        data = _load(sweep_dir / f"{dataset}_workshop_test_eval.json")
        if data is None or "aggregated" not in data:
            print(f"[noise-curves] skip {dataset}: no {dataset}_workshop_test_eval.json "
                  f"or it predates the 'aggregated' section")
            continue
        nr = data["aggregated"].get("noise_robustness")
        if nr is None:
            print(f"[noise-curves] skip {dataset}: no noise_robustness block")
            continue
        noise_stds = data["noise_stds"]
        quantities = sorted(q for q in nr if "jepa" in nr[q] and "mae" in nr[q])
        # Two schemas coexist across datasets: rayleigh_benard was regenerated
        # this session with the clean-fit/no-refit ridge+MLP noise protocol
        # (per-std dict has "ridge_r2_mean"; frozen_layer lives in the raw
        # per-checkpoint section, same layer reused across every noise level).
        # active_matter/shear_flow still carry the older matched-noise
        # protocol (per-std dict has "mean" and "frozen_layers" directly).
        std0 = str(noise_stds[0])
        new_schema = quantities and "ridge_r2_mean" in nr[quantities[0]]["jepa"][std0]
        nr_raw = data.get("noise_robustness", {})
        for q in quantities:
            for obj in ("jepa", "mae"):
                if new_schema:
                    layers = [nr_raw[ckpt][q]["frozen_layer"] for ckpt in nr_raw if obj in ckpt and q in nr_raw[ckpt]]
                else:
                    layers = nr[q][obj][std0]["frozen_layers"]
                frozen_records.append((dataset, q, obj, "pooled", layers))

        value_key = "ridge_r2_mean" if new_schema else "mean"
        jepa_curves = {q: [nr[q]["jepa"][str(s)][value_key] for s in noise_stds] for q in quantities}
        mae_curves = {q: [nr[q]["mae"][str(s)][value_key] for s in noise_stds] for q in quantities}
        panels.append((dataset, noise_stds, jepa_curves, mae_curves))

    _write_frozen_layers(out_dir, frozen_records, filename="frozen_layers_noise.txt")
    if not panels:
        print("[noise-curves] no panels to plot")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (dataset, noise_stds, jepa_curves, mae_curves) in zip(axes, panels):
        _depth_or_noise_panel(ax, noise_stds, jepa_curves, mae_curves, r"Input noise $\sigma$ (z-scored units)", xscale="symlog")
        ax.set_xticks(noise_stds)
        ax.set_xticklabels([str(s) for s in noise_stds], fontsize=11)
        ax.set_title(_nice_dataset(dataset), fontsize=15)
    axes[0].set_ylabel(r"$R^2$ (mean over variables)", fontsize=14)
    axes[0].set_ylim(-0.05, 1.02)
    axes[0].legend(fontsize=12, loc="lower left")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig4_noise_curves.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_noise_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[noise-curves] wrote fig4_noise_curves.{{pdf,png}} ({len(panels)} panels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="sweep_results")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--datasets", nargs="+", default=["active_matter", "shear_flow", "rayleigh_benard"])
    args = ap.parse_args()

    sweep_dir, out_dir = Path(args.sweep_dir), Path(args.out_dir)
    plot_variable_time_heatmap(sweep_dir, out_dir, args.datasets)
    plot_depth_curves(sweep_dir, out_dir, args.datasets)
    plot_noise_curves(sweep_dir, out_dir, args.datasets)
    if "rayleigh_benard" in args.datasets:
        plot_rb_detailed_heatmap(sweep_dir, out_dir)
        plot_rb_depth_story(sweep_dir, out_dir)
        plot_rb_noise_story(sweep_dir, out_dir)


if __name__ == "__main__":
    main()

"""Figure 2 redesign: physical-probing heatmaps, Rayleigh-Bénard.

Single merged grid: one shared row-label column (physical quantities:
local/differential, then coupled/integrated, then the token-only targets
divergence and signed vorticity at the bottom), with two column blocks --
Pooled and Token -- each spanning t+0, t+8, t+32. Token-only rows have no
pooled target by construction (spatial mean ~0), shown as an em-dash in the
pooled block.

Per cell:
  background shading = Delta R^2 = R^2_{Ray-JEPA} - R^2_{Ray-vMAE}  (the
    scientific comparison the paper makes -- this is what the eye scans)
  text annotation    = the two absolute scores, "Ray-JEPA / Ray-vMAE"

A one-line key above the grid explains both encodings; a compact vertical
colorbar sits directly to the right. Cells where BOTH objectives fall below
LOW_SIGNAL_THRESHOLD are greyed out (the delta between two near-noise
numbers is not a meaningful comparison); cells whose |delta| does not clear
the combined nested SE of the two means are shaded at reduced opacity
(absolute scores stay fully legible either way).

All numbers come from the "aggregated" section: across-encoder-seed means
at each cell's own held-out-valid-selected frozen layer, evaluated once on
the held-out test split.

Usage: uv run python paper-plots/plot_fig2_probe_heatmaps.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from style import (COUPLED_ROWS, GAPS, LABEL, LOCAL_ROWS, SWEEP_JSON,
                   TOKEN_ONLY_ROWS, apply_rcparams, nice, save)

DELTA_CLIP = 0.2          # color range for Delta R^2; text still shows true values
LOW_SIGNAL_THRESHOLD = 0.15
GREY = "#cdcdcd"
COL_LABELS = ["$t{+}0$", "$t{+}8$", "$t{+}32$"]


def build_grids(agg_family_by_col, rows):
    """agg_family_by_col: list of 3 aggregated {quantity: {obj: stats}} dicts,
    one per column (t+0, t+8, t+32). Returns (jepa, mae, se) grids; rows with
    no entry in a source stay NaN."""
    jepa = np.full((len(rows), 3), np.nan)
    mae = np.full((len(rows), 3), np.nan)
    se = np.full((len(rows), 3), np.nan)
    for i, q in enumerate(rows):
        for j, src in enumerate(agg_family_by_col):
            ej, em = src.get(q, {}).get("jepa"), src.get(q, {}).get("mae")
            if ej is None or em is None:
                continue
            jepa[i, j], mae[i, j] = ej["mean"], em["mean"]
            se[i, j] = np.sqrt(ej["nested_se"] ** 2 + em["nested_se"] ** 2)
    return jepa, mae, se


def fmt(v: float) -> str:
    # Pathological long-horizon MLP R^2 can be deeply negative; keep it honest
    # but compact.
    if v <= -9.95:
        return f"{v:.0f}"
    return f"{v:.2f}"


def draw_block(ax, rows, jepa, mae, se, dividers, title, show_row_labels):
    delta = jepa - mae
    both_low = (jepa < LOW_SIGNAL_THRESHOLD) & (mae < LOW_SIGNAL_THRESHOLD)
    not_sig = ~np.isnan(se) & (np.abs(delta) < se)
    alpha = np.where(not_sig, 0.35, 1.0)
    ax.imshow(np.clip(delta, -DELTA_CLIP, DELTA_CLIP), aspect="auto",
              cmap="RdBu", vmin=-DELTA_CLIP, vmax=DELTA_CLIP, alpha=alpha)
    for i in range(len(rows)):
        for j in range(3):
            if np.isnan(delta[i, j]):
                # Structurally-absent target (token-only row, pooled block).
                ax.text(j, i, "—", ha="center", va="center", fontsize=9,
                        color="0.65", zorder=2)
                continue
            if both_low[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=GREY, edgecolor="none", zorder=1))
                color = "0.45"
            else:
                strong = (not not_sig[i, j]) and abs(delta[i, j]) > 0.75 * DELTA_CLIP
                color = "white" if strong else "black"
            ax.text(j, i, f"{fmt(jepa[i, j])} / {fmt(mae[i, j])}",
                    ha="center", va="center", fontsize=8.5, color=color, zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(COL_LABELS, fontsize=11)
    ax.xaxis.tick_top()
    if show_row_labels:
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([nice(r) for r in rows], fontsize=10.5)
    else:
        ax.set_yticks([])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for boundary in dividers:
        ax.axhline(boundary - 0.5, color="0.25", lw=1.1)
    ax.set_title(title, fontsize=12, pad=30)


def main():
    apply_rcparams()
    data = json.loads(SWEEP_JSON.read_text())
    agg = data["aggregated"]

    pooled_cols = [agg["contemporaneous"]["pooled"]] + \
        [agg["forecast"][str(g)]["pooled"] for g in GAPS]
    token_cols = [agg["contemporaneous"]["token"]] + \
        [agg["forecast"][str(g)]["token"] for g in GAPS]

    # One shared row set: token-only rows appear in both blocks, but stay NaN
    # (rendered as an em-dash) in the pooled one.
    rows = LOCAL_ROWS + COUPLED_ROWS + TOKEN_ONLY_ROWS
    dividers = (len(LOCAL_ROWS), len(LOCAL_ROWS) + len(COUPLED_ROWS))

    p_jepa, p_mae, p_se = build_grids(pooled_cols, rows)
    t_jepa, t_mae, t_se = build_grids(token_cols, rows)

    # Explicit inch-based layout: label column, two adjacent 3-column blocks,
    # vertical group labels on the right, horizontal colorbar below.
    CELL_W, CELL_H = 0.98, 0.40
    LABEL_W, BLOCK_GAP, GROUP_W = 1.85, 0.22, 0.45
    TOP, BOTTOM = 1.05, 0.75  # key line + block titles / colorbar strip
    grid_w = 3 * CELL_W
    grid_h = len(rows) * CELL_H
    fig_w = LABEL_W + grid_w + BLOCK_GAP + grid_w + GROUP_W
    fig_h = TOP + grid_h + BOTTOM

    fig = plt.figure(figsize=(fig_w, fig_h))
    y0 = BOTTOM / fig_h
    ax_p = fig.add_axes((LABEL_W / fig_w, y0, grid_w / fig_w, grid_h / fig_h))
    ax_t = fig.add_axes(((LABEL_W + grid_w + BLOCK_GAP) / fig_w, y0,
                         grid_w / fig_w, grid_h / fig_h))

    draw_block(ax_p, rows, p_jepa, p_mae, p_se, dividers, "Pooled", show_row_labels=True)
    draw_block(ax_t, rows, t_jepa, t_mae, t_se, dividers, "Token", show_row_labels=False)

    # Vertical group labels along the right edge, one per row family.
    group_spans = [
        ("local /\ndifferential", 0, len(LOCAL_ROWS)),
        ("coupled /\nintegrated", len(LOCAL_ROWS), len(LOCAL_ROWS) + len(COUPLED_ROWS)),
        ("token-\nonly", len(LOCAL_ROWS) + len(COUPLED_ROWS), len(rows)),
    ]
    for text, lo, hi in group_spans:
        ax_t.text(2.72, (lo + hi - 1) / 2, text, rotation=270, ha="left",
                  va="center", fontsize=9.5, color="0.3", linespacing=1.1,
                  clip_on=False, multialignment="center")

    # One-line key, centered over the grid.
    fig.text((LABEL_W + grid_w + BLOCK_GAP / 2) / fig_w, (fig_h - 0.18) / fig_h,
             rf"values: {LABEL['jepa']} / {LABEL['mae']} $R^2$   $\cdot$   "
             rf"shading: $\Delta R^2$ ({LABEL['jepa']} $-$ {LABEL['mae']})",
             ha="center", va="center", fontsize=10.5)

    # Horizontal colorbar centered under the two blocks.
    cbar_w = 2.2
    cax = fig.add_axes(((LABEL_W + grid_w + BLOCK_GAP / 2 - cbar_w / 2) / fig_w,
                        0.34 / fig_h, cbar_w / fig_w, 0.15 / fig_h))
    sm = plt.cm.ScalarMappable(cmap="RdBu",
                               norm=plt.Normalize(-DELTA_CLIP, DELTA_CLIP))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$\Delta R^2$", fontsize=10.5)
    cbar.set_ticks([-DELTA_CLIP, 0, DELTA_CLIP])
    cbar.ax.set_xticklabels([rf"$\leq -{DELTA_CLIP}$", "0", rf"$\geq {DELTA_CLIP}$"])
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)

    save(fig, "fig2_probe_heatmaps")


if __name__ == "__main__":
    main()

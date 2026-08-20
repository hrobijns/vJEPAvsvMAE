"""Figure 3 redesign: where does the objective-induced difference emerge?

Main panel: mean R^2 across the 7 pooled physical quantities vs encoder
depth at t+32 (the forecast family, where the depth story lives), JEPA vs
MAE, with SEM bands across the 3 encoder seeds. Below it, a small aligned
panel showing Delta R^2 = R^2_JEPA - R^2_MAE vs layer directly, so the
reader does not have to infer the gap from the distance between curves:
early layers ~similar, deeper layers -> growing JEPA advantage.

Uncertainty: each encoder seed contributes one curve (its valid_r2_curve
averaged over quantities); bands are +/- SEM across seeds. The delta band
combines the two SEMs in quadrature (JEPA and MAE seeds are independent
runs, not pairs).

Curves are the held-out-valid layer sweep computed as a byproduct of the
frozen-layer selection (genuinely held-out; full curve, so no argmax
selection bias).

Also writes fig3_appendix_depth_per_quantity: the same two-line plot for
each pooled quantity separately (small multiples, shared axes).

Usage: uv run python paper-plots/plot_fig3_depth.py [--gap 32]
"""

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from style import (COLORS, LABEL, OBJECTIVES, SWEEP_JSON, apply_rcparams,
                   layer_labels, nice, save)


def per_seed_curves(data, gap: int):
    """Returns {obj: array (n_seeds, n_quantities, n_layers)} of held-out-valid
    R^2 layer curves, plus the quantity list."""
    raw = data["forecast"][str(gap)]["pooled"] if gap else data["contemporaneous"]["pooled"]
    out = {}
    quantities = None
    for obj in OBJECTIVES:
        ckpts = sorted(k for k in raw if f"_{obj}_" in k)
        qs = sorted(raw[ckpts[0]].keys())
        quantities = qs if quantities is None else quantities
        out[obj] = np.array([[raw[c][q]["valid_r2_curve"] for q in qs] for c in ckpts])
    return out, quantities


def mean_sem(curves_2d):
    """curves_2d: (n_seeds, n_layers). SEM = std/sqrt(n) across seeds."""
    m = curves_2d.mean(0)
    sem = curves_2d.std(0, ddof=1) / np.sqrt(curves_2d.shape[0])
    return m, sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=32, help="forecast gap (0 = contemporaneous)")
    args = ap.parse_args()

    apply_rcparams()
    data = json.loads(SWEEP_JSON.read_text())
    curves, quantities = per_seed_curves(data, args.gap)
    n_layers = curves["jepa"].shape[-1]
    x = np.arange(n_layers)
    labels = layer_labels(n_layers)
    tag = f"$t{{+}}{args.gap}$" if args.gap else "$t{+}0$"

    # ---- main figure: mean-over-quantities + Delta R^2 strip ----
    stats = {obj: mean_sem(curves[obj].mean(axis=1)) for obj in OBJECTIVES}

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(5.4, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.12})

    for obj in OBJECTIVES:  # fills first so no band obscures a line
        m, sem = stats[obj]
        ax.fill_between(x, m - sem, m + sem, alpha=0.25, color=COLORS[obj], zorder=1)
    for obj in OBJECTIVES:
        m, _ = stats[obj]
        ax.plot(x, m, color=COLORS[obj], lw=1.8, marker="o", markersize=4,
                label=LABEL[obj], zorder=2)
    ax.set_ylabel(r"$R^2$ (mean over quantities)")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, 0.94, tag, transform=ax.transAxes, fontsize=11, color="0.3",
            ha="left", va="top")

    (mj, sj), (mm, sm) = stats["jepa"], stats["mae"]
    delta = mj - mm
    dsem = np.sqrt(sj ** 2 + sm ** 2)
    axd.axhline(0, color="0.6", lw=0.8, zorder=1)
    axd.fill_between(x, delta - dsem, delta + dsem, alpha=0.25, color="0.4", zorder=1)
    axd.plot(x, delta, color="0.15", lw=1.8, marker="o", markersize=4, zorder=2)
    axd.set_ylabel(r"$\Delta R^2$")
    axd.set_xlabel("Encoder layer")
    axd.set_xticks(x)
    axd.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    axd.yaxis.set_major_locator(ticker.MaxNLocator(3))

    save(fig, f"fig3_depth_gap{args.gap}")

    # ---- appendix: per-quantity small multiples ----
    n_cols = 4
    n_rows = -(-len(quantities) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.5 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for k, q in enumerate(quantities):
        axq = axes[k]
        for obj in OBJECTIVES:
            m, sem = mean_sem(curves[obj][:, k, :])
            axq.fill_between(x, m - sem, m + sem, alpha=0.25, color=COLORS[obj], zorder=1)
        for obj in OBJECTIVES:
            m, _ = mean_sem(curves[obj][:, k, :])
            axq.plot(x, m, color=COLORS[obj], lw=1.5, marker="o", markersize=3,
                     label=LABEL[obj], zorder=2)
        axq.set_title(nice(q), fontsize=11)
        axq.set_xticks(x[::3])
        axq.set_xticklabels([labels[j] for j in range(0, n_layers, 3)], fontsize=9)
    for axq in axes[len(quantities):]:
        axq.axis("off")
    for r in range(n_rows):
        axes[r * n_cols].set_ylabel(r"$R^2$", fontsize=11)
    for k in range(len(quantities)):
        if k >= len(quantities) - n_cols:
            axes[k].set_xlabel("Encoder layer", fontsize=11)
    axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    save(fig, f"fig3_appendix_depth_per_quantity_gap{args.gap}")


if __name__ == "__main__":
    main()

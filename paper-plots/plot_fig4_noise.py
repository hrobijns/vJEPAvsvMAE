"""Figure 4 redesign: how robust is the representation to observational noise?

Two panels sharing an x-axis (linear -- no symlog exaggeration of the
low-noise region):

  (a) absolute performance: mean ridge Pearson r across the 7 pooled
      quantities vs injected Gaussian noise std, JEPA vs MAE, SEM band
      across the 3 encoder seeds.
  (b) degradation from clean: r(sigma) - r(0), each seed's curve baselined
      against its own clean value before averaging. This makes the
      robustness claim direct: even if the two objectives started at
      slightly different clean performance, (b) shows who loses more.

Metric is Pearson r, not R^2: under the clean-fit protocol (probe fitted on
clean features, evaluated on noisy features) R^2 goes arbitrarily negative
at high noise (seed means below -100 in the raw JSON) and is not comparable
once deeply negative, while Pearson r stays interpretable throughout.
Ridge (deterministic) rather than the MLP head, so the only variance shown
is encoder-seed variance. Each quantity's frozen layer is fixed from its
own contemporaneous selection and reused across all noise levels.

Also writes fig4_appendix_noise_per_quantity: per-quantity small multiples
(the per-quantity texture -- where the JEPA edge is large vs where it
compresses -- for the appendix).

Usage: uv run python paper-plots/plot_fig4_noise.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from style import (COLORS, LABEL, OBJECTIVES, SWEEP_JSON, apply_rcparams,
                   nice, save)


def per_seed_curves(data):
    """Returns {obj: array (n_seeds, n_quantities, n_stds)} of ridge Pearson r,
    plus (quantities, noise_stds)."""
    raw = data["noise_robustness"]
    stds = data["noise_stds"]
    out, quantities = {}, None
    for obj in OBJECTIVES:
        ckpts = sorted(k for k in raw if f"_{obj}_" in k)
        qs = sorted(raw[ckpts[0]].keys())
        quantities = qs if quantities is None else quantities
        out[obj] = np.array([[[raw[c][q]["by_noise"][str(s)]["ridge_pearson_r"]
                               for s in stds] for q in qs] for c in ckpts])
    return out, quantities, stds


def mean_sem(curves_2d):
    m = curves_2d.mean(0)
    sem = curves_2d.std(0, ddof=1) / np.sqrt(curves_2d.shape[0])
    return m, sem


def main():
    apply_rcparams()
    data = json.loads(SWEEP_JSON.read_text())
    curves, quantities, stds = per_seed_curves(data)
    x = np.array(stds)

    # ---- main figure: (a) absolute, (b) change from clean ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    seed_means = {obj: curves[obj].mean(axis=1) for obj in OBJECTIVES}  # (seeds, stds)

    for obj in OBJECTIVES:
        m, sem = mean_sem(seed_means[obj])
        ax_a.fill_between(x, m - sem, m + sem, alpha=0.25, color=COLORS[obj], zorder=1)
    for obj in OBJECTIVES:
        m, _ = mean_sem(seed_means[obj])
        ax_a.plot(x, m, color=COLORS[obj], lw=1.8, marker="o", markersize=4,
                  label=LABEL[obj], zorder=2)
    ax_a.set_xlabel(r"Input noise $\sigma$ (z-scored units)")
    ax_a.set_ylabel(r"Pearson $r$ (mean over quantities)")
    ax_a.set_title("(a) Absolute performance", fontsize=12, loc="left")
    ax_a.legend(frameon=False, loc="lower left")

    retained = {obj: seed_means[obj] - seed_means[obj][:, :1] for obj in OBJECTIVES}
    for obj in OBJECTIVES:
        m, sem = mean_sem(retained[obj])
        ax_b.fill_between(x, m - sem, m + sem, alpha=0.25, color=COLORS[obj], zorder=1)
    for obj in OBJECTIVES:
        m, _ = mean_sem(retained[obj])
        ax_b.plot(x, m, color=COLORS[obj], lw=1.8, marker="o", markersize=4,
                  label=LABEL[obj], zorder=2)
    ax_b.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax_b.set_xlabel(r"Input noise $\sigma$ (z-scored units)")
    ax_b.set_ylabel(r"$r(\sigma) - r(0)$")
    ax_b.set_title("(b) Degradation from clean", fontsize=12, loc="left")

    fig.tight_layout()
    save(fig, "fig4_noise")

    # ---- appendix: per-quantity small multiples (absolute r) ----
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
    for axq in axes[len(quantities):]:
        axq.axis("off")
    for r in range(n_rows):
        axes[r * n_cols].set_ylabel(r"Pearson $r$", fontsize=11)
    for k in range(len(quantities)):
        if k >= len(quantities) - n_cols:
            axes[k].set_xlabel(r"Noise $\sigma$", fontsize=11)
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_appendix_noise_per_quantity")


if __name__ == "__main__":
    main()

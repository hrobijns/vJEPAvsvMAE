"""Render the JEPA-vs-MAE training loss curve figure from sweep_results/training_history.csv.

Regenerate the source CSV with scripts/extract_training_history.py (requires
the local pod_logs/wandb/ offline-run logs, which aren't committed).
"""

import csv
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.7,
        "axes.linewidth": 0.8,
        "font.size": 10,
    }
)

BLUE = "#2a78d6"  # JEPA
ORANGE = "#eb6834"  # MAE

IN_CSV = "sweep_results/training_history.csv"
OUT_PNG = "docs/figures/training_curves.png"
OUT_PDF = "docs/figures/training_curves.pdf"

DATASETS = [
    ("active_matter", "Active Matter"),
    ("shear_flow", "Shear Flow"),
    ("rayleigh_benard", "Rayleigh–Bénard"),
]


def load_history(path):
    history = defaultdict(lambda: {"step": [], "loss": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["method"])
            history[key]["step"].append(int(row["step"]))
            history[key]["loss"].append(float(row["loss"]))
    return history


def smooth(y, k=9):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")[: len(y)]


def main():
    history = load_history(IN_CSV)

    fig, axes = plt.subplots(3, 2, figsize=(8.5, 10.2), sharex=True)

    for row, (ds_key, ds_label) in enumerate(DATASETS):
        for col, (method, color, title) in enumerate(
            [("jepa", BLUE, "V-JEPA"), ("mae", ORANGE, "VideoMAE")]
        ):
            ax = axes[row, col]
            steps = history[(ds_key, method)]["step"]
            loss = history[(ds_key, method)]["loss"]
            loss_smooth = smooth(loss)

            ax.plot(steps, loss, color=color, alpha=0.22, linewidth=1.0)
            ax.plot(steps, loss_smooth, color=color, linewidth=2.0, solid_capstyle="round")

            ax.set_yscale("log")
            ax.set_xlim(0, 100_000)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#c3c2b7")
            ax.spines["bottom"].set_color("#c3c2b7")

            if row == 0:
                ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
            if row == 2:
                ax.set_xlabel("Training step")
                ax.xaxis.set_major_formatter(
                    lambda x, pos: f"{int(x / 1000)}k" if x > 0 else "0"
                )
            if col == 0:
                ax.text(
                    -0.30,
                    0.5,
                    ds_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                    color="#0b0b0b",
                )
            ax.set_ylabel("loss (log scale)", fontsize=8.5, color="#52514e")

    fig.tight_layout(rect=[0.03, 0, 1, 0.90])
    fig.suptitle(
        "Training loss curves: V-JEPA vs. VideoMAE encoders on The Well",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.955,
        "Identical ViT-S encoder, 100k steps, per dataset. Thin line = raw logged value; bold = smoothed trend.",
        ha="center",
        fontsize=9,
        color="#52514e",
    )

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    print("wrote", OUT_PNG)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()

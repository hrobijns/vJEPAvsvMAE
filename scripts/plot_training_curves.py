"""Render training-curve figures for the JEPA-vs-MAE encoder runs.

Two data sources:
  - `runs/*/history.jsonl`, written directly by src/train.py (train + val
    rows, including feature-std collapse diagnostics) -- the source for any
    run trained after the val-loop rigor pass (see plot_run_diagnostics()).
    No wandb dependency; always present locally for a run that's happened.
  - `sweep_results/training_history.csv`, scraped from wandb offline logs by
    scripts/extract_training_history.py -- loss/lr/grad_norm only, for the
    original 6 pre-val-loop runs (see main(), unchanged).

Usage:
    uv run python scripts/plot_training_curves.py                # original 6-run CSV figure
    uv run python scripts/plot_training_curves.py --runs "runs/*_seed*"  # jsonl multi-seed figures
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Multi-seed loss + collapse-diagnostic figures from runs/*/history.jsonl
# ---------------------------------------------------------------------------

RUN_NAME_RE = re.compile(r"^(?P<dataset>active_matter|shear_flow|rayleigh_benard|debug)_(?P<objective>jepa|mae)_seed(?P<seed>\d+)$")

FEAT_STD_KEYS = ["context_feat_std", "target_feat_std", "pred_feat_std"]


def load_history_jsonl(path: Path) -> dict:
    """phase -> metric -> list of (step, value)."""
    out = {"train": defaultdict(list), "val": defaultdict(list)}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        phase = row["phase"]
        step = row["step"]
        for k, v in row.items():
            if k in ("step", "phase"):
                continue
            out[phase][k].append((step, v))
    return out


def discover_runs(pattern: str) -> dict:
    """{(dataset, objective, seed): history dict} for every runs/*/history.jsonl matching pattern."""
    runs = {}
    for run_dir in sorted(glob.glob(pattern)):
        run_dir = Path(run_dir)
        m = RUN_NAME_RE.match(run_dir.name)
        history_path = run_dir / "history.jsonl"
        if not m or not history_path.exists():
            continue
        key = (m["dataset"], m["objective"], int(m["seed"]))
        runs[key] = load_history_jsonl(history_path)
    return runs


def plot_run_diagnostics(runs: dict, out_dir: str = "docs/figures"):
    """One loss figure + one collapse-diagnostic figure per (dataset, objective)
    pair, overlaying all available seeds. Evidence against trivial
    representational collapse, not proof of representational health more
    broadly -- see the plan's methodology note on collapse language."""
    pairs = sorted({(ds, obj) for ds, obj, _ in runs})
    seed_colors = {1: "#2a78d6", 2: "#eb6834", 3: "#3aa66b"}
    os.makedirs(out_dir, exist_ok=True)

    for ds, obj in pairs:
        seeds = sorted(s for d, o, s in runs if d == ds and o == obj)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for seed in seeds:
            h = runs[(ds, obj, seed)]
            color = seed_colors.get(seed, "#666")
            if h["train"]["loss"]:
                steps, vals = zip(*h["train"]["loss"])
                ax.plot(steps, vals, color=color, alpha=0.25, linewidth=1.0)
            if h["val"]["loss"]:
                steps, vals = zip(*h["val"]["loss"])
                ax.plot(steps, vals, color=color, linewidth=2.0, marker="o", markersize=3,
                         label=f"seed {seed}")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss (log scale)")
        ax.set_title(f"{ds} / {obj} — train (thin) + val (bold, markers) loss")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{ds}_{obj}_loss.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        any_data = False
        for seed in seeds:
            h = runs[(ds, obj, seed)]
            color = seed_colors.get(seed, "#666")
            for i, key in enumerate(FEAT_STD_KEYS):
                if not h["val"][key]:
                    continue
                any_data = True
                steps, vals = zip(*h["val"][key])
                ax.plot(steps, vals, color=color, linestyle=["-", "--", ":"][i % 3],
                         marker="o", markersize=3, label=f"seed {seed} {key}")
        if any_data:
            ax.set_xlabel("step")
            ax.set_ylabel("val feature std (per-dim mean)")
            ax.set_title(f"{ds} / {obj} — representation-collapse diagnostic (val)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(f"{out_dir}/{ds}_{obj}_feat_std.png", dpi=180)
        plt.close(fig)

        print(f"wrote {out_dir}/{ds}_{obj}_loss.png (+ _feat_std.png if diagnostics present), seeds={seeds}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None,
                     help="glob for run dirs, e.g. 'runs/*_seed*' — reads history.jsonl from each; "
                          "omit to render the original 6-run CSV figure instead")
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    if args.runs:
        runs = discover_runs(args.runs)
        if not runs:
            raise SystemExit(f"no runs/*/history.jsonl matched {args.runs!r}")
        plot_run_diagnostics(runs, args.out_dir)
    else:
        main()

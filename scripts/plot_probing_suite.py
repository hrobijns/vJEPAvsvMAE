"""Workshop figures for the probing suite: reads sweep_results/*.json (as
produced by analyze_encoders.py --out, analyze_noise_robustness.py --out,
analyze_encoders_local.py --out, forecast_content_probe.py --out) and
produces the 4 "core" plots from docs/LINEAR_PROBE.md's experiment design:

  1. emergence-depth: per-quantity R^2-vs-layer curves (JEPA vs MAE overlaid),
     plus a summary scatter of each quantity's "emergence layer" (first layer
     reaching 80% of that quantity's own peak R^2) — formalizes the
     qualitative "MAE peaks early / JEPA peaks late" observation into a
     citable number.
  2. noise x layer heatmap: R^2 as a function of (layer, noise std) for each
     quantity, JEPA vs MAE side by side — shows whether robustness is a
     final-layer-only property or holds at intermediate depths too.
  3. token linear-vs-MLP: bar comparison of per-token linear vs per-token MLP
     R^2 at the final layer, per quantity — tests whether any quantity is
     present nonlinearly-but-not-linearly at the token level.
  4. forecast skill vs. difficulty: skill score (persistence-relative, see
     skill_score() in analyze_encoders.py) vs. R^2_persistence per quantity,
     one point per (quantity, gap), JEPA vs MAE — visualizes whether an
     objective's forecast advantage concentrates in genuinely hard-to-forecast
     quantities (low persistence-R^2 = real headroom) or easy/already-static
     ones (high persistence-R^2 = little headroom either way to show skill).

Each plot function degrades gracefully (skips + prints a note) if its input
JSON is missing for a given dataset, so this can be run against a partial
sweep_results/ directory.

Usage:
    uv run python scripts/plot_probing_suite.py \
        --sweep-dir sweep_results --out-dir reports/figures \
        --datasets active_matter shear_flow rayleigh_benard
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBJECTIVES = ("jepa", "mae")
COLORS = {"jepa": "#3b6fd4", "mae": "#d4703b"}


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _ckpt_key(all_results: dict, dataset: str, objective: str) -> str | None:
    """all_results is keyed by checkpoint PATH, not objective name — find the
    key whose path contains the objective string."""
    for k in all_results:
        if objective in Path(k).stem:
            return k
    return None


def layer_labels(n_layers: int) -> list[str]:
    return [f"L{i}" for i in range(n_layers - 1)] + ["norm"]


def emergence_layer(r2_by_layer: list[float], frac: float = 0.8) -> int | None:
    """First layer index where R^2 >= frac * max(R^2 across layers).
    Returns None if the quantity never reaches a positive peak (no real
    signal to speak of an "emergence depth" for)."""
    peak = max(r2_by_layer)
    if peak <= 0:
        return None
    threshold = frac * peak
    for i, v in enumerate(r2_by_layer):
        if v >= threshold:
            return i
    return None


def plot_emergence_depth(sweep_dir: Path, out_dir: Path, datasets: list[str]):
    for dataset in datasets:
        data = _load(sweep_dir / f"{dataset}_pooled.json")
        if data is None:
            print(f"[emergence-depth] skip {dataset}: no {dataset}_pooled.json")
            continue
        jepa_key = _ckpt_key(data, dataset, "jepa")
        mae_key = _ckpt_key(data, dataset, "mae")
        if jepa_key is None or mae_key is None:
            print(f"[emergence-depth] skip {dataset}: missing jepa/mae checkpoint key")
            continue
        jepa_layers, mae_layers = data[jepa_key]["layers"], data[mae_key]["layers"]
        n_layers = len(jepa_layers)
        quantities = list(jepa_layers[0].keys())
        labels = layer_labels(n_layers)

        # (a) small multiples: R^2 vs layer, one panel per quantity
        n_q = len(quantities)
        ncols = 3
        nrows = (n_q + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for i, q in enumerate(quantities):
            ax = axes[i // ncols][i % ncols]
            jepa_r2 = [jepa_layers[l][q] for l in range(n_layers)]
            mae_r2 = [mae_layers[l][q] for l in range(n_layers)]
            ax.plot(range(n_layers), jepa_r2, color=COLORS["jepa"], marker="o", ms=3, label="JEPA")
            ax.plot(range(n_layers), mae_r2, color=COLORS["mae"], marker="o", ms=3, label="MAE")
            ax.set_title(q, fontsize=9)
            ax.set_xticks(range(0, n_layers, max(1, n_layers // 6)))
            ax.set_xticklabels([labels[j] for j in range(0, n_layers, max(1, n_layers // 6))], fontsize=7)
            ax.axhline(0, color="gray", lw=0.5)
        for i in range(n_q, nrows * ncols):
            axes[i // ncols][i % ncols].axis("off")
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"{dataset}: R^2 vs. layer depth", fontsize=12)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{dataset}_emergence_curves.png", dpi=150)
        plt.close(fig)

        # (b) summary: emergence layer per quantity, JEPA vs MAE
        fig, ax = plt.subplots(figsize=(6, 0.4 * n_q + 1.5))
        for i, q in enumerate(quantities):
            jepa_e = emergence_layer([jepa_layers[l][q] for l in range(n_layers)])
            mae_e = emergence_layer([mae_layers[l][q] for l in range(n_layers)])
            if jepa_e is not None:
                ax.scatter(jepa_e, i, color=COLORS["jepa"], zorder=3, label="JEPA" if i == 0 else None)
            if mae_e is not None:
                ax.scatter(mae_e, i, color=COLORS["mae"], zorder=3, label="MAE" if i == 0 else None)
        ax.set_yticks(range(n_q))
        ax.set_yticklabels(quantities, fontsize=8)
        ax.set_xlabel("emergence layer (first layer >= 80% of peak R^2)")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.legend(fontsize=8)
        fig.suptitle(f"{dataset}: where does each quantity become decodable?", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_emergence_summary.png", dpi=150)
        plt.close(fig)
        print(f"[emergence-depth] wrote {dataset}_emergence_curves.png, {dataset}_emergence_summary.png")


def plot_noise_layer_heatmap(sweep_dir: Path, out_dir: Path, datasets: list[str]):
    for dataset in datasets:
        data = _load(sweep_dir / f"{dataset}_noise_robustness.json")
        if data is None:
            print(f"[noise-heatmap] skip {dataset}: no {dataset}_noise_robustness.json")
            continue
        jepa_key = _ckpt_key(data, dataset, "jepa")
        mae_key = _ckpt_key(data, dataset, "mae")
        if jepa_key is None or mae_key is None:
            print(f"[noise-heatmap] skip {dataset}: missing jepa/mae checkpoint key")
            continue
        if "by_noise" not in data[jepa_key]:
            print(f"[noise-heatmap] skip {dataset}: old (final-layer-only) schema, re-run "
                  f"analyze_noise_robustness.py to get the layer sweep")
            continue

        noise_stds = data[jepa_key]["noise_stds"]
        n_layers = data[jepa_key]["n_layers"]
        quantities = list(data[jepa_key]["by_noise"][str(noise_stds[0])][0].keys())

        for q in quantities:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
            for ax, key, objective in ((axes[0], jepa_key, "jepa"), (axes[1], mae_key, "mae")):
                grid = [[data[key]["by_noise"][str(std)][l][q] for l in range(n_layers)] for std in noise_stds]
                im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1, origin="lower")
                ax.set_title(objective.upper())
                ax.set_xlabel("layer")
                ax.set_xticks(range(0, n_layers, max(1, n_layers // 6)))
                ax.set_xticklabels(layer_labels(n_layers)[::max(1, n_layers // 6)], fontsize=7)
                ax.set_yticks(range(len(noise_stds)))
                ax.set_yticklabels([str(s) for s in noise_stds])
            axes[0].set_ylabel("noise std")
            fig.colorbar(im, ax=axes, label="R^2", fraction=0.03)
            fig.suptitle(f"{dataset}: {q} — R^2 vs. (layer, noise std)", fontsize=11)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{dataset}_noise_heatmap_{q}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"[noise-heatmap] wrote {len(quantities)} heatmaps for {dataset}")


def plot_token_linear_vs_mlp(sweep_dir: Path, out_dir: Path, datasets: list[str]):
    for dataset in datasets:
        data = _load(sweep_dir / f"{dataset}_nonpooled.json")
        if data is None:
            print(f"[token-mlp] skip {dataset}: no {dataset}_nonpooled.json")
            continue
        jepa_key = _ckpt_key(data, dataset, "jepa")
        mae_key = _ckpt_key(data, dataset, "mae")
        if jepa_key is None or mae_key is None:
            print(f"[token-mlp] skip {dataset}: missing jepa/mae checkpoint key")
            continue
        if "mlp_token" not in data[jepa_key]:
            print(f"[token-mlp] skip {dataset}: no mlp_token block (re-run without --skip-mlp)")
            continue

        quantities = list(data[jepa_key]["token_linear"]["layers"][-1].keys())
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(quantities)), 4))
        x = range(len(quantities))
        width = 0.2
        for i, (key, objective) in enumerate(((jepa_key, "jepa"), (mae_key, "mae"))):
            lin = [data[key]["token_linear"]["layers"][-1][q] for q in quantities]
            mlp = [data[key]["mlp_token"]["layers"][-1][q] for q in quantities]
            offset = (i - 0.5) * 2 * width
            ax.bar([xi + offset - width / 2 for xi in x], lin, width, color=COLORS[objective], alpha=0.5,
                   label=f"{objective.upper()} linear")
            ax.bar([xi + offset + width / 2 for xi in x], mlp, width, color=COLORS[objective], alpha=1.0,
                   label=f"{objective.upper()} MLP")
        ax.set_xticks(list(x))
        ax.set_xticklabels(quantities, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel("R^2 (final layer, per-token)")
        ax.legend(fontsize=8, ncol=2)
        fig.suptitle(f"{dataset}: per-token linear vs. MLP readout", fontsize=11)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{dataset}_token_linear_vs_mlp.png", dpi=150)
        plt.close(fig)
        print(f"[token-mlp] wrote {dataset}_token_linear_vs_mlp.png")


def plot_forecast_skill_difficulty(sweep_dir: Path, out_dir: Path, datasets: list):
    """4th core plot: skill score vs. persistence-R^2 ("how much does this
    quantity change into the future, i.e. how much headroom is there above
    the naive baseline") per quantity, one point per (quantity, gap), JEPA
    vs. MAE. Uses the MLP skill score (final layer, clean input) since
    LINEAR_PROBE.md's token-MLP-vs-linear plot already established that
    linear-only readouts can undersell what's actually decodable."""
    for dataset in datasets:
        data = _load(sweep_dir / f"{dataset}_forecast.json")
        if data is None:
            print(f"[forecast-skill] skip {dataset}: no {dataset}_forecast.json")
            continue
        jepa_key = _ckpt_key(data, dataset, "jepa")
        mae_key = _ckpt_key(data, dataset, "mae")
        if jepa_key is None or mae_key is None:
            print(f"[forecast-skill] skip {dataset}: missing jepa/mae checkpoint key")
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        for key, objective in ((jepa_key, "jepa"), (mae_key, "mae")):
            xs, ys = [], []
            for gap in data[key]["gaps"]:
                p = data[key]["pooled"][str(gap)]
                persistence = p["naive_persistence"]
                clean_std = str(p["noise_stds"][0])
                final_layer = p["by_noise"][clean_std][-1]
                for q in persistence:
                    xs.append(persistence[q])
                    ys.append(final_layer[q]["skill_mlp"])
            ax.scatter(xs, ys, color=COLORS[objective], alpha=0.7, label=objective.upper(), zorder=3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(1, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("R^2_persistence (higher = less headroom / more static quantity)")
        ax.set_ylabel("skill score (MLP probe vs. persistence, final layer, clean input)")
        ax.legend(fontsize=8)
        fig.suptitle(f"{dataset}: forecast skill vs. quantity difficulty", fontsize=11)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{dataset}_forecast_skill_difficulty.png", dpi=150)
        plt.close(fig)
        print(f"[forecast-skill] wrote {dataset}_forecast_skill_difficulty.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="sweep_results")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--datasets", nargs="+", default=["active_matter", "shear_flow", "rayleigh_benard"])
    args = ap.parse_args()

    sweep_dir, out_dir = Path(args.sweep_dir), Path(args.out_dir)
    plot_emergence_depth(sweep_dir, out_dir, args.datasets)
    plot_noise_layer_heatmap(sweep_dir, out_dir, args.datasets)
    plot_token_linear_vs_mlp(sweep_dir, out_dir, args.datasets)
    plot_forecast_skill_difficulty(sweep_dir, out_dir, args.datasets)


if __name__ == "__main__":
    main()

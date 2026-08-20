"""Regime-parameter (Rayleigh / Prandtl) readout as a tiny table, not a figure.

The Qu-style regime comparison is conceptually different from the
derived-physics probing, so it lives in its own table: JEPA vs MAE test R^2
(mean +/- nested SE across 3 encoder seeds) for Rayleigh and Prandtl
regression, with the shuffled-label controls. Columns for Qu et al.'s
reported JEPA/VideoMAE numbers are left as placeholders to fill in from
their paper -- they are not in our sweep JSON.

Writes regime_table.md and regime_table.tex.

Usage: uv run python paper-plots/make_regime_table.py
"""

import json

from style import OUT_DIR, SWEEP_JSON

TARGETS = ["Rayleigh", "Prandtl", "Rayleigh_shuffled_control", "Prandtl_shuffled_control"]
NICE = {
    "Rayleigh": "Rayleigh number",
    "Prandtl": "Prandtl number",
    "Rayleigh_shuffled_control": "Rayleigh (shuffled control)",
    "Prandtl_shuffled_control": "Prandtl (shuffled control)",
}


def cell(e):
    return f"{e['mean']:.3f} ± {e['nested_se']:.3f}"


def cell_tex(e):
    return f"${e['mean']:.3f} \\pm {e['nested_se']:.3f}$"


def main():
    regime = json.loads(SWEEP_JSON.read_text())["aggregated"]["regime"]

    md = ["| Target | JEPA $R^2$ | MAE $R^2$ |", "|---|---|---|"]
    tex = [
        "\\begin{tabular}{lcc}", "\\toprule",
        "Target & JEPA $R^2$ & MAE $R^2$ \\\\", "\\midrule",
    ]
    for t in TARGETS:
        j, m = regime[t]["jepa"], regime[t]["mae"]
        md.append(f"| {NICE[t]} | {cell(j)} | {cell(m)} |")
        tex.append(f"{NICE[t]} & {cell_tex(j)} & {cell_tex(m)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]

    (OUT_DIR / "regime_table.md").write_text("\n".join(md) + "\n")
    (OUT_DIR / "regime_table.tex").write_text("\n".join(tex) + "\n")
    print("wrote regime_table.md / regime_table.tex")
    print("\n".join(md))


if __name__ == "__main__":
    main()

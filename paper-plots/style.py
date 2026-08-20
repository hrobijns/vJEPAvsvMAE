"""Shared style for the workshop-paper figures (Rayleigh-Bénard only).

All figures read sweep_results/rayleigh_benard_workshop_test_eval.json and
use one visual language: JEPA blue, MAE orange, serif fonts, no chartjunk.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_JSON = REPO_ROOT / "sweep_results" / "rayleigh_benard_workshop_test_eval.json"
OUT_DIR = Path(__file__).resolve().parent

COLORS = {"jepa": "#3b6fd4", "mae": "#d4703b"}
OBJECTIVES = ("jepa", "mae")
GAPS = (8, 32)

# Row ordering: local/differential quantities first, then coupled/integrated
# ones (matches the prose characterization in docs/LINEAR_PROBE.md).
LOCAL_ROWS = ["buoyancy_grad", "pressure_grad_mag", "buoyancy_laplacian"]
COUPLED_ROWS = ["convective_flux", "velocity_buoyancy_coherence", "okubo_weiss", "enstrophy"]
# Token-only: spatial mean is ~0 by construction, so no pooled target exists.
TOKEN_ONLY_ROWS = ["divergence", "vorticity_signed"]

QUANTITY_LABELS = {
    "buoyancy_grad": "Buoyancy grad.",
    "pressure_grad_mag": "Pressure grad. mag.",
    "buoyancy_laplacian": "Buoyancy Laplacian",
    "convective_flux": "Convective flux",
    "velocity_buoyancy_coherence": "Velocity–buoyancy coh.",
    "okubo_weiss": "Okubo–Weiss",
    "enstrophy": "Enstrophy",
    "divergence": "Divergence",
    "vorticity_signed": "Vorticity (signed)",
}

LABEL = {"jepa": "Ray-JEPA", "mae": "Ray-vMAE"}


def nice(q: str) -> str:
    return QUANTITY_LABELS.get(q, q.replace("_", " ").title())


def apply_rcparams():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def layer_labels(n_layers: int) -> list[str]:
    return [f"L{i}" for i in range(n_layers - 1)] + ["norm"]


def save(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png / {stem}.pdf")

"""Deeper JEPA-vs-MAE analysis: harder, derived physics targets + layer-wise probing.

The acceptance-check probe (per-channel squared magnitude) saturates near R^2=1
for both objectives on active_matter — it's close to circular, since a
mean-pooled linear probe can recover input scale/variance almost trivially.
This script probes genuinely derived, nonlinear physical quantities instead
(none of them are stored channels), and does so at every transformer block
depth, not just the final layer, to see where physics becomes decodable.

Targets computed on active_matter's 11 channels (concentration, velocity_x,
velocity_y, D_xx, D_xy, D_yx, D_yy, E_xx, E_xy, E_yx, E_yy):
  - enstrophy:        mean(curl(velocity)^2) over the clip. Vorticity is not a
                       stored channel; requires spatial finite differences.
  - nematic_order:    mean sqrt(D_xx^2 + D_xy^2) over the clip. A nonlinear
                       invariant of the order tensor (low where defects sit).
  - flow_align:       mean(v . D) over the clip. Cross-channel/relational.
  - future_enstrophy: enstrophy at frame (window_end + horizon), i.e. physics
                       the encoder never saw. Tests dynamical predictiveness.

Usage:
    uv run python scripts/analyze_encoders.py \
        --checkpoints runs/active_matter_jepa/encoder_100pct.pt runs/active_matter_mae/encoder_100pct.pt \
        --data-root /workspace/data --dataset active_matter
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vit import build_encoder
from src.data.well import ClipSpec

DATASET_CHANNELS = {
    "active_matter": {
        "concentration": 0, "vx": 1, "vy": 2,
        "Dxx": 3, "Dxy": 4, "Dyx": 5, "Dyy": 6,
        "Exx": 7, "Exy": 8, "Eyx": 9, "Eyy": 10,
    },
    "shear_flow": {"tracer": 0, "pressure": 1, "vx": 2, "vy": 3},
    "rayleigh_benard": {"buoyancy": 0, "pressure": 1, "vx": 2, "vy": 3},
}
CHANNELS = DATASET_CHANNELS["active_matter"]  # back-compat default


def curl2d(vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
    """vx, vy: (..., H, W). Central differences, periodic boundary."""
    dvy_dx = (torch.roll(vy, -1, dims=-1) - torch.roll(vy, 1, dims=-1)) / 2
    dvx_dy = (torch.roll(vx, -1, dims=-2) - torch.roll(vx, 1, dims=-2)) / 2
    return dvy_dx - dvx_dy


def grad2d_sq(field: torch.Tensor) -> torch.Tensor:
    """field: (..., H, W). Squared magnitude of the spatial gradient."""
    dfdx = (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1)) / 2
    dfdy = (torch.roll(field, -1, dims=-2) - torch.roll(field, 1, dims=-2)) / 2
    return dfdx**2 + dfdy**2


def grad2d(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """field: (..., H, W). Returns (df/dx, df/dy), central differences."""
    dfdx = (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1)) / 2
    dfdy = (torch.roll(field, -1, dims=-2) - torch.roll(field, 1, dims=-2)) / 2
    return dfdx, dfdy


def divergence2d(vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
    """vx, vy: (..., H, W). dvx/dx + dvy/dy, central differences, periodic boundary."""
    dvx_dx = (torch.roll(vx, -1, dims=-1) - torch.roll(vx, 1, dims=-1)) / 2
    dvy_dy = (torch.roll(vy, -1, dims=-2) - torch.roll(vy, 1, dims=-2)) / 2
    return dvx_dx + dvy_dy


def laplacian2d(field: torch.Tensor) -> torch.Tensor:
    """field: (..., H, W). 5-point stencil, unit grid spacing, periodic boundary.

    Approximation for rayleigh_benard's y-axis, which is sampled at Chebyshev
    nodes (non-uniform spacing) rather than a uniform grid — exact for
    active_matter/shear_flow (genuinely uniform grids); see docs/LINEAR_PROBE.md.
    """
    lap_x = torch.roll(field, -1, dims=-1) + torch.roll(field, 1, dims=-1) - 2 * field
    lap_y = torch.roll(field, -1, dims=-2) + torch.roll(field, 1, dims=-2) - 2 * field
    return lap_x + lap_y


def tensor_div2d(txx: torch.Tensor, txy: torch.Tensor,
                  tyx: torch.Tensor, tyy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Divergence of a 2x2 tensor field T: (div T)_i = d(T_ij)/dx_j.

    txx,txy,tyx,tyy: (..., H, W). Returns (div_x, div_y), each (..., H, W).
    Central differences, periodic boundary. Used for the active-stress term
    `alpha * div(D)` in active_matter's momentum equation.
    """
    dtxx_dx = (torch.roll(txx, -1, dims=-1) - torch.roll(txx, 1, dims=-1)) / 2
    dtxy_dy = (torch.roll(txy, -1, dims=-2) - torch.roll(txy, 1, dims=-2)) / 2
    dtyx_dx = (torch.roll(tyx, -1, dims=-1) - torch.roll(tyx, 1, dims=-1)) / 2
    dtyy_dy = (torch.roll(tyy, -1, dims=-2) - torch.roll(tyy, 1, dims=-2)) / 2
    return dtxx_dx + dtxy_dy, dtyx_dx + dtyy_dy


def okubo_weiss(vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
    """vx, vy: (..., H, W). Normal-strain^2 + shear-strain^2 - vorticity^2.

    Negative OW => vorticity-dominated (coherent rotating structure);
    positive OW => strain-dominated. Standard 2D turbulence diagnostic.
    """
    dvx_dx = (torch.roll(vx, -1, dims=-1) - torch.roll(vx, 1, dims=-1)) / 2
    dvx_dy = (torch.roll(vx, -1, dims=-2) - torch.roll(vx, 1, dims=-2)) / 2
    dvy_dx = (torch.roll(vy, -1, dims=-1) - torch.roll(vy, 1, dims=-1)) / 2
    dvy_dy = (torch.roll(vy, -1, dims=-2) - torch.roll(vy, 1, dims=-2)) / 2
    normal_strain = dvx_dx - dvy_dy
    shear_strain = dvy_dx + dvx_dy
    vorticity = dvy_dx - dvx_dy
    return normal_strain**2 + shear_strain**2 - vorticity**2


def contemporaneous_targets(clip: torch.Tensor, dataset: str = "active_matter") -> dict:
    """clip: (B, C, T, H, W), normalized fields. Returns {name: (B,)}.

    All targets are derived/nonlinear, not raw stored channels, and each is
    tied to a specific term in the dataset's governing PDE (see
    docs/LINEAR_PROBE.md for the full derivation):
      - enstrophy: mean(curl(velocity)^2) - vorticity isn't a stored channel;
          real turbulent-intensity diagnostic (NOT near-zero — R^2 ~ 0.95-0.99
          in practice), unlike the two targets deliberately excluded below.
      - vorticity_signed / divergence are intentionally NOT pooled targets
          here: on a periodic (or no-slip) domain the spatial mean of
          curl(v) and div(v) is exactly 0 by Stokes'/the divergence theorem,
          regardless of the underlying physics — a degenerate target with no
          signal to probe (confirmed by near-0/negative/NaN R^2 in early
          results). The pointwise fields are not degenerate, so they're kept
          as token-level targets in local_target_maps() instead.
      - active_matter only:
          nematic_order: mean sqrt(Dxx^2+Dxy^2) - nonlinear order-tensor invariant.
          flow_align: mean(v . D) - cross-channel relational coupling.
          active_stress_div_mag: mean |div(D)| - the active-stress term
              `alpha * div(D)` in the momentum equation; alpha (dipole
              strength) is the regime parameter already probed separately,
              this ties it to a concrete local field.
      - shear_flow only:
          tracer_grad: mean |grad(tracer)|^2 - derived differential quantity.
          advective_flux: mean(v . grad(tracer)) - cross-channel relational
              coupling (velocity-tracer advection), the shear_flow analog of
              active_matter's flow_align.
          tracer_laplacian: mean |laplacian(tracer)| - the diffusive term in
              the tracer PDE (`D * laplacian(s)`, D=nu/Sc), distinct from
              tracer_grad (first- vs second-derivative, different
              frequency content).
      - rayleigh_benard only:
          buoyancy_laplacian: mean |laplacian(buoyancy)| - the diffusive term
              in the buoyancy PDE (`kappa * laplacian(b)`), the RB analog of
              shear_flow's tracer_laplacian.
    """
    ch = DATASET_CHANNELS[dataset]
    vx, vy = clip[:, ch["vx"]], clip[:, ch["vy"]]
    vorticity = curl2d(vx, vy)
    out = {
        "enstrophy": (vorticity**2).mean(dim=(1, 2, 3)),
    }

    if dataset == "active_matter":
        dxx, dxy = clip[:, ch["Dxx"]], clip[:, ch["Dxy"]]
        exx, exy = clip[:, ch["Exx"]], clip[:, ch["Exy"]]
        eyx, eyy = clip[:, ch["Eyx"]], clip[:, ch["Eyy"]]
        dyx, dyy = clip[:, ch["Dyx"]], clip[:, ch["Dyy"]]
        out["nematic_order"] = torch.sqrt(dxx**2 + dxy**2 + 1e-8).mean(dim=(1, 2, 3))
        out["flow_align"] = (vx * dxx + vy * dxy).mean(dim=(1, 2, 3))
        dSdx, dSdy = grad2d(dxx)
        out["order_grad_mag"] = torch.sqrt(dSdx**2 + dSdy**2 + 1e-8).mean(dim=(1, 2, 3))
        out["strain_order_align"] = (
            exx * dxx + exy * dxy + eyx * dyx + eyy * dyy
        ).mean(dim=(1, 2, 3))
        div_dx, div_dy = tensor_div2d(dxx, dxy, dyx, dyy)
        out["active_stress_div_mag"] = torch.sqrt(
            div_dx**2 + div_dy**2 + 1e-8
        ).mean(dim=(1, 2, 3))
    elif dataset == "shear_flow":
        tracer = clip[:, ch["tracer"]]
        pressure = clip[:, ch["pressure"]]
        dtdx = (torch.roll(tracer, -1, dims=-1) - torch.roll(tracer, 1, dims=-1)) / 2
        dtdy = (torch.roll(tracer, -1, dims=-2) - torch.roll(tracer, 1, dims=-2)) / 2
        out["tracer_grad"] = grad2d_sq(tracer).mean(dim=(1, 2, 3))
        out["advective_flux"] = (vx * dtdx + vy * dtdy).mean(dim=(1, 2, 3))
        dvx_dx, dvx_dy = grad2d(vx)
        dvy_dx, dvy_dy = grad2d(vy)
        strain_rate_sq = dvx_dx**2 + dvy_dy**2 + 0.5 * (dvx_dy + dvy_dx) ** 2
        out["strain_rate_mag"] = torch.sqrt(strain_rate_sq + 1e-8).mean(dim=(1, 2, 3))
        out["okubo_weiss"] = okubo_weiss(vx, vy).mean(dim=(1, 2, 3))
        out["pressure_grad_mag"] = torch.sqrt(grad2d_sq(pressure) + 1e-8).mean(dim=(1, 2, 3))
        out["tracer_laplacian"] = laplacian2d(tracer).abs().mean(dim=(1, 2, 3))
    elif dataset == "rayleigh_benard":
        buoyancy = clip[:, ch["buoyancy"]]
        pressure = clip[:, ch["pressure"]]
        out["buoyancy_grad"] = grad2d_sq(buoyancy).mean(dim=(1, 2, 3))
        # canonical RB convective/buoyancy flux <v_y * buoyancy>, the standard
        # order parameter for convective heat transport in this system.
        out["convective_flux"] = (vy * buoyancy).mean(dim=(1, 2, 3))
        out["okubo_weiss"] = okubo_weiss(vx, vy).mean(dim=(1, 2, 3))
        speed = torch.sqrt(vx**2 + vy**2 + 1e-8)
        b_mag = torch.sqrt(buoyancy**2 + 1e-8)
        out["velocity_buoyancy_coherence"] = (
            (vy * buoyancy) / (speed * b_mag + 1e-6)
        ).mean(dim=(1, 2, 3))
        out["pressure_grad_mag"] = torch.sqrt(grad2d_sq(pressure) + 1e-8).mean(dim=(1, 2, 3))
        out["buoyancy_laplacian"] = laplacian2d(buoyancy).abs().mean(dim=(1, 2, 3))
    else:
        raise ValueError(f"no target definitions for dataset {dataset!r}")
    return out


def future_enstrophy(mm: np.ndarray, traj: int, window_end: int, horizon: int,
                      dataset: str = "active_matter") -> float:
    """mm: raw (N,C,T,H,W) memmap. Enstrophy at a single future frame."""
    ch = DATASET_CHANNELS[dataset]
    t = window_end + horizon
    vx = torch.from_numpy(np.array(mm[traj, ch["vx"], t])).float()
    vy = torch.from_numpy(np.array(mm[traj, ch["vy"], t])).float()
    vort = curl2d(vx, vy)
    return (vort**2).mean().item()


def ridge_r2(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-2) -> float:
    n = x.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    folds = perm.chunk(5)
    r2s = []
    for i in range(5):
        val = folds[i]
        tr = torch.cat([folds[j] for j in range(5) if j != i])
        xm, xs = x[tr].mean(0), x[tr].std(0) + 1e-8
        xtr, xval = (x[tr] - xm) / xs, (x[val] - xm) / xs
        ym = y[tr].mean()
        a = xtr.T @ xtr + lam * xtr.size(0) * torch.eye(x.size(1))
        w = torch.linalg.solve(a, xtr.T @ (y[tr] - ym))
        pred = xval @ w + ym
        ss_res = ((pred - y[val]) ** 2).sum()
        ss_tot = ((y[val] - y[val].mean()) ** 2).sum()
        r2s.append((1 - ss_res / ss_tot).item())
    return sum(r2s) / len(r2s)


def ridge_r2_grouped(x: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, lam: float = 1e-2) -> float:
    """Same as ridge_r2, but 5-fold CV splits by GROUP id (e.g. trajectory
    index) so that no two rows sharing a group ever land in different folds.
    Needed wherever many rows come from the same trajectory and share an
    (near-)identical target — e.g. per-token regime probing, where every
    token of every sampled window inherits one constant per-trajectory
    value. A plain random per-row split (ridge_r2) would leak trajectory
    identity across folds and inflate R^2.

    Degenerate folds (validation groups that happen to all share the same
    target value — a real risk for active_matter's alpha/zeta, which only
    take 5-9 discrete values, so a small group count can by chance land an
    entire validation fold on one value) are skipped rather than contributing
    a ss_tot=0 -> -inf/nan: with grouped CV the usual per-row law-of-large-
    numbers guarantee that a random fold spans many target values doesn't
    hold, since whole groups (not rows) move together."""
    uniq = torch.unique(groups)
    perm = torch.randperm(uniq.size(0), generator=torch.Generator().manual_seed(0))
    group_folds = perm.chunk(5)
    r2s = []
    for i in range(5):
        val_groups = uniq[group_folds[i]]
        val_mask = torch.isin(groups, val_groups)
        tr_mask = ~val_mask
        xtr_raw, ytr = x[tr_mask], y[tr_mask]
        xval_raw, yval = x[val_mask], y[val_mask]
        ss_tot = ((yval - yval.mean()) ** 2).sum()
        if ss_tot < 1e-8:
            continue  # degenerate fold: validation groups all share one target value
        xm, xs = xtr_raw.mean(0), xtr_raw.std(0) + 1e-8
        xtr, xval = (xtr_raw - xm) / xs, (xval_raw - xm) / xs
        ym = ytr.mean()
        a = xtr.T @ xtr + lam * xtr.size(0) * torch.eye(x.size(1))
        w = torch.linalg.solve(a, xtr.T @ (ytr - ym))
        pred = xval @ w + ym
        ss_res = ((pred - yval) ** 2).sum()
        r2s.append((1 - ss_res / ss_tot).item())
    return sum(r2s) / len(r2s) if r2s else float("nan")


def skill_score(r2_probe: float, r2_baseline: float, eps: float = 1e-3) -> float:
    """Persistence-relative forecast skill (Murphy 1988): how much of the
    remaining gap between a naive baseline (typically persistence — "assume
    nothing changes") and a perfect ceiling does the probe close?

        skill = 1 - (1 - r2_probe) / (1 - r2_baseline)

    0 -> exactly as good as the baseline. 1 -> closes the entire remaining
    gap. Negative -> worse than the baseline (this happens in practice, e.g.
    fed_back rollout dropping below persistence mid-chain). Algebraically
    equivalent to 1 - MSE_probe/MSE_baseline (the target-variance term in R^2's
    denominator cancels), so it needs no extra encoder/probe computation
    beyond R^2 values the suite already produces.

    Exists specifically because raw R^2 isn't comparable *across* physics
    quantities that evolve at very different rates: a near-static quantity
    looks "easy" (high R^2) for any method including the naive baseline,
    while a fast-changing one looks "hard" even for a good model — averaging
    raw R^2 across quantities conflates "the model is good" with "this
    variable barely changes anyway." Dividing by the baseline's own headroom
    (1 - r2_baseline) normalizes that away.

    NaN (not raised) when r2_baseline is within `eps` of a perfect 1.0 —
    i.e. the baseline already explains ~all the variance, so there's no
    headroom left to score skill against (same degenerate-target guard
    philosophy as ridge_r2_grouped's skipped folds)."""
    denom = 1 - r2_baseline
    if abs(denom) < eps:
        return float("nan")
    return 1 - (1 - r2_probe) / denom


@torch.no_grad()
def layerwise_features(encoder, clip: torch.Tensor) -> list[torch.Tensor]:
    """Returns list of (B, D) mean-pooled features, one per block + final norm."""
    tokens = encoder.tokenize(clip)
    feats = []
    x = tokens
    for blk in encoder.blocks:
        x = blk(x)
        feats.append(x.mean(dim=1).float().cpu())
    feats.append(encoder.norm(x).mean(dim=1).float().cpu())
    return feats


def build_dataset(base: str, name: str, n_offsets: int, horizon: int, n_frames: int = 8,
                   max_trajectories: int | None = None, split: str = "train"):
    """Sample `n_offsets` starting windows per trajectory; return clips + targets.

    `max_trajectories` subsamples (evenly spaced across the full trajectory
    range, for coverage across regime parameters rather than a biased prefix)
    down to a fixed cap — this whole function materializes every sampled
    clip AND several full-resolution intermediate tensors (curl, gradients,
    laplacian, ...) inside contemporaneous_targets() at once, so on datasets
    with many trajectories (e.g. shear_flow's 896, vs. active_matter's 175)
    the default of "every trajectory" can exceed a memory-constrained pod's
    limits even at n_offsets=1. None (default) keeps every trajectory,
    matching the original behavior.

    `split` selects which preprocessed memmap to read (default "train", the
    only split every probing script used historically — see
    docs/OVERVIEW.md's "held-out generalization check" note for why "valid"
    is also supported now)."""
    d = Path(base) / "memmap" / name
    mm = np.load(d / f"{split}.npy", mmap_mode="r")
    n_traj, c, t, h, w = mm.shape
    max_off = t - n_frames - horizon
    offsets = np.linspace(0, max_off, n_offsets).astype(int)
    trajectories = list(range(n_traj))
    if max_trajectories is not None and max_trajectories < n_traj:
        idx = np.linspace(0, n_traj - 1, max_trajectories).astype(int)
        trajectories = sorted(set(idx.tolist()))

    clips = []
    for traj in trajectories:
        for off in offsets:
            window = np.array(mm[traj, :, off : off + n_frames])
            clips.append(torch.from_numpy(window).float())
    clips = torch.stack(clips)  # (N, C, T, H, W)

    contemp = contemporaneous_targets(clips, dataset=name)

    future = []
    for traj in trajectories:
        for off in offsets:
            future.append(future_enstrophy(mm, traj, off + n_frames - 1, horizon, dataset=name))
    future = torch.tensor(future, dtype=torch.float32)

    return clips, contemp, future


def load_checkpoint_encoder(ckpt_path: str, device: torch.device):
    """Loads an encoder-only checkpoint (encoder_*pct.pt or full latest.pt's
    'encoder' key). Returns (encoder, spec)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    spec = ClipSpec(**ckpt["spec"])
    encoder = build_encoder(spec, ckpt["config"].get("encoder", {})).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    return encoder, spec


def compute_layerwise_features_batched(encoder, clips: torch.Tensor, batch_size: int = 16) -> list[torch.Tensor]:
    """Runs layerwise_features over clips in batches. Returns list of (N, D)
    mean-pooled features, one per block + final norm — shared by any script
    that needs pooled per-layer features (physics probing, regime probing)."""
    device = next(encoder.parameters()).device
    n_layers = len(encoder.blocks) + 1
    per_layer_feats = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for start in range(0, clips.size(0), batch_size):
            batch = clips[start : start + batch_size].to(device)
            feats = layerwise_features(encoder, batch)
            for i, f in enumerate(feats):
                per_layer_feats[i].append(f)
    return [torch.cat(fs) for fs in per_layer_feats]


def analyze_checkpoint(ckpt_path: str, clips: torch.Tensor, contemp: dict, future: torch.Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _spec = load_checkpoint_encoder(ckpt_path, device)
    per_layer_feats = compute_layerwise_features_batched(encoder, clips)

    results = {"n_layers": len(per_layer_feats), "n_samples": clips.size(0), "layers": []}
    targets = {**contemp, "future_enstrophy": future}
    for layer_idx, feats in enumerate(per_layer_feats):
        layer_r2 = {}
        for tname, y in targets.items():
            layer_r2[tname] = ridge_r2(feats, y)
        results["layers"].append(layer_r2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="active_matter")
    ap.add_argument("--n-offsets", type=int, default=3, help="windows per trajectory")
    ap.add_argument("--horizon", type=int, default=16, help="future prediction horizon (frames)")
    ap.add_argument("--max-traj", type=int, default=None,
                     help="cap trajectories used (evenly subsampled) — build_dataset() materializes "
                          "everything at once, so large-trajectory-count datasets can exceed memory "
                          "on constrained pods; default uses every trajectory")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"],
                     help="which preprocessed memmap split to probe (default train; use valid for a "
                          "held-out generalization check against the pretraining data)")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    print(f"building probe dataset ({args.split}): {args.n_offsets} offsets/traj, horizon={args.horizon}, max_traj={args.max_traj}")
    clips, contemp, future = build_dataset(args.data_root, args.dataset, args.n_offsets, args.horizon,
                                            max_trajectories=args.max_traj, split=args.split)
    print(f"{clips.size(0)} samples total")
    for k, v in contemp.items():
        print(f"  {k}: mean={v.mean():.4f} std={v.std():.4f}")
    print(f"  future_enstrophy: mean={future.mean():.4f} std={future.std():.4f}")

    all_results = {}
    for ckpt_path in args.checkpoints:
        print(f"\n=== {ckpt_path} ===")
        res = analyze_checkpoint(ckpt_path, clips, contemp, future)
        all_results[ckpt_path] = res
        for i, layer_r2 in enumerate(res["layers"]):
            tag = f"block_{i+1}" if i < res["n_layers"] - 1 else "final_norm"
            line = "  ".join(f"{k}={v:.3f}" for k, v in layer_r2.items())
            print(f"  {tag:12s} {line}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

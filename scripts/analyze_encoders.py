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


def contemporaneous_targets(clip: torch.Tensor, dataset: str = "active_matter") -> dict:
    """clip: (B, C, T, H, W), normalized fields. Returns {name: (B,)}.

    All targets are derived/nonlinear, not raw stored channels:
      - enstrophy: mean(curl(velocity)^2) - vorticity isn't a stored channel.
      - active_matter only:
          nematic_order: mean sqrt(Dxx^2+Dxy^2) - nonlinear order-tensor invariant.
          flow_align: mean(v . D) - cross-channel relational coupling.
      - shear_flow only:
          tracer_grad: mean |grad(tracer)|^2 - derived differential quantity.
          advective_flux: mean(v . grad(tracer)) - cross-channel relational
              coupling (velocity-tracer advection), the shear_flow analog of
              active_matter's flow_align.
    """
    ch = DATASET_CHANNELS[dataset]
    vx, vy = clip[:, ch["vx"]], clip[:, ch["vy"]]
    vorticity = curl2d(vx, vy)
    out = {"enstrophy": (vorticity**2).mean(dim=(1, 2, 3))}

    if dataset == "active_matter":
        dxx, dxy = clip[:, ch["Dxx"]], clip[:, ch["Dxy"]]
        out["nematic_order"] = torch.sqrt(dxx**2 + dxy**2 + 1e-8).mean(dim=(1, 2, 3))
        out["flow_align"] = (vx * dxx + vy * dxy).mean(dim=(1, 2, 3))
    elif dataset == "shear_flow":
        tracer = clip[:, ch["tracer"]]
        dtdx = (torch.roll(tracer, -1, dims=-1) - torch.roll(tracer, 1, dims=-1)) / 2
        dtdy = (torch.roll(tracer, -1, dims=-2) - torch.roll(tracer, 1, dims=-2)) / 2
        out["tracer_grad"] = grad2d_sq(tracer).mean(dim=(1, 2, 3))
        out["advective_flux"] = (vx * dtdx + vy * dtdy).mean(dim=(1, 2, 3))
    elif dataset == "rayleigh_benard":
        buoyancy = clip[:, ch["buoyancy"]]
        out["buoyancy_grad"] = grad2d_sq(buoyancy).mean(dim=(1, 2, 3))
        # canonical RB convective/buoyancy flux <v_y * buoyancy>, the standard
        # order parameter for convective heat transport in this system.
        out["convective_flux"] = (vy * buoyancy).mean(dim=(1, 2, 3))
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


def build_dataset(base: str, name: str, n_offsets: int, horizon: int, n_frames: int = 8):
    """Sample `n_offsets` starting windows per trajectory; return clips + targets."""
    d = Path(base) / "memmap" / name
    mm = np.load(d / "train.npy", mmap_mode="r")
    n_traj, c, t, h, w = mm.shape
    max_off = t - n_frames - horizon
    offsets = np.linspace(0, max_off, n_offsets).astype(int)

    clips = []
    for traj in range(n_traj):
        for off in offsets:
            window = np.array(mm[traj, :, off : off + n_frames])
            clips.append(torch.from_numpy(window).float())
    clips = torch.stack(clips)  # (N, C, T, H, W)

    contemp = contemporaneous_targets(clips, dataset=name)

    future = []
    for traj in range(n_traj):
        for off in offsets:
            future.append(future_enstrophy(mm, traj, off + n_frames - 1, horizon, dataset=name))
    future = torch.tensor(future, dtype=torch.float32)

    return clips, contemp, future


def analyze_checkpoint(ckpt_path: str, clips: torch.Tensor, contemp: dict, future: torch.Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    spec = ClipSpec(**ckpt["spec"])
    encoder = build_encoder(spec, ckpt["config"].get("encoder", {})).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])

    n_layers = encoder.blocks.__len__() + 1
    per_layer_feats = [[] for _ in range(n_layers)]
    batch_size = 16
    with torch.no_grad():
        for start in range(0, clips.size(0), batch_size):
            batch = clips[start : start + batch_size].to(device)
            feats = layerwise_features(encoder, batch)
            for i, f in enumerate(feats):
                per_layer_feats[i].append(f)
    per_layer_feats = [torch.cat(fs) for fs in per_layer_feats]

    results = {"n_layers": n_layers, "n_samples": clips.size(0), "layers": []}
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
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    print(f"building probe dataset: {args.n_offsets} offsets/traj, horizon={args.horizon}")
    clips, contemp, future = build_dataset(args.data_root, args.dataset, args.n_offsets, args.horizon)
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

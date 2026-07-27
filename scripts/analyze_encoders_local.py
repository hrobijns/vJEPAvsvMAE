"""Non-pooled probes: does mean-pooling (not the representation) explain JEPA's
flow_align collapse in analyze_encoders.py?

Two complementary tests, both avoiding global mean-pooling:

1. Per-token linear probe. Each of the ~1024 tokens gets its own LOCAL target
   (vorticity^2 / order magnitude / flow-order coupling, averaged over exactly
   that token's own space-time footprint via avg_pool3d matching the ViT's own
   patchify geometry). We test whether a token's raw feature vector linearly
   predicts the physics AT THAT EXACT LOCATION - no pooling of features or
   targets. If this recovers strong JEPA performance where the pooled probe
   collapsed, pooling was destroying real local structure. If it still
   collapses, the effect is real, not a pooling artifact.

2. Small nonlinear (2-layer MLP) probe on the same globally-pooled features
   used in analyze_encoders.py. Tests whether the information exists but
   needs a nonlinear readout (matching the reference paper's finding that
   some physical variables require "coordinated multi-feature intervention",
   i.e. aren't linearly accessible even when present).

Usage:
    uv run python scripts/analyze_encoders_local.py \
        --checkpoints local_runs/active_matter_jepa/encoder_100pct.pt local_runs/active_matter_mae/encoder_100pct.pt \
        --data-root local_data --dataset active_matter
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vit import build_encoder
from src.data.well import ClipSpec
from scripts.analyze_encoders import (
    DATASET_CHANNELS, curl2d, ridge_r2, build_dataset,
)


def local_target_maps(clip: torch.Tensor, patch_t: int, patch_h: int, patch_w: int,
                       dataset: str = "active_matter") -> dict:
    """Per-token local targets: (B, N) each, N = grid_t*grid_h*grid_w, matching
    the ViT's own tokenize() flatten order (t slowest, then h, then w)."""
    ch = DATASET_CHANNELS[dataset]
    vx, vy = clip[:, ch["vx"]], clip[:, ch["vy"]]
    vort = curl2d(vx, vy)
    fields = {"enstrophy": vort**2}

    if dataset == "active_matter":
        dxx, dxy = clip[:, ch["Dxx"]], clip[:, ch["Dxy"]]
        fields["nematic_order"] = torch.sqrt(dxx**2 + dxy**2 + 1e-8)
        fields["flow_align"] = vx * dxx + vy * dxy
    elif dataset == "shear_flow":
        tracer = clip[:, ch["tracer"]]
        dtdx = (torch.roll(tracer, -1, dims=-1) - torch.roll(tracer, 1, dims=-1)) / 2
        dtdy = (torch.roll(tracer, -1, dims=-2) - torch.roll(tracer, 1, dims=-2)) / 2
        fields["tracer_grad"] = dtdx**2 + dtdy**2
        fields["advective_flux"] = vx * dtdx + vy * dtdy
    elif dataset == "rayleigh_benard":
        buoyancy = clip[:, ch["buoyancy"]]
        dbdx = (torch.roll(buoyancy, -1, dims=-1) - torch.roll(buoyancy, 1, dims=-1)) / 2
        dbdy = (torch.roll(buoyancy, -1, dims=-2) - torch.roll(buoyancy, 1, dims=-2)) / 2
        fields["buoyancy_grad"] = dbdx**2 + dbdy**2
        fields["convective_flux"] = vy * buoyancy
    else:
        raise ValueError(f"no local target definitions for dataset {dataset!r}")

    out = {}
    for name, field in fields.items():
        pooled = F.avg_pool3d(field.unsqueeze(1), kernel_size=(patch_t, patch_h, patch_w))
        out[name] = pooled.squeeze(1).flatten(1)  # (B, N)
    return out


@torch.no_grad()
def layerwise_token_features(encoder, clip: torch.Tensor) -> list[torch.Tensor]:
    """Per-token (unpooled) features per layer: list of (B, N, D)."""
    tokens = encoder.tokenize(clip)
    feats = []
    x = tokens
    for blk in encoder.blocks:
        x = blk(x)
        feats.append(x.float().cpu())
    feats.append(encoder.norm(x).float().cpu())
    return feats


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_r2(x: torch.Tensor, y: torch.Tensor, hidden: int = 128, steps: int = 2000,
           lr: float = 1e-2, seed: int = 0) -> float:
    """Single train/val split (80/20), Adam + cosine schedule, standardized I/O.

    Hyperparameters were calibrated on a synthetic near-linear signal to reach
    R^2 ~= 0.97 (matching ceiling) rather than stalling partway — an
    undertrained MLP would falsely look like "no benefit from nonlinearity".
    """
    g = torch.Generator().manual_seed(seed)
    n = x.size(0)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, n // 5)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    xm, xs = x[tr_idx].mean(0), x[tr_idx].std(0) + 1e-8
    ym, ys = y[tr_idx].mean(), y[tr_idx].std() + 1e-8
    xtr, xval = (x[tr_idx] - xm) / xs, (x[val_idx] - xm) / xs
    ytr, yval = (y[tr_idx] - ym) / ys, (y[val_idx] - ym) / ys

    torch.manual_seed(seed)
    model = TinyMLP(x.size(1), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        pred = model(xtr)
        loss = F.mse_loss(pred, ytr)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(xval)
    ss_res = ((pred_val - yval) ** 2).sum()
    ss_tot = ((yval - yval.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def analyze_token_level(ckpt_path: str, clips: torch.Tensor, contemp: dict, max_clips: int,
                         dataset: str = "active_matter"):
    """Per-token linear probe: no pooling of features or targets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    spec = ClipSpec(**ckpt["spec"])
    encoder = build_encoder(spec, ckpt["config"].get("encoder", {})).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    pt, ph, pw = encoder.patch_size

    sub = clips[:max_clips]
    targets = local_target_maps(sub, pt, ph, pw, dataset=dataset)  # each (B, N)

    n_layers = len(encoder.blocks) + 1
    per_layer = [[] for _ in range(n_layers)]
    batch_size = 8
    with torch.no_grad():
        for start in range(0, sub.size(0), batch_size):
            batch = sub[start : start + batch_size].to(device)
            feats = layerwise_token_features(encoder, batch)  # list of (b, N, D)
            for i, f in enumerate(feats):
                per_layer[i].append(f)
    per_layer = [torch.cat(fs) for fs in per_layer]  # (n_clips, N, D)

    results = {"n_layers": n_layers, "n_clips": sub.size(0), "layers": []}
    for layer_idx, feats in enumerate(per_layer):
        n_clips, n_tok, d = feats.shape
        x_flat = feats.reshape(-1, d)  # (n_clips*N, D)
        layer_r2 = {}
        for tname, y in targets.items():
            y_flat = y.reshape(-1)
            layer_r2[tname] = ridge_r2(x_flat, y_flat, lam=1e-2)
        results["layers"].append(layer_r2)
    return results


def analyze_mlp_level(ckpt_path: str, clips: torch.Tensor, contemp: dict):
    """Nonlinear (small MLP) probe on globally pooled features, same targets
    as the original pooled ridge probe in analyze_encoders.py."""
    from scripts.analyze_encoders import layerwise_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    spec = ClipSpec(**ckpt["spec"])
    encoder = build_encoder(spec, ckpt["config"].get("encoder", {})).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])

    n_layers = len(encoder.blocks) + 1
    per_layer = [[] for _ in range(n_layers)]
    batch_size = 16
    with torch.no_grad():
        for start in range(0, clips.size(0), batch_size):
            batch = clips[start : start + batch_size].to(device)
            feats = layerwise_features(encoder, batch)  # list of (b, D)
            for i, f in enumerate(feats):
                per_layer[i].append(f)
    per_layer = [torch.cat(fs) for fs in per_layer]

    results = {"n_layers": n_layers, "layers": []}
    for layer_idx, feats in enumerate(per_layer):
        layer_r2 = {}
        for tname, y in contemp.items():
            layer_r2[tname] = mlp_r2(feats, y, seed=layer_idx)
        results["layers"].append(layer_r2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="active_matter")
    ap.add_argument("--n-offsets", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--token-max-clips", type=int, default=64,
                     help="clips used for per-token probe (each yields ~1024 token-samples)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("building probe dataset...")
    clips, contemp, _future = build_dataset(args.data_root, args.dataset, args.n_offsets, args.horizon)
    print(f"{clips.size(0)} clips total; token probe uses first {args.token_max_clips}")

    all_results = {}
    for ckpt_path in args.checkpoints:
        print(f"\n=== {ckpt_path} ===")
        tok_res = analyze_token_level(ckpt_path, clips, contemp, args.token_max_clips, dataset=args.dataset)
        mlp_res = analyze_mlp_level(ckpt_path, clips, contemp)
        all_results[ckpt_path] = {"token_linear": tok_res, "mlp_pooled": mlp_res}

        print("  -- per-token LINEAR probe (no pooling) --")
        for i, layer_r2 in enumerate(tok_res["layers"]):
            tag = f"block_{i+1}" if i < tok_res["n_layers"] - 1 else "final_norm"
            line = "  ".join(f"{k}={v:.3f}" for k, v in layer_r2.items())
            print(f"    {tag:12s} {line}")

        print("  -- pooled NONLINEAR (MLP) probe --")
        for i, layer_r2 in enumerate(mlp_res["layers"]):
            tag = f"block_{i+1}" if i < mlp_res["n_layers"] - 1 else "final_norm"
            line = "  ".join(f"{k}={v:.3f}" for k, v in layer_r2.items())
            print(f"    {tag:12s} {line}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

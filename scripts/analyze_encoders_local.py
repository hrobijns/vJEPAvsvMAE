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
import copy
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
    DATASET_CHANNELS, curl2d, grad2d, divergence2d, laplacian2d, tensor_div2d,
    okubo_weiss, ridge_r2, build_dataset,
)


def local_target_maps(clip: torch.Tensor, patch_t: int, patch_h: int, patch_w: int,
                       dataset: str = "active_matter") -> dict:
    """Per-token local targets: (B, N) each, N = grid_t*grid_h*grid_w, matching
    the ViT's own tokenize() flatten order (t slowest, then h, then w)."""
    ch = DATASET_CHANNELS[dataset]
    vx, vy = clip[:, ch["vx"]], clip[:, ch["vy"]]
    vort = curl2d(vx, vy)
    fields = {
        "enstrophy": vort**2,
        "vorticity_signed": vort,
        "divergence": divergence2d(vx, vy),
    }

    if dataset == "active_matter":
        dxx, dxy = clip[:, ch["Dxx"]], clip[:, ch["Dxy"]]
        exx, exy = clip[:, ch["Exx"]], clip[:, ch["Exy"]]
        eyx, eyy = clip[:, ch["Eyx"]], clip[:, ch["Eyy"]]
        dyx, dyy = clip[:, ch["Dyx"]], clip[:, ch["Dyy"]]
        fields["nematic_order"] = torch.sqrt(dxx**2 + dxy**2 + 1e-8)
        fields["flow_align"] = vx * dxx + vy * dxy
        dSdx, dSdy = grad2d(dxx)
        fields["order_grad_mag"] = torch.sqrt(dSdx**2 + dSdy**2 + 1e-8)
        fields["strain_order_align"] = exx * dxx + exy * dxy + eyx * dyx + eyy * dyy
        div_dx, div_dy = tensor_div2d(dxx, dxy, dyx, dyy)
        fields["active_stress_div_mag"] = torch.sqrt(div_dx**2 + div_dy**2 + 1e-8)
    elif dataset == "shear_flow":
        tracer = clip[:, ch["tracer"]]
        pressure = clip[:, ch["pressure"]]
        dtdx = (torch.roll(tracer, -1, dims=-1) - torch.roll(tracer, 1, dims=-1)) / 2
        dtdy = (torch.roll(tracer, -1, dims=-2) - torch.roll(tracer, 1, dims=-2)) / 2
        fields["tracer_grad"] = dtdx**2 + dtdy**2
        fields["advective_flux"] = vx * dtdx + vy * dtdy
        dvx_dx, dvx_dy = grad2d(vx)
        dvy_dx, dvy_dy = grad2d(vy)
        strain_rate_sq = dvx_dx**2 + dvy_dy**2 + 0.5 * (dvx_dy + dvy_dx) ** 2
        fields["strain_rate_mag"] = torch.sqrt(strain_rate_sq + 1e-8)
        fields["okubo_weiss"] = okubo_weiss(vx, vy)
        dpdx, dpdy = grad2d(pressure)
        fields["pressure_grad_mag"] = torch.sqrt(dpdx**2 + dpdy**2 + 1e-8)
        fields["tracer_laplacian"] = laplacian2d(tracer).abs()
    elif dataset == "rayleigh_benard":
        buoyancy = clip[:, ch["buoyancy"]]
        pressure = clip[:, ch["pressure"]]
        dbdx = (torch.roll(buoyancy, -1, dims=-1) - torch.roll(buoyancy, 1, dims=-1)) / 2
        dbdy = (torch.roll(buoyancy, -1, dims=-2) - torch.roll(buoyancy, 1, dims=-2)) / 2
        fields["buoyancy_grad"] = dbdx**2 + dbdy**2
        fields["convective_flux"] = vy * buoyancy
        fields["okubo_weiss"] = okubo_weiss(vx, vy)
        speed = torch.sqrt(vx**2 + vy**2 + 1e-8)
        b_mag = torch.sqrt(buoyancy**2 + 1e-8)
        fields["velocity_buoyancy_coherence"] = (vy * buoyancy) / (speed * b_mag + 1e-6)
        dpdx, dpdy = grad2d(pressure)
        fields["pressure_grad_mag"] = torch.sqrt(dpdx**2 + dpdy**2 + 1e-8)
        fields["buoyancy_laplacian"] = laplacian2d(buoyancy).abs()
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
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_r2(x: torch.Tensor, y: torch.Tensor, hidden: int = 128, steps: int = 2000,
           lr: float = 1e-2, seed: int = 0) -> float:
    """Single train/val split (80/20), Adam + cosine schedule, standardized I/O.

    Hyperparameters were calibrated on a synthetic near-linear signal to reach
    R^2 ~= 0.97 (matching ceiling) rather than stalling partway — an
    undertrained MLP would falsely look like "no benefit from nonlinearity".

    Runs on GPU when available: unlike ridge_r2's single closed-form solve,
    this is `steps` sequential Adam iterations — the one part of the probing
    suite that's actually compute-bound rather than I/O-bound, so leaving it
    on CPU (as the rest of the suite does, since `layerwise_*features()`
    already moves everything to `.cpu()` right after the encoder forward
    pass) stalls badly at the scale of a full sweep (8 targets x 13 layers x
    2 checkpoints x 3 datasets).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = x.to(device), y.to(device)
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
    model = TinyMLP(x.size(1), hidden).to(device)
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


def mlp_fit_eval(x_train: torch.Tensor, y_train: torch.Tensor,
                  x_test: torch.Tensor, y_test: torch.Tensor, hidden: int = 128,
                  steps: int = 2000, lr: float = 1e-2, seed: int = 0) -> float:
    """Fit-train/eval-test counterpart to mlp_r2() — trains on the full
    (x_train, y_train) with no internal val split (fixed step count, no
    early stopping, matching mlp_r2()'s schedule), evaluates once on
    (x_test, y_test). Same fixed hyperparameters as mlp_r2() (calibrated
    against a synthetic ceiling target); layer/architecture choices must
    already be frozen from train-CV before calling this."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    xm, xs = x_train.mean(0), x_train.std(0) + 1e-8
    ym, ys = y_train.mean(), y_train.std() + 1e-8
    xtr, xte = (x_train - xm) / xs, (x_test - xm) / xs
    ytr, yte = (y_train - ym) / ys, (y_test - ym) / ys

    torch.manual_seed(seed)
    model = TinyMLP(x_train.size(1), hidden).to(device)
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
        pred_te = model(xte)
    ss_res = ((pred_te - yte) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def _train_mlp_early_stop(x_fit: torch.Tensor, y_fit: torch.Tensor, hidden: int = 128,
                           dropout: float = 0.1, lr: float = 1e-2, weight_decay: float = 1e-4,
                           seed: int = 0, min_steps: int = 150, max_steps: int = 4000,
                           patience: int = 100, check_every: int = 20):
    """Train one TinyMLP with early stopping monitored on a small internal holdout
    carved from x_fit/y_fit (never valid or test data — a further split of
    whatever "fit" set was passed in). Replaces the old fixed-2000-step
    schedule with an actual convergence check: keeps the best-monitor-loss
    state dict, restores it before returning. Returns (model, xm, xs, ym, ys)
    so the caller can standardize an eval set with the SAME (inner-train-only)
    statistics used during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_fit, y_fit = x_fit.to(device), y_fit.to(device)
    g = torch.Generator().manual_seed(seed)
    n = x_fit.size(0)
    perm = torch.randperm(n, generator=g)
    n_mon = max(1, n // 10)
    mon_idx, tr_idx = perm[:n_mon], perm[n_mon:]

    xm, xs = x_fit[tr_idx].mean(0), x_fit[tr_idx].std(0) + 1e-8
    ym, ys = y_fit[tr_idx].mean(), y_fit[tr_idx].std() + 1e-8
    xtr, xmon = (x_fit[tr_idx] - xm) / xs, (x_fit[mon_idx] - xm) / xs
    ytr, ymon = (y_fit[tr_idx] - ym) / ys, (y_fit[mon_idx] - ym) / ys

    torch.manual_seed(seed)
    model = TinyMLP(x_fit.size(1), hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    best_loss, best_state, best_step = float("inf"), None, 0
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        loss = F.mse_loss(model(xtr), ytr)
        loss.backward()
        opt.step()
        sched.step()

        if step % check_every == 0 or step == max_steps - 1:
            model.eval()
            with torch.no_grad():
                mon_loss = F.mse_loss(model(xmon), ymon).item()
            if mon_loss < best_loss:
                best_loss, best_state, best_step = mon_loss, copy.deepcopy(model.state_dict()), step
            elif step >= min_steps and step - best_step >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, xm, xs, ym, ys


def mlp_multiseed(x_fit: torch.Tensor, y_fit: torch.Tensor, x_eval: torch.Tensor, y_eval: torch.Tensor,
                   seeds=(0, 1, 2, 3, 4), hidden: int = 128, dropout: float = 0.1, lr: float = 1e-2,
                   weight_decay: float = 1e-4, min_steps: int = 150, max_steps: int = 4000,
                   patience: int = 100) -> dict:
    """Fit N independently-seeded TinyMLPs on (x_fit, y_fit) with early
    stopping (see _train_mlp_early_stop), score each on (x_eval, y_eval),
    return {"mean": float, "std": float, "per_seed": [float, ...]}.

    Same function serves two roles depending on what's passed: (train, valid)
    for layer selection, (train, test) for final reporting -- the two
    numbers are produced by identical code, just different data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_eval, y_eval = x_eval.to(device), y_eval.to(device)
    r2s = []
    for seed in seeds:
        model, xm, xs, ym, ys = _train_mlp_early_stop(
            x_fit, y_fit, hidden=hidden, dropout=dropout, lr=lr, weight_decay=weight_decay,
            seed=seed, min_steps=min_steps, max_steps=max_steps, patience=patience,
        )
        xev = (x_eval - xm) / xs
        yev = (y_eval - ym) / ys
        with torch.no_grad():
            pred = model(xev)
        ss_res = ((pred - yev) ** 2).sum()
        ss_tot = ((yev - yev.mean()) ** 2).sum()
        r2s.append((1 - ss_res / ss_tot).item())
    r2_t = torch.tensor(r2s)
    return {"mean": r2_t.mean().item(), "std": r2_t.std().item(), "per_seed": r2s}


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


def analyze_token_mlp_level(ckpt_path: str, clips: torch.Tensor, contemp: dict, max_clips: int,
                             dataset: str = "active_matter"):
    """Per-token nonlinear (small MLP) probe: identical features/targets to
    analyze_token_level's per-token LINEAR probe, swapping ridge_r2 for
    mlp_r2 as the readout. analyze_mlp_level's pooled MLP was flagged
    noisy/unreliable — per arXiv:2602.07050 (attentive/per-token probes
    recover structure that pooled-then-nonlinear probes miss because
    mean-pooling destroys the local structure a small MLP needs), this tests
    whether that was a pooling artifact rather than a real absence of
    nonlinear-only signal at the token level."""
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
        x_flat = feats.reshape(-1, d)
        layer_r2 = {}
        for tname, y in targets.items():
            y_flat = y.reshape(-1)
            layer_r2[tname] = mlp_r2(x_flat, y_flat, seed=layer_idx)
        results["layers"].append(layer_r2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="active_matter")
    ap.add_argument("--n-offsets", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--max-traj", type=int, default=None,
                     help="cap trajectories used (evenly subsampled) — see analyze_encoders.py's "
                          "build_dataset() docstring; default uses every trajectory")
    ap.add_argument("--token-max-clips", type=int, default=64,
                     help="clips used for per-token linear probe (each yields ~1024 token-samples)")
    ap.add_argument("--token-mlp-max-clips", type=int, default=8,
                     help="clips used for per-token MLP probe (more expensive than linear: "
                          "2000 Adam steps x n_targets x n_layers, so a smaller default)")
    ap.add_argument("--skip-mlp", action="store_true",
                     help="skip both nonlinear MLP probes (pooled + token)")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"],
                     help="which preprocessed memmap split to probe (default train; use valid for a "
                          "held-out generalization check against the pretraining data)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"building probe dataset ({args.split})...")
    clips, contemp, _future = build_dataset(args.data_root, args.dataset, args.n_offsets, args.horizon,
                                             max_trajectories=args.max_traj, split=args.split)
    print(f"{clips.size(0)} clips total; token probe uses first {args.token_max_clips}")

    all_results = {}
    for ckpt_path in args.checkpoints:
        print(f"\n=== {ckpt_path} ===")
        tok_res = analyze_token_level(ckpt_path, clips, contemp, args.token_max_clips, dataset=args.dataset)
        all_results[ckpt_path] = {"token_linear": tok_res}

        print("  -- per-token LINEAR probe (no pooling) --")
        for i, layer_r2 in enumerate(tok_res["layers"]):
            tag = f"block_{i+1}" if i < tok_res["n_layers"] - 1 else "final_norm"
            line = "  ".join(f"{k}={v:.3f}" for k, v in layer_r2.items())
            print(f"    {tag:12s} {line}")

        if args.skip_mlp:
            continue
        mlp_res = analyze_mlp_level(ckpt_path, clips, contemp)
        all_results[ckpt_path]["mlp_pooled"] = mlp_res
        print("  -- pooled NONLINEAR (MLP) probe --")
        for i, layer_r2 in enumerate(mlp_res["layers"]):
            tag = f"block_{i+1}" if i < mlp_res["n_layers"] - 1 else "final_norm"
            line = "  ".join(f"{k}={v:.3f}" for k, v in layer_r2.items())
            print(f"    {tag:12s} {line}")

        tok_mlp_res = analyze_token_mlp_level(
            ckpt_path, clips, contemp, args.token_mlp_max_clips, dataset=args.dataset
        )
        all_results[ckpt_path]["mlp_token"] = tok_mlp_res
        print("  -- per-token NONLINEAR (MLP) probe --")
        for i, layer_r2 in enumerate(tok_mlp_res["layers"]):
            tag = f"block_{i+1}" if i < tok_mlp_res["n_layers"] - 1 else "final_norm"
            line = "  ".join(f"{k}={v:.3f}" for k, v in layer_r2.items())
            print(f"    {tag:12s} {line}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""End-of-run acceptance check: ridge-probe frozen encoder features.

Predicts a per-clip scalar (mean squared field magnitude, an energy proxy)
from mean-pooled frozen encoder features on the valid split, and compares
against a baseline probe on per-channel clip means (the "pixel-mean" baseline).
A healthy encoder should clearly beat the baseline R^2.

Usage:
    uv run python scripts/linear_probe.py runs/active_matter_jepa/encoder_100pct.pt \
        --data-root /workspace/data [--n-clips 512] [--split valid]
"""

import argparse
import os

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.well import ClipSpec, WellClipDataset
from src.models.vit import build_encoder


def ridge_r2(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-3):
    """Closed-form ridge regression, 5-fold CV, returns mean R^2."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n-clips", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg, spec = ckpt["config"], ClipSpec(**ckpt["spec"])
    encoder = build_encoder(spec, cfg.get("encoder", {})).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])

    dataset = WellClipDataset(
        base_path=os.path.expanduser(args.data_root),
        dataset_name=cfg["data"]["dataset_name"],
        split=args.split,
        n_frames=cfg["data"].get("n_frames", 8),
    )
    n = min(args.n_clips, len(dataset))
    idx = torch.linspace(0, len(dataset) - 1, n).long().tolist()

    feats, pix_means, targets = [], [], []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            clips = torch.stack(
                [dataset[i]["clip"] for i in idx[start : start + args.batch_size]]
            ).to(device)
            f = encoder(clips)  # (B, N, D), no masking
            feats.append(f.mean(dim=1).float().cpu())
            pix_means.append(clips.mean(dim=(2, 3, 4)).float().cpu())
            targets.append((clips**2).mean(dim=(1, 2, 3, 4)).float().cpu())

    x = torch.cat(feats)
    xb = torch.cat(pix_means)
    y = torch.cat(targets)

    r2_encoder = ridge_r2(x, y)
    r2_baseline = ridge_r2(xb, y)
    print(f"probe target: mean squared field magnitude ({n} clips, {args.split})")
    print(f"encoder features R^2:   {r2_encoder:.4f}  (dim {x.size(1)})")
    print(f"pixel-mean baseline R^2: {r2_baseline:.4f}  (dim {xb.size(1)})")
    verdict = "PASS" if r2_encoder > r2_baseline else "FAIL"
    print(f"acceptance: {verdict}")


if __name__ == "__main__":
    main()

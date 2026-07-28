"""Does the encoder's pooled representation know what physical regime it's
looking at? (Reynolds/Schmidt for shear_flow, Rayleigh/Prandtl for
rayleigh_benard, alpha/zeta for active_matter — regime is constant across an
entire trajectory, extracted per-file by extract_regime_metadata.py.)

One feature vector per TRAJECTORY (mean-pooled over a few sampled offsets),
matched 1:1 with one regime value per trajectory — critical for avoiding
leakage: multiple windows from the same trajectory must never land in both
train and val folds of the ridge probe, since they'd share an (almost)
identical target. Reuses compute_layerwise_features_batched() (same forward
passes already paid for elsewhere), so this adds zero extra encoder cost
beyond sampling clips.

Log-transforms Re/Ra/Sc/Pr (span multiple orders of magnitude); regresses
active_matter's alpha/zeta raw. Each target also gets a shuffled-control
column (same features, permuted target) as a leakage sanity check — this
should collapse toward R^2~=0.

Usage:
    uv run python scripts/analyze_regime.py \
        --checkpoints local_runs/shear_flow_jepa/encoder_100pct.pt local_runs/shear_flow_mae/encoder_100pct.pt \
        --data-root /workspace/data --dataset shear_flow
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_encoders import (
    compute_layerwise_features_batched, load_checkpoint_encoder, ridge_r2,
)

LOG_TRANSFORM = {"Reynolds", "Schmidt", "Rayleigh", "Prandtl"}


def build_regime_dataset(base: str, dataset: str, split: str, n_offsets: int, n_frames: int = 8):
    d = Path(base) / "memmap" / dataset
    mm = np.load(d / f"{split}.npy", mmap_mode="r")
    regime_path = d / f"{split}.regime.json"
    if not regime_path.exists():
        raise FileNotFoundError(f"{regime_path} missing — run extract_regime_metadata.py first")
    records = json.loads(regime_path.read_text())
    n_traj, c, t, h, w = mm.shape
    assert len(records) == n_traj, f"{len(records)} regime records != {n_traj} trajectories"

    max_off = t - n_frames
    offsets = np.linspace(0, max_off, n_offsets).astype(int)

    param_names = [k for k in records[0].keys() if k != "file"]
    traj_clips = []
    keep_traj = []
    for traj in range(n_traj):
        if any(records[traj][p] is None for p in param_names):
            continue  # skip trajectories whose filename didn't match the regex
        windows = [torch.from_numpy(np.array(mm[traj, :, off:off + n_frames])).float() for off in offsets]
        traj_clips.append(torch.stack(windows))  # (n_offsets, C, T, H, W)
        keep_traj.append(traj)

    targets = {}
    for p in param_names:
        vals = np.array([records[traj][p] for traj in keep_traj], dtype=np.float64)
        if p in LOG_TRANSFORM:
            vals = np.log10(vals)
        targets[p] = torch.from_numpy(vals).float()

    return traj_clips, targets, param_names


def analyze_regime_checkpoint(ckpt_path: str, traj_clips: list, targets: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _spec = load_checkpoint_encoder(ckpt_path, device)

    n_traj = len(traj_clips)
    n_offsets = traj_clips[0].size(0)
    flat = torch.cat(traj_clips, dim=0)  # (n_traj*n_offsets, C, T, H, W)
    per_layer_feats = compute_layerwise_features_batched(encoder, flat)
    n_layers = len(per_layer_feats)

    # average the n_offsets windows back down to one feature vector per trajectory
    per_layer_traj_feats = [
        feats.view(n_traj, n_offsets, -1).mean(dim=1) for feats in per_layer_feats
    ]

    results = {"n_layers": n_layers, "n_trajectories": n_traj, "layers": []}
    for feats in per_layer_traj_feats:
        layer_r2 = {}
        for tname, y in targets.items():
            layer_r2[tname] = ridge_r2(feats, y)
            perm = torch.randperm(y.size(0), generator=torch.Generator().manual_seed(0))
            layer_r2[f"{tname}_shuffled_control"] = ridge_r2(feats, y[perm])
        results["layers"].append(layer_r2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-offsets", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("building regime probe dataset...")
    traj_clips, targets, param_names = build_regime_dataset(
        args.data_root, args.dataset, args.split, args.n_offsets
    )
    print(f"{len(traj_clips)} trajectories with matched regime params: {param_names}")
    for k, v in targets.items():
        print(f"  {k}: mean={v.mean():.3f} std={v.std():.3f} (log10-transformed: {k in LOG_TRANSFORM})")

    all_results = {}
    for ckpt_path in args.checkpoints:
        print(f"\n=== {ckpt_path} ===")
        res = analyze_regime_checkpoint(ckpt_path, traj_clips, targets)
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

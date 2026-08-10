"""Representation-stability probe: does the frozen encoder's physics
decodability degrade gracefully under Gaussian noise injected into the INPUT
physical variables, or does it collapse abruptly?

Same probe machinery as analyze_encoders.py (final-layer pooled features,
ridge R^2 against derived physics targets) but swept over injected input
noise levels instead of layer depth. Targets are always computed on the
CLEAN clip (the true underlying physics never changes) — only the encoder's
input is corrupted, so a dropping R^2 curve reflects the representation
losing track of real physics under noise, not the target itself moving.

Noise is added in normalized input space (channels are already per-channel
z-score normalized to ~unit variance, so --noise-stds are directly
comparable to that unit scale, e.g. 0.5 means half a channel standard
deviation of injected noise).

Deliberately does NOT use analyze_encoders.py's build_dataset(): that helper
materializes every sampled clip into one big stacked tensor and then runs
contemporaneous_targets() on the whole thing at once, which for shear_flow
(2688 samples) and rayleigh_benard (4200 samples) exceeded this project's
pod memory limits and got silently OOM-killed with no output. Instead, each
chunk is read from the memmap exactly ONCE and immediately used for the
physics targets AND every (checkpoint, noise level) encoding pass, rather
than re-reading it from the slow network-mounted memmap once per pass — the
data loop is the outer loop, checkpoints/noise levels are inner — so peak
memory is independent of sample count and total disk I/O is independent of
how many checkpoints/noise levels are swept.

Usage:
    uv run python scripts/analyze_noise_robustness.py \
        --checkpoints checkpoints/active_matter_jepa.pt checkpoints/active_matter_mae.pt \
        --data-root /workspace/data --dataset active_matter \
        --noise-stds 0 0.1 0.25 0.5 1.0 2.0 \
        --out sweep_results/active_matter_noise.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_encoders import contemporaneous_targets, load_checkpoint_encoder, ridge_r2

N_FRAMES = 8


def sample_seeds(mm: np.ndarray, n_offsets: int, n_frames: int = N_FRAMES) -> list:
    n_traj, _, t, _, _ = mm.shape
    max_off = t - n_frames
    offsets = np.linspace(0, max_off, n_offsets).astype(int)
    return [(traj, off) for traj in range(n_traj) for off in offsets]


def load_batch(mm: np.ndarray, seeds: list, n_frames: int = N_FRAMES) -> torch.Tensor:
    clips = [torch.from_numpy(np.array(mm[traj, :, off : off + n_frames])).float() for traj, off in seeds]
    return torch.stack(clips)


@torch.no_grad()
def run_sweep(mm: np.ndarray, seeds: list, dataset: str, encoders: dict, noise_stds: list,
              batch_size: int = 16) -> tuple[dict, dict]:
    """Single pass over the data: each batch is loaded from the memmap once
    and reused for the physics targets and every (checkpoint, noise level)
    combination. Returns (targets, feats) where feats[ckpt_path][str(std)] is
    the pooled feature tensor for that checkpoint/noise level."""
    targets_acc: dict = {}
    feats_acc = {name: {std: [] for std in noise_stds} for name in encoders}

    for start in range(0, len(seeds), batch_size):
        clip = load_batch(mm, seeds[start : start + batch_size])
        for k, v in contemporaneous_targets(clip, dataset=dataset).items():
            targets_acc.setdefault(k, []).append(v)

        for name, encoder in encoders.items():
            device = next(encoder.parameters()).device
            batch = clip.to(device)
            for std in noise_stds:
                noisy = batch + std * torch.randn_like(batch) if std > 0 else batch
                feats_acc[name][std].append(encoder(noisy).mean(dim=1).float().cpu())

    targets = {k: torch.cat(v) for k, v in targets_acc.items()}
    feats = {name: {str(std): torch.cat(v) for std, v in per_std.items()} for name, per_std in feats_acc.items()}
    return targets, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="active_matter")
    ap.add_argument("--n-offsets", type=int, default=3, help="windows per trajectory")
    ap.add_argument("--noise-stds", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1.0, 2.0],
                     help="Gaussian noise std added to z-score-normalized input clips")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm = np.load(Path(args.data_root) / "memmap" / args.dataset / "train.npy", mmap_mode="r")
    seeds = sample_seeds(mm, args.n_offsets)
    print(f"{len(seeds)} samples total")

    encoders = {ckpt: load_checkpoint_encoder(ckpt, device)[0] for ckpt in args.checkpoints}
    targets, feats = run_sweep(mm, seeds, args.dataset, encoders, args.noise_stds)
    for k, v in targets.items():
        print(f"  {k}: mean={v.mean():.4f} std={v.std():.4f}")

    all_results = {}
    for ckpt_path in args.checkpoints:
        print(f"\n=== {ckpt_path} ===")
        results = {}
        for std in args.noise_stds:
            f = feats[ckpt_path][str(std)]
            results[str(std)] = {tname: ridge_r2(f, y) for tname, y in targets.items()}
        all_results[ckpt_path] = results
        for std_str, r2s in results.items():
            line = "  ".join(f"{k}={v:.3f}" for k, v in r2s.items())
            print(f"  noise_std={std_str:>6s}  {line}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

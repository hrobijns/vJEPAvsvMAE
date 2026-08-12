"""Representation-stability probe: does the frozen encoder's physics
decodability degrade gracefully under Gaussian noise injected into the INPUT
physical variables, or does it collapse abruptly?

Same probe machinery as analyze_encoders.py (ridge R^2 against derived
physics targets) but swept over BOTH injected input noise level AND layer
depth (all blocks + final norm), producing a 2D noise x layer grid per
quantity — not just a final-layer noise curve — so robustness can be read
off at any depth, e.g. "is there an intermediate layer whose representation
degrades more gracefully than the output?" Targets are always computed on
the CLEAN clip (the true underlying physics never changes) — only the
encoder's input is corrupted, so a dropping R^2 curve reflects the
representation losing track of real physics under noise, not the target
itself moving.

Both pooled AND per-token variants are computed. The per-token variant is
capped to the first `--token-max-samples` clips (token tensors are ~n_tokens
x larger per sample) — for those capped batches, the per-token features are
computed once via layerwise_token_features() and the pooled features are
derived by mean-pooling them (rather than a second, separate forward pass),
so every batch still costs exactly one forward pass per (checkpoint, noise
level).

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

from scripts.analyze_encoders import (
    contemporaneous_targets, layerwise_features, load_checkpoint_encoder, ridge_r2,
)
from scripts.analyze_encoders_local import layerwise_token_features, local_target_maps

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
              batch_size: int = 16, token_max_samples: int = 64) -> tuple[dict, dict, dict, dict]:
    """Single pass over the data: each batch is loaded from the memmap once
    and reused for the physics targets and every (checkpoint, noise level)
    combination. Returns (targets, feats, tok_targets, tok_feats):
      - feats[ckpt_path][str(std)]: LIST of per-layer POOLED feature tensors
        (one per block + final norm), for the full sample set.
      - tok_feats[ckpt_path][str(std)]: LIST of per-layer PER-TOKEN feature
        tensors (b, N, D), capped to the first `token_max_samples` clips.
    For capped batches, per-token features are computed once (encoder called
    once per checkpoint/noise-level) and pooled features are derived by
    mean-pooling them, so token-level collection never doubles the forward-
    pass cost."""
    targets_acc: dict = {}
    tok_targets_acc: dict = {}
    feats_acc = {name: {std: None for std in noise_stds} for name in encoders}
    tok_feats_acc = {name: {std: None for std in noise_stds} for name in encoders}
    patch_size = next(iter(encoders.values())).patch_size
    n_tok_seen = 0

    for start in range(0, len(seeds), batch_size):
        clip = load_batch(mm, seeds[start : start + batch_size])
        use_token = n_tok_seen < token_max_samples
        for k, v in contemporaneous_targets(clip, dataset=dataset).items():
            targets_acc.setdefault(k, []).append(v)
        if use_token:
            for k, v in local_target_maps(clip, *patch_size, dataset=dataset).items():
                tok_targets_acc.setdefault(k, []).append(v)

        for name, encoder in encoders.items():
            device = next(encoder.parameters()).device
            batch = clip.to(device)
            for std in noise_stds:
                noisy = batch + std * torch.randn_like(batch) if std > 0 else batch
                if use_token:
                    tok_layer_feats = layerwise_token_features(encoder, noisy)  # list of (b, N, D)
                    layer_feats = [f.mean(dim=1) for f in tok_layer_feats]
                    if tok_feats_acc[name][std] is None:
                        tok_feats_acc[name][std] = [[] for _ in tok_layer_feats]
                    for i, f in enumerate(tok_layer_feats):
                        tok_feats_acc[name][std][i].append(f)
                else:
                    layer_feats = layerwise_features(encoder, noisy)  # list of (b, D)
                if feats_acc[name][std] is None:
                    feats_acc[name][std] = [[] for _ in layer_feats]
                for i, f in enumerate(layer_feats):
                    feats_acc[name][std][i].append(f)

        if use_token:
            n_tok_seen += clip.size(0)

    targets = {k: torch.cat(v) for k, v in targets_acc.items()}
    tok_targets = {k: torch.cat(v) for k, v in tok_targets_acc.items()}
    feats = {
        name: {str(std): [torch.cat(fs) for fs in per_layer] for std, per_layer in per_std.items()}
        for name, per_std in feats_acc.items()
    }
    tok_feats = {
        name: {str(std): [torch.cat(fs) for fs in per_layer] for std, per_layer in per_std.items()}
        for name, per_std in tok_feats_acc.items()
    }
    return targets, feats, tok_targets, tok_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="active_matter")
    ap.add_argument("--n-offsets", type=int, default=3, help="windows per trajectory")
    ap.add_argument("--noise-stds", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1.0, 2.0],
                     help="Gaussian noise std added to z-score-normalized input clips")
    ap.add_argument("--token-max-samples", type=int, default=16,
                     help="clips used for the per-token noise-robustness variant "
                          "(token tensors are ~n_tokens x larger than pooled, and this "
                          "script keeps them uncollapsed across every checkpoint x noise-level "
                          "combo until the end, so memory scales fast with this value — "
                          "keep it modest on memory-constrained pods)")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"],
                     help="which preprocessed memmap split to probe (default train; use valid for a "
                          "held-out generalization check against the pretraining data)")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm = np.load(Path(args.data_root) / "memmap" / args.dataset / f"{args.split}.npy", mmap_mode="r")
    seeds = sample_seeds(mm, args.n_offsets)
    print(f"{len(seeds)} samples total")

    encoders = {ckpt: load_checkpoint_encoder(ckpt, device)[0] for ckpt in args.checkpoints}
    targets, feats, tok_targets, tok_feats = run_sweep(
        mm, seeds, args.dataset, encoders, args.noise_stds, token_max_samples=args.token_max_samples
    )
    for k, v in targets.items():
        print(f"  {k}: mean={v.mean():.4f} std={v.std():.4f}")

    all_results = {}
    for ckpt_path in args.checkpoints:
        print(f"\n=== {ckpt_path} ===")
        by_noise = {}
        for std in args.noise_stds:
            layer_feats = feats[ckpt_path][str(std)]
            by_noise[str(std)] = [
                {tname: ridge_r2(f, y) for tname, y in targets.items()} for f in layer_feats
            ]
        n_layers = len(next(iter(by_noise.values())))
        for std_str, layers in by_noise.items():
            for i, r2s in enumerate(layers):
                tag = f"block_{i+1}" if i < n_layers - 1 else "final_norm"
                line = "  ".join(f"{k}={v:.3f}" for k, v in r2s.items())
                print(f"  noise_std={std_str:>6s}  {tag:12s} {line}")

        tok_by_noise = {}
        for std in args.noise_stds:
            tok_layer_feats = tok_feats[ckpt_path][str(std)]
            tok_by_noise[str(std)] = [
                {tname: ridge_r2(f.reshape(-1, f.size(-1)), y.reshape(-1))
                 for tname, y in tok_targets.items()}
                for f in tok_layer_feats
            ]
        print("  -- per-token --")
        for std_str, layers in tok_by_noise.items():
            for i, r2s in enumerate(layers):
                tag = f"block_{i+1}" if i < n_layers - 1 else "final_norm"
                line = "  ".join(f"{k}={v:.3f}" for k, v in r2s.items())
                print(f"    noise_std={std_str:>6s}  {tag:12s} {line}")

        all_results[ckpt_path] = {
            "n_layers": n_layers, "noise_stds": args.noise_stds, "by_noise": by_noise,
            "token": {"n_token_samples": args.token_max_samples, "by_noise": tok_by_noise},
        }

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

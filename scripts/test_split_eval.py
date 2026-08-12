"""Held-out **test**-split evaluation: freezes layer choice from the
existing **train**-CV sweep results, then fits probes on train and evaluates
ONCE on test trajectories that the encoder never pretrained on and the probe
never used for any selection. This is the fix for a specific rigor gap in
the rest of the suite: `analyze_encoders.py`/`analyze_encoders_local.py`
report "best-layer R^2" by taking argmax over 13 layers of train-CV curves
and reporting that max directly — an honest per-layer estimate, but the
argmax-over-13 operation is itself a form of selection that isn't validated
against independent data. Here, the layer (and, for noise robustness, the
final layer already used elsewhere) is chosen from train-CV *before* any
test data is touched, then frozen.

Three sections, matching the three headline analyses this was scoped to
(see docs/LINEAR_PROBE.md's "Held-out test-split evaluation"):
1. Pooled contemporaneous physics (linear + MLP).
2. Per-token physics (linear + MLP) — where `rayleigh_benard`'s coupled-
   quantity JEPA/MAE gap lives.
3. Noise robustness at the fixed final layer (no layer search happens for
   this metric anywhere in the suite, so no freezing step is needed here —
   only the fit-train/eval-test split itself is new).

Does NOT retrain or fine-tune any encoder — frozen checkpoints only, same
ones used everywhere else in the suite.

Usage:
    uv run python scripts/test_split_eval.py \
        --checkpoints checkpoints/active_matter_jepa.pt checkpoints/active_matter_mae.pt \
        --data-root /workspace/data --dataset active_matter \
        --train-sweep sweep_results/active_matter_pooled.json \
        --train-sweep-token sweep_results/active_matter_nonpooled.json \
        --out sweep_results/active_matter_test_eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_encoders import (
    contemporaneous_targets, layerwise_features, load_checkpoint_encoder, ridge_fit_eval,
)
from scripts.analyze_encoders_local import layerwise_token_features, local_target_maps, mlp_fit_eval

N_FRAMES = 8


def sample_seeds(mm: np.ndarray, n_offsets: int, n_frames: int = N_FRAMES,
                  max_trajectories: int | None = None) -> list:
    n_traj, _, t, _, _ = mm.shape
    max_off = t - n_frames
    offsets = np.linspace(0, max_off, n_offsets).astype(int)
    trajectories = list(range(n_traj))
    if max_trajectories is not None and max_trajectories < n_traj:
        idx = np.linspace(0, n_traj - 1, max_trajectories).astype(int)
        trajectories = sorted(set(idx.tolist()))
    return [(traj, off) for traj in trajectories for off in offsets]


def load_batch(mm: np.ndarray, seeds: list, n_frames: int = N_FRAMES) -> torch.Tensor:
    clips = [torch.from_numpy(np.array(mm[traj, :, off : off + n_frames])).float() for traj, off in seeds]
    return torch.stack(clips)


@torch.no_grad()
def collect_pooled(mm: np.ndarray, seeds: list, dataset: str, encoders: dict,
                    batch_size: int = 16) -> tuple[dict, dict]:
    """One pass over `seeds`: returns (targets, feats) where targets maps
    quantity name -> (N,) tensor and feats maps ckpt_path -> list of
    per-layer (N, D) pooled feature tensors."""
    targets_acc: dict = {}
    feats_acc = {name: None for name in encoders}
    for start in range(0, len(seeds), batch_size):
        clip = load_batch(mm, seeds[start : start + batch_size])
        for k, v in contemporaneous_targets(clip, dataset=dataset).items():
            targets_acc.setdefault(k, []).append(v)
        for name, encoder in encoders.items():
            device = next(encoder.parameters()).device
            layer_feats = layerwise_features(encoder, clip.to(device))
            if feats_acc[name] is None:
                feats_acc[name] = [[] for _ in layer_feats]
            for i, f in enumerate(layer_feats):
                feats_acc[name][i].append(f)
    targets = {k: torch.cat(v) for k, v in targets_acc.items()}
    feats = {name: [torch.cat(fs) for fs in per_layer] for name, per_layer in feats_acc.items()}
    return targets, feats


@torch.no_grad()
def collect_token(mm: np.ndarray, seeds: list, dataset: str, encoders: dict, patch_size: tuple,
                   batch_size: int = 8) -> tuple[dict, dict]:
    """Same as collect_pooled but per-token: targets map quantity ->
    (n_clips, n_tok) tensor, feats map ckpt_path -> list of per-layer
    (n_clips, n_tok, D) tensors."""
    targets_acc: dict = {}
    feats_acc = {name: None for name in encoders}
    for start in range(0, len(seeds), batch_size):
        clip = load_batch(mm, seeds[start : start + batch_size])
        for k, v in local_target_maps(clip, *patch_size, dataset=dataset).items():
            targets_acc.setdefault(k, []).append(v)
        for name, encoder in encoders.items():
            device = next(encoder.parameters()).device
            layer_feats = layerwise_token_features(encoder, clip.to(device))  # list of (b, N, D)
            if feats_acc[name] is None:
                feats_acc[name] = [[] for _ in layer_feats]
            for i, f in enumerate(layer_feats):
                feats_acc[name][i].append(f)
    targets = {k: torch.cat(v) for k, v in targets_acc.items()}
    feats = {name: [torch.cat(fs) for fs in per_layer] for name, per_layer in feats_acc.items()}
    return targets, feats


@torch.no_grad()
def collect_pooled_noise(mm: np.ndarray, seeds: list, dataset: str, encoders: dict, noise_stds: list,
                          batch_size: int = 16) -> tuple[dict, dict]:
    """Like collect_pooled but sweeps injected input noise; targets are
    always computed on the CLEAN clip (only the encoder's input is
    corrupted)."""
    targets_acc: dict = {}
    feats_acc = {name: {std: None for std in noise_stds} for name in encoders}
    for start in range(0, len(seeds), batch_size):
        clip = load_batch(mm, seeds[start : start + batch_size])
        for k, v in contemporaneous_targets(clip, dataset=dataset).items():
            targets_acc.setdefault(k, []).append(v)
        for name, encoder in encoders.items():
            device = next(encoder.parameters()).device
            batch = clip.to(device)
            for std in noise_stds:
                noisy = batch + std * torch.randn_like(batch) if std > 0 else batch
                layer_feats = layerwise_features(encoder, noisy)
                if feats_acc[name][std] is None:
                    feats_acc[name][std] = [[] for _ in layer_feats]
                for i, f in enumerate(layer_feats):
                    feats_acc[name][std][i].append(f)
    targets = {k: torch.cat(v) for k, v in targets_acc.items()}
    feats = {
        name: {str(std): [torch.cat(fs) for fs in per_layer] for std, per_layer in per_std.items()}
        for name, per_std in feats_acc.items()
    }
    return targets, feats


def frozen_best_layers_pooled(sweep_json_path: str, ckpt_names: list) -> dict:
    """From analyze_encoders.py's pooled train-CV sweep JSON
    ({ckpt: {"layers": [{quantity: r2, ...}, ...]}}), returns
    {ckpt_name: {quantity: best_layer_idx}} via argmax over the train-CV
    curve — frozen before any test data is touched."""
    d = json.loads(Path(sweep_json_path).read_text())
    out = {}
    for ckpt_path, res in d.items():
        name = next(n for n in ckpt_names if n in ckpt_path)
        layers = res["layers"]
        quantities = [q for q in layers[0].keys() if "shuffled" not in q]
        out[name] = {q: max(range(len(layers)), key=lambda i: layers[i][q]) for q in quantities}
    return out


def frozen_best_layers_token(sweep_json_path: str, ckpt_names: list) -> dict:
    """Same idea, from analyze_encoders_local.py's nonpooled sweep JSON's
    `token_linear` curve (used to select the layer for both the linear and
    the MLP token-level test-eval, for simplicity/consistency)."""
    d = json.loads(Path(sweep_json_path).read_text())
    out = {}
    for ckpt_path, res in d.items():
        name = next(n for n in ckpt_names if n in ckpt_path)
        layers = res["token_linear"]["layers"]
        quantities = list(layers[0].keys())
        out[name] = {q: max(range(len(layers)), key=lambda i: layers[i][q]) for q in quantities}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-offsets", type=int, default=3)
    ap.add_argument("--train-max-traj", type=int, default=None)
    ap.add_argument("--test-max-traj", type=int, default=None)
    ap.add_argument("--token-max-clips", type=int, default=16,
                     help="clips used for the token-level train/test fit+eval (both linear and MLP)")
    ap.add_argument("--noise-stds", type=float, nargs="+", default=[0, 0.5, 1.0, 2.0])
    ap.add_argument("--train-sweep", required=True,
                     help="existing train-CV pooled sweep JSON, for frozen best-layer selection")
    ap.add_argument("--train-sweep-token", default=None,
                     help="existing train-CV token sweep JSON (nonpooled.json), for frozen token-layer selection")
    ap.add_argument("--skip-token", action="store_true")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = {}
    for ckpt_path in args.checkpoints:
        enc, _spec = load_checkpoint_encoder(ckpt_path, device)
        encoders[ckpt_path] = enc
    ckpt_names = list(encoders.keys())
    patch_size = next(iter(encoders.values())).patch_size

    mm_train = np.load(Path(args.data_root) / "memmap" / args.dataset / "train.npy", mmap_mode="r")
    mm_test = np.load(Path(args.data_root) / "memmap" / args.dataset / "test.npy", mmap_mode="r")
    print(f"{args.dataset}: train={mm_train.shape[0]} traj, test={mm_test.shape[0]} traj", flush=True)

    results: dict = {"n_train_traj": int(mm_train.shape[0]), "n_test_traj": int(mm_test.shape[0])}

    # ---- 1. Pooled contemporaneous physics ----
    seeds_train = sample_seeds(mm_train, args.n_offsets, max_trajectories=args.train_max_traj)
    seeds_test = sample_seeds(mm_test, args.n_offsets, max_trajectories=args.test_max_traj)
    print(f"pooled: n_train={len(seeds_train)} n_test={len(seeds_test)}", flush=True)

    targets_train, feats_train = collect_pooled(mm_train, seeds_train, args.dataset, encoders)
    targets_test, feats_test = collect_pooled(mm_test, seeds_test, args.dataset, encoders)
    frozen_layers = frozen_best_layers_pooled(args.train_sweep, ckpt_names)

    pooled_out = {"n_samples_train": len(seeds_train), "n_samples_test": len(seeds_test)}
    for ckpt_path in ckpt_names:
        per_q = {}
        for q, layer_idx in frozen_layers[ckpt_path].items():
            if q not in targets_train or q not in targets_test:
                continue
            xtr, xte = feats_train[ckpt_path][layer_idx], feats_test[ckpt_path][layer_idx]
            ytr, yte = targets_train[q], targets_test[q]
            entry = {"frozen_layer": layer_idx, "test_r2_linear": ridge_fit_eval(xtr, ytr, xte, yte)}
            if not args.skip_mlp:
                entry["test_r2_mlp"] = mlp_fit_eval(xtr, ytr, xte, yte)
            per_q[q] = entry
        pooled_out[ckpt_path] = per_q
        print(f"  {ckpt_path}: " + "  ".join(f"{q}={v['test_r2_linear']:.3f}(L{v['frozen_layer']})" for q, v in per_q.items()), flush=True)
    results["pooled"] = pooled_out

    # ---- 2. Per-token physics ----
    if not args.skip_token and args.train_sweep_token:
        tok_seeds_train = seeds_train[: args.token_max_clips]
        tok_seeds_test = seeds_test[: args.token_max_clips]
        print(f"token: n_train_clips={len(tok_seeds_train)} n_test_clips={len(tok_seeds_test)}", flush=True)
        ttargets_train, tfeats_train = collect_token(mm_train, tok_seeds_train, args.dataset, encoders, patch_size)
        ttargets_test, tfeats_test = collect_token(mm_test, tok_seeds_test, args.dataset, encoders, patch_size)
        frozen_tok_layers = frozen_best_layers_token(args.train_sweep_token, ckpt_names)

        token_out = {"n_clips_train": len(tok_seeds_train), "n_clips_test": len(tok_seeds_test)}
        for ckpt_path in ckpt_names:
            per_q = {}
            for q, layer_idx in frozen_tok_layers[ckpt_path].items():
                if q not in ttargets_train or q not in ttargets_test:
                    continue
                xtr = tfeats_train[ckpt_path][layer_idx].reshape(-1, tfeats_train[ckpt_path][layer_idx].size(-1))
                xte = tfeats_test[ckpt_path][layer_idx].reshape(-1, tfeats_test[ckpt_path][layer_idx].size(-1))
                ytr = ttargets_train[q].reshape(-1)
                yte = ttargets_test[q].reshape(-1)
                entry = {"frozen_layer": layer_idx, "test_r2_linear": ridge_fit_eval(xtr, ytr, xte, yte)}
                if not args.skip_mlp:
                    entry["test_r2_mlp"] = mlp_fit_eval(xtr, ytr, xte, yte)
                per_q[q] = entry
            token_out[ckpt_path] = per_q
            print(f"  {ckpt_path}: " + "  ".join(f"{q}={v['test_r2_linear']:.3f}(L{v['frozen_layer']})" for q, v in per_q.items()), flush=True)
        results["token"] = token_out

    # ---- 3. Noise robustness (fixed final layer, no selection needed) ----
    if not args.skip_noise:
        print(f"noise: stds={args.noise_stds}", flush=True)
        ntargets_train, nfeats_train = collect_pooled_noise(mm_train, seeds_train, args.dataset, encoders, args.noise_stds)
        ntargets_test, nfeats_test = collect_pooled_noise(mm_test, seeds_test, args.dataset, encoders, args.noise_stds)

        noise_out = {"noise_stds": args.noise_stds}
        for ckpt_path in ckpt_names:
            by_std = {}
            n_layers = len(next(iter(nfeats_train[ckpt_path].values())))
            final_layer = n_layers - 1
            for std in args.noise_stds:
                skey = str(std)
                xtr = nfeats_train[ckpt_path][skey][final_layer]
                xte = nfeats_test[ckpt_path][skey][final_layer]
                per_q = {}
                for q in ntargets_train:
                    if q not in ntargets_test:
                        continue
                    per_q[q] = ridge_fit_eval(xtr, ntargets_train[q], xte, ntargets_test[q])
                by_std[skey] = per_q
            noise_out[ckpt_path] = {"final_layer": final_layer, "by_noise": by_std}
            print(f"  {ckpt_path} (L{final_layer}): " + "  ".join(
                f"std={s}:{sum(by_std[str(s)].values())/len(by_std[str(s)]):.3f}" for s in args.noise_stds), flush=True)
        results["noise_robustness"] = noise_out

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

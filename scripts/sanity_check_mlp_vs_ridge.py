"""Sanity check: did mlp_multiseed() actually train properly, or is the
early-stopping speedup (see analyze_encoders_local.py, min_steps cut from
500->150 for a ~19x wall-clock win) undertrained relative to a closed-form
ridge fit on the same data?

Ridge regression has no training dynamics -- it's the exact global optimum
of its own (different, L2-penalized linear) objective, so it makes a clean
lower-bound sanity check: a properly-trained MLP (strictly more expressive
than a linear map, given enough capacity) should not be SYSTEMATICALLY worse
than ridge at the same frozen layer. If it is, that's undertraining, not a
real "no benefit from nonlinearity" finding -- exactly the failure mode
mlp_r2()'s original docstring flagged and calibrated against, on a synthetic
signal only. This script checks it on the real held-out data instead.

Does NOT retrain or reselect anything: reads workshop_test_eval.py's
already-frozen `frozen_layer` per quantity from its output JSON, recomputes
features ONCE at that layer for the exact same fit/test trajectory split
(same train_valid_trajectory_split, same --train-max-traj/--n-offsets/
--token-max-clips the original run used), and fits one ridge probe per
quantity via ridge_fit_eval -- no sweep, no seeds, cheap.

Usage:
    uv run python scripts/sanity_check_mlp_vs_ridge.py \
        --checkpoints checkpoints/active_matter_jepa.pt checkpoints/active_matter_mae.pt \
        --data-root /workspace/data --dataset active_matter \
        --workshop-json sweep_results/active_matter_workshop_test_eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_encoders import load_checkpoint_encoder, ridge_fit_eval
from scripts.analyze_encoders_local import mlp_multiseed
from scripts.workshop_test_eval import (
    ROLES, collect_pooled, collect_token, sample_context_seeds, sample_forecast_pairs,
    train_valid_trajectory_split,
)


def _ckpt_display(ckpt_path: str) -> str:
    return Path(ckpt_path).stem


def _new_mlp(xtr, ytr, xte, yte, mlp_kwargs):
    if mlp_kwargs is None:
        return None, None
    r = mlp_multiseed(xtr, ytr, xte, yte, seeds=(0, 1, 2, 3, 4), **mlp_kwargs)
    return r["mean"], r["std"]


def check_pooled(mm_train, mm_test, fit_traj, dataset, encoders, n_offsets, n_frames,
                  train_max_traj, test_max_traj, entries: dict, mlp_kwargs=None) -> list:
    seeds_train = sample_context_seeds(mm_train, n_offsets, n_frames, trajectories=fit_traj,
                                        max_trajectories=train_max_traj)
    seeds_test = sample_context_seeds(mm_test, n_offsets, n_frames, max_trajectories=test_max_traj)
    targets_train, feats_train = collect_pooled(mm_train, seeds_train, dataset, encoders)
    targets_test, feats_test = collect_pooled(mm_test, seeds_test, dataset, encoders)
    rows = []
    for ckpt in encoders:
        name = _ckpt_display(ckpt)
        for q, e in entries.get(ckpt, {}).items():
            if q not in targets_train or q not in targets_test:
                continue
            layer = e["frozen_layer"]
            xtr, xte = feats_train[ckpt][layer], feats_test[ckpt][layer]
            ridge_r2 = ridge_fit_eval(xtr, targets_train[q], xte, targets_test[q])
            new_mean, new_std = _new_mlp(xtr, targets_train[q], xte, targets_test[q], mlp_kwargs)
            rows.append((name, q, layer, e["test_r2_mean"], e["test_r2_std"], ridge_r2, new_mean, new_std))
    return rows


def check_token(mm_train, mm_test, fit_traj, dataset, encoders, patch_size, n_offsets, n_frames,
                 train_max_traj, test_max_traj, token_max_clips, entries: dict, mlp_kwargs=None) -> list:
    seeds_train = sample_context_seeds(mm_train, n_offsets, n_frames, trajectories=fit_traj,
                                        max_trajectories=train_max_traj)[:token_max_clips]
    seeds_test = sample_context_seeds(mm_test, n_offsets, n_frames, max_trajectories=test_max_traj)[:token_max_clips]
    targets_train, feats_train = collect_token(mm_train, seeds_train, dataset, encoders, patch_size)
    targets_test, feats_test = collect_token(mm_test, seeds_test, dataset, encoders, patch_size)
    rows = []
    for ckpt in encoders:
        name = _ckpt_display(ckpt)
        for q, e in entries.get(ckpt, {}).items():
            if q not in targets_train or q not in targets_test:
                continue
            layer = e["frozen_layer"]
            xtr = feats_train[ckpt][layer].reshape(-1, feats_train[ckpt][layer].size(-1))
            xte = feats_test[ckpt][layer].reshape(-1, feats_test[ckpt][layer].size(-1))
            ytr, yte = targets_train[q].reshape(-1), targets_test[q].reshape(-1)
            ridge_r2 = ridge_fit_eval(xtr, ytr, xte, yte)
            new_mean, new_std = _new_mlp(xtr, ytr, xte, yte, mlp_kwargs)
            rows.append((name, q, layer, e["test_r2_mean"], e["test_r2_std"], ridge_r2, new_mean, new_std))
    return rows


def check_forecast_pooled(mm_train, mm_test, fit_traj, dataset, encoders, n_offsets, gap, max_gap, n_frames,
                           train_max_traj, test_max_traj, entries: dict, mlp_kwargs=None) -> list:
    pairs_train = sample_forecast_pairs(mm_train, n_offsets, max_gap, n_frames, trajectories=fit_traj,
                                         max_trajectories=train_max_traj)
    pairs_test = sample_forecast_pairs(mm_test, n_offsets, max_gap, n_frames, max_trajectories=test_max_traj)
    offset_fn = lambda tr, off, g=gap: off + n_frames + g
    ctx_train, cfeats_train = collect_pooled(mm_train, pairs_train, dataset, encoders)
    ctx_test, cfeats_test = collect_pooled(mm_test, pairs_test, dataset, encoders)
    fut_train, ffeats_train = collect_pooled(mm_train, pairs_train, dataset, encoders, offset_fn=offset_fn)
    fut_test, ffeats_test = collect_pooled(mm_test, pairs_test, dataset, encoders, offset_fn=offset_fn)
    rows = []
    for ckpt in encoders:
        name = _ckpt_display(ckpt)
        for q, e in entries.get(ckpt, {}).items():
            if q not in fut_train or q not in fut_test:
                continue
            layer = e["frozen_layer"]
            xtr, xte = cfeats_train[ckpt][layer], cfeats_test[ckpt][layer]
            ridge_r2 = ridge_fit_eval(xtr, fut_train[q], xte, fut_test[q])
            new_mean, new_std = _new_mlp(xtr, fut_train[q], xte, fut_test[q], mlp_kwargs)
            rows.append((name, q, layer, e["test_r2_mean"], e["test_r2_std"], ridge_r2, new_mean, new_std))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workshop-json", required=True)
    ap.add_argument("--n-offsets", type=int, default=3)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--valid-stride", type=int, default=8)
    ap.add_argument("--train-max-traj", type=int, default=None)
    ap.add_argument("--test-max-traj", type=int, default=None)
    ap.add_argument("--token-max-clips", type=int, default=16)
    ap.add_argument("--gaps", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--weight-decay", type=float, default=None,
                     help="if set, also fit a candidate-hyperparameter MLP at the same frozen layer")
    ap.add_argument("--dropout", type=float, default=None)
    args = ap.parse_args()

    mlp_kwargs = None
    if args.weight_decay is not None or args.dropout is not None:
        mlp_kwargs = {}
        if args.weight_decay is not None:
            mlp_kwargs["weight_decay"] = args.weight_decay
        if args.dropout is not None:
            mlp_kwargs["dropout"] = args.dropout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = {}
    for ckpt_path in args.checkpoints:
        enc, _spec = load_checkpoint_encoder(ckpt_path, device)
        encoders[ckpt_path] = enc
    patch_size = next(iter(encoders.values())).patch_size

    data = json.loads(Path(args.workshop_json).read_text())
    mm_train = np.load(Path(args.data_root) / "memmap" / args.dataset / "train.npy", mmap_mode="r")
    mm_test = np.load(Path(args.data_root) / "memmap" / args.dataset / "test.npy", mmap_mode="r")
    fit_traj, _valid_traj = train_valid_trajectory_split(mm_train.shape[0], valid_stride=args.valid_stride)
    max_gap = max(args.gaps)

    all_rows = []

    print("== contemporaneous pooled ==", flush=True)
    rows = check_pooled(mm_train, mm_test, fit_traj, args.dataset, encoders, args.n_offsets, args.n_frames,
                         args.train_max_traj, args.test_max_traj, data["contemporaneous"]["pooled"], mlp_kwargs)
    all_rows += [("t+0-pooled",) + r for r in rows]

    print("== contemporaneous token ==", flush=True)
    rows = check_token(mm_train, mm_test, fit_traj, args.dataset, encoders, patch_size, args.n_offsets,
                        args.n_frames, args.train_max_traj, args.test_max_traj, args.token_max_clips,
                        data["contemporaneous"]["token"], mlp_kwargs)
    all_rows += [("t+0-token",) + r for r in rows]

    for gap in args.gaps:
        print(f"== forecast t+{gap} pooled (offsets sized for max_gap={max_gap}, matching production) ==", flush=True)
        rows = check_forecast_pooled(mm_train, mm_test, fit_traj, args.dataset, encoders, args.n_offsets, gap,
                                      max_gap, args.n_frames, args.train_max_traj, args.test_max_traj,
                                      data["forecast"][str(gap)]["pooled"], mlp_kwargs)
        all_rows += [(f"t+{gap}-pooled",) + r for r in rows]

    header = f"\n{'family':<12}{'ckpt':<22}{'quantity':<26}{'L':<4}{'orig_mlp':<11}{'orig_std':<9}{'ridge':<9}"
    if mlp_kwargs:
        header += f"{'new_mlp':<10}{'new_std':<9}{'new-ridge':<10}"
    print(header)
    n_worse_orig, n_worse_new, n_total, worst = 0, 0, 0, []
    for family, ckpt, q, layer, mlp_mean, mlp_std, ridge_r2, new_mean, new_std in all_rows:
        n_total += 1
        orig_diff = mlp_mean - ridge_r2
        flag = ""
        if orig_diff < -0.02:
            n_worse_orig += 1
        line = f"{family:<12}{ckpt:<22}{q:<26}{layer:<4}{mlp_mean:<+11.3f}{mlp_std:<9.3f}{ridge_r2:<+9.3f}"
        if mlp_kwargs:
            new_diff = new_mean - ridge_r2
            if new_diff < -0.02:
                n_worse_new += 1
                flag = "  <-- NEW MLP STILL WORSE"
                worst.append((new_diff, family, ckpt, q, layer, new_mean, ridge_r2))
            line += f"{new_mean:<+10.3f}{new_std:<9.3f}{new_diff:<+10.3f}{flag}"
        print(line)

    print(f"\noriginal MLP: {n_worse_orig}/{n_total} cells >0.02 below ridge.")
    if mlp_kwargs:
        print(f"new MLP ({mlp_kwargs}): {n_worse_new}/{n_total} cells >0.02 below ridge.")
        if worst:
            worst.sort()
            print("Remaining offenders (new_mlp - ridge):")
            for diff, family, ckpt, q, layer, new_mean, ridge_r2 in worst[:10]:
                print(f"  {diff:+.3f}  {family} {ckpt} {q} L{layer}  new_mlp={new_mean:+.3f} ridge={ridge_r2:+.3f}")


if __name__ == "__main__":
    main()

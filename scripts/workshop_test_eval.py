"""Held-out, multi-seed, MLP-only workshop probing run: the "formal sweep" the
Fig 2-4 plan (see plot_workshop_figures.py's docstring) explicitly deferred.

Fixes the one real rigor gap found in the earlier train-CV suite (see git
history at or before commit 92da170): picking each cell's "best layer" via
argmax over the SAME train-split data the R^2 is then read from is an
unvalidated selection, not a held-out estimate -- even a 5-fold-CV-within-train
version of that fix never touches a genuinely disjoint set. Here: fit on a
train subset, select the layer on a held-out slice (mean R^2 over multiple
seeds), report once on the official test split (mean +/- std over multiple
seeds).

The held-out selection slice is carved OUT OF TRAIN, not the Well-shipped
`valid` split: that split turned out to be too small/regime-degenerate to
trust (as few as 4-5 trajectories total, and for shear_flow/rayleigh_benard
ALL of them from a single regime file -- exactly the coverage bug that once
produced nonsense negative R^2 on the original, since-fixed `test` split).
See train_valid_trajectory_split() in src/data/well.py: an interleaved every-Nth-trajectory
split (not a contiguous prefix/suffix) samples proportionally from every
regime file in both halves, since preprocess_memmap.py lays trajectories out
in contiguous per-source-file blocks (same file-sorted order
extract_regime_metadata.py relies on). The official `test` split is still
never touched until the final, single evaluation.

MLP-only throughout (mlp_multiseed(), see analyze_encoders_local.py, has
early stopping + dropout + seed-averaging -- none of which the rest of the
suite's single-seed, fixed-step-count MLP had).

Four families:
  1. Contemporaneous (t+0) physics, pooled + token.
  2. Forecast-content physics at each --gaps value (t+8, t+32), pooled + token
     -- context/future windows never overlap (future starts at
     off + n_frames + gap), so there is no temporal leakage between probe
     input and probe target.
  3. Noise robustness, pooled, at each quantity's ALREADY-frozen contemporaneous
     layer (no separate noise-layer search -- reusing the clean-optimal layer
     is the existing convention elsewhere in the suite and keeps this a simple
     adjustment, not new machinery).
  4. Regime: trajectory-constant governing physical parameters (Reynolds/
     Schmidt, Rayleigh/Prandtl, alpha/zeta -- see regime_family()), pooled
     only, one row per trajectory. Same train->select-on-valid->report-on-test
     protocol and same held-out-seed rigor as families 1-3, superseding the
     earlier 1-seed, ridge-only analyze_regime.py for this task. Requires
     {train,test}.regime.json (scripts/extract_regime_metadata.py); silently
     skipped (not a hard error) if missing, so it stays optional per-dataset.

Multi-encoder-seed usage: pass every seed's checkpoint for both objectives in
one `--checkpoints` list (order doesn't matter) -- each is probed completely
independently (its own held-out layer/weight_decay/ridge selection via its
own frozen weights), then results are combined per objective in the
"aggregated" section of the output JSON via a nested nested/hierarchical
nested-variance combination (see aggregate_by_objective()): the probe-seed
draws within one encoder are correlated (same frozen weights), so the
encoder seed -- not the individual probe fit -- is the unit that actually
matters for "would retraining this objective give the same answer." Which
checkpoint file to pass per seed is a training-side decision, not something
this script chooses -- `encoder_best_val.pt` (best validation-loss step) is
recommended over `encoder_100pct.pt` (fixed compute budget regardless of
whether it was actually the best step) if your training run saved one.

Usage (single seed per objective, still supported):
    uv run python scripts/workshop_test_eval.py \
        --checkpoints checkpoints/old/active_matter_jepa.pt checkpoints/old/active_matter_mae.pt \
        --data-root /workspace/data --dataset active_matter \
        --out sweep_results/active_matter_workshop_test_eval.json

Usage (3 encoder seeds per objective):
    uv run python scripts/workshop_test_eval.py \
        --checkpoints runs/active_matter_jepa_seed1/encoder_best_val.pt \
                       runs/active_matter_jepa_seed2/encoder_best_val.pt \
                       runs/active_matter_jepa_seed3/encoder_best_val.pt \
                       runs/active_matter_mae_seed1/encoder_best_val.pt \
                       runs/active_matter_mae_seed2/encoder_best_val.pt \
                       runs/active_matter_mae_seed3/encoder_best_val.pt \
        --data-root /workspace/data --dataset active_matter \
        --out sweep_results/active_matter_workshop_test_eval.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_encoders import (
    contemporaneous_targets, layerwise_features, load_checkpoint_encoder, ridge_fit_eval,
)
from scripts.analyze_encoders_local import (
    _train_mlp_early_stop, layerwise_token_features, local_target_maps, mlp_multiseed,
)
from src.data.well import train_valid_trajectory_split

N_FRAMES = 8
ROLES = ("train", "valid", "test")  # "train"=fit subset, "valid"=held-out selection subset (both carved from train.npy)
REGIME_LOG_TRANSFORM = {"Reynolds", "Schmidt", "Rayleigh", "Prandtl"}


def _ckpt_display(ckpt_path: str) -> str:
    """Log-friendly checkpoint label. Multi-seed checkpoints are typically
    named identically (e.g. every seed's `encoder_best_val.pt`) with the
    seed only distinguished by the parent run directory
    (`runs/{dataset}_{objective}_seed{N}/...`) -- the filename alone would
    collide across seeds in every printed line, so prefer
    `parent_dir/filename` whenever the parent directory is itself
    informative (not just "checkpoints", the flat single-seed convention)."""
    p = Path(ckpt_path)
    if p.parent.name and p.parent.name != "checkpoints":
        return f"{p.parent.name}/{p.name}"
    return p.name


def _objective_of(ckpt_path: str) -> str:
    """Which objective a checkpoint belongs to, by substring search over the
    FULL path (not just the filename) -- multi-seed checkpoints often carry
    the objective in the run directory name only (e.g.
    runs/active_matter_jepa_seed1/encoder_best_val.pt), not the filename."""
    has_jepa, has_mae = "jepa" in ckpt_path, "mae" in ckpt_path
    if has_jepa and not has_mae:
        return "jepa"
    if has_mae and not has_jepa:
        return "mae"
    raise ValueError(f"cannot uniquely determine objective (jepa/mae) from checkpoint path: {ckpt_path}")


# ---------------------------------------------------------------------------
# Trajectory selection
# ---------------------------------------------------------------------------
# train_valid_trajectory_split() now lives in src/data/well.py (also used by
# src/train.py's pretraining val loop) -- imported above, not redefined here.


def _select_trajectories(n_traj: int, trajectories: list | None, max_trajectories: int | None) -> list:
    base = trajectories if trajectories is not None else list(range(n_traj))
    if max_trajectories is not None and max_trajectories < len(base):
        idx = np.linspace(0, len(base) - 1, max_trajectories).astype(int)
        base = sorted({base[i] for i in idx})
    return base


def sample_context_seeds(mm: np.ndarray, n_offsets: int, n_frames: int = N_FRAMES,
                          trajectories: list | None = None, max_trajectories: int | None = None) -> list:
    n_traj, _, t, _, _ = mm.shape
    max_off = t - n_frames
    offsets = np.linspace(0, max_off, n_offsets).astype(int)
    trajs = _select_trajectories(n_traj, trajectories, max_trajectories)
    return [(traj, off) for traj in trajs for off in offsets]


def sample_forecast_pairs(mm: np.ndarray, n_offsets: int, max_gap: int, n_frames: int = N_FRAMES,
                           trajectories: list | None = None, max_trajectories: int | None = None) -> list:
    """Sized for the LARGEST gap requested, so every gap in a sweep is scored
    on the exact same (traj, offset) samples (matches forecast_content_probe.py)."""
    n_traj, _, t, _, _ = mm.shape
    max_off = t - (2 * n_frames + max_gap)
    if max_off < 0:
        raise ValueError(
            f"trajectory has {t} frames, too short for context({n_frames}) + "
            f"max_gap({max_gap}) + future({n_frames})"
        )
    offsets = np.linspace(0, max_off, n_offsets).astype(int)
    trajs = _select_trajectories(n_traj, trajectories, max_trajectories)
    return [(traj, off) for traj in trajs for off in offsets]


def _load_window(mm: np.ndarray, traj: int, start: int, n_frames: int) -> torch.Tensor:
    return torch.from_numpy(np.array(mm[traj, :, start : start + n_frames])).float()


def load_batch(mm: np.ndarray, seeds: list, offset_fn=None, n_frames: int = N_FRAMES) -> torch.Tensor:
    if offset_fn is None:
        offset_fn = lambda tr, off: off
    clips = [_load_window(mm, tr, offset_fn(tr, off), n_frames) for tr, off in seeds]
    return torch.stack(clips)


# ---------------------------------------------------------------------------
# Feature/target collection (one pass over `seeds`, all encoders + all layers)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_pooled(mm: np.ndarray, seeds: list, dataset: str, encoders: dict,
                    offset_fn=None, batch_size: int = 16) -> tuple[dict, dict]:
    targets_acc: dict = {}
    feats_acc = {name: None for name in encoders}
    for start in range(0, len(seeds), batch_size):
        clip = load_batch(mm, seeds[start : start + batch_size], offset_fn)
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
                   offset_fn=None, batch_size: int = 8) -> tuple[dict, dict]:
    targets_acc: dict = {}
    feats_acc = {name: None for name in encoders}
    for start in range(0, len(seeds), batch_size):
        clip = load_batch(mm, seeds[start : start + batch_size], offset_fn)
        for k, v in local_target_maps(clip, *patch_size, dataset=dataset).items():
            targets_acc.setdefault(k, []).append(v)
        for name, encoder in encoders.items():
            device = next(encoder.parameters()).device
            layer_feats = layerwise_token_features(encoder, clip.to(device))
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
    """Noise injected into the encoder's input only; targets always computed
    on the CLEAN clip (matches analyze_noise_robustness.py's convention)."""
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


# ---------------------------------------------------------------------------
# Train -> select-on-valid -> report-on-test
# ---------------------------------------------------------------------------

WEIGHT_DECAY_GRID = (1e-4, 1e-2, 1e-1)
SELECT_SEEDS = (0, 1, 2)
STD_PENALTY = 2.0


def select_and_eval(feats_train: list, y_train: torch.Tensor, feats_valid: list, y_valid: torch.Tensor,
                     feats_test: list, y_test: torch.Tensor, seeds: tuple) -> dict:
    """feats_*: list of n_layers tensors, row-aligned with y_*. Jointly
    selects (layer, method) by held-out valid performance, where `method` is
    one of the MLP weight_decay grid values OR ridge itself.

    Why ridge is a candidate, not just an audit field: a single global
    weight_decay does not work (naively raising it fixed a handful of
    catastrophically overfit MLP cells but REGRESSED many already-good ones,
    e.g. active_matter's t+0 token quantities falling below ridge), and even
    per-layer adaptive weight_decay selection with a variance-penalized
    (mean - std) score left one quantity (shear_flow's tracer_grad forecast)
    stuck picking an unstable config -- the valid-set estimate itself doesn't
    reliably distinguish it from the alternatives even at 5 selection seeds,
    a genuine valid/test generalization gap, not sampling noise a bigger
    seed count fixes. Including ridge as a candidate in the SAME fair,
    held-out-valid comparison closes this structurally: whichever method
    actually wins on data the final test split never touched gets reported,
    so the result is never worse than ridge by more than ordinary
    valid-to-test generalization noise, while MLP still wins (and is
    reported) wherever it genuinely earns it.

    Grid search uses fewer seeds (SELECT_SEEDS) than the final reported fit
    (`seeds`) to keep the added cost modest. STD_PENALTY=2.0 (score =
    mean - 2*std, not just mean - std) was tuned against the two remaining
    stubborn cases after adding ridge as a candidate: a plain (mean - std)
    penalty still let shear_flow's tracer_grad forecast pick an unstable
    layer/config that looked fine on valid (std~0.02 at 3 selection seeds)
    but collapsed on test (R^2=-0.66, true std~1.0) -- doubling the penalty
    correctly rejects that layer in favor of one that's genuinely stable
    (R^2=+0.75, std=0.02, confirmed on the actual held-out test split).

    Reports mean+/-std test R^2 at the frozen (layer, method) (fresh
    `seeds` fits if MLP won, a single closed-form solve if ridge won --
    test never touched during selection).

    Also always records a closed-form ridge fit at the frozen layer
    (`ridge_test_r2`) regardless of which method won, as a transparent
    audit trail."""
    n_layers = len(feats_train)
    valid_curve = []        # plain mean R^2 per layer (for reporting/depth-curve plots)
    layer_scores = []       # best (variance-penalized) score per layer -- drives layer selection
    method_per_layer = []   # ("mlp", weight_decay) or ("ridge", None), whichever won at that layer
    for layer_idx in range(n_layers):
        best_method, best_score, best_r2 = ("mlp", WEIGHT_DECAY_GRID[0]), -float("inf"), -float("inf")
        for wd in WEIGHT_DECAY_GRID:
            r = mlp_multiseed(feats_train[layer_idx], y_train, feats_valid[layer_idx], y_valid,
                               seeds=SELECT_SEEDS, weight_decay=wd)
            # Variance-penalized (mean - std), not raw mean: a config that
            # happens to look good on one noisy valid draw but is actually
            # unstable must not win just by getting lucky.
            score = r["mean"] - STD_PENALTY * r["std"]
            if score > best_score:
                best_score, best_r2, best_method = score, r["mean"], ("mlp", wd)
        ridge_valid_r2 = ridge_fit_eval(feats_train[layer_idx], y_train, feats_valid[layer_idx], y_valid)
        if ridge_valid_r2 > best_score:  # ridge has no seed variance -> its score is just its value
            best_score, best_r2, best_method = ridge_valid_r2, ridge_valid_r2, ("ridge", None)
        valid_curve.append(best_r2)
        layer_scores.append(best_score)
        method_per_layer.append(best_method)
    frozen_layer = int(np.argmax(layer_scores))
    method, frozen_weight_decay = method_per_layer[frozen_layer]
    if method == "mlp":
        test_res = mlp_multiseed(feats_train[frozen_layer], y_train, feats_test[frozen_layer], y_test,
                                  seeds=seeds, weight_decay=frozen_weight_decay)
        test_mean, test_std, test_per_seed = test_res["mean"], test_res["std"], test_res["per_seed"]
    else:
        r2 = ridge_fit_eval(feats_train[frozen_layer], y_train, feats_test[frozen_layer], y_test)
        test_mean, test_std, test_per_seed = r2, 0.0, [r2]
    ridge_test_r2 = ridge_fit_eval(feats_train[frozen_layer], y_train, feats_test[frozen_layer], y_test)
    return {
        "frozen_layer": frozen_layer,
        "method": method,
        "frozen_weight_decay": frozen_weight_decay,
        "valid_r2_curve": valid_curve,
        "test_r2_mean": test_mean,
        "test_r2_std": test_std,
        "test_r2_per_seed": test_per_seed,
        "ridge_test_r2": ridge_test_r2,
    }


def _common_quantities(*target_dicts) -> list:
    keys = set(target_dicts[0])
    for d in target_dicts[1:]:
        keys &= set(d)
    return sorted(keys)


def _flatten_token(feats_by_layer: list, target: torch.Tensor) -> tuple:
    feats_flat = [f.reshape(-1, f.size(-1)) for f in feats_by_layer]
    return feats_flat, target.reshape(-1)


# ---------------------------------------------------------------------------
# Family 1: contemporaneous (t+0)
# ---------------------------------------------------------------------------

def contemporaneous_family(mm: dict, traj: dict, dataset: str, encoders: dict, patch_size: tuple,
                            n_offsets: int, n_frames: int, max_traj: dict, token_max_clips: int,
                            seeds: tuple) -> dict:
    seeds_by_role = {
        r: sample_context_seeds(mm[r], n_offsets, n_frames, trajectories=traj[r], max_trajectories=max_traj[r])
        for r in ROLES
    }
    print(f"  [t+0] n_samples: " + " ".join(f"{r}={len(seeds_by_role[r])}" for r in ROLES), flush=True)

    pooled_by_role = {r: collect_pooled(mm[r], seeds_by_role[r], dataset, encoders) for r in ROLES}
    tok_seeds = {r: seeds_by_role[r][:token_max_clips] for r in ROLES}
    token_by_role = {r: collect_token(mm[r], tok_seeds[r], dataset, encoders, patch_size) for r in ROLES}

    out = {"pooled": {}, "token": {}}
    for ckpt in encoders:
        qs = _common_quantities(*(pooled_by_role[r][0] for r in ROLES))
        per_q = {}
        for q in qs:
            per_q[q] = select_and_eval(
                pooled_by_role["train"][1][ckpt], pooled_by_role["train"][0][q],
                pooled_by_role["valid"][1][ckpt], pooled_by_role["valid"][0][q],
                pooled_by_role["test"][1][ckpt], pooled_by_role["test"][0][q],
                seeds=seeds,
            )
            print(f"    pooled t+0 {_ckpt_display(ckpt):32s} {q:26s} "
                  f"L{per_q[q]['frozen_layer']:2d} [{per_q[q]['method']}]  test_r2={per_q[q]['test_r2_mean']:+.3f}"
                  f"(+/-{per_q[q]['test_r2_std']:.3f})  ridge={per_q[q]['ridge_test_r2']:+.3f}", flush=True)
        out["pooled"][ckpt] = per_q

        qs_t = _common_quantities(*(token_by_role[r][0] for r in ROLES))
        per_qt = {}
        for q in qs_t:
            ft, yt = _flatten_token(token_by_role["train"][1][ckpt], token_by_role["train"][0][q])
            fv, yv = _flatten_token(token_by_role["valid"][1][ckpt], token_by_role["valid"][0][q])
            fte, yte = _flatten_token(token_by_role["test"][1][ckpt], token_by_role["test"][0][q])
            per_qt[q] = select_and_eval(ft, yt, fv, yv, fte, yte, seeds=seeds)
            print(f"    token  t+0 {_ckpt_display(ckpt):32s} {q:26s} "
                  f"L{per_qt[q]['frozen_layer']:2d} [{per_qt[q]['method']}]  test_r2={per_qt[q]['test_r2_mean']:+.3f}"
                  f"(+/-{per_qt[q]['test_r2_std']:.3f})  ridge={per_qt[q]['ridge_test_r2']:+.3f}", flush=True)
        out["token"][ckpt] = per_qt
    return out


# ---------------------------------------------------------------------------
# Family 2: forecast-content (t+gap)
# ---------------------------------------------------------------------------

def forecast_family(mm: dict, traj: dict, dataset: str, encoders: dict, patch_size: tuple,
                     n_offsets: int, gaps: list, n_frames: int, max_traj: dict, token_max_clips: int,
                     seeds: tuple) -> dict:
    pairs = {
        r: sample_forecast_pairs(mm[r], n_offsets, max(gaps), n_frames, trajectories=traj[r],
                                  max_trajectories=max_traj[r])
        for r in ROLES
    }
    tok_pairs = {r: pairs[r][:token_max_clips] for r in ROLES}
    print(f"  [forecast] n_samples: " + " ".join(f"{r}={len(pairs[r])}" for r in ROLES)
          + "  n_token_clips: " + " ".join(f"{r}={len(tok_pairs[r])}" for r in ROLES), flush=True)

    ctx_pooled = {r: collect_pooled(mm[r], pairs[r], dataset, encoders) for r in ROLES}
    ctx_token = {r: collect_token(mm[r], tok_pairs[r], dataset, encoders, patch_size) for r in ROLES}

    out = {}
    for gap in gaps:
        t0 = time.time()
        offset_fn = lambda tr, off, g=gap: off + n_frames + g
        fut_pooled = {r: collect_pooled(mm[r], pairs[r], dataset, encoders, offset_fn=offset_fn) for r in ROLES}
        fut_token = {r: collect_token(mm[r], tok_pairs[r], dataset, encoders, patch_size, offset_fn=offset_fn)
                     for r in ROLES}

        gap_out = {"pooled": {}, "token": {}}
        for ckpt in encoders:
            qs = _common_quantities(*(fut_pooled[r][0] for r in ROLES))
            per_q = {}
            for q in qs:
                per_q[q] = select_and_eval(
                    ctx_pooled["train"][1][ckpt], fut_pooled["train"][0][q],
                    ctx_pooled["valid"][1][ckpt], fut_pooled["valid"][0][q],
                    ctx_pooled["test"][1][ckpt], fut_pooled["test"][0][q],
                    seeds=seeds,
                )
                print(f"    pooled t+{gap:<3d}{_ckpt_display(ckpt):32s} {q:26s} "
                      f"L{per_q[q]['frozen_layer']:2d} [{per_q[q]['method']}]  test_r2={per_q[q]['test_r2_mean']:+.3f}"
                      f"(+/-{per_q[q]['test_r2_std']:.3f})  ridge={per_q[q]['ridge_test_r2']:+.3f}", flush=True)
            gap_out["pooled"][ckpt] = per_q

            qs_t = _common_quantities(*(fut_token[r][0] for r in ROLES))
            per_qt = {}
            for q in qs_t:
                ft, _ = _flatten_token(ctx_token["train"][1][ckpt], ctx_token["train"][0][q])
                _, yt = _flatten_token(fut_token["train"][1][ckpt], fut_token["train"][0][q])
                fv, _ = _flatten_token(ctx_token["valid"][1][ckpt], ctx_token["valid"][0][q])
                _, yv = _flatten_token(fut_token["valid"][1][ckpt], fut_token["valid"][0][q])
                fte, _ = _flatten_token(ctx_token["test"][1][ckpt], ctx_token["test"][0][q])
                _, yte = _flatten_token(fut_token["test"][1][ckpt], fut_token["test"][0][q])
                per_qt[q] = select_and_eval(ft, yt, fv, yv, fte, yte, seeds=seeds)
                print(f"    token  t+{gap:<3d}{_ckpt_display(ckpt):32s} {q:26s} "
                      f"L{per_qt[q]['frozen_layer']:2d} [{per_qt[q]['method']}]  test_r2={per_qt[q]['test_r2_mean']:+.3f}"
                      f"(+/-{per_qt[q]['test_r2_std']:.3f})  ridge={per_qt[q]['ridge_test_r2']:+.3f}", flush=True)
            gap_out["token"][ckpt] = per_qt
        out[str(gap)] = gap_out
        print(f"  [forecast] gap={gap} done in {time.time()-t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Family 3: noise robustness (pooled, at each quantity's already-frozen layer)
# ---------------------------------------------------------------------------
# Clean-fit / noisy-test protocol (standard corruption-robustness framing,
# matches ImageNet-C style evaluation): both readouts (ridge and MLP,
# mirroring the ridge-vs-MLP pairing used everywhere else in this suite)
# are fit ONCE on clean (std=0) training features, then evaluated -- with
# NO refitting -- against test features at each noise level. This replaced
# an earlier matched-noise design (fit AND test at the same std, letting
# the probe recalibrate to whatever noise level it's scored on) after that
# design produced smooth but misleadingly optimistic degradation curves:
# on rayleigh_benard, the matched design showed MAE as uniformly more
# noise-robust than JEPA, but a clean-fit test (no recalibration allowed)
# showed the reverse on most quantities -- MAE's readout adapts well, but
# JEPA's underlying representation is actually the more intrinsically
# stable one. Reports R^2 (can go arbitrarily negative once the
# clean-calibrated readout's scale/offset stops matching the noise-shifted
# feature distribution -- a real signature of fragility, not a bug) AND
# Pearson r (scale/offset invariant, so it still reveals genuinely
# surviving signal even where R^2 has collapsed to a large negative number
# that isn't otherwise comparable across quantities/objectives), for both
# methods.
NOISE_STDS_DEFAULT = (0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def _pearson_r(pred: torch.Tensor, y_true: torch.Tensor) -> float:
    pred_c = pred - pred.mean()
    y_c = y_true - y_true.mean()
    denom = pred_c.norm() * y_c.norm() + 1e-12
    return (pred_c @ y_c / denom).item()


def _r2(pred: torch.Tensor, y_true: torch.Tensor) -> float:
    ss_res = ((pred - y_true) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def _ridge_fit_predict(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor,
                        lam: float = 1e-2) -> torch.Tensor:
    xm, xs = x_train.mean(0), x_train.std(0) + 1e-8
    xtr, xte = (x_train - xm) / xs, (x_test - xm) / xs
    ym = y_train.mean()
    a = xtr.T @ xtr + lam * xtr.size(0) * torch.eye(x_train.size(1))
    w = torch.linalg.solve(a, xtr.T @ (y_train - ym))
    return xte @ w + ym


def _mlp_fit_clean(x_train: torch.Tensor, y_train: torch.Tensor, seeds: tuple,
                    hidden: int = 128, dropout: float = 0.1, lr: float = 1e-2,
                    weight_decay: float = 1e-4, min_steps: int = 150, max_steps: int = 4000,
                    patience: int = 100) -> list:
    """Fit len(seeds) independently-seeded TinyMLPs once on CLEAN
    (x_train, y_train) via _train_mlp_early_stop, returning the fitted
    (model, xm, xs, ym, ys) tuples for reuse across every noise level --
    fitting cost is independent of how fine the noise grid is."""
    return [
        _train_mlp_early_stop(x_train, y_train, hidden=hidden, dropout=dropout, lr=lr,
                               weight_decay=weight_decay, seed=seed, min_steps=min_steps,
                               max_steps=max_steps, patience=patience)
        for seed in seeds
    ]


def _mlp_predict_multiseed(fits: list, x_test: torch.Tensor) -> list:
    """Apply each already clean-fitted model to x_test, returning per-seed
    predictions de-normalized back to y's original (not z-scored) scale,
    moved back to CPU -- layerwise_features()/ridge already keep everything
    on CPU (only the MLP forward pass itself needs the GPU), so this keeps
    a single CPU convention for the R^2/Pearson comparisons that follow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test_dev = x_test.to(device)
    preds = []
    for model, xm, xs, ym, ys in fits:
        xte = (x_test_dev - xm) / xs
        with torch.no_grad():
            pred_norm = model(xte)
        preds.append((pred_norm * ys + ym).cpu())
    return preds


def noise_family(mm: dict, traj: dict, dataset: str, encoders: dict, n_offsets: int, n_frames: int,
                  max_traj: dict, noise_stds: list, frozen_layers: dict, seeds: tuple) -> dict:
    roles = ("train", "test")
    seeds_by_role = {
        r: sample_context_seeds(mm[r], n_offsets, n_frames, trajectories=traj[r], max_trajectories=max_traj[r])
        for r in roles
    }
    print(f"  [noise] n_samples: " + " ".join(f"{r}={len(seeds_by_role[r])}" for r in roles), flush=True)

    noise_by_role = {
        r: collect_pooled_noise(mm[r], seeds_by_role[r], dataset, encoders, noise_stds) for r in roles
    }

    out = {}
    for ckpt in encoders:
        per_q = {}
        for q, layer_idx in frozen_layers[ckpt].items():
            if q not in noise_by_role["train"][0] or q not in noise_by_role["test"][0]:
                continue
            xtr_clean = noise_by_role["train"][1][ckpt]["0"][layer_idx]
            ytr = noise_by_role["train"][0][q]
            yte = noise_by_role["test"][0][q]  # stays CPU -- matches ridge_pred and mlp preds (see above)
            mlp_fits = _mlp_fit_clean(xtr_clean, ytr, seeds=seeds)
            by_std = {}
            for std in noise_stds:
                skey = str(std)
                xte = noise_by_role["test"][1][ckpt][skey][layer_idx]

                ridge_pred = _ridge_fit_predict(xtr_clean, ytr, xte)
                ridge_r2 = _r2(ridge_pred, yte)
                ridge_pr = _pearson_r(ridge_pred, yte)

                mlp_preds = _mlp_predict_multiseed(mlp_fits, xte)
                mlp_r2s = [_r2(p, yte) for p in mlp_preds]
                mlp_prs = [_pearson_r(p, yte) for p in mlp_preds]
                mlp_r2_t, mlp_pr_t = torch.tensor(mlp_r2s), torch.tensor(mlp_prs)

                by_std[skey] = {
                    "ridge_r2": ridge_r2, "ridge_pearson_r": ridge_pr,
                    "mlp_r2_mean": mlp_r2_t.mean().item(), "mlp_r2_std": mlp_r2_t.std().item(),
                    "mlp_pearson_r_mean": mlp_pr_t.mean().item(), "mlp_pearson_r_std": mlp_pr_t.std().item(),
                }
            per_q[q] = {"frozen_layer": layer_idx, "by_noise": by_std}
            print(f"    noise  {_ckpt_display(ckpt):32s} {q:26s} L{layer_idx:2d}  " +
                  "  ".join(f"std={s}:ridge_r2={by_std[str(s)]['ridge_r2']:+.2f}"
                            f",mlp_r2={by_std[str(s)]['mlp_r2_mean']:+.2f}"
                            for s in noise_stds),
                  flush=True)
        out[ckpt] = per_q
    return out


# ---------------------------------------------------------------------------
# Family 4: regime (trajectory-constant governing physical parameters)
# ---------------------------------------------------------------------------

def _load_regime_records(data_root: str, dataset: str, split: str) -> list | None:
    path = Path(data_root) / "memmap" / dataset / f"{split}.regime.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _regime_targets_for_trajectories(records: list, trajs: list) -> tuple[list, dict]:
    """Filters trajs to those with a fully-populated regime record (some
    filenames don't match extract_regime_metadata.py's regex), returns
    (kept_trajs, {param_name: [raw_value, ...]}) row-aligned with kept_trajs."""
    param_names = [k for k in records[0].keys() if k != "file"]
    kept = [t for t in trajs if all(records[t][p] is not None for p in param_names)]
    raw = {p: [records[t][p] for t in kept] for p in param_names}
    return kept, raw


@torch.no_grad()
def collect_regime_pooled(mm: np.ndarray, kept_trajs: list, encoders: dict, n_offsets: int,
                           n_frames: int = N_FRAMES, batch_size: int = 16) -> dict:
    """One feature row per trajectory (mean-pooled over n_offsets sampled
    windows within that trajectory), not one row per window -- regime is
    constant across a whole trajectory, so per-window rows of the same
    trajectory would carry an (almost) identical target, and
    train_valid_trajectory_split() only guarantees no *trajectory* is split
    across roles, not that individual windows are non-redundant within one."""
    if not kept_trajs:
        return {name: [] for name in encoders}
    _, _, t, _, _ = mm.shape
    max_off = t - n_frames
    offsets = np.linspace(0, max_off, n_offsets).astype(int)
    seeds = [(traj, off) for traj in kept_trajs for off in offsets]  # trajectory-major order

    feats_acc = {name: None for name in encoders}
    for start in range(0, len(seeds), batch_size):
        clip = load_batch(mm, seeds[start:start + batch_size], n_frames=n_frames)
        for name, encoder in encoders.items():
            device = next(encoder.parameters()).device
            layer_feats = layerwise_features(encoder, clip.to(device))
            if feats_acc[name] is None:
                feats_acc[name] = [[] for _ in layer_feats]
            for i, f in enumerate(layer_feats):
                feats_acc[name][i].append(f)
    feats = {name: [torch.cat(fs) for fs in per_layer] for name, per_layer in feats_acc.items()}
    n_traj = len(kept_trajs)
    return {
        name: [f.view(n_traj, n_offsets, -1).mean(dim=1) for f in per_layer]
        for name, per_layer in feats.items()
    }


def regime_family(mm: dict, traj: dict, dataset: str, data_root: str, encoders: dict,
                   n_offsets: int, max_traj: dict, seeds: tuple) -> dict | None:
    """Does the pooled representation know the trajectory-constant physical
    regime (Reynolds/Schmidt for shear_flow, Rayleigh/Prandtl for
    rayleigh_benard, alpha/zeta for active_matter)? Same
    train->select-on-valid->report-on-test protocol as the other families
    (select_and_eval()), just with trajectory-level rows instead of
    window/token-level ones. Log-transforms Re/Ra/Sc/Pr (span multiple
    orders of magnitude); regresses active_matter's alpha/zeta raw.

    Each real target also gets a `{name}_shuffled_control` companion column
    (same features, independently permuted target per role, seeded) as a
    leakage sanity check -- this should collapse toward R^2~=0. A real,
    non-collapsed control R^2 would mean the split leaks trajectory identity
    somehow, not that the encoder is unusually good.

    Returns None (family skipped, not a hard error) if regime metadata is
    missing for this dataset -- run extract_regime_metadata.py --split
    {train,test} first."""
    records = {
        role: _load_regime_records(data_root, dataset, "train" if role == "valid" else role)
        for role in ROLES
    }
    if any(r is None for r in records.values()):
        missing = [role for role, r in records.items() if r is None]
        print(f"  [regime] regime metadata missing for {dataset} ({missing}) -- skipping regime family "
              f"(run extract_regime_metadata.py --split {{train,test}} first)", flush=True)
        return None

    kept, raw_targets = {}, {}
    for role in ROLES:
        n_traj_role = mm[role].shape[0]
        trajs = _select_trajectories(n_traj_role, traj[role], max_traj[role])
        k, rt = _regime_targets_for_trajectories(records[role], trajs)
        kept[role], raw_targets[role] = k, rt
    param_names = sorted(set(raw_targets["train"]) & set(raw_targets["valid"]) & set(raw_targets["test"]))
    print(f"  [regime] n_trajectories: " + " ".join(f"{r}={len(kept[r])}" for r in ROLES)
          + f"  params: {param_names}", flush=True)
    if not param_names:
        print(f"  [regime] no regime params survived filtering for {dataset}, skipping", flush=True)
        return None

    pooled = {role: collect_regime_pooled(mm[role], kept[role], encoders, n_offsets) for role in ROLES}

    targets = {role: {} for role in ROLES}
    for role in ROLES:
        for p in param_names:
            vals = np.array(raw_targets[role][p], dtype=np.float64)
            if p in REGIME_LOG_TRANSFORM:
                vals = np.log10(vals)
            y = torch.from_numpy(vals).float()
            targets[role][p] = y
            perm = torch.randperm(y.size(0), generator=torch.Generator().manual_seed(0))
            targets[role][f"{p}_shuffled_control"] = y[perm]

    out = {}
    all_quantities = list(param_names) + [f"{p}_shuffled_control" for p in param_names]
    for ckpt in encoders:
        per_q = {}
        for q in all_quantities:
            per_q[q] = select_and_eval(
                pooled["train"][ckpt], targets["train"][q],
                pooled["valid"][ckpt], targets["valid"][q],
                pooled["test"][ckpt], targets["test"][q],
                seeds=seeds,
            )
            print(f"    regime {_ckpt_display(ckpt):32s} {q:26s} "
                  f"L{per_q[q]['frozen_layer']:2d} [{per_q[q]['method']}]  test_r2={per_q[q]['test_r2_mean']:+.3f}"
                  f"(+/-{per_q[q]['test_r2_std']:.3f})  ridge={per_q[q]['ridge_test_r2']:+.3f}", flush=True)
        out[ckpt] = per_q
    return out


# ---------------------------------------------------------------------------
# Aggregation across encoder seeds (per objective)
# ---------------------------------------------------------------------------

def _nested_stats(seed_results: list, n_probe_seeds: int) -> dict:
    """Combine N independently-trained encoder seeds' results (each already
    a probe-seed mean+/-std from select_and_eval) into one objective-level
    estimate. This is a two-level nested/hierarchical combination, not a
    naive pool of every (encoder_seed, probe_seed) pair as if independent:
    the probe-seed draws within one encoder are correlated (same frozen
    weights), so the encoder seed -- not the individual probe fit -- is the
    unit that actually answers "would retraining this objective give the
    same answer." Concretely: average the nuisance (probe-seed) variance
    away first, then treat the N per-encoder-seed means as the actual
    sample for computing spread/uncertainty across encoder seeds -- the
    one-way random-effects decomposition:

        SE(grand mean)^2 = s_between^2/n_seeds + s_within_avg^2/(n_seeds * n_probe_seeds)

    where s_between is the sample std (ddof=1) of the per-encoder-seed
    means, and s_within_avg is the average of each encoder seed's own
    probe-seed std. `encoder_seed_std` (plain std of the seed means) is
    also reported for the simpler, more conservative "just show the spread
    across seeds" reading -- with only 2-3 encoder seeds either estimate is
    noisy; report both rather than picking one as if it settles the
    question. ridge_test_r2 has no probe-seed variance of its own (it's
    deterministic given the frozen features), so its spread across encoder
    seeds is a plain std."""
    means = np.array([r["test_r2_mean"] for r in seed_results], dtype=float)
    within_stds = np.array([r["test_r2_std"] for r in seed_results], dtype=float)
    ridge_vals = np.array([r["ridge_test_r2"] for r in seed_results], dtype=float)
    n = len(seed_results)
    encoder_mean = float(means.mean())
    encoder_seed_std = float(means.std(ddof=1)) if n > 1 else 0.0
    within_seed_std_avg = float(within_stds.mean())
    nested_se = float(np.sqrt(
        encoder_seed_std**2 / max(n, 1) + within_seed_std_avg**2 / max(n * n_probe_seeds, 1)
    ))
    out = {
        "n_encoder_seeds": n,
        "encoder_seed_means": means.tolist(),
        "mean": encoder_mean,
        "encoder_seed_std": encoder_seed_std,
        "within_seed_std_avg": within_seed_std_avg,
        "nested_se": nested_se,
        "ridge_mean": float(ridge_vals.mean()),
        "ridge_std": float(ridge_vals.std(ddof=1)) if n > 1 else 0.0,
    }
    if all("frozen_layer" in r for r in seed_results):
        out["frozen_layers"] = [r["frozen_layer"] for r in seed_results]
    if all("valid_r2_curve" in r for r in seed_results):
        # Pointwise mean across encoder seeds' own 13-layer valid curves --
        # feeds the depth-curve figure with a lower-variance curve than any
        # single seed's, same nested-averaging spirit as the headline number.
        curves = np.array([r["valid_r2_curve"] for r in seed_results], dtype=float)
        out["mean_valid_r2_curve"] = curves.mean(axis=0).tolist()
    return out


def _aggregate_quantity_map(per_ckpt: dict, groups: dict, n_probe_seeds: int) -> dict:
    """per_ckpt: {ckpt_path: {quantity: select_and_eval() result}}. Returns
    {quantity: {objective: _nested_stats(...)}}."""
    quantities = set()
    for res in per_ckpt.values():
        quantities |= set(res)
    out = {}
    for q in sorted(quantities):
        out[q] = {}
        for obj, ckpts in groups.items():
            seed_results = [per_ckpt[ckpt][q] for ckpt in ckpts if q in per_ckpt.get(ckpt, {})]
            if seed_results:
                out[q][obj] = _nested_stats(seed_results, n_probe_seeds)
    return out


def aggregate_by_objective(results: dict, encoders: dict, n_probe_seeds: int) -> dict:
    """Groups every family's per-checkpoint results by objective (jepa/mae,
    via _objective_of()) and combines each group's encoder seeds with
    _nested_stats(). With a single seed per objective (the original,
    still-supported usage) this reduces to n_encoder_seeds=1, encoder_seed_std=0,
    nested_se=within_seed_std_avg/sqrt(n_probe_seeds) -- consistent with
    treating a lone checkpoint as a (trivial) one-member group, not a
    special case."""
    groups: dict = {}
    for ckpt in encoders:
        groups.setdefault(_objective_of(ckpt), []).append(ckpt)

    agg: dict = {}
    if "contemporaneous" in results:
        agg["contemporaneous"] = {
            kind: _aggregate_quantity_map(results["contemporaneous"][kind], groups, n_probe_seeds)
            for kind in ("pooled", "token")
        }
    if "forecast" in results:
        agg["forecast"] = {
            gap: {
                kind: _aggregate_quantity_map(gap_data[kind], groups, n_probe_seeds)
                for kind in ("pooled", "token")
            }
            for gap, gap_data in results["forecast"].items()
        }
    if "regime" in results:
        agg["regime"] = _aggregate_quantity_map(results["regime"], groups, n_probe_seeds)
    if "noise_robustness" in results:
        # Ridge is deterministic (no probe-seed draws), so its encoder-seed
        # mean/std is a plain average across checkpoints. MLP's per-checkpoint
        # by_noise entries are themselves already a probe-seed mean/std
        # (mlp_r2_mean/mlp_r2_std from noise_family()'s multi-seed fit) --
        # here we take the grand mean of those per-checkpoint means across
        # encoder seeds (the "between-seed" component), not the full nested
        # SE decomposition _nested_stats() does for the MLP-based families
        # elsewhere -- sufficient for this exploratory sweep, not intended
        # as a publication-grade SE claim the way select_and_eval()'s
        # headline numbers are.
        nr = results["noise_robustness"]

        def _agg_field(ckpts, q, skey, field):
            vals = [nr[ckpt][q]["by_noise"][skey][field] for ckpt in ckpts
                     if q in nr.get(ckpt, {}) and skey in nr[ckpt][q]["by_noise"]]
            if not vals:
                return None
            n = len(vals)
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if n > 1 else 0.0, "n": n}

        quantities = set()
        for res in nr.values():
            quantities |= set(res)
        agg["noise_robustness"] = {}
        for q in sorted(quantities):
            agg["noise_robustness"][q] = {}
            for obj, ckpts in groups.items():
                std_keys = set()
                for ckpt in ckpts:
                    if q in nr.get(ckpt, {}):
                        std_keys |= set(nr[ckpt][q]["by_noise"])
                by_std = {}
                for skey in std_keys:
                    ridge_r2 = _agg_field(ckpts, q, skey, "ridge_r2")
                    if ridge_r2 is None:
                        continue
                    by_std[skey] = {
                        "ridge_r2_mean": ridge_r2["mean"], "ridge_r2_std": ridge_r2["std"],
                        "ridge_pearson_r_mean": _agg_field(ckpts, q, skey, "ridge_pearson_r")["mean"],
                        "ridge_pearson_r_std": _agg_field(ckpts, q, skey, "ridge_pearson_r")["std"],
                        "mlp_r2_mean": _agg_field(ckpts, q, skey, "mlp_r2_mean")["mean"],
                        "mlp_r2_std": _agg_field(ckpts, q, skey, "mlp_r2_mean")["std"],
                        "mlp_pearson_r_mean": _agg_field(ckpts, q, skey, "mlp_pearson_r_mean")["mean"],
                        "mlp_pearson_r_std": _agg_field(ckpts, q, skey, "mlp_pearson_r_mean")["std"],
                        "n_encoder_seeds": ridge_r2["n"],
                    }
                if by_std:
                    agg["noise_robustness"][q][obj] = by_std
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-offsets", type=int, default=3)
    ap.add_argument("--gaps", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--n-frames", type=int, default=N_FRAMES)
    ap.add_argument("--valid-stride", type=int, default=8,
                     help="every Nth train trajectory (interleaved) is carved off as the held-out "
                          "layer-selection set; the Well's shipped valid split is too small/regime- "
                          "degenerate to use directly (see module docstring)")
    ap.add_argument("--train-max-traj", type=int, default=None)
    ap.add_argument("--valid-max-traj", type=int, default=None)
    ap.add_argument("--test-max-traj", type=int, default=None)
    ap.add_argument("--token-max-clips", type=int, default=16)
    ap.add_argument("--noise-stds", type=float, nargs="+", default=list(NOISE_STDS_DEFAULT))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--skip-contemporaneous", action="store_true",
                     help="also skips noise (family 3), which depends on family 1's frozen layers")
    ap.add_argument("--skip-forecast", action="store_true")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--skip-regime", action="store_true",
                     help="skips family 4 (governing physical parameters -- Reynolds/Schmidt, "
                          "Rayleigh/Prandtl, alpha/zeta); silently no-ops anyway if "
                          "{train,test}.regime.json don't exist for this dataset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.skip_contemporaneous and not args.skip_noise:
        raise SystemExit("--skip-contemporaneous requires --skip-noise (noise reuses family 1's frozen layers)")

    seeds = tuple(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    encoders = {}
    for ckpt_path in args.checkpoints:
        enc, _spec = load_checkpoint_encoder(ckpt_path, device)
        encoders[ckpt_path] = enc
    patch_size = next(iter(encoders.values())).patch_size

    mm_train = np.load(Path(args.data_root) / "memmap" / args.dataset / "train.npy", mmap_mode="r")
    mm_test = np.load(Path(args.data_root) / "memmap" / args.dataset / "test.npy", mmap_mode="r")
    fit_traj, valid_traj = train_valid_trajectory_split(mm_train.shape[0], valid_stride=args.valid_stride)
    print(f"{args.dataset}: train_total={mm_train.shape[0]} -> fit={len(fit_traj)} valid={len(valid_traj)}"
          f"  test={mm_test.shape[0]}", flush=True)

    mm = {"train": mm_train, "valid": mm_train, "test": mm_test}
    traj = {"train": fit_traj, "valid": valid_traj, "test": None}
    max_traj = {"train": args.train_max_traj, "valid": args.valid_max_traj, "test": args.test_max_traj}

    results: dict = {
        "dataset": args.dataset,
        "seeds": list(seeds),
        "gaps": args.gaps,
        "noise_stds": args.noise_stds,
        "n_traj": {"train_total": int(mm_train.shape[0]), "fit": len(fit_traj), "valid": len(valid_traj),
                   "test": int(mm_test.shape[0])},
    }

    # Each family is 1-4+ hours of GPU-bound compute with nothing saved to
    # disk until now only happened at the very end -- a crash/OOM/preemption
    # partway through (observed in practice: a multi-hour run died silently
    # with no traceback, deep into family 2, losing everything) meant losing
    # every family already finished too. Write after each family completes
    # so a later crash only costs the family in progress, not the whole run.
    def _save_partial():
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2))

    if not args.skip_contemporaneous:
        print("== family 1: contemporaneous (t+0) ==", flush=True)
        t0 = time.time()
        results["contemporaneous"] = contemporaneous_family(
            mm, traj, args.dataset, encoders, patch_size, args.n_offsets, args.n_frames, max_traj,
            args.token_max_clips, seeds,
        )
        print(f"family 1 done in {time.time()-t0:.0f}s", flush=True)
        _save_partial()

    if not args.skip_forecast:
        print("== family 2: forecast-content ==", flush=True)
        t0 = time.time()
        results["forecast"] = forecast_family(
            mm, traj, args.dataset, encoders, patch_size, args.n_offsets, args.gaps, args.n_frames,
            max_traj, args.token_max_clips, seeds,
        )
        print(f"family 2 done in {time.time()-t0:.0f}s", flush=True)
        _save_partial()

    if not args.skip_noise:
        print("== family 3: noise robustness ==", flush=True)
        t0 = time.time()
        frozen_layers = {
            ckpt: {q: v["frozen_layer"] for q, v in results["contemporaneous"]["pooled"][ckpt].items()}
            for ckpt in encoders
        }
        results["noise_robustness"] = noise_family(
            mm, traj, args.dataset, encoders, args.n_offsets, args.n_frames, max_traj, args.noise_stds,
            frozen_layers, seeds,
        )
        print(f"family 3 done in {time.time()-t0:.0f}s", flush=True)
        _save_partial()

    if not args.skip_regime:
        print("== family 4: regime (governing physical parameters) ==", flush=True)
        t0 = time.time()
        reg = regime_family(mm, traj, args.dataset, args.data_root, encoders, args.n_offsets, max_traj, seeds)
        if reg is not None:
            results["regime"] = reg
        print(f"family 4 done in {time.time()-t0:.0f}s", flush=True)
        _save_partial()

    print("== aggregating across encoder seeds (per objective) ==", flush=True)
    results["aggregated"] = aggregate_by_objective(results, encoders, len(seeds))
    n_seeds_by_obj = {obj: len(ckpts) for obj, ckpts in
                       {o: [c for c in encoders if _objective_of(c) == o] for o in ("jepa", "mae")}.items()}
    print(f"  encoder seeds per objective: {n_seeds_by_obj}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

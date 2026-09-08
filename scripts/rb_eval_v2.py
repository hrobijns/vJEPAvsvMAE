"""Corrected Rayleigh--Benard analysis primitives and command-line entry point.

The expensive data/cache/extraction commands are added below these pure,
unit-tested contracts.  Nothing in this module imports the legacy RB target
functions from :mod:`scripts.analyze_encoders`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA_VERSION = "rb-probe-v2"
N_FRAMES = 8
GAPS = (0, 8, 32)
RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def target_start(context_start: int, gap: int) -> int:
    if gap not in GAPS:
        raise ValueError(f"gap must be one of {GAPS}, got {gap}")
    return int(context_start if gap == 0 else context_start + N_FRAMES + gap)


def pooled_offsets(stratum: str, *, onset: int) -> list[int]:
    if stratum == "original_support":
        lo, hi = 0, 101 - (2 * N_FRAMES + max(GAPS))
    elif stratum == "developed":
        lo, hi = int(onset), 200 - (2 * N_FRAMES + max(GAPS))
    else:
        raise ValueError(f"unknown stratum {stratum!r}")
    if lo > hi:
        raise ValueError(f"onset {lo} leaves no room for the t+32 target (last start {hi})")
    return np.linspace(lo, hi, 3).round().astype(int).tolist()


def token_offset(stratum: str, *, onset: int, replicate: int) -> int:
    if replicate not in range(5):
        raise ValueError("replicate must be 0..4")
    if stratum == "original_support":
        lo, hi = 0, 53
    elif stratum == "developed":
        lo, hi = int(onset), 152
    else:
        raise ValueError(f"unknown stratum {stratum!r}")
    if lo > hi:
        raise ValueError(f"onset {lo} leaves no room for the t+32 target")
    return int(np.rint(np.linspace(lo, hi, 5)[replicate]))


def regime_folds(records: list[dict]) -> list[dict[str, list[int]]]:
    by_regime: dict[tuple[float, float], list[int]] = {}
    for row, record in enumerate(records):
        try:
            key = (float(record["Rayleigh"]), float(record["Prandtl"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"record {row} lacks numeric Rayleigh/Prandtl") from exc
        by_regime.setdefault(key, []).append(row)
    if len(by_regime) != 35:
        raise ValueError(f"expected 35 RB regimes, found {len(by_regime)}")
    if any(len(rows) != 5 for rows in by_regime.values()):
        counts = sorted({len(rows) for rows in by_regime.values()})
        raise ValueError(f"expected five trajectories per regime, found counts {counts}")
    ordered = {key: sorted(rows) for key, rows in sorted(by_regime.items())}
    folds = []
    all_rows = set(range(len(records)))
    for replicate in range(5):
        select = sorted(rows[replicate] for rows in ordered.values())
        folds.append({"fit": sorted(all_rows - set(select)), "select": select})
    return folds


def token_indices(*, trajectory: int, n_tokens: int = 1024, n_select: int = 64) -> np.ndarray:
    if not 0 < n_select <= n_tokens:
        raise ValueError("n_select must be in 1..n_tokens")
    seed = np.random.SeedSequence([20260825, int(trajectory)])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_tokens, size=n_select, replace=False))


def group_folds(groups: np.ndarray, n_folds: int = 5) -> list[dict[str, np.ndarray]]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if unique.size < n_folds:
        raise ValueError(f"need at least {n_folds} groups, found {unique.size}")
    buckets = np.array_split(unique, n_folds)
    folds = []
    for held in buckets:
        select = np.flatnonzero(np.isin(groups, held))
        fit = np.flatnonzero(~np.isin(groups, held))
        folds.append({"fit": fit, "select": select})
    return folds


def _ridge_predict(x_fit: torch.Tensor, y_fit: torch.Tensor, x_eval: torch.Tensor, alpha: float) -> torch.Tensor:
    x_fit = x_fit.double()
    y_fit = y_fit.double()
    x_eval = x_eval.double()
    mean = x_fit.mean(dim=0)
    std = x_fit.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    x_train = (x_fit - mean) / std
    x_test = (x_eval - mean) / std
    y_mean = y_fit.mean()
    eye = torch.eye(x_train.shape[1], dtype=x_train.dtype, device=x_train.device)
    gram = x_train.T @ x_train + float(alpha) * x_train.shape[0] * eye
    weights = torch.linalg.solve(gram, x_train.T @ (y_fit - y_mean))
    return x_test @ weights + y_mean


def r2_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction, target = prediction.double(), target.double()
    total = (target - target.mean()).square().sum()
    if total <= 0:
        return float("nan")
    return float(1.0 - (prediction - target).square().sum() / total)


def pearson_r(prediction: torch.Tensor, target: torch.Tensor) -> float:
    p = prediction.double() - prediction.double().mean()
    y = target.double() - target.double().mean()
    denom = p.norm() * y.norm()
    return float((p @ y) / denom) if denom > 0 else float("nan")


def select_ridge_cv(
    features_by_layer: list[torch.Tensor],
    target: torch.Tensor,
    folds: list[dict[str, np.ndarray]],
    *,
    alphas: Iterable[float] = RIDGE_ALPHAS,
) -> dict:
    best = None
    curves = []
    for layer, features in enumerate(features_by_layer):
        layer_best = None
        for alpha in alphas:
            scores = []
            for fold in folds:
                fit = torch.as_tensor(fold["fit"], dtype=torch.long)
                select = torch.as_tensor(fold["select"], dtype=torch.long)
                pred = _ridge_predict(features[fit], target[fit], features[select], float(alpha))
                scores.append(r2_score(pred, target[select]))
            score = float(np.nanmean(scores))
            candidate = {"layer": layer, "alpha": float(alpha), "cv_r2": score, "fold_r2": scores}
            if layer_best is None or score > layer_best["cv_r2"]:
                layer_best = candidate
            if best is None or score > best["cv_r2"]:
                best = candidate
        curves.append(layer_best)
    assert best is not None
    return {**best, "layer_curve": curves}


def nuisance_basis(*, rayleigh: np.ndarray, prandtl: np.ndarray, age: np.ndarray) -> np.ndarray:
    ra = np.log10(np.asarray(rayleigh, dtype=np.float64))
    pr = np.log10(np.asarray(prandtl, dtype=np.float64))
    t = np.asarray(age, dtype=np.float64)
    if not (ra.shape == pr.shape == t.shape):
        raise ValueError("rayleigh, prandtl, and age must have the same shape")
    return np.column_stack(
        [np.ones_like(ra), ra, pr, t, ra**2, pr**2, t**2, ra * pr, ra * t, pr * t]
    )


def add_standardized_noise(clip: torch.Tensor, *, sigma: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=clip.device).manual_seed(int(seed))
    noise = torch.randn(clip.shape, dtype=clip.dtype, device=clip.device, generator=generator)
    return clip + float(sigma) * noise


def validate_checkpoint_payload(payload: dict) -> dict:
    cfg = payload.get("config", {})
    objective = cfg.get("objective_name")
    seed = cfg.get("seed")
    expected_lr = {"jepa": 5e-5, "mae": 1e-4}
    if objective not in expected_lr:
        raise ValueError(f"objective_name must be jepa or mae, got {objective!r}")
    if seed not in (1, 2, 3):
        raise ValueError(f"seed must be one of 1,2,3, got {seed!r}")
    if payload.get("step") != 100_000 or cfg.get("optim", {}).get("total_steps") != 100_000:
        raise ValueError("checkpoint must be the final 100000-step endpoint")
    if cfg.get("data", {}).get("dataset_name") != "rayleigh_benard":
        raise ValueError("checkpoint dataset must be rayleigh_benard")
    encoder = cfg.get("encoder", {})
    expected_encoder = {
        "patch_t": 2,
        "patch_h": 16,
        "patch_w": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    }
    for key, expected in expected_encoder.items():
        if encoder.get(key) != expected:
            raise ValueError(f"encoder {key} must be {expected}, got {encoder.get(key)!r}")
    lr = float(cfg.get("optim", {}).get("lr", float("nan")))
    if not np.isclose(lr, expected_lr[objective], rtol=0, atol=1e-12):
        raise ValueError(f"{objective} lr must be {expected_lr[objective]}, got {lr}")
    head = cfg.get("objective", {})
    expected_head = (
        {
            "mask_ratio": 0.9,
            "predictor_dim": 192,
            "predictor_depth": 6,
            "predictor_heads": 6,
            "ema_start": 0.996,
            "ema_end": 1.0,
        }
        if objective == "jepa"
        else {
            "mask_ratio": 0.9,
            "decoder_dim": 192,
            "decoder_depth": 4,
            "decoder_heads": 6,
            "norm_pix": True,
        }
    )
    for key, expected in expected_head.items():
        value = head.get(key)
        matches = np.isclose(value, expected) if isinstance(expected, float) and value is not None else value == expected
        if not matches:
            raise ValueError(f"{key} must be {expected}, got {value!r}")
    spec = payload.get("spec", {})
    if spec != {"n_channels": 4, "n_frames": 8, "height": 512, "width": 128}:
        raise ValueError(f"unexpected checkpoint input spec {spec!r}")
    return {"objective": objective, "seed": int(seed), "step": 100_000, "lr": lr}


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_envelope(results: dict, *, manifest: dict, provenance: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": canonical_hash(manifest),
        "provenance": provenance or {},
        "results": results,
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cli_validate_checkpoint(paths: list[str]) -> int:
    seen = set()
    rows = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        try:
            meta = validate_checkpoint_payload(payload)
        except ValueError as exc:
            raise SystemExit(f"checkpoint validation failed for {path}: {exc}") from exc
        key = (meta["objective"], meta["seed"])
        if key in seen:
            raise SystemExit(f"duplicate checkpoint for {key}")
        seen.add(key)
        rows.append({"path": str(Path(path)), "sha256": sha256_file(path), **meta})
    expected = {(objective, seed) for objective in ("jepa", "mae") for seed in (1, 2, 3)}
    if seen != expected:
        raise SystemExit(f"checkpoint set incomplete: missing {sorted(expected - seen)}")
    print(json.dumps({"schema_version": SCHEMA_VERSION, "git_sha": _git_sha(), "checkpoints": rows}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-checkpoints", help="validate the six final RB endpoints")
    validate.add_argument("checkpoints", nargs="+")

    prepare = sub.add_parser("prepare-cache", help="cache one complete official split")
    prepare.add_argument("--base", required=True, help="The Well data root")
    prepare.add_argument("--split", required=True, choices=("valid", "test"))
    prepare.add_argument("--cache-root", default="sweep_results/rb_v2/cache")
    prepare.add_argument("--skip-source-sha256", action="store_true")
    prepare.add_argument("--skip-cache-sha256", action="store_true")

    extract = sub.add_parser("extract-features", help="extract one checkpoint's features")
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--cache-root", default="sweep_results/rb_v2/cache")
    extract.add_argument("--feature-root", default="sweep_results/rb_v2/features")
    extract.add_argument("--splits", nargs="+", choices=("valid", "test"), default=["valid", "test"])
    extract.add_argument("--batch-size", type=int, default=4)

    fit = sub.add_parser("fit-probes", help="fit valid-selected probes for one checkpoint")
    fit.add_argument("--feature-dir", required=True)
    fit.add_argument("--cache-root", default="sweep_results/rb_v2/cache")
    fit.add_argument("--output", required=True)
    fit.add_argument("--ridge-only", action="store_true")
    fit.add_argument("--mlp-max-steps", type=int, default=2000)
    fit.add_argument("--n-boot", type=int, default=1000)

    selected = sub.add_parser(
        "fit-selected-probes",
        help="select MLP layers and Ridge-versus-MLP families for one checkpoint",
    )
    selected.add_argument("--feature-dir", required=True)
    selected.add_argument("--cache-root", default="sweep_results/rb_v2/cache")
    selected.add_argument("--output", required=True)
    selected.add_argument("--mlp-max-steps", type=int, default=2000)
    selected.add_argument("--n-boot", type=int, default=1000)

    selected_noise = sub.add_parser(
        "fit-selected-noise",
        help="evaluate the saved pooled t+0 probe selections on noisy test features",
    )
    selected_noise.add_argument("--feature-dir", required=True)
    selected_noise.add_argument("--selected-feature-dir", required=True)
    selected_noise.add_argument("--cache-root", required=True)
    selected_noise.add_argument("--selected-result", required=True)
    selected_noise.add_argument("--output", required=True)
    selected_noise.add_argument("--clean-tolerance", type=float, default=1e-6)

    mlp_depth = sub.add_parser(
        "fit-mlp-depth",
        help="evaluate clean-fit pooled MLP probes at every encoder layer",
    )
    mlp_depth.add_argument("--feature-dir", required=True)
    mlp_depth.add_argument("--cache-root", required=True)
    mlp_depth.add_argument("--selected-result", required=True)
    mlp_depth.add_argument("--output", required=True)
    mlp_depth.add_argument("--selected-tolerance", type=float, default=1e-6)

    persistence = sub.add_parser(
        "persistence-baseline",
        help="score current physical targets as predictions of future targets",
    )
    persistence.add_argument("--cache-root", default="sweep_results/rb_v2/cache")
    persistence.add_argument(
        "--output",
        default="sweep_results/rb_v2/persistence_baseline.json",
    )
    persistence.add_argument("--n-boot", type=int, default=1000)

    token_control = sub.add_parser(
        "balanced-token-control",
        help="refit the token-position control with time-balanced validation folds",
    )
    token_control.add_argument("--cache-root", default="sweep_results/rb_v2/cache")
    token_control.add_argument(
        "--output",
        default="sweep_results/rb_v2_selected_probes/token_position_control.json",
    )
    token_control.add_argument("--n-boot", type=int, default=1000)

    aggregate = sub.add_parser("aggregate", help="combine exactly six checkpoint results")
    aggregate.add_argument("results", nargs="+")
    aggregate.add_argument("--output", default="sweep_results/rb_v2/aggregate.json")

    plot = sub.add_parser("plot", help="render paper-ready figures outside paper/")
    plot.add_argument("--aggregate", required=True)
    plot.add_argument("--output-dir", default="sweep_results/rb_v2/figures")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-checkpoints":
        return _cli_validate_checkpoint(args.checkpoints)
    from scripts import rb_pipeline_v2 as pipeline

    if args.command == "prepare-cache":
        path = pipeline.prepare_cache(
            base=args.base,
            split=args.split,
            cache_root=args.cache_root,
            hash_source_files=not args.skip_source_sha256,
            hash_cache_files=not args.skip_cache_sha256,
        )
    elif args.command == "extract-features":
        path = pipeline.extract_checkpoint_features(
            checkpoint=args.checkpoint,
            cache_root=args.cache_root,
            feature_root=args.feature_root,
            splits=args.splits,
            batch_size=args.batch_size,
        )
    elif args.command == "fit-probes":
        path = pipeline.fit_checkpoint_probes(
            feature_dir=args.feature_dir,
            cache_root=args.cache_root,
            output=args.output,
            include_mlp=not args.ridge_only,
            mlp_max_steps=args.mlp_max_steps,
            n_boot=args.n_boot,
        )
    elif args.command == "fit-selected-probes":
        path = pipeline.fit_checkpoint_selected_probes(
            feature_dir=args.feature_dir,
            cache_root=args.cache_root,
            output=args.output,
            mlp_max_steps=args.mlp_max_steps,
            n_boot=args.n_boot,
        )
    elif args.command == "fit-selected-noise":
        path = pipeline.fit_checkpoint_selected_noise(
            feature_dir=args.feature_dir,
            selected_feature_dir=args.selected_feature_dir,
            cache_root=args.cache_root,
            selected_result=args.selected_result,
            output=args.output,
            clean_tolerance=args.clean_tolerance,
        )
    elif args.command == "fit-mlp-depth":
        path = pipeline.fit_checkpoint_mlp_depth(
            feature_dir=args.feature_dir,
            cache_root=args.cache_root,
            selected_result=args.selected_result,
            output=args.output,
            selected_tolerance=args.selected_tolerance,
        )
    elif args.command == "persistence-baseline":
        path = pipeline.compute_persistence_baseline(
            cache_root=args.cache_root,
            output=args.output,
            n_boot=args.n_boot,
        )
    elif args.command == "balanced-token-control":
        path = pipeline.compute_balanced_token_position_control(
            cache_root=args.cache_root,
            output=args.output,
            n_boot=args.n_boot,
        )
    elif args.command == "aggregate":
        path = pipeline.aggregate_result_files(args.results, args.output)
    elif args.command == "plot":
        path = pipeline.plot_aggregate(args.aggregate, args.output_dir)
    else:
        raise AssertionError(args.command)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

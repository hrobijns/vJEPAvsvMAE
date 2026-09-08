"""Executable, split-safe Rayleigh--Benard analysis pipeline.

This module is the I/O and orchestration layer for :mod:`scripts.rb_eval_v2`
and :mod:`scripts.rb_targets_v2`.  The cache contains normalized encoder
inputs, targets derived after returning to physical units, and enough row-
level metadata to reproduce every sample.  It never reads the legacy
101-frame RB memmaps.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.extract_regime_metadata import parse_regime
from scripts.rb_eval_v2 import (
    GAPS,
    N_FRAMES,
    RIDGE_ALPHAS,
    SCHEMA_VERSION,
    _ridge_predict,
    canonical_hash,
    nuisance_basis,
    pearson_r,
    pooled_offsets,
    r2_score,
    sha256_file,
    target_start,
    token_indices,
    token_offset,
    validate_checkpoint_payload,
)
from scripts.rb_targets_v2 import (
    PRIMARY_TARGETS,
    TOKEN_ONLY_TARGETS,
    detect_onset_from_energy,
    kinetic_energy_series,
)

EXPECTED_TRAJECTORIES = 175
EXPECTED_REGIMES = 35
EXPECTED_REPLICATES = 5
EXPECTED_FRAMES = 200
NOISE_SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
NOISE_SEEDS = (0, 1, 2)
POOLED_TARGETS = PRIMARY_TARGETS + ("buoyancy_gradient_energy_bprime",)
TOKEN_TARGETS = PRIMARY_TARGETS + TOKEN_ONLY_TARGETS + ("buoyancy_gradient_energy_bprime",)
PERSISTENCE_TARGETS = (
    "enstrophy",
    "buoyancy_gradient_energy",
    "convective_flux",
    "pressure_gradient_magnitude",
    "buoyancy_laplacian_magnitude",
)
PERSISTENCE_GAPS = (8, 32)


def validate_official_split(records: Sequence[dict]) -> None:
    """Require The Well's complete, regime-balanced RB valid/test split."""
    if len(records) != EXPECTED_TRAJECTORIES:
        raise ValueError(f"official RB split must contain 175 trajectories, found {len(records)}")
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in records:
        try:
            grouped[(float(row["Rayleigh"]), float(row["Prandtl"]))].append(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid RB regime record: {row!r}") from exc
    if len(grouped) != EXPECTED_REGIMES:
        raise ValueError(f"official RB split must contain 35 regimes, found {len(grouped)}")
    bad = {str(key): len(value) for key, value in grouped.items() if len(value) != 5}
    if bad:
        raise ValueError(f"each RB regime must contain five trajectories: {bad}")


def assign_replicates(records: Sequence[dict]) -> list[dict]:
    """Assign stable within-regime replicate ids in source trajectory order."""
    seen: dict[tuple[float, float], int] = defaultdict(int)
    output = []
    for trajectory, record in enumerate(records):
        row = dict(record)
        key = (float(row["Rayleigh"]), float(row["Prandtl"]))
        row["trajectory"] = trajectory
        row["replicate"] = seen[key]
        seen[key] += 1
        output.append(row)
    return output


def _sample_record(record: dict, *, stratum: str, kind: str, onset: int, offset: int) -> dict:
    return {
        "sample": None,
        "trajectory": int(record["trajectory"]),
        "source_file": record["file"],
        "Rayleigh": float(record["Rayleigh"]),
        "Prandtl": float(record["Prandtl"]),
        "replicate": int(record["replicate"]),
        "stratum": stratum,
        "kind": kind,
        "onset": int(onset),
        "context_start": int(offset),
        "age": float(offset / (EXPECTED_FRAMES - 1)),
    }


def build_sample_table(
    records: Sequence[dict], onsets: Sequence[int], stratum: str, kind: str
) -> list[dict]:
    if len(records) != len(onsets):
        raise ValueError("records and onsets must have the same length")
    if kind not in {"pooled", "token"}:
        raise ValueError("kind must be pooled or token")
    output = []
    for record, onset in zip(records, onsets):
        if kind == "pooled":
            offsets = pooled_offsets(stratum, onset=int(onset))
        else:
            offsets = [
                token_offset(stratum, onset=int(onset), replicate=int(record["replicate"]))
            ]
        for offset in offsets:
            row = _sample_record(record, stratum=stratum, kind=kind, onset=int(onset), offset=offset)
            row["sample"] = len(output)
            output.append(row)
    return output


def _channel_view(values: torch.Tensor, ndim: int) -> torch.Tensor:
    if values.numel() != 4:
        raise ValueError(f"RB normalization requires four flattened channels, got {values.numel()}")
    return values.reshape(1, 4, *([1] * (ndim - 2)))


def normalize_clip(raw: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    if raw.ndim not in (4, 5) or raw.shape[-4] != 4:
        raise ValueError(f"expected (4,T,X,Y) or (B,4,T,X,Y), got {tuple(raw.shape)}")
    batched = raw.unsqueeze(0) if raw.ndim == 4 else raw
    mean = _channel_view(torch.as_tensor(means, dtype=batched.dtype, device=batched.device), 5)
    std = _channel_view(torch.as_tensor(stds, dtype=batched.dtype, device=batched.device), 5)
    if torch.any(std <= 0):
        raise ValueError("normalization standard deviations must be positive")
    result = (batched - mean) / std
    return result.squeeze(0) if raw.ndim == 4 else result


def denormalize_clip(clip: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    if clip.ndim not in (4, 5) or clip.shape[-4] != 4:
        raise ValueError(f"expected (4,T,X,Y) or (B,4,T,X,Y), got {tuple(clip.shape)}")
    batched = clip.unsqueeze(0) if clip.ndim == 4 else clip
    mean = _channel_view(torch.as_tensor(means, dtype=batched.dtype, device=batched.device), 5)
    std = _channel_view(torch.as_tensor(stds, dtype=batched.dtype, device=batched.device), 5)
    result = batched * std + mean
    return result.squeeze(0) if clip.ndim == 4 else result


@torch.no_grad()
def layerwise_features(
    encoder: nn.Module, clip: torch.Tensor, *, token_positions: np.ndarray | torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(B,L,D)`` pooled and optional ``(B,K,L,D)`` token features."""
    x = encoder.tokenize(clip)
    states = []
    for block in encoder.blocks:
        x = block(x)
        states.append(x.float())
    states.append(encoder.norm(x).float())
    pooled = torch.stack([state.mean(dim=1) for state in states], dim=1).cpu()
    if token_positions is None:
        return pooled, None
    positions = torch.as_tensor(token_positions, dtype=torch.long, device=x.device)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)
    if positions.shape[0] != x.shape[0]:
        raise ValueError("token_positions must have one row per clip")
    selected = []
    for state in states:
        gather = positions.unsqueeze(-1).expand(-1, -1, state.shape[-1])
        selected.append(torch.gather(state, 1, gather))
    return pooled, torch.stack(selected, dim=2).cpu()


def paired_noise_batch(
    clips: torch.Tensor,
    *,
    sigma: float,
    corruption_seed: int,
    sample_indices: Sequence[int] | np.ndarray,
) -> torch.Tensor:
    """Add deterministic per-sample noise, identical for every checkpoint."""
    if sigma == 0:
        return clips.clone()
    if len(sample_indices) != clips.shape[0]:
        raise ValueError("sample_indices must align with clips")
    output = clips.clone()
    for row, sample in enumerate(sample_indices):
        seed = int(np.random.SeedSequence([20260825, int(corruption_seed), int(sample)]).generate_state(1)[0])
        generator = torch.Generator(device=clips.device).manual_seed(seed)
        noise = torch.randn(
            clips[row].shape, dtype=clips.dtype, device=clips.device, generator=generator
        )
        output[row].add_(noise, alpha=float(sigma))
    return output


def atomic_write_json(path: str | Path, value: object, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _r2_np(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.square(prediction - target).sum()
    total = np.square(target - target.mean()).sum()
    return float(1 - residual / total) if total > 0 else float("nan")


def stratified_block_bootstrap_r2(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    regimes: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Trajectory-block bootstrap, sampling trajectories within each regime."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(groups)
    regimes = np.asarray(regimes)
    if not (prediction.shape == target.shape == groups.shape == regimes.shape):
        raise ValueError("prediction, target, groups, and regimes must be row-aligned")
    rng = np.random.default_rng(seed)
    group_to_rows = {group: np.flatnonzero(groups == group) for group in np.unique(groups)}
    regime_groups: dict[object, list] = defaultdict(list)
    for group, rows in group_to_rows.items():
        keys = np.unique(regimes[rows])
        if keys.size != 1:
            raise ValueError(f"trajectory group {group} spans multiple regimes")
        regime_groups[keys[0]].append(group)
    values = []
    for _ in range(int(n_boot)):
        draw_rows = []
        for regime in sorted(regime_groups, key=str):
            candidates = np.asarray(regime_groups[regime])
            sampled = rng.choice(candidates, size=len(candidates), replace=True)
            draw_rows.extend(group_to_rows[group] for group in sampled)
        index = np.concatenate(draw_rows)
        values.append(_r2_np(prediction[index], target[index]))
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n_boot": int(n_boot), "median": None, "ci_low": None, "ci_high": None}
    return {
        "n_boot": int(n_boot),
        "median": float(np.median(finite)),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
    }


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_manifest() -> list[dict]:
    """Hash the exact analysis sources, including uncommitted implementations."""
    directory = Path(__file__).resolve().parent
    names = (
        "rb_eval_v2.py",
        "rb_pipeline_v2.py",
        "rb_targets_v2.py",
        "rb_derivatives.py",
        "rb_quadrature.py",
    )
    return [
        {"path": f"scripts/{name}", "sha256": sha256_file(directory / name)}
        for name in names
    ]


def _source_records(dataset) -> list[dict]:
    records = []
    for file_path, count in zip(dataset.files_paths, dataset.n_trajectories_per_file):
        name = Path(str(file_path)).name
        regime = parse_regime(name, "rayleigh_benard")
        for _ in range(int(count)):
            records.append({"file": name, **regime})
    if len(records) != len(dataset):
        raise ValueError(f"source index has {len(records)} records but dataset has {len(dataset)}")
    return assign_replicates(records)


class RBWellSource:
    """Raw full trajectories plus official normalization metadata."""

    def __init__(self, base: str | Path, split: str):
        from the_well.data import WellDataset
        from the_well.data.normalization import ZScoreNormalization

        base = Path(base)
        well_base = base / "datasets" if (base / "datasets" / "rayleigh_benard").is_dir() else base
        kwargs = dict(
            well_base_path=str(well_base),
            well_dataset_name="rayleigh_benard",
            well_split_name=split,
            n_steps_input=1,
            n_steps_output=0,
            max_rollout_steps=EXPECTED_FRAMES - 1,
            full_trajectory_mode=True,
            flatten_tensors=True,
            return_grid=False,
            boundary_return_type=None,
        )
        self.raw = WellDataset(use_normalization=False, **kwargs)
        normalized = WellDataset(
            use_normalization=True, normalization_type=ZScoreNormalization, **kwargs
        )
        self.means = normalized.norm.flattened_means["variable"].float().cpu()
        self.stds = normalized.norm.flattened_stds["variable"].float().cpu()
        self.field_names = list(normalized.core_field_names)
        self.records = _source_records(self.raw)
        self.split = split
        validate_official_split(self.records)
        if self.field_names != ["buoyancy", "pressure", "velocity"]:
            raise ValueError(
                "unexpected RB field order; expected buoyancy, pressure, velocity, "
                f"found {self.field_names!r}"
            )
        if self.means.numel() != 4 or self.stds.numel() != 4:
            raise ValueError(
                f"expected four flattened RB variables, got means={self.means.numel()} stds={self.stds.numel()}"
            )

    def __len__(self) -> int:
        return len(self.raw)

    def trajectory(self, index: int) -> torch.Tensor:
        sample = self.raw[index]
        fields = torch.cat([sample["input_fields"], sample["output_fields"]], dim=0)
        raw = fields.permute(3, 0, 1, 2).contiguous().float()
        if raw.shape != (4, EXPECTED_FRAMES, 512, 128):
            raise ValueError(
                f"trajectory {index} expected (4,200,512,128), got {tuple(raw.shape)}"
            )
        if not torch.isfinite(raw).all():
            raise ValueError(f"trajectory {index} contains non-finite values")
        return raw

    def source_manifest(self, *, hash_files: bool) -> list[dict]:
        rows = []
        for file_path in self.raw.files_paths:
            path = Path(str(file_path))
            row = {"path": str(path), "name": path.name}
            if path.exists():
                stat = path.stat()
                row.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                if hash_files:
                    row["sha256"] = sha256_file(path)
            elif hash_files:
                raise FileNotFoundError(f"cannot hash non-local source {path}")
            rows.append(row)
        return rows


def _allocate_cache_arrays(root: Path, n_trajectories: int) -> dict:
    arrays = {}
    for stratum in ("original_support", "developed"):
        directory = root / stratum
        directory.mkdir(parents=True)
        for kind, n_samples, targets in (
            ("pooled", n_trajectories * 3, POOLED_TARGETS),
            ("token", n_trajectories, TOKEN_TARGETS),
        ):
            shape = (n_samples, 4, N_FRAMES, 512, 128)
            arrays[(stratum, kind, "context")] = np.lib.format.open_memmap(
                directory / f"{kind}_context.npy", mode="w+", dtype=np.float16, shape=shape
            )
            if kind == "token":
                arrays[(stratum, kind, "positions")] = np.lib.format.open_memmap(
                    directory / "token_positions.npy", mode="w+", dtype=np.int16, shape=(n_samples, 64)
                )
            for gap in GAPS:
                for target in targets:
                    target_shape = (n_samples,) if kind == "pooled" else (n_samples, 64)
                    arrays[(stratum, kind, gap, target)] = np.lib.format.open_memmap(
                        directory / f"{kind}_gap{gap}_{target}.npy",
                        mode="w+",
                        dtype=np.float64,
                        shape=target_shape,
                    )
    return arrays


def _target_values(raw_clip: torch.Tensor, kind: str, positions: np.ndarray | None) -> dict[str, np.ndarray]:
    from scripts import rb_derivatives as derivatives
    from scripts.rb_targets_v2 import (
        buoyancy_fluctuation,
        pooled_targets,
        token_targets,
        volume_mean,
        weighted_patch_mean,
    )

    batch = raw_clip.unsqueeze(0).to(torch.float64)
    bprime_gradient = derivatives.grad_sq(buoyancy_fluctuation(batch[:, 0]))
    if kind == "pooled":
        values = {name: value.cpu().numpy() for name, value in pooled_targets(batch).items()}
        values["buoyancy_gradient_energy_bprime"] = volume_mean(bprime_gradient).cpu().numpy()
        return values
    full = {name: value.cpu().numpy() for name, value in token_targets(batch).items()}
    full["buoyancy_gradient_energy_bprime"] = weighted_patch_mean(
        bprime_gradient, (2, 16, 16)
    ).cpu().numpy()
    assert positions is not None
    return {name: value[:, positions] for name, value in full.items()}


def prepare_cache(
    *,
    base: str | Path,
    split: str,
    cache_root: str | Path,
    hash_source_files: bool = True,
    hash_cache_files: bool = True,
) -> Path:
    """Materialize one immutable official-split cache."""
    if split not in {"valid", "test"}:
        raise ValueError("RB-v2 only permits official valid or test splits")
    destination = Path(cache_root) / split
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite completed cache {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{split}.partial.{os.getpid()}"
    if stage.exists():
        raise FileExistsError(f"staging directory already exists: {stage}")
    stage.mkdir()
    source = RBWellSource(base, split)
    arrays = _allocate_cache_arrays(stage, len(source))
    sample_tables = {
        (stratum, kind): []
        for stratum in ("original_support", "developed")
        for kind in ("pooled", "token")
    }
    onsets = []
    try:
        for trajectory in range(len(source)):
            raw = source.trajectory(trajectory)
            onset = int(detect_onset_from_energy(kinetic_energy_series(raw)))
            onsets.append(onset)
            record = source.records[trajectory]
            for stratum in ("original_support", "developed"):
                for kind in ("pooled", "token"):
                    if kind == "pooled":
                        offsets = pooled_offsets(stratum, onset=onset)
                    else:
                        offsets = [
                            token_offset(
                                stratum,
                                onset=onset,
                                replicate=int(record["replicate"]),
                            )
                        ]
                    for offset in offsets:
                        row_index = len(sample_tables[(stratum, kind)])
                        row = _sample_record(
                            record, stratum=stratum, kind=kind, onset=onset, offset=offset
                        )
                        row["sample"] = row_index
                        sample_tables[(stratum, kind)].append(row)
                        context = raw[:, offset : offset + N_FRAMES]
                        arrays[(stratum, kind, "context")][row_index] = normalize_clip(
                            context, source.means, source.stds
                        ).half().numpy()
                        positions = None
                        if kind == "token":
                            positions = token_indices(trajectory=trajectory)
                            arrays[(stratum, kind, "positions")][row_index] = positions
                        for gap in GAPS:
                            start = target_start(offset, gap)
                            target_clip = raw[:, start : start + N_FRAMES]
                            values = _target_values(target_clip, kind, positions)
                            for target, value in values.items():
                                arrays[(stratum, kind, gap, target)][row_index] = value[0]
            print(f"{split}: cached trajectory {trajectory + 1}/{len(source)}", flush=True)
        for array in arrays.values():
            array.flush()
        for (stratum, kind), rows in sample_tables.items():
            atomic_write_json(stage / stratum / f"{kind}_samples.json", rows)
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        cache_files = []
        for path in files:
            stat = path.stat()
            row = {"path": str(path.relative_to(stage)), "size": stat.st_size}
            if hash_cache_files:
                row["sha256"] = sha256_file(path)
            cache_files.append(row)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "rb-target-cache",
            "split": split,
            "git_sha": git_sha(),
            "analysis_code": code_manifest(),
            "frames": EXPECTED_FRAMES,
            "context_frames": N_FRAMES,
            "gaps": list(GAPS),
            "strata": ["original_support", "developed"],
            "normalization": {
                "source": "The Well stats.yaml / ZScoreNormalization",
                "field_names": source.field_names,
                "flattened_means": source.means.tolist(),
                "flattened_stds": source.stds.tolist(),
            },
            "onsets": onsets,
            "trajectory_records": source.records,
            "source_files": source.source_manifest(hash_files=hash_source_files),
            "cache_files": cache_files,
        }
        manifest["manifest_sha256"] = canonical_hash(manifest)
        atomic_write_json(stage / "manifest.json", manifest)
        stage.rename(destination)
    except BaseException:
        print(f"cache preparation failed; partial data retained at {stage}", flush=True)
        raise
    return destination


def _open_cache(cache_root: str | Path, split: str) -> tuple[Path, dict]:
    root = Path(cache_root) / split
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("split") != split:
        raise ValueError(f"incompatible cache manifest at {root}")
    check = dict(manifest)
    claimed = check.pop("manifest_sha256")
    if canonical_hash(check) != claimed:
        raise ValueError(f"cache manifest hash mismatch at {root}")
    return root, manifest


def _checkpoint_id(meta: dict) -> str:
    return f"{meta['objective']}_seed{meta['seed']}"


def extract_checkpoint_features(
    *,
    checkpoint: str | Path,
    cache_root: str | Path,
    feature_root: str | Path,
    splits: Sequence[str] = ("valid", "test"),
    batch_size: int = 4,
    include_noise: bool = True,
) -> Path:
    from scripts.load_encoder import load_encoder

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = validate_checkpoint_payload(payload)
    encoder, _, _ = load_encoder(str(checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    destination = Path(feature_root) / _checkpoint_id(meta)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite completed features {destination}")
    stage = destination.parent / f".{destination.name}.partial.{os.getpid()}"
    stage.mkdir(parents=True)
    inventory = []
    try:
        for split in splits:
            cache, manifest = _open_cache(cache_root, split)
            for stratum in ("original_support", "developed"):
                for kind in ("pooled", "token"):
                    contexts = np.load(cache / stratum / f"{kind}_context.npy", mmap_mode="r")
                    positions = (
                        np.load(cache / stratum / "token_positions.npy", mmap_mode="r")
                        if kind == "token"
                        else None
                    )
                    output_dir = stage / split / stratum
                    output_dir.mkdir(parents=True, exist_ok=True)
                    n_layers = len(encoder.blocks) + 1
                    if kind == "pooled":
                        shape = (len(contexts), n_layers, encoder.embed_dim)
                    else:
                        shape = (len(contexts), 64, n_layers, encoder.embed_dim)
                    output_path = output_dir / f"{kind}.npy"
                    output = np.lib.format.open_memmap(
                        output_path, mode="w+", dtype=np.float32, shape=shape
                    )
                    for start in range(0, len(contexts), batch_size):
                        stop = min(start + batch_size, len(contexts))
                        batch = torch.from_numpy(np.array(contexts[start:stop])).float().to(device)
                        pooled, tokens = layerwise_features(
                            encoder,
                            batch,
                            token_positions=None if positions is None else positions[start:stop],
                        )
                        output[start:stop] = pooled.numpy() if kind == "pooled" else tokens.numpy()
                    output.flush()
                    inventory.append({"path": str(output_path.relative_to(stage)), "sha256": sha256_file(output_path)})

                if include_noise and split == "test":
                    contexts = np.load(cache / stratum / "pooled_context.npy", mmap_mode="r")
                    output_dir = stage / split / stratum
                    for sigma in NOISE_SIGMAS:
                        for corruption_seed in NOISE_SEEDS:
                            output_path = output_dir / f"pooled_noise_sigma{sigma:g}_seed{corruption_seed}.npy"
                            output = np.lib.format.open_memmap(
                                output_path,
                                mode="w+",
                                dtype=np.float32,
                                shape=(len(contexts), len(encoder.blocks) + 1, encoder.embed_dim),
                            )
                            for start in range(0, len(contexts), batch_size):
                                stop = min(start + batch_size, len(contexts))
                                batch = torch.from_numpy(np.array(contexts[start:stop])).float().to(device)
                                noisy = paired_noise_batch(
                                    batch,
                                    sigma=sigma,
                                    corruption_seed=corruption_seed,
                                    sample_indices=np.arange(start, stop),
                                )
                                pooled, _ = layerwise_features(encoder, noisy)
                                output[start:stop] = pooled.numpy()
                            output.flush()
                            inventory.append(
                                {"path": str(output_path.relative_to(stage)), "sha256": sha256_file(output_path)}
                            )
            inventory.append(
                {
                    "cache_split": split,
                    "cache_manifest_sha256": manifest["manifest_sha256"],
                }
            )
        feature_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "rb-checkpoint-features",
            "git_sha": git_sha(),
            "analysis_code": code_manifest(),
            "checkpoint": {
                "path": str(Path(checkpoint)),
                "sha256": sha256_file(checkpoint),
                **meta,
            },
            "batch_size": batch_size,
            "noise_sigmas": list(NOISE_SIGMAS) if include_noise else [],
            "noise_seeds": list(NOISE_SEEDS) if include_noise else [],
            "files": inventory,
        }
        feature_manifest["manifest_sha256"] = canonical_hash(feature_manifest)
        atomic_write_json(stage / "manifest.json", feature_manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(destination)
    except BaseException:
        print(f"feature extraction failed; partial data retained at {stage}", flush=True)
        raise
    return destination


def _balanced_regime_folds(samples: Sequence[dict]) -> list[dict[str, np.ndarray]]:
    """Hold out one trajectory per regime while balancing replicate-linked times."""
    trajectories = np.asarray([int(row["trajectory"]) for row in samples])
    trajectory_metadata: dict[int, tuple[tuple[float, float], int]] = {}
    for row in samples:
        trajectory = int(row["trajectory"])
        metadata = (
            (float(row["Rayleigh"]), float(row["Prandtl"])),
            int(row["replicate"]),
        )
        previous = trajectory_metadata.setdefault(trajectory, metadata)
        if previous != metadata:
            raise ValueError(f"trajectory {trajectory} has inconsistent regime or replicate metadata")
    by_regime: dict[tuple[float, float], dict[int, int]] = defaultdict(dict)
    for trajectory, (regime, replicate) in trajectory_metadata.items():
        if replicate in by_regime[regime]:
            raise ValueError(f"regime {regime} has duplicate replicate {replicate}")
        by_regime[regime][replicate] = trajectory
    expected_replicates = set(range(EXPECTED_REPLICATES))
    for regime, members in by_regime.items():
        if set(members) != expected_replicates:
            raise ValueError(f"regime {regime} does not contain replicates 0..4")

    folds = []
    ordered_regimes = sorted(by_regime)
    for fold in range(EXPECTED_REPLICATES):
        selected_trajectories = {
            by_regime[regime][(fold + regime_index) % EXPECTED_REPLICATES]
            for regime_index, regime in enumerate(ordered_regimes)
        }
        select = np.flatnonzero(np.isin(trajectories, list(selected_trajectories)))
        fit = np.flatnonzero(~np.isin(trajectories, list(selected_trajectories)))
        if select.size == 0 or fit.size == 0:
            raise ValueError(f"empty official-valid fold {fold}")
        folds.append({"fit": fit, "select": select})
    return folds


def _balanced_fold_ids(samples: Sequence[dict]) -> np.ndarray:
    fold_ids = np.full(len(samples), -1, dtype=np.int64)
    for fold, indices in enumerate(_balanced_regime_folds(samples)):
        if np.any(fold_ids[indices["select"]] != -1):
            raise ValueError("validation folds overlap")
        fold_ids[indices["select"]] = fold
    if np.any(fold_ids < 0):
        raise ValueError("validation folds do not cover every sample")
    return fold_ids


def _sample_bootstrap_labels(samples: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray([int(row["trajectory"]) for row in samples])
    regimes = np.asarray(
        [f"{float(row['Rayleigh']):.17g}|{float(row['Prandtl']):.17g}" for row in samples]
    )
    return groups, regimes


def compute_persistence_baseline(
    *,
    cache_root: str | Path,
    output: str | Path,
    n_boot: int = 1000,
) -> Path:
    """Score the current physical target as a prediction of each future target."""
    cache, manifest = _open_cache(cache_root, "test")
    stratum = "original_support"
    rows = []
    for representation in ("pooled", "token"):
        samples = _load_samples(cache, stratum, representation)
        for target in PERSISTENCE_TARGETS:
            current_raw = np.load(
                cache / stratum / f"{representation}_gap0_{target}.npy",
                mmap_mode="r",
            )
            if representation == "pooled":
                if current_raw.shape != (len(samples),):
                    raise ValueError(
                        f"pooled persistence target {target} has shape {current_raw.shape}; "
                        f"expected {(len(samples),)}"
                    )
                aligned_samples = samples
            else:
                if current_raw.ndim != 2 or current_raw.shape[0] != len(samples):
                    raise ValueError(
                        f"token persistence target {target} has shape {current_raw.shape}; "
                        f"expected ({len(samples)}, n_tokens)"
                    )
                aligned_samples = _repeat_samples(samples, int(current_raw.shape[1]))
            prediction = np.asarray(current_raw, dtype=np.float64).reshape(-1)
            groups, regimes = _sample_bootstrap_labels(aligned_samples)
            for gap in PERSISTENCE_GAPS:
                future_raw = np.load(
                    cache / stratum / f"{representation}_gap{gap}_{target}.npy",
                    mmap_mode="r",
                )
                if future_raw.shape != current_raw.shape:
                    raise ValueError(
                        f"persistence targets are not aligned for {representation}/{target}/gap{gap}: "
                        f"current={current_raw.shape}, future={future_raw.shape}"
                    )
                future = np.asarray(future_raw, dtype=np.float64).reshape(-1)
                residual = prediction - future
                cell = {
                    "family": "physics",
                    "method": "persistence",
                    "representation": representation,
                    "stratum": stratum,
                    "gap": int(gap),
                    "target": target,
                    "metric": "r2",
                }
                rows.append(
                    {
                        "objective": "shared",
                        **cell,
                        "cell_id": _cell_id(**cell),
                        "n_test": int(future.size),
                        "test_r2": _r2_np(prediction, future),
                        "test_log_mse": float(
                            math.log10(np.mean(np.square(residual)) + 1e-300)
                        ),
                        "bootstrap": stratified_block_bootstrap_r2(
                            prediction,
                            future,
                            groups,
                            regimes,
                            n_boot=n_boot,
                            seed=20260825,
                        ),
                    }
                )

    summary = []
    for representation in ("pooled", "token"):
        for gap in PERSISTENCE_GAPS:
            values = [
                row
                for row in rows
                if row["representation"] == representation and row["gap"] == gap
            ]
            summary.append(
                {
                    "method": "persistence",
                    "representation": representation,
                    "stratum": stratum,
                    "gap": int(gap),
                    "five_target_r2_mean": float(
                        np.mean([row["test_r2"] for row in values])
                    ),
                    "test_r2_by_target": {
                        row["target"]: row["test_r2"] for row in values
                    },
                }
            )

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "rb-persistence-baseline",
        "git_sha": git_sha(),
        "analysis_code": code_manifest(),
        "cache_manifest": {
            "path": str(cache / "manifest.json"),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "protocol": {
            "report_split": "test",
            "stratum": stratum,
            "prediction": "copy the aligned gap0 physical target",
            "gaps": list(PERSISTENCE_GAPS),
            "targets": list(PERSISTENCE_TARGETS),
            "r2_reference": "test-set target mean",
            "token_alignment": "same cached token index in current and future clips",
            "bootstrap": "trajectory blocks resampled within each regime",
            "n_boot": int(n_boot),
        },
        "rows": rows,
        "summary": summary,
    }
    output = Path(output)
    atomic_write_json(output, result)
    return output


def compute_balanced_token_position_control(
    *,
    cache_root: str | Path,
    output: str | Path,
    n_boot: int = 1000,
) -> Path:
    """Refit the deterministic token-position control with balanced folds."""
    valid_cache, valid_manifest = _open_cache(cache_root, "valid")
    test_cache, test_manifest = _open_cache(cache_root, "test")
    stratum = "original_support"
    valid_samples_base = _load_samples(valid_cache, stratum, "token")
    test_samples_base = _load_samples(test_cache, stratum, "token")
    valid_positions = np.load(valid_cache / stratum / "token_positions.npy", mmap_mode="r")
    test_positions = np.load(test_cache / stratum / "token_positions.npy", mmap_mode="r")
    if valid_positions.ndim != 2 or test_positions.ndim != 2:
        raise ValueError("token positions must be (samples,tokens)")
    if valid_positions.shape[1] != test_positions.shape[1]:
        raise ValueError("valid and test token counts differ")
    valid_samples = _repeat_samples(valid_samples_base, int(valid_positions.shape[1]))
    test_samples = _repeat_samples(test_samples_base, int(test_positions.shape[1]))
    valid_features = _as_layers(token_position_basis(valid_positions).reshape(-1, 10))
    test_features = _as_layers(token_position_basis(test_positions).reshape(-1, 10))
    valid_targets = {
        f"gap{gap}:{target}": np.load(
            valid_cache / stratum / f"token_gap{gap}_{target}.npy", mmap_mode="r"
        ).reshape(-1)
        for gap in GAPS
        for target in PERSISTENCE_TARGETS
    }
    test_targets = {
        f"gap{gap}:{target}": np.load(
            test_cache / stratum / f"token_gap{gap}_{target}.npy", mmap_mode="r"
        ).reshape(-1)
        for gap in GAPS
        for target in PERSISTENCE_TARGETS
    }
    fitted, _ = fit_ridge_cells_many(
        valid_features,
        valid_targets,
        test_features,
        test_targets,
        valid_samples=valid_samples,
        test_samples=test_samples,
        n_boot=n_boot,
    )
    rows = []
    for gap in GAPS:
        for target in PERSISTENCE_TARGETS:
            cell = {
                "family": "physics",
                "method": "ridge_position",
                "representation": "token",
                "stratum": stratum,
                "gap": int(gap),
                "target": target,
            }
            rows.append(
                {
                    "objective": "shared",
                    **cell,
                    "cell_id": _cell_id(**cell),
                    **fitted[f"gap{gap}:{target}"],
                }
            )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "rb-balanced-token-position-control",
        "git_sha": git_sha(),
        "analysis_code": code_manifest(),
        "cache_manifests": {
            "valid": valid_manifest["manifest_sha256"],
            "test": test_manifest["manifest_sha256"],
        },
        "protocol": {
            "selection_split": "valid",
            "report_split": "test",
            "stratum": stratum,
            "valid_folds": "one trajectory per regime with sampling times balanced across folds",
            "ridge_alphas": list(RIDGE_ALPHAS),
            "targets": list(PERSISTENCE_TARGETS),
            "gaps": list(GAPS),
        },
        "rows": rows,
    }
    output = Path(output)
    atomic_write_json(output, result)
    return output


def fit_ridge_cell(
    valid_features: np.ndarray | torch.Tensor,
    valid_target: np.ndarray | torch.Tensor,
    test_features: np.ndarray | torch.Tensor,
    test_target: np.ndarray | torch.Tensor,
    *,
    valid_samples: Sequence[dict],
    test_samples: Sequence[dict],
    alphas: Iterable[float] = RIDGE_ALPHAS,
    n_boot: int = 1000,
) -> tuple[dict, np.ndarray]:
    """Single-target wrapper around :func:`fit_ridge_cells_many`."""
    results, predictions = fit_ridge_cells_many(
        valid_features,
        {"target": valid_target},
        test_features,
        {"target": test_target},
        valid_samples=valid_samples,
        test_samples=test_samples,
        alphas=alphas,
        n_boot=n_boot,
    )
    return results["target"], predictions["target"]


def _ridge_design(
    fit_features: torch.Tensor, eval_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = fit_features.mean(dim=0)
    std = fit_features.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    fit = (fit_features - mean) / std
    evaluate = (eval_features - mean) / std
    gram = fit.T @ fit
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    return fit, evaluate, eigenvalues.clamp_min(0), eigenvectors


def fit_ridge_cells_many(
    valid_features: np.ndarray | torch.Tensor,
    valid_targets: dict[str, np.ndarray | torch.Tensor],
    test_features: np.ndarray | torch.Tensor,
    test_targets: dict[str, np.ndarray | torch.Tensor],
    *,
    valid_samples: Sequence[dict],
    test_samples: Sequence[dict],
    alphas: Iterable[float] = RIDGE_ALPHAS,
    n_boot: int = 1000,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Batched ridge selection, sharing each layer/fold factorization.

    All targets must share the same feature rows. This changes only runtime:
    it is algebraically the same standardized ridge fit and valid-only search
    as fitting every cell separately.
    """
    x_valid = torch.as_tensor(np.asarray(valid_features), dtype=torch.float32)
    x_test = torch.as_tensor(np.asarray(test_features), dtype=torch.float32)
    if x_valid.ndim != 3 or x_test.ndim != 3:
        raise ValueError("features must be (rows,layers,dimensions)")
    names = list(valid_targets)
    if not names or set(names) != set(test_targets):
        raise ValueError("valid_targets and test_targets must have the same non-empty keys")
    y_valid = torch.column_stack(
        [torch.as_tensor(np.asarray(valid_targets[name]), dtype=torch.float64).reshape(-1) for name in names]
    )
    y_test = torch.column_stack(
        [torch.as_tensor(np.asarray(test_targets[name]), dtype=torch.float64).reshape(-1) for name in names]
    )
    if not (x_valid.shape[0] == y_valid.shape[0] == len(valid_samples)):
        raise ValueError("valid features, targets, and sample metadata are not aligned")
    if not (x_test.shape[0] == y_test.shape[0] == len(test_samples)):
        raise ValueError("test features, targets, and sample metadata are not aligned")
    alphas = tuple(float(alpha) for alpha in alphas)
    folds = _balanced_regime_folds(valid_samples)
    n_layers, n_targets = x_valid.shape[1], len(names)
    cv = np.full((n_layers, len(alphas), n_targets, len(folds)), np.nan)
    x_valid64 = x_valid.double()
    x_test64 = x_test.double()
    for layer in range(n_layers):
        for fold_index, fold in enumerate(folds):
            fit_index = torch.as_tensor(fold["fit"], dtype=torch.long)
            select_index = torch.as_tensor(fold["select"], dtype=torch.long)
            fit, evaluate, eigenvalues, eigenvectors = _ridge_design(
                x_valid64[fit_index, layer], x_valid64[select_index, layer]
            )
            target_fit = y_valid[fit_index]
            target_mean = target_fit.mean(dim=0)
            projected = eigenvectors.T @ (fit.T @ (target_fit - target_mean))
            target_select = y_valid[select_index]
            total = (target_select - target_select.mean(dim=0)).square().sum(dim=0)
            for alpha_index, alpha in enumerate(alphas):
                weights = eigenvectors @ (
                    projected / (eigenvalues[:, None] + alpha * fit.shape[0])
                )
                prediction = evaluate @ weights + target_mean
                residual = (prediction - target_select).square().sum(dim=0)
                scores = 1.0 - residual / total
                scores = torch.where(total > 0, scores, torch.full_like(scores, float("nan")))
                cv[layer, alpha_index, :, fold_index] = scores.cpu().numpy()
    cv_mean = np.nanmean(cv, axis=3)
    layer_alpha = np.nanargmax(cv_mean.reshape(n_layers * len(alphas), n_targets), axis=0)
    selected_layers = layer_alpha // len(alphas)
    selected_alpha_indices = layer_alpha % len(alphas)
    depth_alpha_indices = np.nanargmax(cv_mean, axis=1)
    depth_predictions: list[list[np.ndarray | None]] = [
        [None for _ in range(n_targets)] for _ in range(n_layers)
    ]
    for layer in range(n_layers):
        fit, evaluate, eigenvalues, eigenvectors = _ridge_design(
            x_valid64[:, layer], x_test64[:, layer]
        )
        target_mean = y_valid.mean(dim=0)
        projected = eigenvectors.T @ (fit.T @ (y_valid - target_mean))
        for alpha_index in np.unique(depth_alpha_indices[layer]):
            target_indices = np.flatnonzero(depth_alpha_indices[layer] == alpha_index)
            weights = eigenvectors @ (
                projected[:, target_indices]
                / (eigenvalues[:, None] + alphas[int(alpha_index)] * fit.shape[0])
            )
            prediction = (evaluate @ weights + target_mean[target_indices]).cpu().numpy()
            for column, target_index in enumerate(target_indices):
                depth_predictions[layer][int(target_index)] = prediction[:, column]

    groups, regimes = _sample_bootstrap_labels(test_samples)
    results, predictions = {}, {}
    for target_index, name in enumerate(names):
        layer = int(selected_layers[target_index])
        alpha_index = int(selected_alpha_indices[target_index])
        prediction = np.asarray(depth_predictions[layer][target_index])
        target = y_test[:, target_index].numpy()
        depth_curve = []
        for depth_layer in range(n_layers):
            depth_prediction = np.asarray(depth_predictions[depth_layer][target_index])
            depth_alpha_index = int(depth_alpha_indices[depth_layer, target_index])
            depth_curve.append(
                {
                    "layer": depth_layer,
                    "alpha": alphas[depth_alpha_index],
                    "valid_cv_r2": float(cv_mean[depth_layer, depth_alpha_index, target_index]),
                    "test_r2": _r2_np(depth_prediction, target),
                }
            )
        prediction_t = torch.from_numpy(prediction)
        target_t = y_test[:, target_index]
        results[name] = {
            "selected_layer": layer,
            "selected_alpha": alphas[alpha_index],
            "valid_cv_r2": float(cv_mean[layer, alpha_index, target_index]),
            "test_r2": _r2_np(prediction, target),
            "test_pearson_r": pearson_r(prediction_t, target_t),
            "test_log_mse": float(math.log10(np.mean(np.square(prediction - target)) + 1e-300)),
            "depth_curve": depth_curve,
            "bootstrap": stratified_block_bootstrap_r2(
                prediction, target, groups, regimes, n_boot=n_boot, seed=20260825
            ),
        }
        predictions[name] = prediction
    return results, predictions


def aggregate_seed_rows(rows: Sequence[dict]) -> list[dict]:
    """Aggregate scientific units without treating probe fits as replicates."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    shared_controls: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("method") in {"ridge_nuisance", "ridge_position"}:
            shared_controls[str(row["cell_id"])].append(row)
        else:
            grouped[(str(row["objective"]), str(row["cell_id"]))].append(row)
    output = []
    for (objective, cell_id), values in sorted(grouped.items()):
        seeds = sorted(int(row["checkpoint_seed"]) for row in values)
        if seeds != [1, 2, 3]:
            raise ValueError(
                f"{objective}/{cell_id} requires exactly three checkpoint seeds 1,2,3; found {seeds}"
            )
        scores = np.asarray([float(row["test_r2"]) for row in values], dtype=float)
        summary = {
            "objective": objective,
            "cell_id": cell_id,
            "n_checkpoint_seeds": 3,
            "test_r2_mean": float(np.mean(scores)),
            "test_r2_std": float(np.std(scores, ddof=1)),
            "test_r2_per_seed": {str(row["checkpoint_seed"]): float(row["test_r2"]) for row in values},
        }
        exemplar = values[0]
        for key in (
            "family",
            "method",
            "representation",
            "stratum",
            "gap",
            "target",
            "metric",
            "sigma",
        ):
            if key in exemplar:
                summary[key] = exemplar[key]
        for metric in ("test_log_mse", "test_pearson_r"):
            if all(metric in row for row in values):
                metric_values = np.asarray([float(row[metric]) for row in values])
                summary[f"{metric}_mean"] = float(np.mean(metric_values))
                summary[f"{metric}_std"] = float(np.std(metric_values, ddof=1))
        if exemplar.get("method") in {"selected_probe", "selected_probe_clean_fit"}:
            summary["selected_method_per_seed"] = {
                str(row["checkpoint_seed"]): row["selected_method"] for row in values
            }
            summary["selected_layer_per_seed"] = {
                str(row["checkpoint_seed"]): int(row["selected_layer"]) for row in values
            }
        output.append(summary)
    for cell_id, values in sorted(shared_controls.items()):
        scores = np.asarray([float(row["test_r2"]) for row in values])
        if not np.allclose(scores, scores[0]):
            raise ValueError(f"shared control differs across checkpoints: {cell_id}")
        exemplar = values[0]
        summary = {
            "objective": "shared",
            "cell_id": cell_id,
            "test_r2_mean": float(scores[0]),
        }
        for key in ("family", "method", "representation", "stratum", "gap", "target", "metric"):
            if key in exemplar:
                summary[key] = exemplar[key]
        for metric in ("test_log_mse", "test_pearson_r"):
            if metric in exemplar:
                summary[f"{metric}_mean"] = float(exemplar[metric])
        if "bootstrap" in exemplar:
            summary["bootstrap"] = exemplar["bootstrap"]
        output.append(summary)
    return output


class _ProbeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def _standardize_fit(
    features: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-8)
    target_mean = target.mean()
    target_std = target.std(unbiased=False).clamp_min(1e-8)
    return (features - mean) / std, (target - target_mean) / target_std, mean, std, target_mean, target_std


def _select_mlp_steps(
    features: torch.Tensor,
    target: torch.Tensor,
    fold_ids: np.ndarray,
    *,
    seed: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_steps: int,
    min_steps: int,
    patience: int,
    check_every: int,
) -> int:
    return _select_mlp_steps_and_score(
        features,
        target,
        fold_ids,
        seed=seed,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
        min_steps=min_steps,
        patience=patience,
        check_every=check_every,
    )["selected_steps"]


def _select_mlp_steps_and_score(
    features: torch.Tensor,
    target: torch.Tensor,
    fold_ids: np.ndarray,
    *,
    seed: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_steps: int,
    min_steps: int,
    patience: int,
    check_every: int,
) -> dict:
    """Select stopping time on one validation fold and retain its R2."""
    if min_steps > max_steps:
        raise ValueError("min_steps cannot exceed max_steps")
    monitor = torch.as_tensor(fold_ids == (seed % EXPECTED_REPLICATES), device=features.device)
    train = ~monitor
    if monitor.sum() == 0 or train.sum() == 0:
        raise ValueError("MLP inner split is empty")
    x_train, y_train, mean, std, target_mean, target_std = _standardize_fit(
        features[train], target[train]
    )
    x_monitor = (features[monitor] - mean) / std
    y_monitor = (target[monitor] - target_mean) / target_std
    torch.manual_seed(seed)
    model = _ProbeMLP(features.shape[1], hidden, dropout).to(features.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_r2 = float("-inf")
    best_step = min_steps
    last_improvement = 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if step % check_every == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                monitor_prediction = model(x_monitor)
                monitor_loss = float(F.mse_loss(monitor_prediction, y_monitor))
            if step >= min_steps and monitor_loss < best_loss:
                best_loss = monitor_loss
                best_r2 = r2_score(monitor_prediction.detach().cpu(), y_monitor.detach().cpu())
                best_step = step
                last_improvement = step
            elif step >= min_steps and step - last_improvement >= patience:
                break
    return {
        "selected_steps": int(best_step),
        "valid_r2": float(best_r2),
        "monitor_fold": int(seed % EXPECTED_REPLICATES),
    }


def select_probe_family(ridge: dict, mlp: dict) -> dict:
    """Choose the probe family from validation scores without consulting test."""
    candidates = (ridge, mlp)
    for candidate in candidates:
        if not np.isfinite(float(candidate.get("valid_cv_r2", float("nan")))):
            raise ValueError("probe candidate lacks a finite valid_cv_r2")
        if "method" not in candidate or "test_r2" not in candidate:
            raise ValueError("probe candidate lacks method or test_r2")
    # Stable tie break keeps the simpler Ridge probe when validation is exact.
    selected = max(candidates, key=lambda row: (float(row["valid_cv_r2"]), row is ridge))
    return {
        **selected,
        "selected_method": selected["method"],
        "selection_rule": "maximum validation CV R2",
        "selection_candidates": {
            row["method"]: {
                "valid_cv_r2": float(row["valid_cv_r2"]),
                "test_r2": float(row["test_r2"]),
                "selected_layer": int(row["selected_layer"]),
            }
            for row in candidates
        },
    }


def fit_mlp_cell(
    valid_features: np.ndarray | torch.Tensor,
    valid_target: np.ndarray | torch.Tensor,
    test_features: np.ndarray | torch.Tensor,
    test_target: np.ndarray | torch.Tensor,
    *,
    valid_samples: Sequence[dict],
    test_samples: Sequence[dict],
    layer: int,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    hidden: int = 128,
    dropout: float = 0.1,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    max_steps: int = 2000,
    min_steps: int = 150,
    patience: int = 100,
    check_every: int = 20,
    n_boot: int = 1000,
) -> tuple[dict, np.ndarray]:
    """Choose training duration inside valid, refit all valid, score test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_all = torch.as_tensor(np.asarray(valid_features), dtype=torch.float32, device=device)
    test_all = torch.as_tensor(np.asarray(test_features), dtype=torch.float32, device=device)
    if valid_all.ndim != 3 or test_all.ndim != 3:
        raise ValueError("MLP features must be (rows,layers,dimensions)")
    valid = valid_all[:, int(layer)]
    test = test_all[:, int(layer)]
    y_valid = torch.as_tensor(np.asarray(valid_target), dtype=torch.float32, device=device).reshape(-1)
    y_test = torch.as_tensor(np.asarray(test_target), dtype=torch.float32, device=device).reshape(-1)
    fold_ids = _balanced_fold_ids(valid_samples)
    if valid.shape[0] != len(valid_samples) or test.shape[0] != len(test_samples):
        raise ValueError("MLP feature rows and sample metadata are not aligned")
    predictions = []
    selected_steps = []
    for seed in seeds:
        steps = _select_mlp_steps(
            valid,
            y_valid,
            fold_ids,
            seed=int(seed),
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            max_steps=max_steps,
            min_steps=min_steps,
            patience=patience,
            check_every=check_every,
        )
        selected_steps.append(steps)
        x_train, y_train, mean, std, target_mean, target_std = _standardize_fit(valid, y_valid)
        torch.manual_seed(int(seed))
        model = _ProbeMLP(valid.shape[1], hidden, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x_train), y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model((test - mean) / std) * target_std + target_mean
        predictions.append(prediction.double().cpu())
    prediction_stack = torch.stack(predictions)
    mean_prediction = prediction_stack.mean(dim=0)
    per_seed_r2 = [r2_score(prediction, y_test.cpu()) for prediction in prediction_stack]
    pred_np = mean_prediction.numpy()
    target_np = y_test.double().cpu().numpy()
    groups, regimes = _sample_bootstrap_labels(test_samples)
    result = {
        "selected_layer": int(layer),
        "layer_selection": "ridge_valid_cv",
        "probe_seeds": [int(seed) for seed in seeds],
        "selected_steps": selected_steps,
        "test_r2": r2_score(mean_prediction, y_test.cpu()),
        "test_r2_per_probe_seed": per_seed_r2,
        "test_r2_probe_seed_std": float(np.std(per_seed_r2, ddof=1)) if len(per_seed_r2) > 1 else 0.0,
        "test_pearson_r": pearson_r(mean_prediction, y_test.cpu()),
        "test_log_mse": float(math.log10(np.mean(np.square(pred_np - target_np)) + 1e-300)),
        "bootstrap": stratified_block_bootstrap_r2(
            pred_np, target_np, groups, regimes, n_boot=n_boot, seed=20260825
        ),
    }
    return result, pred_np


def fit_mlp_selected_layer_cell(
    valid_features: np.ndarray | torch.Tensor,
    valid_target: np.ndarray | torch.Tensor,
    test_features: np.ndarray | torch.Tensor,
    test_target: np.ndarray | torch.Tensor,
    *,
    valid_samples: Sequence[dict],
    test_samples: Sequence[dict],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    hidden: int = 128,
    dropout: float = 0.1,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    max_steps: int = 2000,
    min_steps: int = 150,
    patience: int = 100,
    check_every: int = 20,
    n_boot: int = 1000,
) -> tuple[dict, np.ndarray]:
    """Select an MLP layer using validation folds, then score test once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_all = torch.as_tensor(np.asarray(valid_features), dtype=torch.float32, device=device)
    test_all = torch.as_tensor(np.asarray(test_features), dtype=torch.float32, device=device)
    if valid_all.ndim != 3 or test_all.ndim != 3:
        raise ValueError("MLP features must be (rows,layers,dimensions)")
    if valid_all.shape[1:] != test_all.shape[1:]:
        raise ValueError("valid and test MLP layer/dimension shapes differ")
    y_valid = torch.as_tensor(np.asarray(valid_target), dtype=torch.float32, device=device).reshape(-1)
    y_test = torch.as_tensor(np.asarray(test_target), dtype=torch.float32, device=device).reshape(-1)
    if valid_all.shape[0] != y_valid.numel() or test_all.shape[0] != y_test.numel():
        raise ValueError("MLP feature and target rows are not aligned")
    if valid_all.shape[0] != len(valid_samples) or test_all.shape[0] != len(test_samples):
        raise ValueError("MLP feature rows and sample metadata are not aligned")
    seeds = tuple(int(seed) for seed in seeds)
    if sorted(seed % EXPECTED_REPLICATES for seed in seeds) != list(range(EXPECTED_REPLICATES)):
        raise ValueError("MLP layer selection requires one seed for each validation replicate")
    fold_ids = _balanced_fold_ids(valid_samples)

    layer_curve = []
    for layer in range(valid_all.shape[1]):
        fold_results = [
            _select_mlp_steps_and_score(
                valid_all[:, layer],
                y_valid,
                fold_ids,
                seed=seed,
                hidden=hidden,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                max_steps=max_steps,
                min_steps=min_steps,
                patience=patience,
                check_every=check_every,
            )
            for seed in seeds
        ]
        layer_curve.append(
            {
                "layer": int(layer),
                "valid_cv_r2": float(np.mean([row["valid_r2"] for row in fold_results])),
                "valid_fold_r2": [float(row["valid_r2"]) for row in fold_results],
                "selected_steps": [int(row["selected_steps"]) for row in fold_results],
            }
        )
    selected = max(layer_curve, key=lambda row: row["valid_cv_r2"])
    layer = int(selected["layer"])
    valid = valid_all[:, layer]
    test = test_all[:, layer]
    x_train, y_train, mean, std, target_mean, target_std = _standardize_fit(valid, y_valid)

    predictions = []
    for seed, steps in zip(seeds, selected["selected_steps"]):
        torch.manual_seed(seed)
        model = _ProbeMLP(valid.shape[1], hidden, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(int(steps)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x_train), y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model((test - mean) / std) * target_std + target_mean
        predictions.append(prediction.double().cpu())

    prediction_stack = torch.stack(predictions)
    mean_prediction = prediction_stack.mean(dim=0)
    per_seed_r2 = [r2_score(prediction, y_test.cpu()) for prediction in prediction_stack]
    pred_np = mean_prediction.numpy()
    target_np = y_test.double().cpu().numpy()
    groups, regimes = _sample_bootstrap_labels(test_samples)
    result = {
        "selected_layer": layer,
        "layer_selection": "mlp_valid_cv",
        "valid_cv_r2": float(selected["valid_cv_r2"]),
        "valid_fold_r2": list(selected["valid_fold_r2"]),
        "layer_curve": layer_curve,
        "probe_seeds": list(seeds),
        "selected_steps": list(selected["selected_steps"]),
        "test_r2": r2_score(mean_prediction, y_test.cpu()),
        "test_r2_per_probe_seed": per_seed_r2,
        "test_r2_probe_seed_std": float(np.std(per_seed_r2, ddof=1)) if len(per_seed_r2) > 1 else 0.0,
        "test_pearson_r": pearson_r(mean_prediction, y_test.cpu()),
        "test_log_mse": float(math.log10(np.mean(np.square(pred_np - target_np)) + 1e-300)),
        "bootstrap": stratified_block_bootstrap_r2(
            pred_np, target_np, groups, regimes, n_boot=n_boot, seed=20260825
        ),
    }
    return result, pred_np


def token_position_basis(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    time = positions // (32 * 8)
    remainder = positions % (32 * 8)
    x = remainder // 8
    y = remainder % 8
    axes = [
        2 * time / 3 - 1,
        2 * x / 31 - 1,
        2 * y / 7 - 1,
    ]
    a, b, c = [axis.astype(np.float64) for axis in axes]
    return np.stack(
        [np.ones_like(a), a, b, c, a**2, b**2, c**2, a * b, a * c, b * c], axis=-1
    )


def _load_samples(cache: Path, stratum: str, kind: str) -> list[dict]:
    return json.loads((cache / stratum / f"{kind}_samples.json").read_text())


def _repeat_samples(samples: Sequence[dict], repeats: int) -> list[dict]:
    return [row for row in samples for _ in range(repeats)]


def _nuisance_from_samples(samples: Sequence[dict]) -> np.ndarray:
    return nuisance_basis(
        rayleigh=np.asarray([row["Rayleigh"] for row in samples]),
        prandtl=np.asarray([row["Prandtl"] for row in samples]),
        age=np.asarray([row["age"] for row in samples]),
    )


def _as_layers(features: np.ndarray) -> np.ndarray:
    return features[:, None, :] if features.ndim == 2 else features


def _cell_id(**parts) -> str:
    order = ("family", "method", "representation", "stratum", "gap", "target", "sigma")
    return "|".join(f"{key}={parts[key]}" for key in order if key in parts and parts[key] is not None)


def _annotate(row: dict, *, objective: str, checkpoint_seed: int, **cell) -> dict:
    return {
        "objective": objective,
        "checkpoint_seed": int(checkpoint_seed),
        **cell,
        "cell_id": _cell_id(**cell),
        **row,
    }


def _trajectory_average(features: np.ndarray, samples: Sequence[dict]) -> tuple[np.ndarray, list[dict]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[int(sample["trajectory"])].append(index)
    values, rows = [], []
    for trajectory in sorted(grouped):
        indices = grouped[trajectory]
        values.append(np.asarray(features[indices]).mean(axis=0))
        rows.append(dict(samples[indices[0]]))
    return np.stack(values), rows


def _evaluate_clean_fit_noise(
    clean_noise_models: dict[tuple[str, str], dict],
    *,
    feature_dir: Path,
    objective: str,
    checkpoint_seed: int,
) -> list[dict]:
    rows = []
    for (stratum, target), model in clean_noise_models.items():
        layer, alpha = model["layer"], model["alpha"]
        x_valid = torch.as_tensor(np.asarray(model["valid_features"][:, layer]), dtype=torch.float32)
        y_valid = torch.as_tensor(np.asarray(model["valid_target"]), dtype=torch.float64)
        y_test = torch.as_tensor(np.asarray(model["test_target"]), dtype=torch.float64)
        for sigma in NOISE_SIGMAS:
            scores, correlations = [], []
            for corruption_seed in NOISE_SEEDS:
                path = (
                    feature_dir
                    / "test"
                    / stratum
                    / f"pooled_noise_sigma{sigma:g}_seed{corruption_seed}.npy"
                )
                if not path.exists():
                    raise FileNotFoundError(f"noise feature missing: {path}")
                noisy = np.load(path, mmap_mode="r")[:, layer]
                prediction = _ridge_predict(
                    x_valid,
                    y_valid,
                    torch.as_tensor(np.asarray(noisy), dtype=torch.float32),
                    alpha,
                )
                scores.append(r2_score(prediction, y_test))
                correlations.append(pearson_r(prediction, y_test))
            base_cell = {
                "family": "noise",
                "method": "ridge_encoder_clean_fit",
                "representation": "pooled",
                "stratum": stratum,
                "gap": 0,
                "target": target,
                "sigma": float(sigma),
            }
            rows.append(
                _annotate(
                    {
                        "selected_layer": int(layer),
                        "selected_alpha": float(alpha),
                        "test_r2": float(np.mean(scores)),
                        "test_r2_corruption_seed_std": float(np.std(scores, ddof=1)),
                        "test_r2_per_corruption_seed": scores,
                        "test_pearson_r": float(np.mean(correlations)),
                    },
                    objective=objective,
                    checkpoint_seed=checkpoint_seed,
                    **base_cell,
                )
            )
    return rows


def fit_checkpoint_probes(
    *,
    feature_dir: str | Path,
    cache_root: str | Path,
    output: str | Path,
    include_mlp: bool = True,
    mlp_max_steps: int = 2000,
    n_boot: int = 1000,
) -> Path:
    """Run every prespecified cell for one frozen encoder checkpoint."""
    feature_dir = Path(feature_dir)
    feature_manifest = json.loads((feature_dir / "manifest.json").read_text())
    if feature_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"incompatible feature manifest at {feature_dir}")
    feature_check = dict(feature_manifest)
    feature_claim = feature_check.pop("manifest_sha256", None)
    if feature_claim is None or canonical_hash(feature_check) != feature_claim:
        raise ValueError(f"feature manifest hash mismatch at {feature_dir}")
    checkpoint = feature_manifest["checkpoint"]
    objective, checkpoint_seed = checkpoint["objective"], int(checkpoint["seed"])
    if feature_dir.name != _checkpoint_id(checkpoint):
        raise ValueError(
            f"feature directory {feature_dir.name!r} does not match checkpoint {_checkpoint_id(checkpoint)!r}"
        )
    valid_cache, valid_manifest = _open_cache(cache_root, "valid")
    test_cache, test_manifest = _open_cache(cache_root, "test")
    feature_cache_manifests = {
        row["cache_split"]: row["cache_manifest_sha256"]
        for row in feature_manifest["files"]
        if "cache_split" in row
    }
    expected_cache_manifests = {
        "valid": valid_manifest["manifest_sha256"],
        "test": test_manifest["manifest_sha256"],
    }
    if feature_cache_manifests != expected_cache_manifests:
        raise ValueError("features were not extracted from the current valid/test cache manifests")
    rows = []
    clean_noise_models = {}

    for stratum in ("original_support", "developed"):
        for representation, targets in (("pooled", POOLED_TARGETS), ("token", TOKEN_TARGETS)):
            valid_samples_base = _load_samples(valid_cache, stratum, representation)
            test_samples_base = _load_samples(test_cache, stratum, representation)
            valid_features_raw = np.load(
                feature_dir / "valid" / stratum / f"{representation}.npy", mmap_mode="r"
            )
            test_features_raw = np.load(
                feature_dir / "test" / stratum / f"{representation}.npy", mmap_mode="r"
            )
            if representation == "token":
                n_tokens = valid_features_raw.shape[1]
                valid_features = np.asarray(valid_features_raw).reshape(
                    -1, valid_features_raw.shape[2], valid_features_raw.shape[3]
                )
                test_features = np.asarray(test_features_raw).reshape(
                    -1, test_features_raw.shape[2], test_features_raw.shape[3]
                )
                valid_samples = _repeat_samples(valid_samples_base, n_tokens)
                test_samples = _repeat_samples(test_samples_base, n_tokens)
                valid_positions = np.load(
                    valid_cache / stratum / "token_positions.npy", mmap_mode="r"
                )
                test_positions = np.load(
                    test_cache / stratum / "token_positions.npy", mmap_mode="r"
                )
                valid_baseline = token_position_basis(valid_positions).reshape(-1, 10)
                test_baseline = token_position_basis(test_positions).reshape(-1, 10)
            else:
                valid_features, test_features = valid_features_raw, test_features_raw
                valid_samples, test_samples = valid_samples_base, test_samples_base
                valid_baseline = _nuisance_from_samples(valid_samples)
                test_baseline = _nuisance_from_samples(test_samples)

            valid_targets, test_targets, cell_keys = {}, {}, {}
            for gap in GAPS:
                for target in targets:
                    key = f"gap{gap}:{target}"
                    valid_targets[key] = np.load(
                        valid_cache / stratum / f"{representation}_gap{gap}_{target}.npy",
                        mmap_mode="r",
                    ).reshape(-1)
                    test_targets[key] = np.load(
                        test_cache / stratum / f"{representation}_gap{gap}_{target}.npy",
                        mmap_mode="r",
                    ).reshape(-1)
                    cell_keys[key] = (gap, target)

            encoder_results, _ = fit_ridge_cells_many(
                valid_features,
                valid_targets,
                test_features,
                test_targets,
                valid_samples=valid_samples,
                test_samples=test_samples,
                n_boot=n_boot,
            )
            baseline_results, _ = fit_ridge_cells_many(
                _as_layers(valid_baseline),
                valid_targets,
                _as_layers(test_baseline),
                test_targets,
                valid_samples=valid_samples,
                test_samples=test_samples,
                n_boot=n_boot,
            )
            combined_results = None
            if representation == "pooled":
                valid_combined = np.concatenate(
                    [
                        np.asarray(valid_features),
                        np.repeat(valid_baseline[:, None, :], valid_features.shape[1], axis=1),
                    ],
                    axis=2,
                )
                test_combined = np.concatenate(
                    [
                        np.asarray(test_features),
                        np.repeat(test_baseline[:, None, :], test_features.shape[1], axis=1),
                    ],
                    axis=2,
                )
                combined_results, _ = fit_ridge_cells_many(
                    valid_combined,
                    valid_targets,
                    test_combined,
                    test_targets,
                    valid_samples=valid_samples,
                    test_samples=test_samples,
                    n_boot=n_boot,
                )

            for key, (gap, target) in cell_keys.items():
                valid_target = valid_targets[key]
                test_target = test_targets[key]
                base_cell = {
                    "family": "physics",
                    "representation": representation,
                    "stratum": stratum,
                    "gap": int(gap),
                    "target": target,
                }
                encoder_result = encoder_results[key]
                rows.append(
                    _annotate(
                        encoder_result,
                        objective=objective,
                        checkpoint_seed=checkpoint_seed,
                        method="ridge_encoder",
                        **base_cell,
                    )
                )
                baseline_result = baseline_results[key]
                rows.append(
                    _annotate(
                        baseline_result,
                        objective=objective,
                        checkpoint_seed=checkpoint_seed,
                        method="ridge_nuisance" if representation == "pooled" else "ridge_position",
                        **base_cell,
                    )
                )
                if representation == "pooled":
                    assert combined_results is not None
                    combined_result = combined_results[key]
                    rows.append(
                        _annotate(
                            combined_result,
                            objective=objective,
                            checkpoint_seed=checkpoint_seed,
                            method="ridge_encoder_plus_nuisance",
                            **base_cell,
                        )
                    )
                if include_mlp:
                    mlp_result, _ = fit_mlp_cell(
                        valid_features,
                        valid_target,
                        test_features,
                        test_target,
                        valid_samples=valid_samples,
                        test_samples=test_samples,
                        layer=encoder_result["selected_layer"],
                        max_steps=mlp_max_steps,
                        n_boot=n_boot,
                    )
                    rows.append(
                        _annotate(
                            mlp_result,
                            objective=objective,
                            checkpoint_seed=checkpoint_seed,
                            method="mlp_encoder",
                            **base_cell,
                        )
                    )
                if representation == "pooled" and gap == 0:
                    clean_noise_models[(stratum, target)] = {
                        "layer": encoder_result["selected_layer"],
                        "alpha": encoder_result["selected_alpha"],
                        "valid_features": valid_features,
                        "valid_target": valid_target,
                        "test_target": test_target,
                        "test_samples": test_samples,
                    }

        valid_pooled = np.load(feature_dir / "valid" / stratum / "pooled.npy", mmap_mode="r")
        test_pooled = np.load(feature_dir / "test" / stratum / "pooled.npy", mmap_mode="r")
        valid_pooled_samples = _load_samples(valid_cache, stratum, "pooled")
        test_pooled_samples = _load_samples(test_cache, stratum, "pooled")
        valid_regime_features, valid_regime_samples = _trajectory_average(
            valid_pooled, valid_pooled_samples
        )
        test_regime_features, test_regime_samples = _trajectory_average(
            test_pooled, test_pooled_samples
        )
        regime_parameters = {"log10_Rayleigh": "Rayleigh", "log10_Prandtl": "Prandtl"}
        valid_regime_targets = {
            target: np.log10([row[parameter] for row in valid_regime_samples])
            for target, parameter in regime_parameters.items()
        }
        test_regime_targets = {
            target: np.log10([row[parameter] for row in test_regime_samples])
            for target, parameter in regime_parameters.items()
        }
        regime_ridge_results, _ = fit_ridge_cells_many(
            valid_regime_features,
            valid_regime_targets,
            test_regime_features,
            test_regime_targets,
            valid_samples=valid_regime_samples,
            test_samples=test_regime_samples,
            n_boot=n_boot,
        )
        for target in regime_parameters:
            valid_target = valid_regime_targets[target]
            test_target = test_regime_targets[target]
            base_cell = {
                "family": "regime",
                "representation": "pooled",
                "stratum": stratum,
                "gap": None,
                "target": target,
            }
            ridge_result = regime_ridge_results[target]
            rows.append(
                _annotate(
                    ridge_result,
                    objective=objective,
                    checkpoint_seed=checkpoint_seed,
                    method="ridge_encoder",
                    **base_cell,
                )
            )
            if include_mlp:
                mlp_result, _ = fit_mlp_cell(
                    valid_regime_features,
                    valid_target,
                    test_regime_features,
                    test_target,
                    valid_samples=valid_regime_samples,
                    test_samples=test_regime_samples,
                    layer=ridge_result["selected_layer"],
                    max_steps=mlp_max_steps,
                    n_boot=n_boot,
                )
                rows.append(
                    _annotate(
                        mlp_result,
                        objective=objective,
                        checkpoint_seed=checkpoint_seed,
                        method="mlp_encoder",
                        **base_cell,
                    )
                )

    rows.extend(
        _evaluate_clean_fit_noise(
            clean_noise_models,
            feature_dir=feature_dir,
            objective=objective,
            checkpoint_seed=checkpoint_seed,
        )
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "rb-checkpoint-probes",
        "git_sha": git_sha(),
        "analysis_code": code_manifest(),
        "checkpoint": checkpoint,
        "feature_manifest_sha256": feature_manifest["manifest_sha256"],
        "cache_manifests": {
            "valid": valid_manifest["manifest_sha256"],
            "test": test_manifest["manifest_sha256"],
        },
        "protocol": {
            "selection_split": "valid",
            "report_split": "test",
            "valid_folds": "one trajectory replicate per each of 35 regimes",
            "ridge_alphas": list(RIDGE_ALPHAS),
            "mlp_layer_selection": "ridge-valid-CV layer; never test",
            "checkpoint_seed_is_scientific_unit": True,
            "formal_significance_test": None,
        },
        "rows": rows,
    }
    output = Path(output)
    atomic_write_json(output, result)
    return output


def fit_checkpoint_selected_probes(
    *,
    feature_dir: str | Path,
    cache_root: str | Path,
    output: str | Path,
    mlp_max_steps: int = 2000,
    n_boot: int = 1000,
) -> Path:
    """Select Ridge-versus-MLP probes without using test scores.

    This lean follow-up reuses frozen features and cached targets. It
    evaluates only the five paper targets on the manuscript sampling support.
    """
    feature_dir = Path(feature_dir)
    feature_manifest = json.loads((feature_dir / "manifest.json").read_text())
    if feature_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"incompatible feature manifest at {feature_dir}")
    feature_check = dict(feature_manifest)
    feature_claim = feature_check.pop("manifest_sha256", None)
    if feature_claim is None or canonical_hash(feature_check) != feature_claim:
        raise ValueError(f"feature manifest hash mismatch at {feature_dir}")
    checkpoint = feature_manifest["checkpoint"]
    objective, checkpoint_seed = checkpoint["objective"], int(checkpoint["seed"])
    if feature_dir.name != _checkpoint_id(checkpoint):
        raise ValueError(
            f"feature directory {feature_dir.name!r} does not match checkpoint "
            f"{_checkpoint_id(checkpoint)!r}"
        )

    valid_cache, valid_manifest = _open_cache(cache_root, "valid")
    test_cache, test_manifest = _open_cache(cache_root, "test")
    expected_cache_manifests = {
        "valid": valid_manifest["manifest_sha256"],
        "test": test_manifest["manifest_sha256"],
    }
    feature_cache_manifests = {
        row["cache_split"]: row["cache_manifest_sha256"]
        for row in feature_manifest["files"]
        if "cache_split" in row
    }
    if feature_cache_manifests != expected_cache_manifests:
        raise ValueError("features were not extracted from the current valid/test cache manifests")

    rows = []
    stratum = "original_support"
    for representation in ("pooled", "token"):
        valid_samples_base = _load_samples(valid_cache, stratum, representation)
        test_samples_base = _load_samples(test_cache, stratum, representation)
        valid_features_raw = np.load(
            feature_dir / "valid" / stratum / f"{representation}.npy", mmap_mode="r"
        )
        test_features_raw = np.load(
            feature_dir / "test" / stratum / f"{representation}.npy", mmap_mode="r"
        )
        if representation == "token":
            if valid_features_raw.ndim != 4 or test_features_raw.ndim != 4:
                raise ValueError("token features must be (samples,tokens,layers,dimensions)")
            n_tokens = int(valid_features_raw.shape[1])
            if int(test_features_raw.shape[1]) != n_tokens:
                raise ValueError("valid and test token counts differ")
            valid_features = np.asarray(valid_features_raw).reshape(
                -1, valid_features_raw.shape[2], valid_features_raw.shape[3]
            )
            test_features = np.asarray(test_features_raw).reshape(
                -1, test_features_raw.shape[2], test_features_raw.shape[3]
            )
            valid_samples = _repeat_samples(valid_samples_base, n_tokens)
            test_samples = _repeat_samples(test_samples_base, n_tokens)
        else:
            valid_features, test_features = valid_features_raw, test_features_raw
            valid_samples, test_samples = valid_samples_base, test_samples_base

        valid_targets = {
            f"gap{gap}:{target}": np.load(
                valid_cache / stratum / f"{representation}_gap{gap}_{target}.npy",
                mmap_mode="r",
            ).reshape(-1)
            for gap in GAPS
            for target in PERSISTENCE_TARGETS
        }
        test_targets = {
            f"gap{gap}:{target}": np.load(
                test_cache / stratum / f"{representation}_gap{gap}_{target}.npy",
                mmap_mode="r",
            ).reshape(-1)
            for gap in GAPS
            for target in PERSISTENCE_TARGETS
        }
        ridge_results, _ = fit_ridge_cells_many(
            valid_features,
            valid_targets,
            test_features,
            test_targets,
            valid_samples=valid_samples,
            test_samples=test_samples,
            n_boot=n_boot,
        )
        for gap in GAPS:
            for target in PERSISTENCE_TARGETS:
                key = f"gap{gap}:{target}"
                valid_target = valid_targets[key]
                test_target = test_targets[key]
                base_cell = {
                    "family": "physics",
                    "representation": representation,
                    "stratum": stratum,
                    "gap": int(gap),
                    "target": target,
                }
                mlp_result, _ = fit_mlp_selected_layer_cell(
                    valid_features,
                    valid_target,
                    test_features,
                    test_target,
                    valid_samples=valid_samples,
                    test_samples=test_samples,
                    max_steps=mlp_max_steps,
                    n_boot=n_boot,
                )
                mlp_row = _annotate(
                    mlp_result,
                    objective=objective,
                    checkpoint_seed=checkpoint_seed,
                    method="mlp_encoder_selected",
                    **base_cell,
                )
                ridge_row = _annotate(
                    ridge_results[key],
                    objective=objective,
                    checkpoint_seed=checkpoint_seed,
                    method="ridge_encoder",
                    **base_cell,
                )
                rows.append(ridge_row)
                rows.append(mlp_row)
                selected = select_probe_family(ridge_row, mlp_row)
                selected["method"] = "selected_probe"
                selected["cell_id"] = _cell_id(method="selected_probe", **base_cell)
                rows.append(selected)

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "rb-selected-probes",
        "git_sha": git_sha(),
        "analysis_code": code_manifest(),
        "checkpoint": checkpoint,
        "feature_manifest_sha256": feature_manifest["manifest_sha256"],
        "cache_manifests": expected_cache_manifests,
        "protocol": {
            "selection_split": "valid",
            "report_split": "test",
            "stratum": stratum,
            "targets": list(PERSISTENCE_TARGETS),
            "representations": ["pooled", "token"],
            "gaps": list(GAPS),
            "valid_folds": "one trajectory per regime with sampling times balanced across folds",
            "ridge_selection": "maximum mean validation-fold R2 over layers and penalties",
            "mlp_layer_selection": "maximum mean validation-fold R2 over layers",
            "probe_family_selection": "maximum validation CV R2; Ridge wins exact ties",
            "test_access_during_selection": False,
            "checkpoint_seed_is_scientific_unit": True,
            "formal_significance_test": None,
        },
        "rows": rows,
    }
    output = Path(output)
    atomic_write_json(output, result)
    return output


def _load_feature_manifest(feature_dir: Path) -> dict:
    manifest = json.loads((feature_dir / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != "rb-checkpoint-features"
    ):
        raise ValueError(f"incompatible feature manifest at {feature_dir}")
    check = dict(manifest)
    claim = check.pop("manifest_sha256", None)
    if claim is None or canonical_hash(check) != claim:
        raise ValueError(f"feature manifest hash mismatch at {feature_dir}")
    if feature_dir.name != _checkpoint_id(manifest["checkpoint"]):
        raise ValueError(
            f"feature directory {feature_dir.name!r} does not match checkpoint "
            f"{_checkpoint_id(manifest['checkpoint'])!r}"
        )
    return manifest


def _manifest_file_hash(manifest: dict, relative_path: str) -> str:
    matches = [
        row["sha256"]
        for row in manifest.get("files", ())
        if row.get("path") == relative_path and "sha256" in row
    ]
    if len(matches) != 1:
        raise ValueError(f"feature manifest needs exactly one file {relative_path!r}")
    return str(matches[0])


def _fit_mlp_from_selected_row(
    valid_features: np.ndarray,
    valid_target: np.ndarray,
    selected_row: dict,
) -> dict:
    return _fit_mlp_layer_from_steps(
        valid_features,
        valid_target,
        layer=int(selected_row["selected_layer"]),
        seeds=selected_row["probe_seeds"],
        steps=selected_row["selected_steps"],
    )


def _fit_mlp_layer_from_steps(
    valid_features: np.ndarray,
    valid_target: np.ndarray,
    *,
    layer: int,
    seeds: Sequence[int],
    steps: Sequence[int],
) -> dict:
    layer = int(layer)
    seeds = tuple(int(seed) for seed in seeds)
    steps = tuple(int(step) for step in steps)
    if len(seeds) != len(steps) or not seeds:
        raise ValueError("selected MLP row has incompatible probe seeds and stopping steps")
    if not 0 <= layer < valid_features.shape[1]:
        raise ValueError(f"selected MLP layer {layer} is outside the feature array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid = torch.tensor(
        np.asarray(valid_features[:, layer]), dtype=torch.float32, device=device
    )
    target = torch.tensor(
        np.asarray(valid_target), dtype=torch.float32, device=device
    ).reshape(-1)
    x_train, y_train, mean, std, target_mean, target_std = _standardize_fit(valid, target)
    models = []
    for seed, selected_steps in zip(seeds, steps):
        torch.manual_seed(seed)
        model = _ProbeMLP(valid.shape[1], hidden=128, dropout=0.1).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-2, weight_decay=1e-4
        )
        for _ in range(selected_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x_train), y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        models.append(model)
    return {
        "layer": layer,
        "models": models,
        "mean": mean,
        "std": std,
        "target_mean": target_mean,
        "target_std": target_std,
        "probe_seeds": list(seeds),
        "selected_steps": list(steps),
    }


def _predict_selected_mlp(fit: dict, features: np.ndarray) -> torch.Tensor:
    evaluate = torch.tensor(
        np.asarray(features[:, int(fit["layer"])]),
        dtype=torch.float32,
        device=fit["mean"].device,
    )
    predictions = []
    with torch.no_grad():
        standardized = (evaluate - fit["mean"]) / fit["std"]
        for model in fit["models"]:
            predictions.append(
                model(standardized) * fit["target_std"] + fit["target_mean"]
            )
    return torch.stack(predictions).mean(dim=0).double().cpu()


def _noise_scores(
    *,
    feature_dir: Path,
    layer: int,
    target: torch.Tensor,
    predict,
) -> dict[float, dict]:
    output = {}
    for sigma in NOISE_SIGMAS:
        scores, correlations = [], []
        for corruption_seed in NOISE_SEEDS:
            path = (
                feature_dir
                / "test"
                / "original_support"
                / f"pooled_noise_sigma{sigma:g}_seed{corruption_seed}.npy"
            )
            if not path.exists():
                raise FileNotFoundError(f"noise feature missing: {path}")
            features = np.load(path, mmap_mode="r")
            if features.ndim != 3 or not 0 <= int(layer) < features.shape[1]:
                raise ValueError(f"noise feature has incompatible shape at {path}: {features.shape}")
            prediction = predict(features)
            scores.append(r2_score(prediction, target))
            correlations.append(pearson_r(prediction, target))
        output[float(sigma)] = {
            "test_r2": float(np.mean(scores)),
            "test_r2_corruption_seed_std": (
                float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            ),
            "test_r2_per_corruption_seed": scores,
            "test_pearson_r": float(np.mean(correlations)),
            "test_pearson_r_corruption_seed_std": (
                float(np.std(correlations, ddof=1)) if len(correlations) > 1 else 0.0
            ),
            "test_pearson_r_per_corruption_seed": correlations,
        }
    return output


def _assert_clean_noise_endpoint(actual: dict, expected: dict, *, label: str, tolerance: float) -> None:
    for metric in ("test_r2", "test_pearson_r"):
        if not math.isclose(
            float(actual[metric]),
            float(expected[metric]),
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError(
                f"{label} at sigma=0 does not reproduce Figure 1: "
                f"{metric}={actual[metric]} versus {expected[metric]}"
            )


def fit_checkpoint_selected_noise(
    *,
    feature_dir: str | Path,
    selected_feature_dir: str | Path,
    cache_root: str | Path,
    selected_result: str | Path,
    output: str | Path,
    clean_tolerance: float = 1e-6,
) -> Path:
    """Evaluate the exact clean-selected pooled probes on corrupted test inputs."""
    feature_dir = Path(feature_dir)
    selected_feature_dir = Path(selected_feature_dir)
    selected_result = Path(selected_result)
    noise_manifest = _load_feature_manifest(feature_dir)
    selection_manifest = _load_feature_manifest(selected_feature_dir)
    selection = json.loads(selected_result.read_text())
    if (
        selection.get("schema_version") != SCHEMA_VERSION
        or selection.get("kind") != "rb-selected-probes"
    ):
        raise ValueError(f"selected probe result is incompatible: {selected_result}")
    if selection.get("feature_manifest_sha256") != selection_manifest["manifest_sha256"]:
        raise ValueError("selected probe result does not match its feature manifest")

    checkpoint = noise_manifest["checkpoint"]
    checkpoint_key = (checkpoint.get("objective"), int(checkpoint.get("seed")))
    if checkpoint_key != (
        selection["checkpoint"].get("objective"),
        int(selection["checkpoint"].get("seed")),
    ) or checkpoint_key != (
        selection_manifest["checkpoint"].get("objective"),
        int(selection_manifest["checkpoint"].get("seed")),
    ):
        raise ValueError("noise features and selected probes use different checkpoints")
    for key in ("sha256", "step"):
        expected = selection["checkpoint"].get(key)
        if checkpoint.get(key) != expected or selection_manifest["checkpoint"].get(key) != expected:
            raise ValueError(f"checkpoint {key} differs between selected and noise features")

    valid_cache, valid_manifest = _open_cache(cache_root, "valid")
    test_cache, test_manifest = _open_cache(cache_root, "test")
    cache_manifests = {
        "valid": valid_manifest["manifest_sha256"],
        "test": test_manifest["manifest_sha256"],
    }
    if selection.get("cache_manifests") != cache_manifests:
        raise ValueError("selected probes and noise evaluation use different target caches")
    for manifest in (selection_manifest, noise_manifest):
        recorded = {
            row["cache_split"]: row["cache_manifest_sha256"]
            for row in manifest.get("files", ())
            if "cache_split" in row
        }
        if recorded != cache_manifests:
            raise ValueError("feature manifest does not match the current target caches")

    for clean_path in (
        "valid/original_support/pooled.npy",
        "test/original_support/pooled.npy",
    ):
        if _manifest_file_hash(selection_manifest, clean_path) != _manifest_file_hash(
            noise_manifest, clean_path
        ):
            raise ValueError(f"selected and noise features differ for {clean_path}")
    clean_test_hash = _manifest_file_hash(
        selection_manifest, "test/original_support/pooled.npy"
    )
    for corruption_seed in NOISE_SEEDS:
        path = f"test/original_support/pooled_noise_sigma0_seed{corruption_seed}.npy"
        if _manifest_file_hash(noise_manifest, path) != clean_test_hash:
            raise ValueError(f"sigma=0 features differ from the Figure 1 clean test features: {path}")

    if tuple(float(value) for value in noise_manifest.get("noise_sigmas", ())) != tuple(
        float(value) for value in NOISE_SIGMAS
    ) or tuple(int(value) for value in noise_manifest.get("noise_seeds", ())) != tuple(
        int(value) for value in NOISE_SEEDS
    ):
        raise ValueError("noise feature manifest does not contain the required corruption grid")

    objective, checkpoint_seed = checkpoint_key
    selected_rows = [
        row
        for row in selection.get("rows", ())
        if row.get("family") == "physics"
        and row.get("representation") == "pooled"
        and row.get("stratum") == "original_support"
        and row.get("gap") == 0
        and row.get("target") in PERSISTENCE_TARGETS
    ]
    lookup = {
        (row["target"], row["method"]): row
        for row in selected_rows
        if row.get("method") in {"ridge_encoder", "mlp_encoder_selected", "selected_probe"}
    }
    expected_keys = {
        (target, method)
        for target in PERSISTENCE_TARGETS
        for method in ("ridge_encoder", "mlp_encoder_selected", "selected_probe")
    }
    if set(lookup) != expected_keys or len(selected_rows) != len(expected_keys):
        raise ValueError("selected probe result lacks one unique pooled t+0 row per target and family")

    valid_features = np.load(
        feature_dir / "valid" / "original_support" / "pooled.npy", mmap_mode="r"
    )
    rows = []
    family_method = {
        "ridge_encoder": "ridge_encoder_clean_fit",
        "mlp_encoder_selected": "mlp_encoder_clean_fit",
    }
    for target_name in PERSISTENCE_TARGETS:
        valid_target = np.load(
            valid_cache / "original_support" / f"pooled_gap0_{target_name}.npy",
            mmap_mode="r",
        ).reshape(-1)
        test_target = torch.as_tensor(
            np.asarray(
                np.load(
                    test_cache / "original_support" / f"pooled_gap0_{target_name}.npy",
                    mmap_mode="r",
                )
            ),
            dtype=torch.float64,
        ).reshape(-1)
        source_by_family = {
            source_method: lookup[(target_name, source_method)]
            for source_method in family_method
        }
        predictions = {}
        ridge_source = source_by_family["ridge_encoder"]
        ridge_layer = int(ridge_source["selected_layer"])
        ridge_alpha = float(ridge_source["selected_alpha"])
        x_valid = torch.as_tensor(
            np.asarray(valid_features[:, ridge_layer]), dtype=torch.float32
        )
        y_valid = torch.as_tensor(np.asarray(valid_target), dtype=torch.float64)
        predictions["ridge_encoder"] = _noise_scores(
            feature_dir=feature_dir,
            layer=ridge_layer,
            target=test_target,
            predict=lambda features, x=x_valid, y=y_valid, alpha=ridge_alpha, layer=ridge_layer: _ridge_predict(
                x,
                y,
                torch.as_tensor(np.asarray(features[:, layer]), dtype=torch.float32),
                alpha,
            ),
        )

        mlp_source = source_by_family["mlp_encoder_selected"]
        mlp_fit = _fit_mlp_from_selected_row(valid_features, valid_target, mlp_source)
        predictions["mlp_encoder_selected"] = _noise_scores(
            feature_dir=feature_dir,
            layer=int(mlp_fit["layer"]),
            target=test_target,
            predict=lambda features, fit=mlp_fit: _predict_selected_mlp(fit, features),
        )

        family_rows: dict[tuple[str, float], dict] = {}
        for source_method, output_method in family_method.items():
            source = source_by_family[source_method]
            clean = predictions[source_method][0.0]
            _assert_clean_noise_endpoint(
                clean,
                source,
                label=f"{objective}/seed{checkpoint_seed}/{target_name}/{source_method}",
                tolerance=clean_tolerance,
            )
            for sigma in NOISE_SIGMAS:
                base_cell = {
                    "family": "noise",
                    "method": output_method,
                    "representation": "pooled",
                    "stratum": "original_support",
                    "gap": 0,
                    "target": target_name,
                    "sigma": float(sigma),
                }
                metadata = {
                    "selected_layer": int(source["selected_layer"]),
                    "source_selection_method": source_method,
                    **predictions[source_method][float(sigma)],
                }
                if source_method == "ridge_encoder":
                    metadata["selected_alpha"] = float(source["selected_alpha"])
                else:
                    metadata["probe_seeds"] = list(mlp_fit["probe_seeds"])
                    metadata["selected_steps"] = list(mlp_fit["selected_steps"])
                row = _annotate(
                    metadata,
                    objective=objective,
                    checkpoint_seed=checkpoint_seed,
                    **base_cell,
                )
                rows.append(row)
                family_rows[(source_method, float(sigma))] = row

        selected_source = lookup[(target_name, "selected_probe")]
        selected_method = str(selected_source["selected_method"])
        if selected_method not in family_method:
            raise ValueError(f"unknown Figure 1 selected method {selected_method!r}")
        for sigma in NOISE_SIGMAS:
            chosen = family_rows[(selected_method, float(sigma))]
            base_cell = {
                "family": "noise",
                "method": "selected_probe_clean_fit",
                "representation": "pooled",
                "stratum": "original_support",
                "gap": 0,
                "target": target_name,
                "sigma": float(sigma),
            }
            selected_row = _annotate(
                {
                    **{
                        key: value
                        for key, value in chosen.items()
                        if key
                        not in {
                            "objective",
                            "checkpoint_seed",
                            "family",
                            "method",
                            "representation",
                            "stratum",
                            "gap",
                            "target",
                            "sigma",
                            "cell_id",
                        }
                    },
                    "selected_method": selected_method,
                    "selected_layer": int(selected_source["selected_layer"]),
                },
                objective=objective,
                checkpoint_seed=checkpoint_seed,
                **base_cell,
            )
            rows.append(selected_row)
            if float(sigma) == 0.0:
                _assert_clean_noise_endpoint(
                    selected_row,
                    selected_source,
                    label=f"{objective}/seed{checkpoint_seed}/{target_name}/selected_probe",
                    tolerance=clean_tolerance,
                )

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "rb-selected-noise",
        "git_sha": git_sha(),
        "analysis_code": code_manifest(),
        "checkpoint": checkpoint,
        "feature_manifest_sha256": noise_manifest["manifest_sha256"],
        "selection_feature_manifest_sha256": selection_manifest["manifest_sha256"],
        "selection_result": {
            "path": str(selected_result),
            "sha256": sha256_file(selected_result),
        },
        "cache_manifests": cache_manifests,
        "protocol": {
            "selection_source": "saved pooled t+0 Figure 1 validation selection",
            "selection_split": "valid",
            "report_split": "test",
            "stratum": "original_support",
            "targets": list(PERSISTENCE_TARGETS),
            "noise_sigmas": list(NOISE_SIGMAS),
            "noise_seeds": list(NOISE_SEEDS),
            "methods": [
                "ridge_encoder_clean_fit",
                "mlp_encoder_clean_fit",
                "selected_probe_clean_fit",
            ],
            "refit_under_noise": False,
            "reselect_under_noise": False,
            "clean_endpoint_tolerance": float(clean_tolerance),
            "checkpoint_seed_is_scientific_unit": True,
        },
        "rows": rows,
    }
    output = Path(output)
    atomic_write_json(output, result)
    return output


def fit_checkpoint_mlp_depth(
    *,
    feature_dir: str | Path,
    cache_root: str | Path,
    selected_result: str | Path,
    output: str | Path,
    selected_tolerance: float = 1e-6,
) -> Path:
    """Score clean-fit pooled MLP probes on test data at every encoder layer."""
    feature_dir = Path(feature_dir)
    selected_result = Path(selected_result)
    feature_manifest = _load_feature_manifest(feature_dir)
    selection = json.loads(selected_result.read_text())
    if (
        selection.get("schema_version") != SCHEMA_VERSION
        or selection.get("kind") != "rb-selected-probes"
    ):
        raise ValueError(f"selected probe result is incompatible: {selected_result}")
    if selection.get("feature_manifest_sha256") != feature_manifest["manifest_sha256"]:
        raise ValueError("selected probe result does not match its feature manifest")

    checkpoint = feature_manifest["checkpoint"]
    checkpoint_key = (checkpoint.get("objective"), int(checkpoint.get("seed")))
    if checkpoint_key != (
        selection["checkpoint"].get("objective"),
        int(selection["checkpoint"].get("seed")),
    ):
        raise ValueError("MLP depth features and selected probes use different checkpoints")
    for key in ("sha256", "step"):
        if checkpoint.get(key) != selection["checkpoint"].get(key):
            raise ValueError(f"checkpoint {key} differs between features and selected probes")

    valid_cache, valid_manifest = _open_cache(cache_root, "valid")
    test_cache, test_manifest = _open_cache(cache_root, "test")
    cache_manifests = {
        "valid": valid_manifest["manifest_sha256"],
        "test": test_manifest["manifest_sha256"],
    }
    if selection.get("cache_manifests") != cache_manifests:
        raise ValueError("selected probes and MLP depth evaluation use different target caches")
    recorded = {
        row["cache_split"]: row["cache_manifest_sha256"]
        for row in feature_manifest.get("files", ())
        if "cache_split" in row
    }
    if recorded != cache_manifests:
        raise ValueError("feature manifest does not match the current target caches")

    selected_rows = [
        row
        for row in selection.get("rows", ())
        if row.get("family") == "physics"
        and row.get("method") == "mlp_encoder_selected"
        and row.get("representation") == "pooled"
        and row.get("stratum") == "original_support"
        and row.get("gap") in GAPS
        and row.get("target") in PERSISTENCE_TARGETS
    ]
    lookup = {(int(row["gap"]), row["target"]): row for row in selected_rows}
    expected = {(int(gap), target) for gap in GAPS for target in PERSISTENCE_TARGETS}
    if set(lookup) != expected or len(selected_rows) != len(expected):
        raise ValueError("selected probe result lacks one pooled MLP row per target and horizon")

    valid_features = np.load(
        feature_dir / "valid" / "original_support" / "pooled.npy", mmap_mode="r"
    )
    test_features = np.load(
        feature_dir / "test" / "original_support" / "pooled.npy", mmap_mode="r"
    )
    if valid_features.ndim != 3 or test_features.ndim != 3:
        raise ValueError("pooled features must be (rows,layers,dimensions)")
    if valid_features.shape[1:] != test_features.shape[1:]:
        raise ValueError("valid and test pooled feature shapes differ")

    objective, checkpoint_seed = checkpoint_key
    rows = []
    for gap in GAPS:
        for target_name in PERSISTENCE_TARGETS:
            source = lookup[(int(gap), target_name)]
            layer_curve = sorted(source.get("layer_curve", ()), key=lambda row: int(row["layer"]))
            layers = [int(point["layer"]) for point in layer_curve]
            if layers != list(range(valid_features.shape[1])):
                raise ValueError(f"incomplete validation layer curve for gap{gap}/{target_name}")
            valid_target = np.load(
                valid_cache / "original_support" / f"pooled_gap{gap}_{target_name}.npy",
                mmap_mode="r",
            ).reshape(-1)
            test_target = torch.tensor(
                np.asarray(
                    np.load(
                        test_cache / "original_support" / f"pooled_gap{gap}_{target_name}.npy",
                        mmap_mode="r",
                    )
                ),
                dtype=torch.float64,
            ).reshape(-1)
            depth_curve = []
            for point in layer_curve:
                fit = _fit_mlp_layer_from_steps(
                    valid_features,
                    valid_target,
                    layer=int(point["layer"]),
                    seeds=source["probe_seeds"],
                    steps=point["selected_steps"],
                )
                prediction = _predict_selected_mlp(fit, test_features)
                depth_curve.append(
                    {
                        "layer": int(point["layer"]),
                        "selected_steps": list(fit["selected_steps"]),
                        "valid_cv_r2": float(point["valid_cv_r2"]),
                        "test_r2": r2_score(prediction, test_target),
                        "test_pearson_r": pearson_r(prediction, test_target),
                    }
                )
            selected_layer = int(source["selected_layer"])
            selected_point = depth_curve[selected_layer]
            for metric in ("test_r2", "test_pearson_r"):
                if not math.isclose(
                    float(selected_point[metric]),
                    float(source[metric]),
                    rel_tol=selected_tolerance,
                    abs_tol=selected_tolerance,
                ):
                    raise ValueError(
                        f"{objective}/seed{checkpoint_seed}/gap{gap}/{target_name} "
                        f"does not reproduce selected-layer MLP {metric}"
                    )
            rows.append(
                _annotate(
                    {
                        "selected_layer": selected_layer,
                        "selected_steps": list(source["selected_steps"]),
                        "probe_seeds": list(source["probe_seeds"]),
                        "valid_cv_r2": float(source["valid_cv_r2"]),
                        "test_r2": float(selected_point["test_r2"]),
                        "test_pearson_r": float(selected_point["test_pearson_r"]),
                        "test_log_mse": float(source["test_log_mse"]),
                        "bootstrap": source.get("bootstrap"),
                        "depth_curve": depth_curve,
                    },
                    objective=objective,
                    checkpoint_seed=checkpoint_seed,
                    family="physics",
                    method="mlp_encoder_depth",
                    representation="pooled",
                    stratum="original_support",
                    gap=int(gap),
                    target=target_name,
                )
            )

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "rb-mlp-depth",
        "git_sha": git_sha(),
        "analysis_code": code_manifest(),
        "checkpoint": checkpoint,
        "feature_manifest_sha256": feature_manifest["manifest_sha256"],
        "selection_result": {
            "path": str(selected_result),
            "sha256": sha256_file(selected_result),
        },
        "cache_manifests": cache_manifests,
        "protocol": {
            "selection_source": "saved per-layer MLP validation curves",
            "selection_split": "valid",
            "report_split": "test",
            "stratum": "original_support",
            "representation": "pooled",
            "targets": list(PERSISTENCE_TARGETS),
            "gaps": list(GAPS),
            "layer_training_steps": "saved validation-selected stopping time",
            "reselect_on_test": False,
            "selected_endpoint_tolerance": float(selected_tolerance),
            "checkpoint_seed_is_scientific_unit": True,
        },
        "rows": rows,
    }
    output = Path(output)
    atomic_write_json(output, result)
    return output


def aggregate_result_files(paths: Sequence[str | Path], output: str | Path) -> Path:
    results = [json.loads(Path(path).read_text()) for path in paths]
    if len(results) != 6:
        raise ValueError(f"expected six checkpoint result files, found {len(results)}")
    checkpoint_keys = {
        (result["checkpoint"]["objective"], int(result["checkpoint"]["seed"]))
        for result in results
    }
    expected = {(objective, seed) for objective in ("jepa", "mae") for seed in (1, 2, 3)}
    if checkpoint_keys != expected:
        raise ValueError(f"result set mismatch: missing={sorted(expected - checkpoint_keys)}")
    rows = [row for result in results for row in result["rows"]]
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "kind": "rb-aggregate",
        "git_sha": git_sha(),
        "analysis_code": code_manifest(),
        "source_results": [
            {"path": str(Path(path)), "sha256": sha256_file(path)} for path in paths
        ],
        "source_metadata": [
            {
                key: result[key]
                for key in (
                    "kind",
                    "checkpoint",
                    "feature_manifest_sha256",
                    "selection_feature_manifest_sha256",
                    "selection_result",
                    "cache_manifests",
                    "protocol",
                )
                if key in result
            }
            for result in results
        ],
        "summary": aggregate_seed_rows(rows),
        "per_checkpoint_rows": rows,
        "interpretation": {
            "scientific_unit": "checkpoint seed",
            "n_per_objective": 3,
            "formal_significance_test": None,
            "qu_metric": "qualitative only; no converted MSE or attention proxy",
        },
    }
    output = Path(output)
    atomic_write_json(output, aggregate)
    return output


def plot_aggregate(aggregate_path: str | Path, output_dir: str | Path) -> Path:
    """Render fixed, manuscript-facing summaries without touching ``paper/``."""
    import matplotlib.pyplot as plt

    aggregate_path = Path(aggregate_path)
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("schema_version") != SCHEMA_VERSION or aggregate.get("kind") != "rb-aggregate":
        raise ValueError(f"not an RB-v2 aggregate: {aggregate_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        output_dir / "rb_physics_gap0.pdf",
        output_dir / "rb_depth_gap0.pdf",
        output_dir / "rb_noise.pdf",
        output_dir / "rb_summary.tsv",
        output_dir / "plots_manifest.json",
    ]
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite plots: {existing}")

    primary = list(PRIMARY_TARGETS)
    labels = {
        "enstrophy": "Enstrophy",
        "buoyancy_gradient_energy": r"$|\nabla b|^2$",
        "convective_flux": r"$v_y b$",
        "pressure_gradient_magnitude": r"$|\nabla p|$",
        "buoyancy_laplacian_magnitude": r"$|\nabla^2 b|$",
        "deformation_energy": "Deformation",
    }
    colors = {"jepa": "#2b6cb0", "mae": "#c05621"}
    summary = aggregate["summary"]
    physics = [
        row
        for row in summary
        if row.get("family") == "physics"
        and row.get("method") == "ridge_encoder"
        and row.get("representation") == "pooled"
        and row.get("stratum") == "developed"
        and row.get("gap") == 0
        and row.get("target") in primary
    ]
    lookup = {(row["objective"], row["target"]): row for row in physics}
    fig, axis = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    x = np.arange(len(primary))
    width = 0.36
    for offset, objective in ((-width / 2, "jepa"), (width / 2, "mae")):
        values = [lookup[(objective, target)]["test_r2_mean"] for target in primary]
        errors = [lookup[(objective, target)]["test_r2_std"] for target in primary]
        axis.bar(x + offset, values, width, yerr=errors, capsize=3, label=objective.upper(), color=colors[objective])
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_ylabel(r"Official-test $R^2$ (mean $\pm$ checkpoint-seed SD)")
    axis.set_xticks(x, [labels[target] for target in primary], rotation=25, ha="right")
    axis.legend(frameon=False)
    fig.savefig(output_dir / "rb_physics_gap0.pdf")
    plt.close(fig)

    checkpoint_rows = aggregate["per_checkpoint_rows"]
    depth_rows = [
        row
        for row in checkpoint_rows
        if row.get("family") == "physics"
        and row.get("method") == "ridge_encoder"
        and row.get("representation") == "pooled"
        and row.get("stratum") == "developed"
        and row.get("gap") == 0
        and row.get("target") in primary
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True, constrained_layout=True)
    for axis, target in zip(axes.flat, primary):
        for objective in ("jepa", "mae"):
            curves = [
                [point["test_r2"] for point in row["depth_curve"]]
                for row in depth_rows
                if row["objective"] == objective and row["target"] == target
            ]
            values = np.asarray(curves)
            layers = np.arange(1, values.shape[1] + 1)
            axis.plot(layers, values.mean(axis=0), label=objective.upper(), color=colors[objective])
            axis.fill_between(
                layers,
                values.mean(axis=0) - values.std(axis=0, ddof=1),
                values.mean(axis=0) + values.std(axis=0, ddof=1),
                color=colors[objective],
                alpha=0.18,
                linewidth=0,
            )
        axis.axhline(0, color="black", linewidth=0.5)
        axis.set_title(labels[target])
        axis.set_xlabel("Encoder depth")
        axis.set_ylabel(r"Test $R^2$")
    axes.flat[0].legend(frameon=False)
    fig.savefig(output_dir / "rb_depth_gap0.pdf")
    plt.close(fig)

    noise = [
        row
        for row in summary
        if row.get("family") == "noise"
        and row.get("stratum") == "developed"
        and row.get("target") in primary
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True, constrained_layout=True)
    for axis, target in zip(axes.flat, primary):
        for objective in ("jepa", "mae"):
            values = sorted(
                [row for row in noise if row["objective"] == objective and row["target"] == target],
                key=lambda row: row["sigma"],
            )
            axis.plot(
                [row["sigma"] for row in values],
                [row["test_r2_mean"] for row in values],
                marker="o",
                label=objective.upper(),
                color=colors[objective],
            )
        axis.axhline(0, color="black", linewidth=0.5)
        axis.set_title(labels[target])
        axis.set_xlabel(r"Input noise $\sigma$")
        axis.set_ylabel(r"Test $R^2$")
    axes.flat[0].legend(frameon=False)
    fig.savefig(output_dir / "rb_noise.pdf")
    plt.close(fig)

    columns = [
        "objective",
        "family",
        "method",
        "representation",
        "stratum",
        "gap",
        "target",
        "sigma",
        "test_r2_mean",
        "test_r2_std",
        "n_checkpoint_seeds",
    ]
    with (output_dir / "rb_summary.tsv").open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in summary:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")
    products = []
    for path in planned[:-1]:
        products.append({"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "rb-plots",
        "git_sha": git_sha(),
        "analysis_code": code_manifest(),
        "aggregate": {"path": str(aggregate_path), "sha256": sha256_file(aggregate_path)},
        "products": products,
    }
    atomic_write_json(output_dir / "plots_manifest.json", manifest)
    return output_dir

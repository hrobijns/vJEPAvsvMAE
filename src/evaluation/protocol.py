"""Explicit sampling offsets and physical controls."""

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from src.physics.systems import SYSTEMS


@dataclass(frozen=True)
class Protocol:
    dataset: str
    frame_limit: int | None = None
    n_frames: int = 8
    target_offsets: tuple[int, ...] = (0, 8, 16, 40)
    patch: tuple[int, int, int] = (2, 16, 16)
    token_samples: int = 64
    noise_sigmas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
    noise_seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self):
        if (
            self.dataset not in SYSTEMS
            or self.n_frames < 1
            or (self.frame_limit is not None and self.frame_limit < self.n_frames)
        ):
            raise ValueError("invalid dataset or temporal support")
        if (
            len(self.patch) != 3
            or min(self.patch) < 1
            or self.n_frames % self.patch[0]
            or self.token_samples < 1
        ):
            raise ValueError("invalid token geometry")
        if (
            not self.target_offsets
            or min(self.target_offsets) < 0
            or 0 not in self.target_offsets
            or len(set(self.target_offsets)) != len(self.target_offsets)
            or any(0 < offset < self.n_frames for offset in self.target_offsets)
        ):
            raise ValueError(
                "target offsets must include zero and nonoverlapping future clips"
            )
        if (
            not self.noise_sigmas
            or self.noise_sigmas[0] != 0
            or any(not np.isfinite(s) or s < 0 for s in self.noise_sigmas)
            or len(set(self.noise_sigmas)) != len(self.noise_sigmas)
            or not self.noise_seeds
        ):
            raise ValueError("invalid noise grid")

    def to_dict(self):
        return {
            **asdict(self),
            "targets": list(SYSTEMS[self.dataset].targets),
            "target_version": 2,
            "offset_definition": "target_start_minus_context_start",
            "probe_splits": {"fit": "train", "select": "valid", "score": "test"},
        }

    @classmethod
    def from_dict(cls, value):
        if "gaps" in value:
            raise ValueError(
                "legacy gaps are ambiguous; migrate to target_offsets and rebuild caches"
            )
        keys = cls.__dataclass_fields__
        kwargs = {k: v for k, v in value.items() if k in keys}
        for k in ("target_offsets", "patch", "noise_sigmas", "noise_seeds"):
            if k in kwargs:
                kwargs[k] = tuple(kwargs[k])
        return cls(**kwargs)

    def target_start(self, start, offset):
        return start + offset

    def offsets(self, frames, trajectory):
        stop = frames if self.frame_limit is None else min(frames, self.frame_limit)
        extent = self.n_frames + max(self.target_offsets)
        last = stop - extent
        if last < 0:
            raise ValueError(f"{stop} frames cannot contain all requested horizons")
        return {
            "pooled": np.linspace(0, last, 3).round().astype(int).tolist(),
            "token": [int(np.rint(np.linspace(0, last, 5)[trajectory % 5]))],
        }


def token_indices(trajectory, n_tokens, n_select=64):
    if not 0 < n_select <= n_tokens:
        raise ValueError("token sample count exceeds grid")
    rng = np.random.default_rng(np.random.SeedSequence([20260825, int(trajectory)]))
    return np.sort(rng.choice(n_tokens, size=n_select, replace=False))


def polynomial_basis(values):
    values = np.asarray(values, dtype=np.float64)
    columns = [np.ones(len(values)), *values.T, *(values**2).T]
    columns.extend(
        values[:, i] * values[:, j]
        for i in range(values.shape[1])
        for j in range(i + 1, values.shape[1])
    )
    return np.column_stack(columns)


def nuisance_basis(samples, dataset):
    system = SYSTEMS[dataset]
    return polynomial_basis(
        [[*system.regime_values(row["parameters"]), row["age"]] for row in samples]
    )


def position_basis(positions, grid):
    flat = np.asarray(positions).reshape(-1)
    axes = np.unravel_index(flat, grid)
    scaled = np.column_stack(
        [2 * axis / max(n - 1, 1) - 1 for axis, n in zip(axes, grid)]
    )
    return polynomial_basis(scaled)


def trajectory_average(features, samples):
    grouped = defaultdict(list)
    for i, row in enumerate(samples):
        grouped[row["trajectory"]].append(i)
    return (
        np.stack(
            [
                np.asarray(features[ids]).mean(axis=0)
                for _, ids in sorted(grouped.items())
            ]
        ),
        [samples[ids[0]] for _, ids in sorted(grouped.items())],
    )

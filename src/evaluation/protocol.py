"""Explicit sampling, trajectory folds, and physical controls."""

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from src.physics.systems import SYSTEMS


@dataclass(frozen=True)
class Protocol:
    dataset: str
    frame_limit: int | None = 101
    n_frames: int = 8
    gaps: tuple[int, ...] = (0, 8, 32)
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
            not self.gaps
            or min(self.gaps) < 0
            or 0 not in self.gaps
            or len(set(self.gaps)) != len(self.gaps)
        ):
            raise ValueError("gaps must be distinct, nonnegative, and include zero")
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
            "target_version": 1,
            "folds": "trajectory-cyclic-5",
            "gap_definition": "intervening_frames",
        }

    @classmethod
    def from_dict(cls, value):
        keys = cls.__dataclass_fields__
        kwargs = {k: v for k, v in value.items() if k in keys}
        for k in ("gaps", "patch", "noise_sigmas", "noise_seeds"):
            if k in kwargs:
                kwargs[k] = tuple(kwargs[k])
        return cls(**kwargs)

    def target_start(self, start, gap):
        return start if gap == 0 else start + self.n_frames + gap

    def offsets(self, frames, trajectory):
        stop = frames if self.frame_limit is None else min(frames, self.frame_limit)
        extent = max(
            self.n_frames if gap == 0 else 2 * self.n_frames + gap for gap in self.gaps
        )
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


def trajectory_folds(samples, n_folds=5):
    """Rotate held-out replicates by regime to balance sampled times.

    With five runs per regime this reproduces the corrected RB folds exactly.
    Smaller regimes are distributed across folds without inventing replicates.
    All clips and tokens from one trajectory always remain together.
    """
    metadata = {}
    by_regime = defaultdict(dict)
    for row in samples:
        trajectory = row["trajectory_id"]
        value = (tuple(row["regime"]), int(row["replicate"]))
        if metadata.setdefault(trajectory, value) != value:
            raise ValueError("inconsistent trajectory metadata")
        regime, replicate = value
        previous = by_regime[regime].setdefault(replicate, trajectory)
        if previous != trajectory:
            raise ValueError("duplicate replicate within regime")
    assignments = {}
    for regime_index, regime in enumerate(sorted(by_regime)):
        for replicate, trajectory in by_regime[regime].items():
            assignments[trajectory] = (replicate - regime_index) % n_folds
    ids = np.asarray([assignments[row["trajectory_id"]] for row in samples])
    folds = [
        {"fit": np.flatnonzero(ids != fold), "select": np.flatnonzero(ids == fold)}
        for fold in range(n_folds)
    ]
    if any(len(f["fit"]) == 0 or len(f["select"]) == 0 for f in folds):
        raise ValueError("insufficient trajectories for five nonempty grouped folds")
    return folds


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

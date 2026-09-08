"""Trajectory-disjoint training clips from HDF5 or normalized full-rollout caches."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from src.data.source import WellSource, normalize
from src.evaluation.artifacts import canonical_hash, sha256_file

MEMMAP_SCHEMA = "well-training-cache-1"


@dataclass
class ClipSpec:
    n_channels: int
    n_frames: int
    height: int
    width: int


def train_valid_trajectory_split(n_traj, valid_stride=8):
    """Internal pretraining validation; official valid/test remain for probes."""
    if n_traj < 2 or valid_stride < 2:
        raise ValueError("need at least two trajectories and stride >= 2")
    valid = list(range(0, n_traj, valid_stride))
    held = set(valid)
    return [i for i in range(n_traj) if i not in held], valid


def _windows(records, trajectories, n_frames, frame_limit):
    if n_frames < 1 or (frame_limit is not None and frame_limit < n_frames):
        raise ValueError("invalid frame limit")
    ids = list(range(len(records))) if trajectories is None else list(trajectories)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(i < 0 or i >= len(records) for i in ids)
    ):
        raise ValueError("invalid or duplicate trajectory selection")
    counts = [
        (r["frames"] if frame_limit is None else min(r["frames"], frame_limit))
        - n_frames
        + 1
        for r in (records[i] for i in ids)
    ]
    if min(counts) < 1:
        raise ValueError("a trajectory is shorter than a clip")
    return ids, np.cumsum([0, *counts])


class _ClipDataset(torch.utils.data.Dataset):
    def __len__(self):
        return int(self.offsets[-1])

    def window(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        local = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.traj_ids[local], int(index - self.offsets[local])


class WellClipDataset(_ClipDataset):
    def __init__(
        self,
        base_path,
        dataset_name,
        split="train",
        n_frames=8,
        trajectories=None,
        frame_limit=101,
    ):
        self.source = WellSource(base_path, dataset_name, split, n_frames)
        self.records = self.source.records
        self.traj_ids, self.offsets = _windows(
            self.records, trajectories, n_frames, frame_limit
        )
        self.n_traj = len(self.records)
        self.identity = self.source.identity
        self.spec = ClipSpec(len(self.source.channels), n_frames, *self.source.shape)

    def __getitem__(self, index):
        trajectory, start = self.window(index)
        # Match the training cache's fp16 storage precision on either backend.
        clip = normalize(
            self.source.clip(trajectory, start), self.source.means, self.source.stds
        ).half()
        return {"clip": clip}


@lru_cache(maxsize=8)
def _checked_array(path, size, mtime_ns, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"training cache content hash mismatch: {path}")


class MemmapClipDataset(_ClipDataset):
    def __init__(
        self,
        base_path,
        dataset_name,
        split,
        n_frames=8,
        trajectories=None,
        frame_limit=101,
    ):
        directory = Path(base_path).expanduser() / "memmap" / dataset_name
        self.path = directory / f"{split}.npy"
        meta_path = directory / f"{split}.meta.json"
        meta = json.loads(meta_path.read_text())
        claim = meta.get("sha256")
        body = {k: v for k, v in meta.items() if k != "sha256"}
        if canonical_hash(body) != claim or meta.get("schema") != MEMMAP_SCHEMA:
            raise ValueError(
                "invalid training cache metadata; regenerate with python -m src.data.preprocess"
            )
        if (
            meta.get("dataset") != dataset_name
            or meta.get("split") != split
            or meta.get("layout") != "NCTHW"
            or meta.get("dtype") != "float16"
            or meta.get("normalization") != "the_well_zscore"
            or not meta.get("complete")
        ):
            raise ValueError("incompatible or incomplete training cache")
        self.mm = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if (
            self.mm.ndim != 5
            or self.mm.dtype != np.float16
            or list(self.mm.shape) != meta["shape"]
        ):
            raise ValueError("training cache shape/dtype mismatch")
        self.records = meta["source"]["trajectories"]
        n, c, t, h, w = self.mm.shape
        if (
            len(self.records) != n
            or len(meta["source"]["channels"]) != c
            or meta["source"]["shape"] != [h, w]
            or any(r["frames"] != t for r in self.records)
        ):
            raise ValueError("training cache truncation or source metadata mismatch")
        if canonical_hash(meta["source"]) != meta["source_identity"]:
            raise ValueError("training cache source identity mismatch")
        stats = meta["source"]["normalization"]
        if (
            len(stats["means"]) != c
            or len(stats["stds"]) != c
            or not np.isfinite(stats["means"]).all()
            or not np.isfinite(stats["stds"]).all()
            or min(stats["stds"]) <= 0
        ):
            raise ValueError("invalid training cache normalization")
        stat = self.path.stat()
        _checked_array(
            str(self.path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            meta["array_sha256"],
        )
        self.traj_ids, self.offsets = _windows(
            self.records, trajectories, n_frames, frame_limit
        )
        self.n_frames = n_frames
        self.n_traj = n
        self.identity = meta["source_identity"]
        self.spec = ClipSpec(c, n_frames, h, w)

    def __getitem__(self, index):
        trajectory, start = self.window(index)
        return {
            "clip": torch.from_numpy(
                np.array(self.mm[trajectory, :, start : start + self.n_frames])
            )
        }

"""Wrapper around the_well's WellDataset producing (C, T, H, W) clips.

Both objectives (JEPA / MAE) consume identical clips: T consecutive timesteps
of all physical fields at native resolution, z-score normalized per channel
using The Well's precomputed stats.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization


@dataclass
class ClipSpec:
    """Shape metadata inferred from the dataset, needed to build the models."""

    n_channels: int
    n_frames: int
    height: int
    width: int


class WellClipDataset(torch.utils.data.Dataset):
    """Yields dicts with key "clip": float32 tensor of shape (C, T, H, W)."""

    def __init__(
        self,
        base_path: str,
        dataset_name: str,
        split: str = "train",
        n_frames: int = 8,
    ):
        # the-well-download writes to <base>/datasets/<name>/, but WellDataset
        # joins base/name directly; accept either layout.
        if not str(base_path).startswith("hf://"):
            nested = Path(base_path) / "datasets"
            if (nested / dataset_name).is_dir():
                base_path = str(nested)
        self.inner = WellDataset(
            well_base_path=base_path,
            well_dataset_name=dataset_name,
            well_split_name=split,
            n_steps_input=n_frames,
            n_steps_output=0,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            flatten_tensors=True,
            return_grid=False,
            boundary_return_type=None,
        )
        self.n_frames = n_frames

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        sample = self.inner[idx]
        fields = sample["input_fields"]  # (T, H, W, C)
        clip = fields.permute(3, 0, 1, 2).contiguous().float()  # (C, T, H, W)
        return {"clip": clip}

    @property
    def spec(self) -> ClipSpec:
        clip = self[0]["clip"]
        c, t, h, w = clip.shape
        return ClipSpec(n_channels=c, n_frames=t, height=h, width=w)


class MemmapClipDataset(torch.utils.data.Dataset):
    """Fast clip dataset over a memmap produced by scripts/preprocess_memmap.py.

    File layout: float32 .npy of shape (n_traj, C, T, H, W), already normalized
    (channels-first so a clip is a near-contiguous slice copy needing no
    transpose — ~ms instead of the ~0.5 s per item that WellDataset's per-item
    pipeline costs). Window set (stride-1 windows per trajectory) matches
    WellDataset's. A .meta.json sidecar guards against stale old-layout files.
    """

    def __init__(self, base_path: str, dataset_name: str, split: str, n_frames: int = 8):
        d = Path(base_path) / "memmap" / dataset_name
        self.path = d / f"{split}.npy"
        meta = d / f"{split}.meta.json"
        if not self.path.exists() or not meta.exists():
            raise FileNotFoundError(
                f"{self.path} (+ meta) missing — run scripts/preprocess_memmap.py first"
            )
        if '"NCTHW"' not in meta.read_text():
            raise ValueError(f"{self.path} has stale layout — re-run preprocessing")
        self.mm = np.load(self.path, mmap_mode="r")
        self.n_frames = n_frames
        n_traj, _, t, _, _ = self.mm.shape
        self.windows_per_traj = t - n_frames + 1
        self.length = n_traj * self.windows_per_traj

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        traj, off = divmod(idx, self.windows_per_traj)
        window = np.array(self.mm[traj, :, off : off + self.n_frames])  # (C,T,H,W)
        return {"clip": torch.from_numpy(window)}

    @property
    def spec(self) -> ClipSpec:
        _, c, _, h, w = self.mm.shape
        return ClipSpec(n_channels=c, n_frames=self.n_frames, height=h, width=w)

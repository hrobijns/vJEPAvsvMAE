"""Wrapper around the_well's WellDataset producing (C, T, H, W) clips.

Both objectives (JEPA / MAE) consume identical clips: T consecutive timesteps
of all physical fields at native resolution, z-score normalized per channel
using The Well's precomputed stats.
"""

from dataclasses import dataclass
from pathlib import Path

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

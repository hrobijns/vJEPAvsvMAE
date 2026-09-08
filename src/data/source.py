"""Local Well HDF5 clips, their identities, and normalization metadata."""

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization

from src.evaluation.artifacts import canonical_hash, sha256_file
from src.physics.systems import SYSTEMS


def well_root(base, dataset):
    base = Path(base).expanduser()
    return base / "datasets" if (base / "datasets" / dataset).is_dir() else base


def normalize(raw, means, stds):
    means = torch.as_tensor(means, dtype=raw.dtype, device=raw.device)
    stds = torch.as_tensor(stds, dtype=raw.dtype, device=raw.device)
    if (
        raw.shape[-4] != means.numel()
        or means.shape != stds.shape
        or not torch.isfinite(means).all()
        or not torch.isfinite(stds).all()
        or torch.any(stds <= 0)
    ):
        raise ValueError("invalid channel normalization")
    return (raw - means.reshape(-1, 1, 1, 1)) / stds.reshape(-1, 1, 1, 1)


class WellSource:
    """One ordered trajectory index, shared by training and analysis.

    Use window mode to avoid The Well's full-trajectory mode truncating every
    file to the length of the shortest file. No simulation data is normalized
    until explicitly requested by the caller.
    """

    def __init__(self, base, dataset, split, n_frames=8):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"invalid split: {split}")
        self.dataset, self.split, self.n_frames = dataset, split, n_frames
        self.root = well_root(base, dataset)
        kwargs = dict(
            well_base_path=str(self.root),
            well_dataset_name=dataset,
            well_split_name=split,
            n_steps_input=n_frames,
            n_steps_output=0,
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
        self.channels = [
            name for rank in range(3) for name in self.raw.field_names[rank]
        ]
        self.shape = tuple(self.raw.metadata.spatial_resolution)
        self.records, self.window_bases, self.sources = [], [], []
        counts = defaultdict(int)
        system = SYSTEMS.get(dataset)
        for file_index, filename in enumerate(self.raw.files_paths):
            path = Path(filename)
            stat = path.stat()
            self.sources.append(
                dict(name=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            )
            with h5py.File(path, "r") as handle:
                axes = [
                    str(axis) for axis in handle["dimensions"].attrs["spatial_dims"]
                ]
                if len(axes) != 2:
                    raise ValueError("only two spatial dimensions are supported")
                fields = [
                    f"{field}_{suffix}" if suffix else field
                    for rank in range(3)
                    for field in handle[f"t{rank}_fields"].attrs["field_names"]
                    if handle[f"t{rank}_fields"][field].attrs["time_varying"]
                    for suffix in (
                        [""]
                        if rank == 0
                        else axes
                        if rank == 1
                        else [a + b for a in axes for b in axes]
                    )
                ]
                if fields != self.channels:
                    raise ValueError(f"channel order changes in {path}")
                frames = handle["dimensions/time"].shape[-1]
                params = {}
                if system:
                    for key in system.parameters:
                        value = (
                            handle["scalars"][key][()]
                            if key in handle["scalars"]
                            else handle.attrs[key]
                        )
                        if np.asarray(value).size != 1:
                            raise ValueError(
                                f"expected a file-level regime parameter: {key}"
                            )
                        params[key] = float(np.asarray(value).item())
                        if key in handle.attrs and not np.isclose(
                            params[key], float(handle.attrs[key]), rtol=1e-12
                        ):
                            raise ValueError(f"conflicting regime metadata: {key}")
                    system.regime_values(params)
                    if axes[0] != "x" or axes[1] not in ("y", "z"):
                        raise ValueError(f"unexpected physical axis order: {axes}")
                    if dataset != "rayleigh_benard":
                        for axis, length in zip(axes, system.lengths):
                            grid = np.asarray(handle[f"dimensions/{axis}"]).reshape(-1)
                            delta = np.diff(grid)
                            if (
                                len(delta) < 1
                                or not np.allclose(
                                    delta, delta[0], rtol=1e-4, atol=1e-7
                                )
                                or delta[0] <= 0
                            ):
                                raise ValueError(
                                    f"expected a uniform periodic {axis} grid"
                                )
                            # Shear-flow exports label both axes [0, 1].
                            # Physical derivatives still use lengths (1, 2).
                            if dataset == "shear_flow" and np.allclose(
                                grid[[0, -1]], (0, 1), rtol=1e-4, atol=1e-7
                            ):
                                continue
                            # Some Well exports include the endpoint in coordinates
                            # while the simulation fields exclude it.
                            if not any(
                                np.isclose(delta[0] * n, length, rtol=1e-4)
                                for n in (len(grid), len(grid) - 1)
                            ):
                                raise ValueError(f"unexpected {axis} domain length")
                    else:
                        if self.channels != [
                            "buoyancy",
                            "pressure",
                            "velocity_x",
                            "velocity_y",
                        ] or self.shape != (512, 128):
                            raise ValueError(
                                "RB channel order or simulation geometry differs"
                            )
                self.axes = axes
            regime = tuple(params.values())
            for local in range(self.raw.n_trajectories_per_file[file_index]):
                trajectory = len(self.records)
                self.records.append(
                    dict(
                        trajectory=trajectory,
                        source_file=path.name,
                        source_trajectory=local,
                        trajectory_id=f"{split}/{path.name}/{local}",
                        frames=int(frames),
                        parameters=dict(params),
                        regime=list(regime),
                        replicate=counts[regime],
                    )
                )
                counts[regime] += 1
                self.window_bases.append(
                    max(0, self.raw.file_index_offsets[file_index])
                    + local * self.raw.n_windows_per_trajectory[file_index]
                )
        if system and len(self.channels) != system.channels:
            raise ValueError(f"unexpected {dataset} channel count")
        self.velocity_channels = (
            tuple(self.channels.index(f"velocity_{axis}") for axis in self.axes)
            if system
            else ()
        )
        self.contract = dict(
            dataset=dataset,
            split=split,
            channels=self.channels,
            normalization=dict(
                means=self.means.tolist(),
                stds=self.stds.tolist(),
                stats_sha256=sha256_file(normalized.normalization_path),
            ),
            shape=list(self.shape),
            sources=self.sources,
            trajectories=self.records,
        )
        self.identity = canonical_hash(self.contract)

    def clip(self, trajectory, start):
        if (
            not 0 <= trajectory < len(self.records)
            or not 0 <= start <= self.records[trajectory]["frames"] - self.n_frames
        ):
            raise IndexError("clip falls outside its trajectory")
        item = self.raw[self.window_bases[trajectory] + start]
        raw = item["input_fields"].permute(3, 0, 1, 2).contiguous().float()
        if not torch.isfinite(raw).all():
            raise ValueError("non-finite physical fields")
        return raw

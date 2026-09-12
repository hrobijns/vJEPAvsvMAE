"""Manufactured physical fields in actual Well HDF5 format."""

from pathlib import Path
import h5py
import numpy as np
import yaml
from src.physics.systems import SYSTEMS


def write_well(root, dataset, split, frames=6, trajectories=5, shape=None):
    system = SYSTEMS[dataset]
    nx, ny = shape or ((512, 128) if dataset == "rayleigh_benard" else (16, 24))
    xx = np.arange(nx, dtype=np.float64) * system.lengths[0] / nx
    yy = (
        (1 - np.cos(np.pi * (np.arange(ny) + 0.5) / ny)) / 2
        if dataset == "rayleigh_benard"
        else np.arange(ny) * system.lengths[1] / ny
    )
    X, Y = xx[:, None], yy[None, :]
    folder = Path(root) / "datasets" / dataset
    (folder / "data" / split).mkdir(parents=True, exist_ok=True)
    names = {
        0: ["concentration"]
        if dataset == "active_matter"
        else ["buoyancy", "pressure"]
        if dataset == "rayleigh_benard"
        else ["tracer", "pressure"],
        1: ["velocity"],
        2: ["D", "E"] if dataset == "active_matter" else [],
    }
    stats = {key: {} for key in ("mean", "std", "mean_delta", "std_delta")}
    for rank, fields in names.items():
        for field in fields:
            dims = (2,) * rank
            for key in stats:
                value = np.full(dims, 0.0 if "mean" in key else 2.0)
                stats[key][field] = value.tolist()
    (folder / "stats.yaml").write_text(yaml.safe_dump(stats))
    path = folder / "data" / split / "manufactured.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs.update(
            dataset_name=dataset,
            grid_type="cartesian",
            n_spatial_dims=2,
            n_trajectories=trajectories,
            simulation_parameters=system.parameters,
        )
        f.create_group("boundary_conditions")
        d = f.create_group("dimensions")
        d.attrs["spatial_dims"] = ["x", "y"]
        for key, value in [("x", xx), ("y", yy), ("time", np.arange(frames) * 0.25)]:
            # Reproduce the exported labels; fields below use physical X, Y.
            if dataset == "shear_flow" and key in ("x", "y"):
                value = np.linspace(0, 1, len(value), dtype=np.float32)
            d.create_dataset(key, data=value)
            d[key].attrs["sample_varying"] = False
        scalars = f.create_group("scalars")
        scalars.attrs["field_names"] = list(system.parameters)
        params = (
            (-2.0, 3.0)
            if dataset == "active_matter"
            else (1e6, 1.0)
            if dataset == "rayleigh_benard"
            else (100.0, 1.0)
        )
        for key, value in zip(system.parameters, params):
            f.attrs[key] = value
            a = scalars.create_dataset(key, data=value)
            a.attrs.update(sample_varying=False, time_varying=False)
        for rank, fields in names.items():
            g = f.create_group(f"t{rank}_fields")
            g.attrs["field_names"] = fields
            for field in fields:
                dims = (trajectories, frames, nx, ny) + (2,) * rank
                a = g.create_dataset(
                    field, shape=dims, dtype="float32", compression="lzf"
                )
                a.attrs.update(
                    time_varying=True, sample_varying=True, dim_varying=[True, True]
                )
                for trajectory in range(trajectories):
                    for t in range(frames):
                        amp = (
                            1
                            + 0.1 * trajectory
                            + 0.03 * t
                            + (0.025 if split == "test" else 0.0)
                        )
                        if field == "velocity":
                            u = (
                                amp
                                * np.cos(2 * np.pi * Y / system.lengths[1])
                                * np.ones_like(X)
                            )
                            v = (
                                amp
                                * np.sin(2 * np.pi * X / system.lengths[0])
                                * np.ones_like(Y)
                            )
                            a[trajectory, t] = np.stack((u, v), axis=-1)
                        elif field == "concentration":
                            a[trajectory, t] = 1 + 0.01 * np.cos(
                                2 * np.pi * X / system.lengths[0]
                            ) * np.ones_like(Y)
                        elif field == "D":
                            c = 1 + 0.01 * np.cos(2 * np.pi * X / system.lengths[0])
                            strength = (
                                0.1
                                * amp
                                * (1 + 0.2 * np.cos(2 * np.pi * Y / system.lengths[1]))
                            )
                            qxx = strength * np.cos(2 * np.pi * X / system.lengths[0])
                            qxy = strength * np.sin(2 * np.pi * X / system.lengths[0])
                            tensor = np.empty((nx, ny, 2, 2))
                            tensor[..., 0, 0] = c * (0.5 + qxx)
                            tensor[..., 1, 1] = c * (0.5 - qxx)
                            tensor[..., 0, 1] = tensor[..., 1, 0] = c * qxy
                            a[trajectory, t] = tensor
                        elif rank == 0:
                            a[trajectory, t] = (
                                amp
                                * np.sin(2 * np.pi * X / system.lengths[0])
                                * np.ones_like(Y)
                                + Y
                            )
                        else:
                            a[trajectory, t] = np.zeros((nx, ny, 2, 2), dtype="float32")
    return path

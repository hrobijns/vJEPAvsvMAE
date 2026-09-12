"""Physical target caches with normalized copies of encoder contexts."""

from pathlib import Path

import numpy as np

from src.data.source import WellSource, normalize
from src.evaluation.artifacts import Artifact, seal, staged_directory, write_json
from src.evaluation.protocol import token_indices
from src.physics.systems import SYSTEMS


def prepare_cache(base, split, cache_root, protocol):
    if split not in ("train", "valid", "test"):
        raise ValueError("analysis uses official train, valid and test splits")
    source = WellSource(base, protocol.dataset, split, protocol.n_frames)
    system = SYSTEMS[protocol.dataset]
    dimensions = (protocol.n_frames, *source.shape)
    if any(size % patch for size, patch in zip(dimensions, protocol.patch)):
        raise ValueError("input dimensions are not divisible by patches")
    grid = tuple(size // patch for size, patch in zip(dimensions, protocol.patch))
    samples = {kind: [] for kind in ("pooled", "token")}
    for record in source.records:
        for kind, offsets in protocol.offsets(
            record["frames"], record["trajectory"]
        ).items():
            for start in offsets:
                samples[kind].append(
                    {
                        **record,
                        "sample": len(samples[kind]),
                        "context_start": start,
                        "age": float(start / max(record["frames"] - 1, 1)),
                    }
                )
    destination = Path(cache_root) / split
    with staged_directory(destination) as stage:
        for kind, rows in samples.items():
            folder = stage / kind
            folder.mkdir()
            shape = (len(rows), len(source.channels), *dimensions)
            contexts = np.lib.format.open_memmap(
                folder / "context.npy", mode="w+", dtype=np.float16, shape=shape
            )
            positions = None
            if kind == "token":
                positions = np.stack(
                    [
                        token_indices(
                            row["trajectory"],
                            int(np.prod(grid)),
                            protocol.token_samples,
                        )
                        for row in rows
                    ]
                )
                np.save(folder / "positions.npy", positions)
            targets = {
                (offset, target): np.lib.format.open_memmap(
                    folder / f"offset{offset}_{target}.npy",
                    mode="w+",
                    dtype=np.float64,
                    shape=(len(rows),)
                    if kind == "pooled"
                    else (len(rows), protocol.token_samples),
                )
                for offset in protocol.target_offsets
                for target in system.targets
            }
            for i, row in enumerate(rows):
                context = source.clip(row["trajectory"], row["context_start"])
                contexts[i] = (
                    normalize(context, source.means, source.stds).half().numpy()
                )
                for offset in protocol.target_offsets:
                    raw = (
                        context
                        if offset == 0
                        else source.clip(
                            row["trajectory"],
                            protocol.target_start(row["context_start"], offset),
                        )
                    )
                    fields = system.fields(
                        raw.unsqueeze(0), source.velocity_channels, source.channels
                    )
                    for target, field in fields.items():
                        values = system.reduce(
                            field, protocol.patch if kind == "token" else None
                        )[0].numpy()
                        targets[offset, target][i] = (
                            values if positions is None else values[positions[i]]
                        )
                if (i + 1) % 25 == 0 or i + 1 == len(rows):
                    print(f"{split}/{kind}: {i + 1}/{len(rows)} contexts", flush=True)
            contexts.flush()
            for array in targets.values():
                array.flush()
            write_json(folder / "samples.json", rows)
        seal(
            stage,
            "cache",
            protocol=protocol.to_dict(),
            split=split,
            source=source.contract,
            source_identity=source.identity,
            grid=list(grid),
        )
    return destination


def open_caches(root, splits=("train", "valid")):
    caches = tuple(Artifact(Path(root) / split, "cache") for split in splits)
    for split, cache in zip(splits, caches):
        if cache.manifest["split"] != split:
            raise ValueError("cache split identity mismatch")
        for key in ("protocol", "grid"):
            if cache.manifest[key] != caches[0].manifest[key]:
                raise ValueError(f"cache {key} differs")
        for key in ("channels", "shape", "normalization"):
            if cache.manifest["source"][key] != caches[0].manifest["source"][key]:
                raise ValueError(f"cache {key} differs")
    return caches

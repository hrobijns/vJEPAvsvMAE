"""Build a complete normalized fp16 training cache; experiment sampling is separate."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.source import WellSource, normalize
from src.data.well import MEMMAP_SCHEMA
from src.evaluation.artifacts import canonical_hash, sha256_file, write_json


def preprocess(base, dataset, split="train"):
    source = WellSource(base, dataset, split)
    frames = {r["frames"] for r in source.records}
    if len(frames) != 1:
        raise ValueError(
            "rectangular training caches require equal trajectory lengths; use HDF5 training for variable lengths"
        )
    frames = frames.pop()
    source = WellSource(base, dataset, split, n_frames=frames)
    shape = (len(source.records), len(source.channels), frames, *source.shape)
    directory = Path(base).expanduser() / "memmap" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{split}.npy"
    meta = directory / f"{split}.meta.json"
    progress = directory / f"{split}.progress.json"
    partial = directory / f".{split}.partial.npy"
    if output.exists() or meta.exists():
        raise FileExistsError(f"completed or ambiguous cache exists: {output}")
    start = 0
    if partial.exists() != progress.exists():
        raise ValueError("inconsistent partial training cache")
    if progress.exists():
        saved = json.loads(progress.read_text())
        if saved["source_identity"] != source.identity or saved["shape"] != list(shape):
            raise ValueError("partial cache does not match source")
        start = saved["next_index"]
        if not 0 <= start <= shape[0]:
            raise ValueError("invalid resume position")
        mm = np.load(partial, mmap_mode="r+")
        if mm.shape != shape or mm.dtype != np.float16:
            raise ValueError("invalid partial cache")
    else:
        mm = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float16, shape=shape
        )
    for i in range(start, shape[0]):
        mm[i] = normalize(source.clip(i, 0), source.means, source.stds).half().numpy()
        mm.flush()
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                dict(
                    source_identity=source.identity, shape=list(shape), next_index=i + 1
                )
            )
        )
        temporary.replace(progress)
        print(f"{dataset}/{split}: {i + 1}/{shape[0]}", flush=True)
    mm.flush()
    payload = dict(
        schema=MEMMAP_SCHEMA,
        dataset=dataset,
        split=split,
        layout="NCTHW",
        dtype="float16",
        shape=list(shape),
        normalization="the_well_zscore",
        complete=True,
        source=source.contract,
        source_identity=source.identity,
        array_sha256=sha256_file(partial),
    )
    payload["sha256"] = canonical_hash(payload)
    partial.rename(output)
    write_json(meta, payload)
    progress.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    args = parser.parse_args()
    print(preprocess(args.base, args.dataset, args.split))


if __name__ == "__main__":
    main()

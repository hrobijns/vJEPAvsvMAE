"""One-time conversion: Well HDF5 -> normalized float16 memmap for training.

the_well's WellDataset costs ~0.5s CPU per clip (per-item flattening and
preprocessing), far too slow for millions of clip fetches. This script pays
that cost once per trajectory (full_trajectory_mode), storing normalized
trajectories as a single .npy of shape (n_traj, C, T, H, W) that
MemmapClipDataset (src/data/well.py) can slice in ~ms.

Rayleigh--Benard is required to contain the full 200-frame trajectory.  The
old implicit ``max_rollout_steps=100`` produced a silently truncated
101-frame file and is rejected here.  This training cache is normalized and
must not be used to derive physical-unit targets; use ``rb_eval_v2.py
prepare-cache`` for the corrected analysis cache.

Usage:
    uv run python scripts/preprocess_memmap.py --base /workspace/data \
        --dataset active_matter --split train
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization


EXPECTED_TRAJECTORY_FRAMES = {"rayleigh_benard": 200}
MEMMAP_SCHEMA_VERSION = "well-memmap-v2"


def full_trajectory_max_rollout(dataset: str) -> int:
    """Output-frame cap passed to The Well in full-trajectory mode.

    The loader defaults to 100 output steps; with one input step that silently
    creates the legacy 101-frame RB cache.  Known datasets get an exact cap;
    unknown datasets use a deliberately high value and are bounded by their
    own HDF5 metadata inside WellDataset.
    """
    expected = EXPECTED_TRAJECTORY_FRAMES.get(dataset)
    return expected - 1 if expected is not None else 1_000_000


def validate_trajectory_length(dataset: str, frames: int) -> None:
    expected = EXPECTED_TRAJECTORY_FRAMES.get(dataset)
    if expected is not None and frames != expected:
        raise ValueError(
            f"{dataset} expected {expected} frames, got {frames}; this is likely the "
            "legacy max_rollout_steps=100 truncation"
        )


def validate_memmap_state(output: Path, metadata: Path, progress: Path) -> str:
    """Return ``new`` or ``resume``; never overwrite ambiguous/completed data."""
    if metadata.exists():
        raise FileExistsError(
            f"completed memmap already exists ({output} + {metadata}); move it aside to rebuild"
        )
    if output.exists() and progress.exists():
        return "resume"
    if output.exists() != progress.exists():
        raise FileExistsError(
            f"inconsistent partial cache state: output={output.exists()} progress={progress.exists()}"
        )
    return "new"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_contract(dataset) -> dict:
    files = []
    for value in dataset.files_paths:
        path = Path(str(value))
        row = {"path": str(path), "name": path.name}
        if path.exists():
            stat = path.stat()
            row.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        files.append(row)
    stats = Path(str(dataset.normalization_path))
    return {
        "source_files": files,
        "stats_path": str(stats),
        "stats_sha256": _sha256(stats) if stats.exists() else None,
    }


def _contract_hash(contract: dict) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def memmap_metadata(
    *, dataset: str, split: str, shape: tuple[int, ...], source_contract: dict
) -> dict:
    return {
        "schema_version": MEMMAP_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "layout": "NCTHW",
        "dtype": "float16",
        "shape": list(shape),
        "normalization": "the_well_zscore",
        "physical_targets_safe": False,
        "source_contract": source_contract,
    }


def retry_io(fn, *args, retries=8, delay=3.0, **kwargs):
    """The network filesystem backing /workspace has shown transient write
    failures (observed: "Disk quota Exceeded" that cleared on the very next
    retry) severe enough to raise an uncaught OSError and kill the whole
    process. Retry a few times with a short pause before giving up for real."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            if attempt == retries - 1:
                raise
            print(f"WARNING: transient I/O error ({e}), retrying in {delay}s "
                  f"(attempt {attempt+1}/{retries})", flush=True)
            time.sleep(delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    base = Path(args.base)
    if (base / "datasets" / args.dataset).is_dir():
        well_base = base / "datasets"
    else:
        well_base = base

    ds = WellDataset(
        well_base_path=str(well_base),
        well_dataset_name=args.dataset,
        well_split_name=args.split,
        n_steps_input=1,
        n_steps_output=0,
        max_rollout_steps=full_trajectory_max_rollout(args.dataset),
        full_trajectory_mode=True,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        flatten_tensors=True,
        return_grid=False,
        boundary_return_type=None,
    )
    n = len(ds)
    first = ds[0]
    # (T,H,W,C) -> (C,T,H,W): stored channels-first so training clips are
    # near-contiguous slices needing no per-item transpose.
    traj0 = torch.cat([first["input_fields"], first["output_fields"]], dim=0)
    traj0 = traj0.permute(3, 0, 1, 2).contiguous()
    c, t, h, w = traj0.shape
    validate_trajectory_length(args.dataset, t)
    out_dir = base / "memmap" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.npy"
    progress_path = out_dir / f"{args.split}.progress.json"
    metadata_path = out_dir / f"{args.split}.meta.json"
    state = validate_memmap_state(out_path, metadata_path, progress_path)
    contract = source_contract(ds)
    contract_sha256 = _contract_hash(contract)
    print(f"{args.dataset}/{args.split}: {n} trajectories of C,T,H,W=({c},{t},{h},{w}) "
          f"-> {out_path} ({n*t*h*w*c*2/1e9:.1f} GB)", flush=True)

    # Interruptions (pod-level process kills, seen recurring on some hosts)
    # otherwise cost the entire run -- this can take 20-30 min for the larger
    # datasets, so resume from wherever the last run left off rather than
    # starting trajectory 0 over from scratch every time.
    start_i = 0
    if state == "resume":
        prog = json.loads(progress_path.read_text())
        if (
            prog.get("shape") == [n, c, t, h, w]
            and prog.get("source_contract_sha256") == contract_sha256
        ):
            start_i = prog["next_index"]
            print(f"resuming from trajectory {start_i}/{n}", flush=True)
        else:
            raise ValueError(
                "partial cache does not match the current shape/source/stats; move it aside and restart"
            )
    if start_i == 0:
        # fp16 storage: halves batch bytes through the pin-memory/H2D path
        # (the throughput bottleneck); z-scored fields lose nothing
        # meaningful at fp16. Training casts back to fp32 on-GPU.
        mm = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float16, shape=(n, c, t, h, w)
        )
        mm[0] = traj0.to(torch.float16).numpy()
        start_i = 1
        progress_path.write_text(json.dumps({
            "shape": [n, c, t, h, w],
            "next_index": start_i,
            "source_contract_sha256": contract_sha256,
        }))
    else:
        mm = np.lib.format.open_memmap(out_path, mode="r+")

    for i in range(start_i, n):
        item = ds[i]
        traj = torch.cat([item["input_fields"], item["output_fields"]], dim=0)
        traj = traj.permute(3, 0, 1, 2)
        assert traj.shape == (c, t, h, w), (i, traj.shape)
        retry_io(lambda: mm.__setitem__(i, traj.to(torch.float16).numpy()))
        if i % 5 == 0 or i == n - 1:
            retry_io(mm.flush)
            retry_io(progress_path.write_text, json.dumps({
                "shape": [n, c, t, h, w],
                "next_index": i + 1,
                "source_contract_sha256": contract_sha256,
            }))
            print(f"  {i+1}/{n}", flush=True)
    retry_io(mm.flush)
    retry_io(
        metadata_path.write_text,
        json.dumps(memmap_metadata(
            dataset=args.dataset,
            split=args.split,
            shape=(n, c, t, h, w),
            source_contract=contract,
        )),
    )
    progress_path.unlink(missing_ok=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

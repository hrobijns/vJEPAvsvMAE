"""One-time conversion: Well HDF5 -> flat float32 memmap for fast training.

the_well's WellDataset costs ~0.5s CPU per clip (per-item flattening and
preprocessing), far too slow for millions of clip fetches. This script pays
that cost once per trajectory (full_trajectory_mode), storing normalized
trajectories as a single .npy of shape (n_traj, T, H, W, C) that
MemmapClipDataset (src/data/well.py) can slice in ~ms.

Usage:
    uv run python scripts/preprocess_memmap.py --base /workspace/data \
        --dataset active_matter --split train
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization


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
    out_dir = base / "memmap" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.npy"
    progress_path = out_dir / f"{args.split}.progress.json"
    print(f"{args.dataset}/{args.split}: {n} trajectories of C,T,H,W=({c},{t},{h},{w}) "
          f"-> {out_path} ({n*t*h*w*c*4/1e9:.1f} GB)", flush=True)

    # Interruptions (pod-level process kills, seen recurring on some hosts)
    # otherwise cost the entire run -- this can take 20-30 min for the larger
    # datasets, so resume from wherever the last run left off rather than
    # starting trajectory 0 over from scratch every time.
    start_i = 0
    if out_path.exists() and progress_path.exists():
        prog = json.loads(progress_path.read_text())
        if prog.get("shape") == [n, c, t, h, w]:
            start_i = prog["next_index"]
            print(f"resuming from trajectory {start_i}/{n}", flush=True)
    if start_i == 0:
        # fp16 storage: halves batch bytes through the pin-memory/H2D path
        # (the throughput bottleneck); z-scored fields lose nothing
        # meaningful at fp16. Training casts back to fp32 on-GPU.
        mm = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float16, shape=(n, c, t, h, w)
        )
        mm[0] = traj0.to(torch.float16).numpy()
        start_i = 1
        progress_path.write_text(json.dumps({"shape": [n, c, t, h, w], "next_index": start_i}))
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
            retry_io(progress_path.write_text, json.dumps({"shape": [n, c, t, h, w], "next_index": i + 1}))
            print(f"  {i+1}/{n}", flush=True)
    retry_io(mm.flush)
    retry_io(
        (out_dir / f"{args.split}.meta.json").write_text,
        json.dumps({"layout": "NCTHW", "dtype": "float16", "shape": [n, c, t, h, w]}),
    )
    progress_path.unlink(missing_ok=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

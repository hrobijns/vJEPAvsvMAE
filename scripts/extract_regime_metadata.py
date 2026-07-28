"""Reconstructs which raw HDF5 file (and therefore which regime parameters —
Reynolds/Schmidt, Rayleigh/Prandtl, activity/alignment) each memmap
trajectory row came from, without reading any field data.

preprocess_memmap.py iterates WellDataset(..., full_trajectory_mode=True) by
index 0..n-1, in exactly the order determined by `files_paths` (sorted) +
`n_trajectories_per_file` (cumulative offsets). Rebuilding the same
WellDataset here (same kwargs, no normalization needed) reconstructs that
same file-to-row mapping deterministically and cheaply — regex-parses regime
parameters out of each filename, writes a sidecar index-aligned with the
memmap.

Usage:
    uv run python scripts/extract_regime_metadata.py \
        --base /workspace/data --dataset shear_flow --split train
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from the_well.data import WellDataset

# Patterns are best-effort based on The Well's documented per-file naming
# convention; verify against `--dry-run` output on real filenames before
# trusting downstream regime probing (the script reports match rate and
# example unmatched names either way).
REGIME_PATTERNS = {
    "shear_flow": {
        "Reynolds": r"[Rr]eynolds[_-]?([0-9.eE+-]+)",
        "Schmidt": r"[Ss]chmidt[_-]?([0-9.eE+-]+)",
    },
    "rayleigh_benard": {
        "Rayleigh": r"[Rr]ayleigh[_-]?([0-9.eE+-]+)",
        "Prandtl": r"[Pp]randtl[_-]?([0-9.eE+-]+)",
    },
    "active_matter": {
        "alpha": r"[Aa]lpha[_-]?([0-9.eE+-]+)",
        "zeta": r"[Zz]eta[_-]?([0-9.eE+-]+)",
    },
}


def parse_regime(filename: str, dataset: str) -> dict:
    # strip the extension first: the capture group's charset includes "."
    # (needed for decimals like "15.0"), so left on a trailing ".hdf5" it
    # would happily eat that separator dot too and break float() parsing.
    stem = re.sub(r"\.hdf5$", "", filename)
    out = {}
    for name, pattern in REGIME_PATTERNS.get(dataset, {}).items():
        m = re.search(pattern, stem)
        out[name] = float(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    base = Path(args.base)
    well_base = base / "datasets" if (base / "datasets" / args.dataset).is_dir() else base

    ds = WellDataset(
        well_base_path=str(well_base),
        well_dataset_name=args.dataset,
        well_split_name=args.split,
        n_steps_input=1,
        n_steps_output=0,
        full_trajectory_mode=True,
        use_normalization=False,
        flatten_tensors=True,
        return_grid=False,
        boundary_return_type=None,
    )

    files = ds.files_paths
    counts = ds.n_trajectories_per_file
    assert sum(counts) == len(ds), f"trajectory count mismatch: {sum(counts)} vs {len(ds)}"

    records = []
    unmatched_names = []
    for file_path, n_traj in zip(files, counts):
        name = Path(str(file_path)).name
        regime = parse_regime(name, args.dataset)
        if regime and all(v is None for v in regime.values()):
            unmatched_names.append(name)
        for _ in range(n_traj):
            records.append({"file": name, **regime})

    assert len(records) == len(ds), f"record count {len(records)} != dataset length {len(ds)}"

    out_dir = base / "memmap" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.regime.json"
    out_path.write_text(json.dumps(records))

    print(f"{args.dataset}/{args.split}: {len(records)} trajectories from {len(files)} files "
          f"({len(unmatched_names)} files unmatched by regex) -> {out_path}")
    if unmatched_names:
        print("  example unmatched filenames (regex needs adjusting):")
        for name in unmatched_names[:5]:
            print(f"    {name}")


if __name__ == "__main__":
    main()

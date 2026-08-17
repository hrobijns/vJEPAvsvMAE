"""Pick the LR with the best validation loss from an LR mini-sweep and record
it in configs/tuned_lr.json. Selection is by validation loss only (never
test-split or probing metrics) -- see the plan's methodology note on LR
selection protocol. Called by scripts/run_lr_minisweep.sh after all candidate
runs for a (dataset, objective) pair finish.

Usage:
    uv run python scripts/pick_lr.py --dataset active_matter --objective jepa \
        --candidates 0.00005:runs_lrsweep/active_matter_jepa/lr_0.5x/active_matter_jepa_seed0 \
                     0.0001:runs_lrsweep/active_matter_jepa/lr_1x/active_matter_jepa_seed0 \
                     0.0002:runs_lrsweep/active_matter_jepa/lr_2x/active_matter_jepa_seed0
"""

import argparse
import json
from pathlib import Path


def best_val_loss(run_dir: Path) -> float:
    losses = []
    for line in (run_dir / "history.jsonl").read_text().splitlines():
        # Network-filesystem reads have occasionally returned a torn/empty
        # line for the most recently written entry (observed on /workspace's
        # MooseFS mount) -- skip lines that don't parse rather than crash,
        # since every real entry is one self-contained JSON object per line.
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["phase"] == "val":
            losses.append(row["loss"])
    if not losses:
        raise ValueError(f"no val rows found in {run_dir / 'history.jsonl'}")
    return min(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--candidates", nargs="+", required=True, help="lr:run_dir pairs")
    ap.add_argument("--out", default="configs/tuned_lr.json")
    args = ap.parse_args()

    scored = []
    for c in args.candidates:
        lr_str, run_dir = c.split(":", 1)
        lr = float(lr_str)
        val = best_val_loss(Path(run_dir))
        scored.append((lr, val))
        print(f"  lr={lr:.2e}  best_val_loss={val:.5f}  ({run_dir})")

    best_lr, best_val = min(scored, key=lambda x: x[1])
    print(f"selected lr={best_lr:.2e} for {args.dataset}/{args.objective} (val_loss={best_val:.5f})")

    out_path = Path(args.out)
    table = json.loads(out_path.read_text()) if out_path.exists() else {}
    table.setdefault(args.dataset, {})[args.objective] = best_lr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Plot the trainer's JSONL loss history; no historical result files are needed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fig, axes = plt.subplots(
        len(args.runs), 1, figsize=(8, 3 * len(args.runs)), squeeze=False
    )
    for axis, run in zip(axes[:, 0], args.runs):
        rows = [
            json.loads(line)
            for line in (run / "history.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for split in ("train", "val"):
            values = [r for r in rows if r["phase"] == split and "loss" in r]
            axis.plot(
                [r["step"] for r in values], [r["loss"] for r in values], label=split
            )
        axis.set_title(run.name)
        axis.set_xlabel("Optimization step")
        axis.set_ylabel("Objective loss")
        axis.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()

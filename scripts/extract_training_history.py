"""Extract per-step loss/lr/grad_norm history from local wandb offline-run logs.

Reads the binary .wandb datastore files under pod_logs/wandb/ (gitignored,
RunPod artifacts — not present unless you have the original training logs)
and writes a small committed CSV that downstream plotting can use without
needing the raw logs.
"""

import csv
import glob
import json
import os

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

RUN_DIR_GLOB = "pod_logs/wandb/offline-run-*"
OUT_CSV = "sweep_results/training_history.csv"
FIELDS = ["dataset", "method", "step", "loss", "lr", "grad_norm"]


def parse_run(wandb_file: str) -> list[dict]:
    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            key = ".".join(item.nested_key) if item.nested_key else item.key
            row[key] = json.loads(item.value_json)
        rows.append(row)
    return rows


def main():
    run_dirs = sorted(glob.glob(RUN_DIR_GLOB))
    if not run_dirs:
        raise SystemExit(f"no run directories found under {RUN_DIR_GLOB}")

    out_rows = []
    for run_dir in run_dirs:
        wandb_files = glob.glob(os.path.join(run_dir, "*.wandb"))
        if not wandb_files:
            continue
        run_name = os.path.basename(wandb_files[0])[len("run-") : -len(".wandb")]
        dataset, method = run_name.rsplit("_", 1)

        history = parse_run(wandb_files[0])
        for row in history:
            out_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "step": row["_step"],
                    "loss": row["loss"],
                    "lr": row.get("lr"),
                    "grad_norm": row.get("grad_norm"),
                }
            )
        print(f"{run_name}: {len(history)} logged steps")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {OUT_CSV} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()

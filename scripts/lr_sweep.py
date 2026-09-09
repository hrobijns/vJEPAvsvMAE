"""Prepare, run, and summarize the agreed four-objective learning-rate pilots."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.artifacts import canonical_hash, provenance, sha256_file
from src.objectives import OBJECTIVES
from src.physics.systems import SYSTEMS
from scripts.stage_training_cache import stage

RATES = {"jepa": (5e-5, 1e-4, 2e-4), "mae": (2.5e-5, 5e-5, 1e-4)}
STEPS = 8000
VALIDATION_STEPS = (2000, 4000, 6000, 8000)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(dataset, base, output, workers=None):
    if workers is not None and workers < 0:
        raise ValueError("workers must be nonnegative")
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("sweep already prepared; use its existing manifest or a new output")
    (output / "configs").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    candidates = []
    # Interleave objectives so an initial group of four jobs covers all models.
    for rate_index in range(3):
        for objective in OBJECTIVES:
            rate = RATES[objective.split("_")[0]][rate_index]
            config = yaml.safe_load((ROOT / "configs" / f"{dataset}_{objective}.yaml").read_text())
            name = f"{dataset}_{objective}_lr{rate:.6g}_seed0"
            config.update(run_name=name, out_dir=str(output / "runs"), seed=0,
                          val_every=2000, save_every=1000)
            config.pop("val_max_batches", None)
            config["data"].update(base_path=str(Path(base).resolve()), memmap=True)
            if workers is not None:
                config["data"]["num_workers"] = workers
            config["optim"].update(lr=rate, total_steps=STEPS, warmup_steps=5000)
            config["wandb"]["enabled"] = False
            path = output / "configs" / f"{len(candidates):02d}_{name}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            candidates.append(dict(index=len(candidates), objective=objective, lr=rate,
                                   config=str(path), config_sha256=canonical_hash(config),
                                   run_dir=str(output / "runs" / name)))
    manifest = dict(dataset=dataset, seed=0, steps=STEPS, validation_steps=VALIDATION_STEPS,
                    selection="minimum recorded internal validation loss in a completed pilot",
                    provenance=provenance(), launcher_sha256=sha256_file(__file__),
                    staging_sha256=sha256_file(ROOT / "scripts/stage_training_cache.py"), candidates=candidates)
    write_json(output / "manifest.json", manifest)
    return manifest


def score_history(rows, steps=STEPS):
    """Require a completed scheduled validation series; retain collapse diagnostics."""
    validation = [row for row in rows if row["phase"] == "val"]
    if [row["step"] for row in validation] != list(VALIDATION_STEPS):
        raise ValueError("incomplete or duplicate validation history")
    for row in rows:
        if not 0 < row["step"] <= steps or not math.isfinite(row["loss"]):
            raise ValueError("non-finite loss or invalid optimizer step")
        for key, value in row.items():
            if key.endswith("_feat_std") and (not math.isfinite(value) or value < 0):
                raise ValueError("invalid feature diagnostics")
    best = min(validation, key=lambda row: row["loss"])
    first = validation[0]
    ratios = {
        key: min(row[key] for row in validation) / value if value > 0 else 0.0
        for key, value in first.items() if key.endswith("_feat_std")
    }
    return dict(best_val_loss=best["loss"], best_val_step=best["step"],
                final_val_loss=validation[-1]["loss"],
                minimum_feature_std_ratio=ratios,
                collapse_flags=[key for key, ratio in ratios.items() if ratio < 0.1],
                validation=validation)


def run(output, index, cache_root=None):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest["provenance"]["code_sha256"] != provenance()["code_sha256"]
            or manifest["launcher_sha256"] != sha256_file(__file__)
            or manifest["staging_sha256"] != sha256_file(ROOT / "scripts/stage_training_cache.py")):
        raise ValueError("source changed since the sweep was prepared")
    candidate = manifest["candidates"][index]
    config = yaml.safe_load(Path(candidate["config"]).read_text())
    if canonical_hash(config) != candidate["config_sha256"]:
        raise ValueError("pilot configuration changed")
    directory = Path(candidate["run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    runtime = copy.deepcopy(config)
    staging = None
    if cache_root is not None:
        staging = stage(config["data"]["base_path"], manifest["dataset"], cache_root)
        runtime["data"]["base_path"] = staging["base_path"]
    runtime_path = directory / "runtime_config.yaml"
    runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False))
    runtime_hash = canonical_hash(runtime)
    invocation = dict(job_id=os.environ.get("SLURM_JOB_ID"),
                      array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                      task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                      host=os.uname().nodename, torch=torch.__version__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                      config_sha256=candidate["config_sha256"], runtime_config_sha256=runtime_hash,
                      staging=staging, provenance=provenance())
    write_json(directory / "invocation.json", invocation)
    subprocess.run([sys.executable, "-u", "-B", "-m", "src.train", "--config", str(runtime_path),
                    "--no-wandb"], cwd=ROOT, check=True)
    checkpoint = torch.load(directory / "encoder_100pct.pt", map_location="cpu", weights_only=False)
    if checkpoint["step"] != STEPS or canonical_hash(checkpoint["config"]) != runtime_hash:
        raise ValueError("final checkpoint does not match the planned pilot")
    rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    result = score_history(rows)
    result.update(config_sha256=candidate["config_sha256"],
                  runtime_config_sha256=runtime_hash, staging=staging,
                  checkpoint_sha256=sha256_file(directory / "encoder_100pct.pt"),
                  history_sha256=sha256_file(directory / "history.jsonl"),
                  data_exposure=checkpoint["data_exposure"],
                  training_identity=checkpoint["training_identity"])
    write_json(directory / "completed.json", result)
    print(json.dumps(dict(objective=candidate["objective"], lr=candidate["lr"],
                          best_val_loss=result["best_val_loss"], collapse_flags=result["collapse_flags"])))


def collect(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    candidates = []
    for candidate in manifest["candidates"]:
        directory = Path(candidate["run_dir"])
        result = json.loads((directory / "completed.json").read_text())
        if (result["config_sha256"] != candidate["config_sha256"]
                or result["history_sha256"] != sha256_file(directory / "history.jsonl")
                or result["checkpoint_sha256"] != sha256_file(directory / "encoder_100pct.pt")):
            raise ValueError("completed pilot artifacts changed")
        candidates.append({**candidate, **result})
    selected = {}
    for objective in OBJECTIVES:
        best = min((c for c in candidates if c["objective"] == objective), key=lambda c: c["best_val_loss"])
        selected[objective] = {key: best[key] for key in ("lr", "best_val_loss", "best_val_step", "run_dir", "collapse_flags")}
    result = dict(dataset=manifest["dataset"], protocol=manifest,
                  diagnostics_review_required=True, selected=selected, candidates=candidates)
    write_json(output / "selection.json", result)
    print(json.dumps(selected, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("--dataset", choices=tuple(SYSTEMS), required=True)
    setup.add_argument("--base", required=True)
    setup.add_argument("--output", required=True)
    setup.add_argument("--workers", type=int, help="freeze this loader worker count for all pilots")
    task = sub.add_parser("run")
    task.add_argument("--output", required=True)
    task.add_argument("--index", type=int, choices=range(12), required=True)
    task.add_argument("--cache-root", help="stage and reuse the full cache on this machine's local disk")
    summary = sub.add_parser("collect")
    summary.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "run": run, "collect": collect}[command](**args)


if __name__ == "__main__":
    main()

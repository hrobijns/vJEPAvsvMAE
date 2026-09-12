#!/usr/bin/env python3
"""Prepare and run independent checkpoint probes, then score selected encoders."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.artifacts import (
    Artifact,
    provenance,
    sha256_file,
    staged_directory,
    write_json,
)
from src.evaluation.cache import prepare_cache
from src.evaluation.features import extract_features
from src.evaluation.pipeline import fit_probes, score_probes
from src.evaluation.protocol import Protocol
from src.evaluation.reporting import aggregate, plot
from src.evaluation.selection import select_checkpoints
from src.objectives import OBJECTIVES


def prepare(handoff, base, output):
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "scripts/probe_sweep.py"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts", "configs"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    bundle = Artifact(handoff, "encoder_handoff")
    index = bundle.json("index.json")
    candidates, identical = [], {}
    for row in index["checkpoints"]:
        key = (
            row["dataset"],
            row["objective"],
            row["seed"],
            row["encoder_state_sha256"],
            row["config_sha256"],
        )
        if key in identical:
            candidates[identical[key]]["aliases"].append(row["path"])
            continue
        identical[key] = len(candidates)
        candidates.append(
            dict(
                dataset=row["dataset"],
                objective=row["objective"],
                seed=row["seed"],
                checkpoint=str((Path(handoff) / row["path"]).resolve()),
                checkpoint_sha256=row["sha256"],
                id=f"{row['dataset']}_{row['objective']}_seed{row['seed']}_{row['sha256'][:12]}",
                aliases=[],
            )
        )
    with staged_directory(output) as stage:
        (stage / "logs").mkdir()
        write_json(
            stage / "study.json",
            dict(
                provenance=provenance(),
                script_sha256=sha256_file(__file__),
                base=str(Path(base).resolve()),
                handoff_sha256=bundle.manifest["sha256"],
                protocols={
                    d: Protocol(d).to_dict()
                    for d in sorted({r["dataset"] for r in candidates})
                },
                candidates=candidates,
            ),
        )
    output = Path(output).resolve()
    subprocess.run(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(output / "source"),
            provenance()["git_commit"],
        ],
        cwd=repo,
        check=True,
        env=os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"},
    )
    (output / "source/.venv").symlink_to(repo / ".venv", target_is_directory=True)
    print(
        json.dumps(
            dict(study=str(output), candidate_jobs=len(candidates), max_concurrency=12)
        ),
        flush=True,
    )


def load(output):
    study = json.loads((output / "study.json").read_text())
    if study["provenance"] != provenance() or study["script_sha256"] != sha256_file(
        __file__
    ):
        raise ValueError("run the frozen study source; code changed after preparation")
    return study


def candidate_paths(output, row):
    cache = output / "cache" / row["dataset"]
    feature_root = output / "features" / row["dataset"]
    feature = (
        feature_root
        / f"{row['objective']}_seed{row['seed']}_{row['checkpoint_sha256'][:12]}"
    )
    return cache, feature_root, feature, output / "fits" / row["id"]


def run(output, study, task):
    row = study["candidates"][task]
    if sha256_file(row["checkpoint"]) != row["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after sweep preparation")
    cache, feature_root, feature, fit = candidate_paths(output, row)
    start = time.monotonic()
    for split in ("train", "valid"):
        if not (feature / split / "manifest.json").exists():
            extract_features(row["checkpoint"], cache, feature_root, split)
    if (fit / "manifest.json").exists():
        Artifact(fit, "probe_fits")
    else:
        fit_probes(feature, cache, fit)
    print(
        json.dumps(
            dict(candidate=row["id"], seconds=time.monotonic() - start, fit=str(fit))
        ),
        flush=True,
    )


def collect(output, study):
    paths = [candidate_paths(output, r)[3] for r in study["candidates"]]
    for row, path in zip(study["candidates"], paths):
        if (
            Artifact(path, "probe_fits").manifest["checkpoint"]["sha256"]
            != row["checkpoint_sha256"]
        ):
            raise ValueError("fit does not match prepared candidate roster")
    select_checkpoints(paths, output / "selection")
    print(output / "selection")


def test(output, study, task):
    choices = Artifact(output / "selection", "checkpoint_selection").json(
        "selections.json"
    )
    chosen = choices[task]
    row = next(
        r
        for r in study["candidates"]
        if r["checkpoint_sha256"] == chosen["checkpoint_sha256"]
    )
    cache, feature_root, feature, fit = candidate_paths(output, row)
    if not (feature / "test/manifest.json").exists():
        extract_features(row["checkpoint"], cache, feature_root, "test")
    score_probes(feature, cache, fit, output / "selection", output / "test" / row["id"])


def report(output, study):
    choices = Artifact(output / "selection", "checkpoint_selection").json(
        "selections.json"
    )
    for dataset in study["protocols"]:
        rows = [
            r
            for r in study["candidates"]
            if r["dataset"] == dataset
            and any(r["checkpoint_sha256"] == s["checkpoint_sha256"] for s in choices)
        ]
        folder = output / "reports" / dataset
        aggregate(
            [output / "test" / r["id"] for r in rows],
            folder / "aggregate",
            objectives=OBJECTIVES,
            seeds=sorted({r["seed"] for r in rows}),
        )
        plot(folder / "aggregate", folder / "plots")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("handoff", "base", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("cache")
    p.add_argument(
        "--dataset",
        required=True,
        choices=("rayleigh_benard", "active_matter", "shear_flow"),
    )
    p.add_argument("--split", required=True, choices=("train", "valid", "test"))
    for name in ("run", "test"):
        p = commands.add_parser(name)
        p.add_argument("--task", required=True, type=int)
    for name in ("collect", "report"):
        commands.add_parser(name)
    for name, p in commands.choices.items():
        if name != "prepare":
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.handoff, args.base, args.output)
        return
    output = Path(args.output).resolve()
    study = load(output)
    if args.command == "cache":
        if args.split == "test":
            Artifact(output / "selection", "checkpoint_selection")
        prepare_cache(
            study["base"],
            args.split,
            output / "cache" / args.dataset,
            Protocol.from_dict(study["protocols"][args.dataset]),
        )
    elif args.command in ("run", "test"):
        (run if args.command == "run" else test)(output, study, args.task)
    else:
        (collect if args.command == "collect" else report)(output, study)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare and run independent checkpoint probes, then score selected encoders."""

import argparse
from collections import defaultdict
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

# Stage 1 compares the common training-fraction milestones only. Objective-
# dependent candidates such as best_val are excluded by this policy, which is
# frozen into every study so a roster from another policy cannot be run.
CANDIDATE_POLICY = dict(
    version=1,
    roster="training_fraction_milestones",
    milestones=[
        dict(candidate=f"{percent:03d}pct", percent_of_total_steps=percent)
        for percent in (25, 50, 75, 100)
    ],
    excluded_candidates=["best_val"],
)
MILESTONES = {
    m["candidate"]: m["percent_of_total_steps"] for m in CANDIDATE_POLICY["milestones"]
}
MILESTONE_ORDER = sorted(MILESTONES, key=MILESTONES.__getitem__)
RUN_IDENTITY = ("config_sha256", "training_identity", "spec", "training_protocol")


def check_objective_roster(keys):
    """Every observed dataset/seed must contribute all compared objectives."""
    objectives = defaultdict(set)
    for dataset, objective, seed in keys:
        objectives[(dataset, seed)].add(objective)
    for key, present in sorted(objectives.items()):
        if present != set(OBJECTIVES):
            raise ValueError(
                f"incomplete objective roster for {key}: {sorted(present)}"
            )


def milestone_checkpoints(index):
    """Validate the declared handoff roster before aliasing can hide a defect."""
    found = defaultdict(dict)
    for row in index["checkpoints"]:
        # Declare the run group first: an excluded label must never make a
        # group, and therefore its missing milestones, disappear silently.
        group = found[(row["dataset"], row["objective"], row["seed"])]
        label = row["candidate"]
        if label in CANDIDATE_POLICY["excluded_candidates"]:
            continue
        if label not in MILESTONES:
            raise ValueError(f"unexpected candidate label {label!r}: {row['path']}")
        if label in group:
            raise ValueError(
                f"duplicate {label} candidate for "
                f"{(row['dataset'], row['objective'], row['seed'])}"
            )
        group[label] = row
    if not found:
        raise ValueError("handoff declares no checkpoints")
    check_objective_roster(found)
    rows = []
    for key, group in sorted(found.items()):
        if set(group) != set(MILESTONES):
            raise ValueError(f"incomplete milestone roster for {key}: {sorted(group)}")
        reference = group[MILESTONE_ORDER[0]]
        for label in MILESTONE_ORDER:
            row = group[label]
            for field in RUN_IDENTITY:
                if row[field] != reference[field]:
                    raise ValueError(f"candidate {field} differs within {key}")
            total = row["training_protocol"]["total_steps"]
            if not isinstance(total, int) or total <= 0:
                raise ValueError(f"missing training budget for {key}: {total}")
            if row["step"] * 100 != MILESTONES[label] * total:
                raise ValueError(
                    f"{label} candidate for {key} is at step {row['step']}, "
                    f"not {MILESTONES[label]}% of {total}"
                )
            rows.append(row)
    return rows


def roster_groups(candidates):
    """Every run group holds exactly one probe candidate per policy milestone."""
    groups = defaultdict(dict)
    for row in candidates:
        key = (row["dataset"], row["objective"], row["seed"])
        for entry in [row, *row["aliases"]]:
            label = entry["candidate"]
            if label not in MILESTONES:
                raise ValueError(f"candidate outside the policy roster: {label!r}")
            if label in groups[key]:
                raise ValueError(f"duplicate {label} candidate for {key}")
            if entry["step"] * 100 != MILESTONES[label] * row["total_steps"]:
                raise ValueError(
                    f"{label} candidate for {key} is at step {entry['step']}, "
                    f"not {MILESTONES[label]}% of {row['total_steps']}"
                )
            groups[key][label] = row
    if not groups:
        raise ValueError("study declares no candidates")
    check_objective_roster(groups)
    for key, group in groups.items():
        if set(group) != set(MILESTONES):
            raise ValueError(f"incomplete milestone roster for {key}: {sorted(group)}")
    return groups


def frozen_candidates(handoff, index, dataset=None):
    rows = milestone_checkpoints(index)
    if dataset is not None:
        available = {row["dataset"] for row in rows}
        if dataset not in available:
            raise ValueError(
                f"requested dataset {dataset!r} is absent from the handoff; "
                f"available datasets: {sorted(available)}"
            )
        rows = [row for row in rows if row["dataset"] == dataset]
    identical, candidates = {}, []
    for row in rows:
        key = (
            row["dataset"],
            row["objective"],
            row["seed"],
            row["encoder_state_sha256"],
            row["config_sha256"],
        )
        entry = dict(candidate=row["candidate"], step=row["step"], path=row["path"])
        if key in identical:
            candidates[identical[key]]["aliases"].append(entry)
            continue
        identical[key] = len(candidates)
        candidates.append(
            dict(
                dataset=row["dataset"],
                objective=row["objective"],
                seed=row["seed"],
                candidate=row["candidate"],
                step=row["step"],
                total_steps=row["training_protocol"]["total_steps"],
                checkpoint=str((Path(handoff) / row["path"]).resolve()),
                checkpoint_sha256=row["sha256"],
                config_sha256=row["config_sha256"],
                id=f"{row['dataset']}_{row['objective']}_seed{row['seed']}"
                f"_{row['candidate']}_{row['sha256'][:12]}",
                aliases=[],
            )
        )
    return candidates


def prepare(handoff, base, output, dataset=None):
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
    candidates = frozen_candidates(
        handoff, bundle.json("index.json"), dataset=dataset
    )
    groups = roster_groups(candidates)
    with staged_directory(output) as stage:
        (stage / "logs").mkdir()
        write_json(
            stage / "study.json",
            dict(
                provenance=provenance(),
                script_sha256=sha256_file(__file__),
                base=str(Path(base).resolve()),
                handoff_sha256=bundle.manifest["sha256"],
                candidate_policy=CANDIDATE_POLICY,
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
            dict(
                study=str(output),
                candidate_jobs=len(candidates),
                run_groups=len(groups),
                max_concurrency=12,
            )
        ),
        flush=True,
    )


def load(output):
    study = json.loads((output / "study.json").read_text())
    if study["provenance"] != provenance() or study["script_sha256"] != sha256_file(
        __file__
    ):
        raise ValueError("run the frozen study source; code changed after preparation")
    if study.get("candidate_policy") != CANDIDATE_POLICY:
        raise ValueError(
            "study candidate policy differs from this source's candidate policy"
        )
    roster_groups(study["candidates"])
    return study


def candidate_paths(output, row):
    cache = output / "cache" / row["dataset"]
    feature_root = output / "features" / row["dataset"]
    feature = (
        feature_root
        / f"{row['objective']}_seed{row['seed']}_{row['checkpoint_sha256'][:12]}"
    )
    return cache, feature_root, feature, output / "fits" / row["id"]


def candidate(study, task):
    rows = study["candidates"]
    if not 0 <= task < len(rows):
        raise ValueError(
            f"candidate task {task} outside the frozen roster of {len(rows)}"
        )
    return rows[task]


def run(output, study, task):
    row = candidate(study, task)
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
    groups = roster_groups(study["candidates"])
    paths = [candidate_paths(output, r)[3] for r in study["candidates"]]
    for row, path in zip(study["candidates"], paths):
        if not (path / "manifest.json").exists():
            raise ValueError(f"frozen candidate has no probe fit: {row['id']}")
        if (
            Artifact(path, "probe_fits").manifest["checkpoint"]["sha256"]
            != row["checkpoint_sha256"]
        ):
            raise ValueError("fit does not match prepared candidate roster")
    select_checkpoints(paths, output / "selection")
    winners = Artifact(output / "selection", "checkpoint_selection").json(
        "selections.json"
    )
    frozen = {r["checkpoint_sha256"] for r in study["candidates"]}
    chosen = [(w["dataset"], w["objective"], w["seed"]) for w in winners]
    if sorted(chosen) != sorted(groups) or not frozen.issuperset(
        w["checkpoint_sha256"] for w in winners
    ):
        raise ValueError(
            "selection must yield exactly one frozen candidate per run group"
        )
    print(output / "selection")


def selected_row(study, chosen):
    row = next(
        (
            r
            for r in study["candidates"]
            if r["checkpoint_sha256"] == chosen["checkpoint_sha256"]
        ),
        None,
    )
    if row is None:
        raise ValueError("selected checkpoint is outside the frozen candidate roster")
    return row


def test(output, study, task):
    choices = Artifact(output / "selection", "checkpoint_selection").json(
        "selections.json"
    )
    if not 0 <= task < len(choices):
        raise ValueError(f"test task {task} outside the {len(choices)} selected runs")
    row = selected_row(study, choices[task])
    cache, feature_root, feature, fit = candidate_paths(output, row)
    if not (feature / "test/manifest.json").exists():
        extract_features(row["checkpoint"], cache, feature_root, "test")
    score_probes(feature, cache, fit, output / "selection", output / "test" / row["id"])


def report(output, study):
    choices = Artifact(output / "selection", "checkpoint_selection").json(
        "selections.json"
    )
    selected = [selected_row(study, choice) for choice in choices]
    for dataset in study["protocols"]:
        rows = [r for r in selected if r["dataset"] == dataset]
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
    p.add_argument(
        "--dataset",
        choices=("rayleigh_benard", "active_matter", "shear_flow"),
        help="freeze only one system; omit to retain the complete handoff",
    )
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
        prepare(args.handoff, args.base, args.output, dataset=args.dataset)
        return
    output = Path(args.output).resolve()
    study = load(output)
    if args.command == "cache":
        if args.split == "test":
            Artifact(output / "selection", "checkpoint_selection")
        if args.dataset not in study["protocols"]:
            raise ValueError(
                f"dataset {args.dataset!r} is outside this frozen study; "
                f"choose one of {sorted(study['protocols'])}"
            )
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

#!/usr/bin/env python3
"""Probe each run's frozen best-validation encoder, then score held-out data."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

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
from src.objectives import OBJECTIVES
from src.physics.systems import SYSTEMS

# The analysis probes one encoder per training run: the state at minimum
# pretraining-validation loss. No downstream checkpoint comparison takes
# place. Validation data still selects probe hyperparameters, probe stopping
# state, and the encoder output inside every fit.
ENCODER_POLICY = dict(
    version=3,
    roster="minimum_pretraining_validation_loss",
    candidate="best_val",
    downstream_checkpoint_selection=False,
)
CANDIDATE = ENCODER_POLICY["candidate"]
RUN_IDENTITY = ("config_sha256", "training_identity", "spec", "training_protocol")
DATASETS = ("rayleigh_benard", "active_matter", "shear_flow")
SPLITS = ("train", "valid", "test")
# Probe fitting budget, frozen with the study so every encoder is fitted and
# validation-selected under identical settings.
PROBE_SETTINGS = dict(
    mlp_max_steps=2000,
    mlp_min_steps=150,
    attentive_epochs=100,
    attentive_min_epochs=15,
    attentive_patience=10,
    attentive_batch_size=32,
)

def selected_physical_targets(dataset, requested=None):
    """Validate a study's ordered physical-target subset."""
    available = list(SYSTEMS[dataset].targets)
    targets = list(available if requested is None else requested)
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(target not in available for target in targets)
    ):
        raise ValueError(
            f"physical targets for {dataset} must be a non-empty unique subset "
            f"of {available}; got {targets}"
        )
    return targets


def check_objective_roster(keys, expected=OBJECTIVES):
    """Every observed dataset/seed must contribute the compared objectives."""
    expected = set(expected)
    objectives = defaultdict(set)
    for dataset, objective, seed in keys:
        objectives[(dataset, seed)].add(objective)
    for key, present in sorted(objectives.items()):
        if present != expected:
            raise ValueError(
                f"incomplete objective roster for {key}: {sorted(present)}"
            )


def best_val_checkpoints(index):
    """Retain each run's best-validation encoder and verify its provenance."""
    runs = defaultdict(list)
    for row in index["checkpoints"]:
        runs[(row["dataset"], row["objective"], row["seed"])].append(row)
    if not runs:
        raise ValueError("handoff declares no checkpoints")
    check_objective_roster(runs)
    rows = []
    for key, group in sorted(runs.items()):
        chosen = [row for row in group if row["candidate"] == CANDIDATE]
        if len(chosen) != 1:
            raise ValueError(
                f"{key} declares {len(chosen)} {CANDIDATE} encoders; "
                "exactly one is required"
            )
        row = chosen[0]
        # Other retained labels are never probed, but a label from a different
        # training run would mean the handoff index itself is unreliable.
        for other in group:
            for field in RUN_IDENTITY:
                if other[field] != row[field]:
                    raise ValueError(f"handoff {field} differs within {key}")
        total = row["training_protocol"]["total_steps"]
        if not isinstance(total, int) or total <= 0:
            raise ValueError(f"missing training budget for {key}: {total}")
        if not 0 < row["step"] <= total:
            raise ValueError(
                f"{CANDIDATE} encoder for {key} is at step {row['step']}, "
                f"outside the training budget {total}"
            )
        rows.append(row)
    return rows


def roster_groups(encoders, objectives=OBJECTIVES):
    """The frozen roster holds exactly one best-validation encoder per run."""
    groups = {}
    for row in encoders:
        if row["candidate"] != CANDIDATE:
            raise ValueError(
                f"encoder outside the frozen roster: {row['candidate']!r}"
            )
        key = (row["dataset"], row["objective"], row["seed"])
        if key in groups:
            raise ValueError(f"duplicate {CANDIDATE} encoder for {key}")
        if row["total_steps"] != row["training_protocol"]["total_steps"]:
            raise ValueError(f"inconsistent training budget for {key}")
        if not 0 < row["step"] <= row["total_steps"]:
            raise ValueError(
                f"{CANDIDATE} encoder for {key} is at step {row['step']}, "
                f"outside the training budget {row['total_steps']}"
            )
        groups[key] = row
    if not groups:
        raise ValueError("study declares no encoders")
    if len({row["checkpoint_sha256"] for row in encoders}) != len(encoders):
        raise ValueError("duplicate encoder checkpoint in the frozen roster")
    check_objective_roster(groups, objectives)
    return groups


def frozen_encoders(handoff, index, dataset=None, objectives=None):
    rows = best_val_checkpoints(index)
    if dataset is not None:
        available = {row["dataset"] for row in rows}
        if dataset not in available:
            raise ValueError(
                f"requested dataset {dataset!r} is absent from the handoff; "
                f"available datasets: {sorted(available)}"
            )
        rows = [row for row in rows if row["dataset"] == dataset]
    if objectives is not None:
        requested = list(objectives)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(objective not in OBJECTIVES for objective in requested)
        ):
            raise ValueError(f"objectives must be a non-empty subset of {OBJECTIVES}")
        rows = [row for row in rows if row["objective"] in requested]
    return [
        dict(
            dataset=row["dataset"],
            objective=row["objective"],
            seed=row["seed"],
            candidate=row["candidate"],
            step=row["step"],
            total_steps=row["training_protocol"]["total_steps"],
            checkpoint=str((Path(handoff) / row["path"]).resolve()),
            checkpoint_sha256=row["sha256"],
            bytes=row["bytes"],
            encoder_state_sha256=row["encoder_state_sha256"],
            config_sha256=row["config_sha256"],
            training_identity=row["training_identity"],
            training_protocol=row["training_protocol"],
            spec=row["spec"],
            data_exposure=row["data_exposure"],
            id=f"{row['dataset']}_{row['objective']}_seed{row['seed']}"
            f"_{CANDIDATE}_{row['sha256'][:12]}",
        )
        for row in rows
    ]


def frozen_protocol(repo, dataset):
    """Freeze the committed evaluation configuration, never library defaults."""
    config = yaml.safe_load(
        (Path(repo) / f"configs/eval_{dataset}.yaml").read_text()
    )
    if (
        not isinstance(config, dict)
        or config.get("dataset") != dataset
        or "frame_limit" not in config
        or set(config) - set(Protocol.__dataclass_fields__)
    ):
        raise ValueError(
            f"configs/eval_{dataset}.yaml must declare dataset {dataset!r}, an "
            "explicit frame_limit, and only supported protocol fields"
        )
    return Protocol.from_dict(config)


def check_input_geometry(protocol, encoders):
    """A frozen encoder only accepts the temporal support it was trained on."""
    for row in encoders:
        if row["spec"]["n_frames"] != protocol.n_frames:
            raise ValueError(
                f"{row['id']} takes {row['spec']['n_frames']}-frame clips but "
                f"configs/eval_{protocol.dataset}.yaml asks for {protocol.n_frames}"
            )


def verify_payloads(encoders):
    """Frozen payloads must be present and unmodified before any other work."""
    for row in encoders:
        path = Path(row["checkpoint"])
        if not path.is_file():
            raise ValueError(f"frozen encoder is missing: {path}")
        size = path.stat().st_size
        if size != row["bytes"]:
            raise ValueError(
                f"{path} holds {size} bytes, not the declared {row['bytes']} "
                "bytes; fetch the Git LFS payload before preparing a study"
            )
        if sha256_file(path) != row["checkpoint_sha256"]:
            raise ValueError(f"frozen encoder content differs: {path}")


def check_clean_source(repo):
    """Freeze committed source only: no modified, staged, or untracked files."""
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "scripts/probe_sweep.py"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "scripts",
            "configs",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(
            "commit or remove uncommitted source before preparing a study:\n"
            + dirty
        )


def prepare(
    handoff, base, output, dataset=None, targets=None, objectives=None
):
    repo = Path(__file__).resolve().parents[1]
    check_clean_source(repo)
    bundle = Artifact(handoff, "encoder_handoff")
    objectives = list(OBJECTIVES if objectives is None else objectives)
    encoders = frozen_encoders(
        handoff,
        bundle.json("index.json"),
        dataset=dataset,
        objectives=objectives,
    )
    groups = roster_groups(encoders, objectives)
    protocols = {}
    for name in sorted({r["dataset"] for r in encoders}):
        protocol = frozen_protocol(repo, name)
        check_input_geometry(protocol, [r for r in encoders if r["dataset"] == name])
        protocols[name] = protocol.to_dict()
    if targets is not None and dataset is None:
        raise ValueError("--targets requires a single --dataset")
    physical_targets = {
        name: selected_physical_targets(
            name, targets if name == dataset else None
        )
        for name in protocols
    }
    verify_payloads(encoders)
    with staged_directory(output) as stage:
        (stage / "logs").mkdir()
        write_json(
            stage / "study.json",
            dict(
                provenance=provenance(),
                script_sha256=sha256_file(__file__),
                base=str(Path(base).resolve()),
                handoff_sha256=bundle.manifest["sha256"],
                encoder_policy=ENCODER_POLICY,
                probe_settings=PROBE_SETTINGS,
                protocols=protocols,
                physical_targets=physical_targets,
                objectives=objectives,
                encoders=encoders,
            ),
        )
    output = Path(output).resolve()
    # Studies are disposable; stale worktree registrations must never block one.
    subprocess.run(["git", "worktree", "prune"], cwd=repo, check=True)
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
    print(
        json.dumps(
            dict(
                study=str(output),
                encoder_jobs=len(encoders),
                run_groups=len(groups),
                datasets=sorted({r["dataset"] for r in encoders}),
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
    if study.get("encoder_policy") != ENCODER_POLICY:
        raise ValueError(
            "study encoder policy differs from this source's encoder policy"
        )
    if study.get("probe_settings") != PROBE_SETTINGS:
        raise ValueError(
            "study probe settings differ from this source's probe settings"
        )
    if set(study.get("physical_targets", {})) != set(study["protocols"]):
        raise ValueError("study physical targets do not match its datasets")
    for dataset, targets in study["physical_targets"].items():
        selected_physical_targets(dataset, targets)
    objectives = study.get("objectives")
    if (
        not isinstance(objectives, list)
        or not objectives
        or len(set(objectives)) != len(objectives)
        or any(objective not in OBJECTIVES for objective in objectives)
    ):
        raise ValueError("study objectives are invalid")
    roster_groups(study["encoders"], objectives)
    return study


def encoder_paths(output, row):
    cache = output / "cache" / row["dataset"]
    feature_root = output / "features" / row["dataset"]
    feature = (
        feature_root
        / f"{row['objective']}_seed{row['seed']}_{row['checkpoint_sha256'][:12]}"
    )
    return cache, feature_root, feature, output / "fits" / row["id"]


def encoder(study, task):
    rows = study["encoders"]
    if not 0 <= task < len(rows):
        raise ValueError(f"task {task} outside the frozen roster of {len(rows)}")
    return rows[task]


def verified_checkpoint(row):
    if sha256_file(row["checkpoint"]) != row["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after sweep preparation")
    return row["checkpoint"]


def sealed(path, kind):
    """A finished stage is a verified artifact, never merely a directory."""
    path = Path(path)
    if not (path / "manifest.json").exists():
        return False
    Artifact(path, kind)
    return True


def fitted(output, row):
    fit = encoder_paths(output, row)[3]
    if not sealed(fit, "probe_fits"):
        return False
    if (
        Artifact(fit, "probe_fits").manifest["checkpoint"]["sha256"]
        != row["checkpoint_sha256"]
    ):
        raise ValueError(f"probe fit does not match the frozen encoder: {row['id']}")
    return True


def completed_fits(output, study):
    """Test data stays sealed until every frozen encoder has a probe fit."""
    for row in study["encoders"]:
        if not fitted(output, row):
            raise ValueError(f"frozen encoder has no probe fit: {row['id']}")
    return [encoder_paths(output, row)[3] for row in study["encoders"]]


def cache(output, study, dataset, split):
    if dataset not in study["protocols"]:
        raise ValueError(
            f"dataset {dataset!r} is outside this frozen study; "
            f"choose one of {sorted(study['protocols'])}"
        )
    if split == "test":
        completed_fits(output, study)
    destination = output / "cache" / dataset / split
    if not sealed(destination, "cache"):
        prepare_cache(
            study["base"],
            split,
            output / "cache" / dataset,
            Protocol.from_dict(study["protocols"][dataset]),
        )
    print(destination, flush=True)


def extracted_features(checkpoint, cache_root, feature_root, feature, splits):
    for split in splits:
        if sealed(feature / split, "features"):
            continue
        extract_features(checkpoint, cache_root, feature_root, split)
        if not sealed(feature / split, "features"):
            raise ValueError(f"extraction sealed no {split} features: {feature}")


def run(output, study, task):
    row = encoder(study, task)
    cache_root, feature_root, feature, fit = encoder_paths(output, row)
    start = time.monotonic()
    if not fitted(output, row):
        checkpoint = verified_checkpoint(row)
        extracted_features(
            checkpoint, cache_root, feature_root, feature, ("train", "valid")
        )
        fit_probes(
            feature,
            cache_root,
            fit,
            physical_targets=study["physical_targets"][row["dataset"]],
            **study["probe_settings"],
        )
        if not fitted(output, row):
            raise ValueError(f"fitting sealed no usable probe fit: {fit}")
    print(
        json.dumps(
            dict(encoder=row["id"], seconds=time.monotonic() - start, fit=str(fit))
        ),
        flush=True,
    )


def test(output, study, task):
    completed_fits(output, study)
    row = encoder(study, task)
    scores = output / "test" / row["id"]
    if not sealed(scores, "probes"):
        checkpoint = verified_checkpoint(row)
        cache_root, feature_root, feature, fit = encoder_paths(output, row)
        extracted_features(checkpoint, cache_root, feature_root, feature, SPLITS)
        score_probes(feature, cache_root, fit, scores)
        if not sealed(scores, "probes"):
            raise ValueError(f"scoring sealed no test artifact: {scores}")
    print(scores, flush=True)


def report(output, study):
    for dataset in sorted(study["protocols"]):
        rows = [r for r in study["encoders"] if r["dataset"] == dataset]
        scores = [output / "test" / r["id"] for r in rows]
        missing = sorted(
            r["id"] for r, p in zip(rows, scores) if not sealed(p, "probes")
        )
        if missing:
            raise ValueError(f"frozen encoders without test scores: {missing}")
        folder = output / "reports" / dataset
        if not sealed(folder / "aggregate", "aggregate"):
            aggregate(
                scores,
                folder / "aggregate",
                objectives=study["objectives"],
                seeds=sorted({r["seed"] for r in rows}),
            )
        if not sealed(folder / "plots", "plots"):
            plot(folder / "aggregate", folder / "plots")
        print(folder / "plots", flush=True)


def status(output, study):
    """Report finished work so an interrupted study resumes without rework."""
    rows = []
    for task, row in enumerate(study["encoders"]):
        feature = encoder_paths(output, row)[2]
        rows.append(
            dict(
                task=task,
                id=row["id"],
                dataset=row["dataset"],
                objective=row["objective"],
                seed=row["seed"],
                step=row["step"],
                feature_dir=str(feature),
                features={
                    split: sealed(feature / split, "features") for split in SPLITS
                },
                fit=fitted(output, row),
                test=sealed(output / "test" / row["id"], "probes"),
            )
        )
    datasets = sorted(study["protocols"])
    print(
        json.dumps(
            dict(
                study=str(output),
                caches={
                    dataset: {
                        split: sealed(output / "cache" / dataset / split, "cache")
                        for split in SPLITS
                    }
                    for dataset in datasets
                },
                pending_fits=[r["task"] for r in rows if not r["fit"]],
                pending_tests=[r["task"] for r in rows if not r["test"]],
                reports={
                    dataset: sealed(
                        output / "reports" / dataset / "plots", "plots"
                    )
                    for dataset in datasets
                },
                encoders=rows,
            ),
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("handoff", "base", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument(
        "--dataset",
        choices=DATASETS,
        help="freeze only one system; omit to retain the complete handoff",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        help="ordered physical-target subset; requires --dataset",
    )
    p.add_argument(
        "--objectives",
        nargs="+",
        choices=OBJECTIVES,
        help="objective subset; defaults to all compared objectives",
    )
    p = commands.add_parser("cache")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--split", required=True, choices=SPLITS)
    for name in ("run", "test"):
        p = commands.add_parser(name)
        p.add_argument("--task", required=True, type=int)
    for name in ("report", "status"):
        commands.add_parser(name)
    for name, p in commands.choices.items():
        if name != "prepare":
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.handoff,
            args.base,
            args.output,
            objectives=args.objectives,
            dataset=args.dataset,
            targets=args.targets,
        )
        return
    output = Path(args.output).resolve()
    study = load(output)
    if args.command == "cache":
        cache(output, study, args.dataset, args.split)
    elif args.command in ("run", "test"):
        (run if args.command == "run" else test)(output, study, args.task)
    else:
        (report if args.command == "report" else status)(output, study)


if __name__ == "__main__":
    main()

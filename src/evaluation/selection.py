"""One validation-selected encoder checkpoint per dataset, objective, and seed."""

from collections import defaultdict
from pathlib import Path
import numpy as np

from src.evaluation.artifacts import Artifact, seal, staged_directory, write_json
from src.evaluation.protocol import Protocol
from src.physics.systems import SYSTEMS

SELECTION_POLICY = dict(
    version=1,
    metric="valid_vrmse",
    direction="minimize",
    present_weight=0.5,
    future_weight=0.5,
    within_group="equal_quantities_equal_global_local",
    future_horizons="equal",
    checkpoint_tie="earlier_step_then_sha256",
)


def balanced_score(rows, protocol):
    """Never silently average a smaller task roster when a metric is undefined."""
    expected = {
        (rep, offset, target)
        for rep in ("pooled", "token")
        for offset in protocol.target_offsets
        for target in SYSTEMS[protocol.dataset].targets
    }
    chosen = [r for r in rows if r["family"] == "physics" and r["method"] == "selected"]
    keys = [(r["representation"], r["target_offset"], r["target"]) for r in chosen]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            "checkpoint selection requires the complete physical task roster"
        )
    future = [o for o in protocol.target_offsets if o > 0]
    if not future:
        raise ValueError("balanced checkpoint selection requires a future horizon")
    groups = {}
    for offset in protocol.target_offsets:
        values = [r.get("valid_vrmse") for r in chosen if r["target_offset"] == offset]
        groups[str(offset)] = (
            float(np.mean(values))
            if all(v is not None and np.isfinite(v) for v in values)
            else None
        )
    score = (
        None
        if any(v is None for v in groups.values())
        else 0.5 * groups["0"] + 0.5 * float(np.mean([groups[str(o)] for o in future]))
    )
    return score, groups


def select_checkpoints(paths, output):
    artifacts = [Artifact(path, "probe_fits") for path in paths]
    if not artifacts:
        raise ValueError("no checkpoint candidates")
    if len({a.manifest["checkpoint"]["sha256"] for a in artifacts}) != len(artifacts):
        raise ValueError("duplicate checkpoint candidate")
    groups, systems = defaultdict(list), {}
    rows = []
    for artifact in artifacts:
        manifest = artifact.manifest
        checkpoint = manifest["checkpoint"]
        dataset = checkpoint["dataset"]
        first = systems.setdefault(dataset, manifest)
        for key in ("protocol", "caches", "probe_settings", "provenance"):
            if manifest[key] != first[key]:
                raise ValueError(f"incompatible candidate {key}")
        for key in ("spec", "encoder", "training_protocol"):
            if checkpoint[key] != first["checkpoint"][key]:
                raise ValueError(f"incompatible candidate checkpoint {key}")
        budget = checkpoint["training_protocol"]["total_steps"]
        if budget is None or not 0 < checkpoint["step"] <= budget:
            raise ValueError(
                "checkpoint step is outside its configured training budget"
            )
        key = (dataset, checkpoint["objective"], checkpoint["seed"])
        if (
            groups[key]
            and checkpoint["config_sha256"] != groups[key][0]["config_sha256"]
        ):
            raise ValueError("candidate configurations differ within a training run")
        score, by_horizon = balanced_score(
            artifact.json("rows.json"), Protocol.from_dict(manifest["protocol"])
        )
        row = dict(
            dataset=dataset,
            objective=checkpoint["objective"],
            seed=checkpoint["seed"],
            step=checkpoint["step"],
            checkpoint_sha256=checkpoint["sha256"],
            config_sha256=checkpoint["config_sha256"],
            fit_sha256=manifest["sha256"],
            fit_path=str(artifact.root.resolve()),
            valid_vrmse=score,
            by_horizon=by_horizon,
            status="ok" if score is not None else "undefined_validation_vrmse",
        )
        groups[key].append(row)
        rows.append(row)
    winners = []
    for key, candidates in sorted(groups.items()):
        finite = [r for r in candidates if r["valid_vrmse"] is not None]
        if not finite:
            raise ValueError(f"no complete finite checkpoint score for {key}")
        winners.append(
            min(
                finite,
                key=lambda r: (r["valid_vrmse"], r["step"], r["checkpoint_sha256"]),
            )
        )
    with staged_directory(output) as stage:
        write_json(stage / "rows.json", rows)
        write_json(stage / "selections.json", winners)
        seal(
            stage,
            "checkpoint_selection",
            policy=SELECTION_POLICY,
            protocols={k: v["protocol"] for k, v in systems.items()},
            candidates=[a.manifest["sha256"] for a in artifacts],
        )
    return Path(output)

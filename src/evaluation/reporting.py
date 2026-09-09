"""Seed summaries and plots derived from the same result rows."""

from collections import defaultdict
from pathlib import Path
import csv

import numpy as np

from src.evaluation.artifacts import Artifact, seal, staged_directory, write_json
from src.objectives import OBJECTIVES

METRICS = ("test_r2", "test_pearson_r", "test_mse", "test_log_mse")


def _summary(values, metric):
    numbers = [v.get(metric) for v in values]
    finite = [float(v) for v in numbers if v is not None and np.isfinite(v)]
    # Do not silently change the number of scientific units for a metric.
    if len(finite) != len(numbers):
        return dict(mean=None, std=None, n=len(numbers), status="undefined_metric")
    return dict(
        mean=float(np.mean(finite)),
        std=float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
        n=len(finite),
        status="ok",
    )


def aggregate(
    paths, output, objectives=OBJECTIVES, seeds=(1, 2, 3), kind="probes"
):
    if (
        not objectives
        or not seeds
        or len(set(objectives)) != len(objectives)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("expected objectives/seeds must be nonempty and distinct")
    artifacts = [Artifact(path, kind) for path in paths]
    expected = {(objective, int(seed)) for objective in objectives for seed in seeds}
    observed = [
        (a.manifest["checkpoint"]["objective"], a.manifest["checkpoint"]["seed"])
        for a in artifacts
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(
            f"incomplete or duplicate run roster: expected {sorted(expected)}, got {observed}"
        )
    first = artifacts[0].manifest
    for artifact in artifacts[1:]:
        for key in ("protocol", "caches", "probe_settings"):
            if artifact.manifest[key] != first[key]:
                raise ValueError(f"incompatible aggregate {key}")
        for key in ("spec", "encoder", "dataset", "training_protocol", "step"):
            if artifact.manifest["checkpoint"].get(key) != first["checkpoint"].get(key):
                raise ValueError(f"incompatible checkpoint {key}")
        if (
            artifact.manifest["provenance"]["code_sha256"]
            != first["provenance"]["code_sha256"]
        ):
            raise ValueError("analysis code differs across results")
    rows = []
    grids = []
    for artifact in artifacts:
        checkpoint = artifact.manifest["checkpoint"]
        items = artifact.json("rows.json")
        keys = [item["cell_id"] for item in items]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("duplicate or empty result cells")
        grids.append(set(keys))
        rows.extend(
            {
                **item,
                "objective": checkpoint["objective"],
                "checkpoint_seed": checkpoint["seed"],
            }
            for item in items
        )
    if any(grid != grids[0] for grid in grids):
        raise ValueError("result cells differ across runs")
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            ("shared" if row.get("shared") else row["objective"], row["cell_id"])
        ].append(row)
    summaries = []
    for (objective, identity), values in sorted(grouped.items()):
        summary = {
            k: v
            for k, v in values[0].items()
            if k
            in (
                "family",
                "representation",
                "method",
                "gap",
                "target",
                "sigma",
                "shared",
            )
        }
        summary.update(objective=objective, cell_id=identity)
        if objective == "shared":
            for metric in METRICS:
                numbers = [v.get(metric) for v in values]
                if any(v is None for v in numbers):
                    if not all(v is None for v in numbers):
                        raise ValueError("shared control metrics disagree")
                elif not np.allclose(
                    numbers, numbers[0], equal_nan=True, atol=1e-10, rtol=1e-8
                ):
                    raise ValueError("shared control metrics disagree")
            values = values[:1]
        summary["selections"] = [
            {
                k: v
                for k, v in value.items()
                if k
                in (
                    "checkpoint_seed",
                    "objective",
                    "selected_layer",
                    "selected_method",
                    "selected_alpha",
                    "selected_steps",
                    "valid_cv_r2",
                    "status",
                    "metric_status",
                )
            }
            for value in values
        ]
        summary["metrics"] = {metric: _summary(values, metric) for metric in METRICS}
        if any("depth_curve" in v for v in values):
            layers = [
                {point["layer"] for point in v.get("depth_curve", [])} for v in values
            ]
            if any(layer != layers[0] for layer in layers):
                raise ValueError("depth curves differ across runs")
            summary["depth_curve"] = [
                dict(
                    layer=layer,
                    metrics={
                        metric: _summary(
                            [
                                next(
                                    point
                                    for point in v["depth_curve"]
                                    if point["layer"] == layer
                                )
                                for v in values
                            ],
                            metric,
                        )
                        for metric in METRICS
                    },
                )
                for layer in sorted(layers[0])
            ]
        summaries.append(summary)
    # Average targets inside each checkpoint before calculating between-seed SD.
    # A target is never treated as another independent training run.
    target_means = []
    averages = defaultdict(list)
    for row in rows:
        if row["family"] not in ("physics", "noise") or row.get("shared"):
            continue
        key = (
            row["objective"],
            row["checkpoint_seed"],
            row["family"],
            row["method"],
            row["representation"],
            row["gap"],
            row.get("sigma"),
        )
        averages[key].append(row)
    by_group = defaultdict(list)
    for key, values in averages.items():
        checkpoint_mean = {
            metric: _summary(values, metric)["mean"] for metric in METRICS
        }
        by_group[(key[0], *key[2:])].append(checkpoint_mean)
    for key, values in sorted(by_group.items(), key=str):
        target_means.append(
            dict(
                zip(
                    ("objective", "family", "method", "representation", "gap", "sigma"),
                    key,
                )
            )
            | {"metrics": {metric: _summary(values, metric) for metric in METRICS}}
        )
    with staged_directory(output) as stage:
        write_json(stage / "rows.json", rows)
        write_json(stage / "summary.json", summaries)
        write_json(stage / "target_means.json", target_means)
        seal(
            stage,
            "aggregate",
            result_kind=kind,
            protocol=first["protocol"],
            caches=first["caches"],
            probe_settings=first["probe_settings"],
            roster=[list(key) for key in sorted(expected)],
            source_results=[a.manifest["sha256"] for a in artifacts],
            scientific_unit="checkpoint_seed",
        )
    return Path(output)


def plot(aggregate_dir, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifact = Artifact(aggregate_dir, "aggregate")
    summaries = artifact.json("summary.json")
    raw = artifact.json("rows.json")
    target_means = artifact.json("target_means.json")
    objectives = sorted(
        {r["objective"] for r in summaries if r["objective"] != "shared"}
    )
    colors = {
        objective: plt.get_cmap("tab10")(i) for i, objective in enumerate(objectives)
    }
    targets = artifact.manifest["protocol"]["targets"]
    gaps = artifact.manifest["protocol"]["gaps"]
    with staged_directory(output) as stage:
        fields = (
            "objective",
            "family",
            "representation",
            "method",
            "gap",
            "target",
            "sigma",
            "metric",
            "mean",
            "std",
            "n",
            "status",
        )
        with (stage / "summary.tsv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in summaries:
                for metric, score in row["metrics"].items():
                    writer.writerow(
                        {key: row.get(key) for key in fields[:7]}
                        | {"metric": metric}
                        | score
                    )
        write_json(stage / "target_means.json", target_means)
        # Main comparison and separate readouts use identical cells and scales.
        if artifact.manifest["result_kind"] == "probes":
            for method in ("selected", "ridge", "mlp"):
                fig, axes = plt.subplots(
                    1,
                    len(objectives),
                    figsize=(6 * len(objectives), max(3, len(targets) * 0.65)),
                    squeeze=False,
                )
                for axis, objective in zip(axes[0], objectives):
                    cells = [
                        r
                        for r in summaries
                        if r["family"] == "physics"
                        and r["method"] == method
                        and r["objective"] == objective
                    ]
                    matrix = np.full((len(targets), 2 * len(gaps)), np.nan)
                    for r in cells:
                        col = (
                            0 if r["representation"] == "pooled" else len(gaps)
                        ) + gaps.index(r["gap"])
                        score = r["metrics"]["test_r2"]["mean"]
                        if score is not None:
                            matrix[targets.index(r["target"]), col] = score
                    axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            axis.text(
                                j,
                                i,
                                f"{matrix[i, j]:.3f}"
                                if np.isfinite(matrix[i, j])
                                else "undefined",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="white"
                                if not np.isfinite(matrix[i, j]) or matrix[i, j] < 0.5
                                else "black",
                            )
                    axis.set_xticks(
                        range(2 * len(gaps)),
                        [
                            f"{rep} +{gap}"
                            for rep in ("pooled", "token")
                            for gap in gaps
                        ],
                        rotation=40,
                        ha="right",
                    )
                    axis.set_yticks(
                        range(len(targets)), [t.replace("_", " ") for t in targets]
                    )
                    axis.set_title(f"{objective}: {method}, test R²")
                fig.tight_layout()
                fig.savefig(stage / f"physics_{method}.pdf")
                plt.close(fig)
            # The paper's depth analysis is pooled, with one panel per target/horizon.
            for method in ("ridge", "mlp"):
                fig, axes = plt.subplots(
                    len(targets),
                    len(gaps),
                    figsize=(4 * len(gaps), 2.5 * len(targets)),
                    squeeze=False,
                )
                for i, target in enumerate(targets):
                    for j, gap in enumerate(gaps):
                        axis = axes[i, j]
                        for row in raw:
                            if (
                                row["family"],
                                row["representation"],
                                row["method"],
                                row["target"],
                                row["gap"],
                            ) != ("physics", "pooled", method, target, gap):
                                continue
                            curve = row.get("depth_curve", [])
                            axis.plot(
                                [p["layer"] for p in curve],
                                [p.get("test_r2") for p in curve],
                                color=colors[row["objective"]],
                                alpha=0.2,
                                lw=0.8,
                            )
                        for row in summaries:
                            if (
                                row["family"],
                                row["representation"],
                                row["method"],
                                row["target"],
                                row["gap"],
                            ) != ("physics", "pooled", method, target, gap):
                                continue
                            curve = row.get("depth_curve", [])
                            axis.plot(
                                [p["layer"] for p in curve],
                                [p["metrics"]["test_r2"]["mean"] for p in curve],
                                label=row["objective"],
                                color=colors[row["objective"]],
                            )
                        axis.set_title(f"{target.replace('_', ' ')} +{gap}", fontsize=9)
                        axis.set_xlabel("Encoder output")
                        axis.set_ylabel("Test R²")
                axes[0, 0].legend()
                fig.tight_layout()
                fig.savefig(stage / f"depth_{method}.pdf")
                plt.close(fig)
        else:
            for metric in ("test_r2", "test_pearson_r"):
                fig, axes = plt.subplots(
                    len(targets), 3, figsize=(12, 2.5 * len(targets)), squeeze=False
                )
                for i, target in enumerate(targets):
                    for j, method in enumerate(("selected", "ridge", "mlp")):
                        axis = axes[i, j]
                        for objective in objectives:
                            points = sorted(
                                [
                                    r
                                    for r in summaries
                                    if r["target"] == target
                                    and r["method"] == method
                                    and r["objective"] == objective
                                ],
                                key=lambda r: r["sigma"],
                            )
                            for seed in sorted(
                                {
                                    r["checkpoint_seed"]
                                    for r in raw
                                    if r["objective"] == objective
                                }
                            ):
                                values = sorted(
                                    [
                                        r
                                        for r in raw
                                        if r["target"] == target
                                        and r["method"] == method
                                        and r["objective"] == objective
                                        and r["checkpoint_seed"] == seed
                                    ],
                                    key=lambda r: r["sigma"],
                                )
                                axis.plot(
                                    [r["sigma"] for r in values],
                                    [r.get(metric) for r in values],
                                    color=colors[objective],
                                    alpha=0.2,
                                    lw=0.8,
                                )
                            axis.plot(
                                [r["sigma"] for r in points],
                                [r["metrics"][metric]["mean"] for r in points],
                                color=colors[objective],
                                label=objective,
                                marker="o",
                            )
                        if metric == "test_r2":
                            axis.set_yscale("symlog", linthresh=1.0)
                        axis.set_title(
                            f"{target.replace('_', ' ')}: {method}", fontsize=9
                        )
                        axis.set_xlabel("Noise sigma")
                        axis.set_ylabel(metric.removeprefix("test_"))
                axes[0, 0].legend()
                fig.tight_layout()
                fig.savefig(stage / f"noise_{metric}.pdf")
                plt.close(fig)
        seal(stage, "plots", aggregate=artifact.manifest["sha256"])
    return Path(output)

"""Seed summaries and plots derived from the same result rows."""

from collections import defaultdict
from pathlib import Path
import csv

import numpy as np

from src.evaluation.artifacts import Artifact, seal, staged_directory, write_json
from src.objectives import OBJECTIVES

METRICS = ("test_vrmse", "test_r2", "test_pearson_r", "test_mse", "test_log_mse")
# Only the governing-parameter probes define an error in train-standardized units.
NORMALIZED_MSE = "test_normalized_mse"
# Probes that read frozen encoder features; everything else is a baseline.
ENCODER_METHODS = ("ridge", "mlp", "attentive")
# The metadata MLP never sees the encoder: pooled gets regime and time, tokens
# additionally get their own spatiotemporal position.
METADATA_METHODS = (
    ("pooled", "regime_time_mlp"),
    ("token", "regime_time_position_mlp"),
)
PHYSICS_METHODS = (
    *ENCODER_METHODS,
    *(method for _, method in METADATA_METHODS),
    "persistence",
)
GOVERNING_LABELS = {
    "Prandtl": r"$\mathrm{Pr}$",
    "log10_Rayleigh": r"$\log_{10}\mathrm{Ra}$",
}
# Paper order first, then anything else a different system contributes.
GOVERNING_ORDER = ("Prandtl", "log10_Rayleigh")


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


def _metric_names(values):
    """Governing rows carry one extra metric; every other cell carries METRICS."""
    extra = (NORMALIZED_MSE,) if any(NORMALIZED_MSE in v for v in values) else ()
    return METRICS + extra


def aggregate(paths, output, objectives=OBJECTIVES, seeds=(1, 2, 3), kind="probes"):
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
    for artifact in artifacts:
        checkpoint = artifact.manifest["checkpoint"]
        budget = checkpoint.get("training_protocol", {}).get("total_steps")
        step = checkpoint.get("step")
        # Each run freezes its own best-validation step, so steps differ across
        # runs, but none of them may sit outside the configured training plan.
        if step is not None and budget is not None and not 0 < step <= budget:
            raise ValueError("checkpoint step exceeds configured training budget")
    for artifact in artifacts[1:]:
        for key in ("protocol", "caches", "probe_settings"):
            if artifact.manifest[key] != first[key]:
                raise ValueError(f"incompatible aggregate {key}")
        for key in ("spec", "encoder", "dataset", "training_protocol"):
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
                "checkpoint_step": checkpoint.get("step"),
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
                "target_offset",
                "target",
                "sigma",
                "shared",
            )
        }
        summary.update(objective=objective, cell_id=identity)
        metric_names = _metric_names(values)
        if objective == "shared":
            for metric in metric_names:
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
                    "selected_alpha",
                    "selected_steps",
                    "valid_r2",
                    "valid_vrmse",
                    "valid_normalized_mse",
                    "checkpoint_step",
                    "status",
                    "metric_status",
                )
            }
            for value in values
        ]
        summary["metrics"] = {
            metric: _summary(values, metric) for metric in metric_names
        }
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
                        metric: _summary(points, metric)
                        for metric in _metric_names(points)
                    },
                )
                for layer, points in (
                    (
                        layer,
                        [
                            next(
                                point
                                for point in v["depth_curve"]
                                if point["layer"] == layer
                            )
                            for v in values
                        ],
                    )
                    for layer in sorted(layers[0])
                )
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
            row["target_offset"],
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
                    (
                        "objective",
                        "family",
                        "method",
                        "representation",
                        "target_offset",
                        "sigma",
                    ),
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


def _column(representation, offset, target_offsets):
    return (
        0 if representation == "pooled" else len(target_offsets)
    ) + target_offsets.index(offset)


def _cells(summaries, targets, target_offsets, score_key, note=None):
    """Score and annotation grids over targets by representation and horizon."""
    shape = (len(targets), 2 * len(target_offsets))
    matrix = np.full(shape, np.nan)
    notes = np.full(shape, "", dtype=object)
    seen = set()
    for row in summaries:
        i, j = (
            targets.index(row["target"]),
            _column(row["representation"], row["target_offset"], target_offsets),
        )
        # One figure cell is one result; never let a later row hide an earlier one.
        if (i, j) in seen:
            raise ValueError(
                "duplicate result cell for "
                f"{row['method']} {row['representation']} "
                f"offset {row['target_offset']} {row['target']}"
            )
        seen.add((i, j))
        score = row["metrics"].get(score_key, {}).get("mean")
        if score is not None:
            matrix[i, j] = score
        if note is not None:
            notes[i, j] = note(row)
    return matrix, notes


def _heatmap(
    plt,
    axis,
    matrix,
    notes,
    targets,
    target_offsets,
    horizons,
    low,
    high,
    metric,
    missing="undefined",
):
    cmap = plt.get_cmap("viridis_r" if metric == "vrmse" else "viridis").with_extremes(
        bad="#eeeeee"
    )
    image = axis.imshow(matrix, vmin=low, vmax=high, cmap=cmap, aspect="auto")
    middle = (low + high) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                text = f"{value:.3f}"
                if notes[i, j]:
                    text += f"\nout {notes[i, j]}"
                dark = value > middle if metric == "vrmse" else value < middle
            else:
                text, dark = missing, False
            axis.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if dark else "black",
            )
    axis.set_xticks(
        range(2 * len(target_offsets)),
        [
            f"{representation} $t+{horizon}$"
            for representation in ("pooled", "token")
            for horizon in horizons
        ],
        rotation=40,
        ha="right",
    )
    axis.set_yticks(range(len(targets)), [t.replace("_", " ") for t in targets])
    return image


def _interval(entry):
    """One seed reports no interval rather than a fabricated zero."""
    if entry is None or entry.get("mean") is None:
        return "undefined"
    mean = entry["mean"]
    std = entry.get("std")
    return f"{mean:.3f}" if std is None else f"{mean:.3f} $\\pm$ {std:.3f}"


def _is_governing(row):
    return row["family"] == "regime" and row["representation"] == "pooled"


def _governing_targets(summaries):
    present = {r["target"] for r in summaries if _is_governing(r)}
    ordered = [t for t in GOVERNING_ORDER if t in present]
    return ordered + sorted(present - set(ordered))


def averaged_normalized_mse(raw, objective, method, targets):
    """Average the parameters inside a seed, then vary only over seeds.

    The paper's governing-parameter score is one number per fit, so the two
    standardized parameter errors are combined before the between-seed SD.
    """
    per_seed = defaultdict(dict)
    for row in raw:
        if (
            _is_governing(row)
            and row["method"] == method
            and row["objective"] == objective
            and row["target"] in targets
        ):
            seed, target = row["checkpoint_seed"], row["target"]
            if target in per_seed[seed]:
                raise ValueError(
                    f"duplicate governing row: {objective} {method} {target} seed {seed}"
                )
            per_seed[seed][target] = row.get(NORMALIZED_MSE)
    means = []
    for seed in sorted(per_seed):
        values = per_seed[seed]
        missing = [target for target in targets if target not in values]
        if missing:
            raise ValueError(
                f"governing parameters {missing} missing for "
                f"{objective} {method} seed {seed}"
            )
        numbers = [values[target] for target in targets]
        means.append(
            {
                "value": float(np.mean(numbers))
                if all(v is not None and np.isfinite(v) for v in numbers)
                else None
            }
        )
    return _summary(means, "value") if means else None


def display_horizons(target_offsets, context_frames):
    """t+0 reads the last context frame; later offsets count frames beyond it."""
    return [
        0 if offset == 0 else offset - context_frames for offset in target_offsets
    ]


def plot(aggregate_dir, output, metric="vrmse"):
    if metric not in ("vrmse", "r2"):
        raise ValueError("plot metric must be vrmse or r2")
    score_key = "test_" + metric
    score_label = "VRMSE" if metric == "vrmse" else "R²"
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
    targets = artifact.manifest.get("probe_settings", {}).get(
        "physical_targets", artifact.manifest["protocol"]["targets"]
    )
    target_offsets = artifact.manifest["protocol"]["target_offsets"]
    context_frames = artifact.manifest["protocol"]["n_frames"]
    horizons = display_horizons(target_offsets, context_frames)
    probe_layers = artifact.manifest.get("probe_settings", {}).get("probe_layers", [])
    probe_outputs = artifact.manifest.get("probe_settings", {}).get("probe_outputs", [])
    # Depth axes and cell annotations both name encoder outputs by layer index.
    if len(probe_outputs) != len(probe_layers):
        raise ValueError(
            f"probe_settings names {len(probe_outputs)} encoder outputs for "
            f"{len(probe_layers)} probe layers"
        )

    def encoder_output_label(layer):
        if layer not in probe_layers:
            return "?"
        output = probe_outputs[probe_layers.index(layer)]
        return "N" if output == "final_norm" else output.removeprefix("block_")

    def selected_output(row):
        labels = [
            encoder_output_label(selection.get("selected_layer"))
            for selection in row["selections"]
        ]
        if not labels:
            return ""
        first = labels[0]
        return first if all(label == first for label in labels) else "/".join(labels)

    with staged_directory(output) as stage:
        fields = (
            "objective",
            "family",
            "representation",
            "method",
            "target_offset",
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
                for metric_name, score in row["metrics"].items():
                    writer.writerow(
                        {key: row.get(key) for key in fields[:7]}
                        | {"metric": metric_name}
                        | score
                    )
        write_json(stage / "target_means.json", target_means)
        # Encoder probes, metadata baselines and persistence share one scale so
        # the separate figures can be read against each other.
        scores = [
            r["metrics"].get(score_key, {}).get("mean")
            for r in summaries
            if r.get("family") == "physics" and r.get("method") in PHYSICS_METHODS
        ]
        finite = [v for v in scores if v is not None and np.isfinite(v)]
        low = min([0.0, *finite])
        high = max([1.0, *finite])
        if artifact.manifest["result_kind"] == "probes":
            methods = [
                method
                for method in ENCODER_METHODS
                if any(
                    r["family"] == "physics" and r["method"] == method
                    for r in summaries
                )
            ]
            fig, axes = plt.subplots(
                max(1, len(methods)),
                len(objectives),
                figsize=(
                    6 * len(objectives),
                    max(3, len(targets) * 0.65) * max(1, len(methods)),
                ),
                squeeze=False,
                layout="constrained",
            )
            image = None
            for row_axes, method in zip(axes, methods):
                for axis, objective in zip(row_axes, objectives):
                    matrix, notes = _cells(
                        [
                            r
                            for r in summaries
                            if r["family"] == "physics"
                            and r["method"] == method
                            and r["objective"] == objective
                        ],
                        targets,
                        target_offsets,
                        score_key,
                        note=selected_output,
                    )
                    image = _heatmap(
                        plt,
                        axis,
                        matrix,
                        notes,
                        targets,
                        target_offsets,
                        horizons,
                        low,
                        high,
                        metric,
                    )
                    axis.set_title(f"{objective}: {method}, test {score_label}")
            if image is not None:
                fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.02).set_label(
                    f"Test {score_label}"
                )
            fig.suptitle(
                "Frozen best-validation encoders: Ridge and MLP probes\n"
                "cell text: test score and validation-selected encoder output"
            )
            fig.savefig(stage / "physics.pdf")
            plt.close(fig)

            # The metadata MLP sees no encoder features, so every objective
            # shares one baseline and it is drawn exactly once.
            metadata_cells = [
                r
                for r in summaries
                if r["family"] == "physics"
                and (r["representation"], r["method"]) in METADATA_METHODS
            ]
            fig, axis = plt.subplots(
                figsize=(8, max(3, len(targets) * 0.65)), layout="constrained"
            )
            matrix, notes = _cells(
                metadata_cells, targets, target_offsets, score_key
            )
            image = _heatmap(
                plt,
                axis,
                matrix,
                notes,
                targets,
                target_offsets,
                horizons,
                low,
                high,
                metric,
                missing="N/A",
            )
            axis.set_title(
                f"Metadata MLP baseline, test {score_label}\n"
                "pooled: regime parameters and time; "
                "token: also normalized token position"
            )
            fig.colorbar(image, ax=axis, pad=0.02).set_label(f"Test {score_label}")
            fig.savefig(stage / "physics_metadata_mlp.pdf")
            plt.close(fig)

            persistence = [
                row
                for row in summaries
                if row["family"] == "physics" and row["method"] == "persistence"
            ]
            fig, axis = plt.subplots(
                figsize=(8, max(3, len(targets) * 0.65)), layout="constrained"
            )
            matrix, notes = _cells(persistence, targets, target_offsets, score_key)
            image = _heatmap(
                plt,
                axis,
                matrix,
                notes,
                targets,
                target_offsets,
                horizons,
                low,
                high,
                metric,
                missing="N/A",
            )
            axis.set_title(
                f"Persistence baseline, test {score_label}\n"
                "future prediction copies the corresponding current target"
            )
            fig.colorbar(image, ax=axis, pad=0.02).set_label(f"Test {score_label}")
            fig.savefig(stage / "physics_persistence.pdf")
            plt.close(fig)

            # The paper's depth analysis is pooled, with one panel per target/horizon.
            for method in methods:
                fig, axes = plt.subplots(
                    len(targets),
                    len(target_offsets),
                    figsize=(4 * len(target_offsets), 2.5 * len(targets)),
                    squeeze=False,
                )
                for i, target in enumerate(targets):
                    for j, target_offset in enumerate(target_offsets):
                        axis = axes[i, j]
                        for row in raw:
                            if (
                                row["family"],
                                row["representation"],
                                row["method"],
                                row["target"],
                                row["target_offset"],
                            ) != ("physics", "pooled", method, target, target_offset):
                                continue
                            curve = row.get("depth_curve", [])
                            axis.plot(
                                [p["layer"] for p in curve],
                                [p.get(score_key) for p in curve],
                                color=colors[row["objective"]],
                                alpha=0.2,
                                lw=0.8,
                                marker="o",
                                markersize=2,
                            )
                        for row in summaries:
                            if (
                                row["family"],
                                row["representation"],
                                row["method"],
                                row["target"],
                                row["target_offset"],
                            ) != ("physics", "pooled", method, target, target_offset):
                                continue
                            curve = row.get("depth_curve", [])
                            axis.plot(
                                [p["layer"] for p in curve],
                                [p["metrics"][score_key]["mean"] for p in curve],
                                label=row["objective"],
                                color=colors[row["objective"]],
                                marker="o",
                                markersize=4,
                            )
                        axis.set_title(
                            f"{target.replace('_', ' ')} $t+{horizons[j]}$",
                            fontsize=9,
                        )
                        axis.set_xlabel("Encoder output")
                        if probe_layers:
                            output_labels = [
                                "N"
                                if output == "final_norm"
                                else output.removeprefix("block_")
                                for output in probe_outputs
                            ]
                            axis.set_xticks(probe_layers, output_labels)
                            if len(probe_layers) == 1:
                                axis.set_xlim(
                                    probe_layers[0] - 0.5, probe_layers[0] + 0.5
                                )
                        axis.set_ylabel(f"Test {score_label}")
                        for rows, color, label in (
                            (persistence, "black", "persistence"),
                            (
                                [
                                    r
                                    for r in summaries
                                    if r["family"] == "physics"
                                    and r["method"] == "regime_time_mlp"
                                ],
                                "tab:gray",
                                "regime+time MLP",
                            ),
                        ):
                            baseline = next(
                                (
                                    r["metrics"][score_key]["mean"]
                                    for r in rows
                                    if r["representation"] == "pooled"
                                    and r["target"] == target
                                    and r["target_offset"] == target_offset
                                ),
                                None,
                            )
                            if baseline is not None:
                                axis.axhline(
                                    baseline,
                                    color=color,
                                    linestyle="--",
                                    linewidth=1,
                                    label=label,
                                )
                handles, labels = [], []
                for axis in axes.ravel():
                    for handle, label in zip(*axis.get_legend_handles_labels()):
                        if label not in labels:
                            handles.append(handle)
                            labels.append(label)
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    ncol=max(1, len(labels)),
                    frameon=False,
                    bbox_to_anchor=(0.5, 1.0),
                )
                # Reserve a fixed strip so the legend never covers panel titles.
                fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
                fig.savefig(stage / f"depth_{method}.pdf")
                plt.close(fig)

            governing_targets = _governing_targets(summaries)
            if governing_targets:
                header = ["Objective", "Probe", "Encoder output"]
                for target in governing_targets:
                    label = GOVERNING_LABELS.get(target, target.replace("_", " "))
                    header += [f"{label} $R^2$", f"{label} nMSE"]
                header.append("Mean nMSE")
                body = []
                for objective in objectives:
                    for method in methods:
                        cells = {}
                        for row in summaries:
                            if (
                                not _is_governing(row)
                                or row["method"] != method
                                or row["objective"] != objective
                            ):
                                continue
                            if row["target"] in cells:
                                raise ValueError(
                                    "duplicate governing summary: "
                                    f"{objective} {method} {row['target']}"
                                )
                            cells[row["target"]] = row
                        if not cells:
                            continue
                        missing = [t for t in governing_targets if t not in cells]
                        if missing:
                            raise ValueError(
                                f"governing parameters {missing} missing for "
                                f"{objective} {method}"
                            )
                        outputs = {selected_output(cell) for cell in cells.values()}
                        line = [
                            objective,
                            method,
                            "/".join(sorted(outputs)),
                        ]
                        for target in governing_targets:
                            entry = cells[target]["metrics"]
                            line += [
                                _interval(entry.get("test_r2")),
                                _interval(entry.get(NORMALIZED_MSE)),
                            ]
                        line.append(
                            _interval(
                                averaged_normalized_mse(
                                    raw, objective, method, governing_targets
                                )
                            )
                        )
                        body.append(line)
                fig, axis = plt.subplots(
                    figsize=(2.0 + 1.55 * len(header), 1.4 + 0.42 * len(body)),
                    layout="constrained",
                )
                axis.axis("off")
                table = axis.table(
                    cellText=body or [["no governing probe rows"] + [""] * (len(header) - 1)],
                    colLabels=header,
                    cellLoc="center",
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.6)
                for (row_index, _), cell in table.get_celld().items():
                    cell.set_edgecolor("#bbbbbb")
                    if row_index == 0:
                        cell.set_facecolor("#e8e8e8")
                        cell.set_text_props(fontweight="bold")
                    elif row_index % 2 == 0:
                        cell.set_facecolor("#f7f7f7")
                seed_counts = sorted(
                    {
                        r["metrics"][NORMALIZED_MSE]["n"]
                        for r in summaries
                        if _is_governing(r) and NORMALIZED_MSE in r["metrics"]
                    }
                )
                axis.set_title(
                    "Governing-parameter probes on frozen best-validation encoders\n"
                    "targets standardized with training statistics; "
                    "nMSE in standardized units",
                    fontsize=11,
                )
                axis.text(
                    0.5,
                    0.0,
                    "Mean nMSE averages both parameters inside a seed before the "
                    "between-seed SD.\n"
                    + (
                        f"Intervals are SD over {'/'.join(str(n) for n in seed_counts)} "
                        "checkpoint seeds; a value without an interval comes from a "
                        "single seed."
                        if seed_counts
                        else "No seed statistics available."
                    ),
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
                fig.savefig(stage / "governing_parameters.pdf")
                plt.close(fig)
        else:
            def is_noise(row):
                return row["family"] == "noise" and row["representation"] == "pooled"

            noise_methods = [
                method
                for method in ENCODER_METHODS
                if any(is_noise(r) and r["method"] == method for r in raw)
            ]
            for metric_name in (score_key, "test_pearson_r"):
                fig, axes = plt.subplots(
                    len(targets),
                    max(1, len(noise_methods)),
                    figsize=(4 * max(1, len(noise_methods)), 2.5 * len(targets)),
                    squeeze=False,
                )
                for i, target in enumerate(targets):
                    for j, method in enumerate(noise_methods):
                        axis = axes[i, j]
                        for objective in objectives:
                            points = sorted(
                                [
                                    r
                                    for r in summaries
                                    if is_noise(r)
                                    and r["target"] == target
                                    and r["method"] == method
                                    and r["objective"] == objective
                                ],
                                key=lambda r: r["sigma"],
                            )
                            for seed in sorted(
                                {
                                    r["checkpoint_seed"]
                                    for r in raw
                                    if is_noise(r) and r["objective"] == objective
                                }
                            ):
                                values = sorted(
                                    [
                                        r
                                        for r in raw
                                        if is_noise(r)
                                        and r["target"] == target
                                        and r["method"] == method
                                        and r["objective"] == objective
                                        and r["checkpoint_seed"] == seed
                                    ],
                                    key=lambda r: r["sigma"],
                                )
                                axis.plot(
                                    [r["sigma"] for r in values],
                                    [r.get(metric_name) for r in values],
                                    color=colors[objective],
                                    alpha=0.2,
                                    lw=0.8,
                                )
                            axis.plot(
                                [r["sigma"] for r in points],
                                [r["metrics"][metric_name]["mean"] for r in points],
                                color=colors[objective],
                                label=objective,
                                marker="o",
                            )
                        if metric_name == "test_r2":
                            axis.set_yscale("symlog", linthresh=1.0)
                        axis.set_title(
                            f"{target.replace('_', ' ')}: {method}", fontsize=9
                        )
                        axis.set_xlabel("Noise sigma")
                        axis.set_ylabel(metric_name.removeprefix("test_"))
                axes[0, 0].legend()
                fig.tight_layout()
                fig.savefig(stage / f"noise_{metric_name}.pdf")
                plt.close(fig)
        seal(stage, "plots", aggregate=artifact.manifest["sha256"], metric=score_key)
    return Path(output)

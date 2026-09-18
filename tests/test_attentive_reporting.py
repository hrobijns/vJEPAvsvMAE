"""Reporting for the best-validation attentive analysis."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.artifacts import Artifact, seal, write_json
from src.evaluation.reporting import (
    aggregate,
    averaged_normalized_mse,
    display_horizons,
    plot,
)

TARGETS = ("enstrophy", "convective_flux")
OFFSETS = (0, 16, 24, 40)
LAYERS = (0, 1, 2)
GOVERNING = ("Prandtl", "log10_Rayleigh")
PROTOCOL = {
    "dataset": "rayleigh_benard",
    "targets": list(TARGETS),
    "target_offsets": list(OFFSETS),
    "n_frames": 8,
}
PROBE_SETTINGS = {
    "probe_layers": list(LAYERS),
    "probe_outputs": ["block_1", "block_2", "final_norm"],
    "selection_metric": "valid_vrmse",
    "attentive_heads": 8,
    "attentive_ffn_hidden": 96,
}
EXPECTED_FIGURES = {
    "depth_attentive.pdf",
    "depth_ridge.pdf",
    "governing_parameters.pdf",
    "physics.pdf",
    "physics_metadata_mlp.pdf",
    "physics_persistence.pdf",
}
DEPRECATED_FIGURES = {
    "physics_selected.pdf",
    "physics_mlp.pdf",
    "workshop_figure1_comparison.pdf",
    "workshop_figure1_comparison.png",
}


def _scores(value, prefix="test"):
    return {
        f"{prefix}_vrmse": value,
        f"{prefix}_r2": 1.0 - value,
        f"{prefix}_pearson_r": 1.0 - value / 2,
        f"{prefix}_mse": value**2,
        f"{prefix}_log_mse": value / 3,
    }


def _cell_id(row):
    keys = ("family", "representation", "target_offset", "target", "method", "sigma")
    return "|".join(f"{k}={row[k]}" for k in keys if k in row)


def _physics_rows(objective, seed):
    rows = []
    offset_base = {"jepa": 0.30, "mae": 0.42}[objective] + 0.04 * seed
    for representation in ("pooled", "token"):
        for offset in OFFSETS:
            for target in TARGETS:
                base = dict(
                    family="physics",
                    representation=representation,
                    target_offset=offset,
                    target=target,
                )
                for index, method in enumerate(("ridge", "attentive")):
                    value = (
                        offset_base
                        + 0.02 * index
                        + 0.01 * OFFSETS.index(offset)
                        + 0.03 * TARGETS.index(target)
                        + (0.0 if representation == "pooled" else 0.015)
                    )
                    rows.append(
                        base
                        | dict(
                            method=method,
                            status="ok",
                            metric_status="ok",
                            selected_layer=LAYERS[(seed + index) % len(LAYERS)],
                            selected_alpha=0.01,
                            valid_vrmse=value + 0.01,
                            valid_r2=1.0 - value,
                            depth_curve=[
                                dict(layer=layer, **_scores(value + 0.01 * layer))
                                for layer in LAYERS
                            ],
                            **_scores(value),
                        )
                    )
                # Encoder-independent baselines: identical for every checkpoint.
                metadata_method = (
                    "regime_time_mlp"
                    if representation == "pooled"
                    else "regime_time_position_mlp"
                )
                rows.append(
                    base
                    | dict(
                        method=metadata_method,
                        shared=True,
                        status="ok",
                        selected_layer=0,
                        depth_curve=[],
                        **_scores(0.80 - 0.05 * TARGETS.index(target)),
                    )
                )
                if offset:
                    rows.append(
                        base
                        | dict(
                            method="persistence",
                            shared=True,
                            status="ok",
                            selected_layer=None,
                            depth_curve=[],
                            **_scores(0.90),
                        )
                    )
    return rows


def _governing_rows(objective, seed):
    rows = []
    for index, method in enumerate(("ridge", "attentive")):
        for position, target in enumerate(GOVERNING):
            value = 0.5 - 0.1 * index + 0.05 * position + 0.02 * seed
            normalized = 0.20 + 0.05 * index + 0.10 * position + 0.01 * seed
            rows.append(
                dict(
                    family="regime",
                    representation="pooled",
                    target_offset=0,
                    target=target,
                    method=method,
                    status="ok",
                    metric_status="ok",
                    selected_layer=LAYERS[-1],
                    valid_normalized_mse=normalized + 0.01,
                    test_normalized_mse=normalized,
                    depth_curve=[
                        dict(
                            layer=layer,
                            test_normalized_mse=normalized + 0.01 * layer,
                            **_scores(value + 0.01 * layer),
                        )
                        for layer in LAYERS
                    ],
                    **_scores(value),
                )
            )
    return rows


def _run(root, objective, seed, step, mutate=None):
    path = Path(root) / f"{objective}_{seed}"
    path.mkdir()
    rows = _physics_rows(objective, seed) + _governing_rows(objective, seed)
    for row in rows:
        row["cell_id"] = _cell_id(row)
    if mutate is not None:
        mutate(rows)
    write_json(path / "rows.json", rows)
    seal(
        path,
        "probes",
        checkpoint=dict(
            objective=objective,
            seed=seed,
            step=step,
            spec={"depth": 12},
            encoder="vit_small",
            dataset="rayleigh_benard",
            training_protocol={"total_steps": 100000},
        ),
        protocol=PROTOCOL,
        caches={"train": "a", "valid": "b", "test": "c"},
        probe_settings=PROBE_SETTINGS,
    )
    return path


def _aggregate(root, steps=((1, 100000), (2, 92000))):
    paths = [
        _run(root, objective, seed, step)
        for objective in ("jepa", "mae")
        for seed, step in steps
    ]
    out = Path(root) / "aggregate"
    aggregate(paths, out, objectives=("jepa", "mae"), seeds=[s for s, _ in steps])
    return out


class ReportingOutputTests(unittest.TestCase):
    def test_plot_emits_contracted_figures_and_no_deprecated_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plots = plot(_aggregate(root), root / "plots")
            produced = {
                path.name
                for path in plots.iterdir()
                if path.suffix in (".pdf", ".png")
            }
            self.assertEqual(produced, EXPECTED_FIGURES)
            self.assertFalse(produced & DEPRECATED_FIGURES)
            for name in EXPECTED_FIGURES:
                with (plots / name).open("rb") as handle:
                    self.assertEqual(handle.read(5), b"%PDF-")
                self.assertGreater((plots / name).stat().st_size, 1000)
            artifact = Artifact(plots, "plots")
            self.assertEqual(artifact.manifest["metric"], "test_vrmse")
            self.assertTrue((plots / "summary.tsv").exists())
            self.assertTrue((plots / "target_means.json").exists())

    def test_summary_tsv_reports_governing_normalized_mse_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plots = plot(_aggregate(root), root / "plots")
            with (plots / "summary.tsv").open() as handle:
                table = list(csv.DictReader(handle, delimiter="\t"))
            normalized = [r for r in table if r["metric"] == "test_normalized_mse"]
            self.assertEqual({r["family"] for r in normalized}, {"regime"})
            self.assertEqual(
                {(r["objective"], r["method"], r["target"]) for r in normalized},
                {
                    (objective, method, target)
                    for objective in ("jepa", "mae")
                    for method in ("ridge", "attentive")
                    for target in GOVERNING
                },
            )
            cell = next(
                r
                for r in normalized
                if r["objective"] == "jepa"
                and r["method"] == "ridge"
                and r["target"] == "Prandtl"
            )
            # 0.20 + 0.01 * seed for seeds 1 and 2.
            self.assertAlmostEqual(float(cell["mean"]), 0.215)
            self.assertAlmostEqual(float(cell["std"]), np.std([0.21, 0.22], ddof=1))
            self.assertEqual(cell["n"], "2")

    def test_metadata_baselines_are_shared_across_objectives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summaries = Artifact(_aggregate(root), "aggregate").json("summary.json")
            metadata = [
                r
                for r in summaries
                if r["method"] in ("regime_time_mlp", "regime_time_position_mlp")
            ]
            self.assertEqual({r["objective"] for r in metadata}, {"shared"})
            self.assertEqual(
                {(r["representation"], r["method"]) for r in metadata},
                {("pooled", "regime_time_mlp"), ("token", "regime_time_position_mlp")},
            )
            # One cell per representation, horizon and target; never per objective.
            self.assertEqual(len(metadata), 2 * len(OFFSETS) * len(TARGETS))
            self.assertEqual(
                {r["metrics"]["test_vrmse"]["n"] for r in metadata}, {1}
            )

    def test_encoder_independent_baseline_must_agree_between_runs(self):
        def bend(rows):
            for row in rows:
                if row["method"] == "regime_time_mlp":
                    row["test_vrmse"] += 0.1
                    break

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                _run(root, "jepa", 1, 100000),
                _run(root, "jepa", 2, 92000, mutate=bend),
            ]
            with self.assertRaisesRegex(ValueError, "shared control metrics disagree"):
                aggregate(paths, root / "out", objectives=("jepa",), seeds=(1, 2))

    def test_best_validation_steps_may_differ_but_stay_inside_the_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summaries = Artifact(
                _aggregate(root, steps=((1, 100000), (2, 71000))), "aggregate"
            ).json("summary.json")
            cell = next(r for r in summaries if r["objective"] == "jepa")
            self.assertEqual(
                sorted(s["checkpoint_step"] for s in cell["selections"]),
                [71000, 100000],
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                _run(root, "jepa", 1, 100000),
                _run(root, "jepa", 2, 100001),
            ]
            with self.assertRaisesRegex(ValueError, "budget"):
                aggregate(paths, root / "out", objectives=("jepa",), seeds=(1, 2))

    def test_averaged_normalized_mse_combines_parameters_inside_a_seed(self):
        raw = [
            dict(
                family="regime",
                representation="pooled",
                method="attentive",
                objective="jepa",
                target=target,
                checkpoint_seed=seed,
                test_normalized_mse=value,
            )
            for seed, values in ((1, (0.10, 0.30)), (2, (0.20, 0.60)))
            for target, value in zip(GOVERNING, values)
        ] + [
            dict(
                family="regime",
                representation="pooled",
                method="ridge",
                objective="jepa",
                target=target,
                checkpoint_seed=1,
                test_normalized_mse=0.5,
            )
            for target in GOVERNING
        ]
        combined = averaged_normalized_mse(raw, "jepa", "attentive", GOVERNING)
        self.assertAlmostEqual(combined["mean"], 0.3)
        self.assertAlmostEqual(combined["std"], np.std([0.2, 0.4], ddof=1))
        self.assertEqual(combined["n"], 2)
        # A single checkpoint reports no between-seed interval.
        single = averaged_normalized_mse(raw, "jepa", "ridge", GOVERNING)
        self.assertAlmostEqual(single["mean"], 0.5)
        self.assertIsNone(single["std"])
        self.assertIsNone(averaged_normalized_mse(raw, "mae", "ridge", GOVERNING))

    def test_horizons_count_frames_beyond_the_context_window(self):
        self.assertEqual(display_horizons(OFFSETS, 8), [0, 8, 16, 32])


if __name__ == "__main__":
    unittest.main()

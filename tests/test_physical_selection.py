"""Analytical targets and validation-only checkpoint selection."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from src.evaluation.artifacts import Artifact, seal, write_json
from src.evaluation.probes import metrics
from src.evaluation.protocol import Protocol
from src.evaluation.selection import (
    balanced_score,
    select_checkpoints,
    SELECTION_POLICY,
)
from src.evaluation.reporting import aggregate
from src.physics.systems import SYSTEMS


class PhysicalSelectionTests(unittest.TestCase):
    def test_nematic_order_normalizes_concentration_and_detects_rotation(self):
        system = SYSTEMS["active_matter"]
        x = torch.arange(32, dtype=torch.float64)[:, None] * 10 / 32
        k, s = 2 * torch.pi / 10, 0.6
        c = (1 + 0.2 * torch.sin(k * x)).expand(32, 32)
        raw = torch.zeros(1, 11, 2, 32, 32, dtype=torch.float64)
        raw[:, 0] = c
        raw[:, 1], raw[:, 2] = 2, 3
        raw[:, 3] = c * (0.5 + 0.5 * s * torch.cos(k * x))
        raw[:, 6] = c * (0.5 - 0.5 * s * torch.cos(k * x))
        raw[:, 4] = raw[:, 5] = c * 0.5 * s * torch.sin(k * x)
        fields = system.fields(raw, (1, 2))
        for target, expected in {
            "kinetic_energy": 6.5,
            "enstrophy": 0,
            "nematic_order": s,
            "nematic_gradient_energy": 0.5 * s * s * k * k,
        }.items():
            torch.testing.assert_close(
                fields[target],
                torch.full_like(fields[target], expected),
                atol=1e-11,
                rtol=1e-11,
            )
            torch.testing.assert_close(
                system.reduce(fields[target], (1, 8, 8)).mean(1),
                system.reduce(fields[target]),
            )

    def test_shear_targets_use_physical_axes_and_frame_spatial_mean(self):
        system = SYSTEMS["shear_flow"]
        x = torch.arange(32, dtype=torch.float64)[:, None] / 32
        y = torch.arange(48, dtype=torch.float64)[None, :] * 2 / 48
        raw = torch.zeros(1, 4, 2, 32, 48, dtype=torch.float64)
        raw[:, 0, 0] = torch.sin(torch.pi * y)
        raw[:, 0, 1] = 5 + torch.sin(torch.pi * y)
        raw[:, 2] = 2
        raw[:, 3] = torch.sin(2 * torch.pi * x)
        fields = system.fields(raw, (2, 3))
        expected = {
            "cross_stream_kinetic_energy": 0.5 * torch.sin(2 * torch.pi * x) ** 2,
            "enstrophy": (2 * torch.pi * torch.cos(2 * torch.pi * x)) ** 2,
            "tracer_variance": torch.sin(torch.pi * y) ** 2,
            "tracer_gradient_energy": (torch.pi * torch.cos(torch.pi * y)) ** 2,
        }
        for name, value in expected.items():
            torch.testing.assert_close(
                fields[name], value.expand(1, 2, 32, 48), atol=1e-10, rtol=1e-10
            )
            torch.testing.assert_close(
                system.reduce(fields[name], (1, 8, 8)).mean(1),
                system.reduce(fields[name]),
            )

    def test_scalar_vrmse_has_expected_scale_and_r2_relationship(self):
        y, p = np.array([0.0, 4.0]), np.array([1.0, 3.0])
        for scale in (1.0, 1e-10, 1000.0):
            result = metrics(p * scale, y * scale)
            self.assertAlmostEqual(result["test_vrmse"], 0.5)
            self.assertAlmostEqual(result["test_vrmse"] ** 2, 1 - result["test_r2"])
        self.assertEqual(metrics(y, y)["test_vrmse"], 0)
        self.assertEqual(metrics(np.full(2, y.mean()), y)["test_vrmse"], 1)
        self.assertTrue(np.isnan(metrics(np.zeros(2), np.ones(2))["test_vrmse"]))

    def test_checkpoint_selection_balances_present_and_future_without_test(self):
        protocol = Protocol("shear_flow")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name, step, present, future in (
                ("a", 25000, 0.1, 0.9),
                ("b", 100000, 0.7, 0.5),
            ):
                rows = [
                    dict(
                        family="physics",
                        method="selected",
                        representation=rep,
                        target_offset=offset,
                        target=target,
                        valid_vrmse=present if offset == 0 else future,
                    )
                    for rep in ("pooled", "token")
                    for offset in protocol.target_offsets
                    for target in SYSTEMS["shear_flow"].targets
                ]
                score, _ = balanced_score(rows, protocol)
                self.assertAlmostEqual(score, (present + future) / 2)
                with self.assertRaisesRegex(ValueError, "roster"):
                    balanced_score(rows[:-1], protocol)
                path = root / name
                path.mkdir()
                write_json(path / "rows.json", rows)
                seal(
                    path,
                    "probe_fits",
                    protocol=protocol.to_dict(),
                    caches={"train": "train", "valid": "valid"},
                    probe_settings={},
                    checkpoint=dict(
                        dataset="shear_flow",
                        objective="jepa",
                        seed=1,
                        step=step,
                        sha256=name,
                        config_sha256="same",
                        spec={},
                        encoder={},
                        training_protocol={"total_steps": 100000},
                    ),
                )
                paths.append(path)
            select_checkpoints(paths, root / "selection")
            selected = Artifact(root / "selection", "checkpoint_selection").json(
                "selections.json"
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["step"], 25000)
            self.assertFalse((root / "test").exists())

    def test_adjacent_and_legacy_horizon_definitions(self):
        p = Protocol("shear_flow")
        self.assertEqual(
            [p.target_start(10, o) for o in p.target_offsets], [10, 18, 26, 50]
        )
        old = Protocol("rayleigh_benard", frame_limit=101, target_offsets=(0, 16, 40))
        self.assertEqual(old.offsets(200, 0)["pooled"], [0, 26, 53])
        with self.assertRaisesRegex(ValueError, "legacy gaps"):
            Protocol.from_dict({"dataset": "shear_flow", "gaps": [0, 8, 32]})

    def test_selected_steps_can_differ_but_training_budgets_still_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed, step, budget in (
                (1, 25000, 100000),
                (2, 100000, 100000),
                (2, 100000, 200000),
            ):
                path = root / f"{seed}_{budget}"
                path.mkdir()
                write_json(
                    path / "rows.json",
                    [dict(cell_id="x", family="regime", target="x", test_vrmse=0.5)],
                )
                seal(
                    path,
                    "probes",
                    checkpoint=dict(
                        objective="jepa",
                        seed=seed,
                        step=step,
                        training_protocol={"total_steps": budget},
                    ),
                    protocol={},
                    caches={},
                    probe_settings={},
                    selection="recorded-selection",
                    selection_policy=SELECTION_POLICY,
                )
                paths.append(path)
            aggregate(paths[:2], root / "valid", objectives=("jepa",), seeds=(1, 2))
            with self.assertRaisesRegex(ValueError, "training_protocol"):
                aggregate(
                    [paths[0], paths[2]],
                    root / "invalid",
                    objectives=("jepa",),
                    seeds=(1, 2),
                )

    def test_mixed_selected_families_preserve_seed_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed, method, error in ((1, "ridge", 0.2), (2, "mlp", 0.6)):
                path = root / str(seed)
                path.mkdir()
                write_json(
                    path / "rows.json",
                    [
                        dict(
                            cell_id="enstrophy",
                            family="physics",
                            representation="token",
                            target_offset=0,
                            target="enstrophy",
                            method="selected",
                            selected_method=method,
                            selected_layer=seed,
                            test_vrmse=error,
                        )
                    ],
                )
                seal(
                    path,
                    "probes",
                    checkpoint=dict(objective="jepa", seed=seed),
                    protocol={},
                    caches={},
                    probe_settings={},
                )
                paths.append(path)
            aggregate(paths, root / "summary", objectives=("jepa",), seeds=(1, 2))
            row = Artifact(root / "summary", "aggregate").json("summary.json")[0]
            self.assertNotIn("depth_curve", row)
            self.assertEqual(
                [
                    (s["checkpoint_seed"], s["selected_method"], s["selected_layer"])
                    for s in row["selections"]
                ],
                [(1, "ridge", 1), (2, "mlp", 2)],
            )
            score = row["metrics"]["test_vrmse"]
            self.assertEqual(score["n"], 2)
            self.assertAlmostEqual(score["mean"], 0.4)
            self.assertAlmostEqual(score["std"], np.std([0.2, 0.6], ddof=1))

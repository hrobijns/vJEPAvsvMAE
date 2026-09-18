"""Analytical physical targets, metrics, and horizon definitions."""

import unittest

import numpy as np
import torch

from src.evaluation.probes import metrics
from src.evaluation.protocol import Protocol
from src.physics.systems import SYSTEMS


class PhysicalTargetTests(unittest.TestCase):
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


    def test_adjacent_and_legacy_horizon_definitions(self):
        p = Protocol("shear_flow")
        self.assertEqual(
            [p.target_start(10, o) for o in p.target_offsets], [10, 26, 34, 50]
        )
        old = Protocol("rayleigh_benard", frame_limit=101, target_offsets=(0, 16, 40))
        self.assertEqual(old.offsets(200, 0)["pooled"], [0, 26, 53])
        with self.assertRaisesRegex(ValueError, "legacy gaps"):
            Protocol.from_dict({"dataset": "shear_flow", "gaps": [0, 8, 32]})

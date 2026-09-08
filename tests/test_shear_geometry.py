"""Check shear-flow physics against a domain fixed independently of SYSTEMS."""

import unittest

import torch

from src.physics.systems import SYSTEMS, periodic_derivative


class ShearGeometryTests(unittest.TestCase):
    def test_known_velocity_on_physical_domain(self):
        # The generator uses x in [0, 1) and y in [-1, 1), periodically.
        # Do not derive these coordinates or the answer from SYSTEMS.lengths.
        x = torch.arange(256, dtype=torch.float64)[:, None] / 256
        y = -1 + 2 * torch.arange(512, dtype=torch.float64)[None, :] / 512
        sx, cx = torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * x)
        sy, cy = torch.sin(torch.pi * y), torch.cos(torch.pi * y)
        u, v = torch.pi * sx * cy, -2 * torch.pi * cx * sy
        du_dx = 2 * torch.pi**2 * cx * cy
        du_dy = -torch.pi**2 * sx * sy
        dv_dx = 4 * torch.pi**2 * sx * sy
        omega = 5 * torch.pi**2 * sx * sy

        torch.testing.assert_close(
            periodic_derivative(u, 0, 1), du_dx, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            periodic_derivative(u, 1, 2), du_dy, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            periodic_derivative(v, 0, 1), dv_dx, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            periodic_derivative(v, 1, 2), -du_dx, atol=1e-10, rtol=1e-10
        )

        raw = torch.zeros(1, 4, 1, 256, 512, dtype=torch.float64)
        raw[0, 2, 0], raw[0, 3, 0] = u, v
        enstrophy = SYSTEMS["shear_flow"].fields(raw, (2, 3))["enstrophy"]
        torch.testing.assert_close(
            enstrophy[0, 0], omega.square(), atol=1e-9, rtol=1e-10
        )
        self.assertAlmostEqual(
            float(SYSTEMS["shear_flow"].reduce(enstrophy)[0]),
            25 * torch.pi**4 / 4,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()

"""The three supported systems and their physical target definitions."""

from dataclasses import dataclass

import numpy as np
import torch

from src.physics import rayleigh_benard as rb


@dataclass(frozen=True)
class System:
    name: str
    parameters: tuple[str, str]
    log_parameters: bool
    lengths: tuple[float, float]
    channels: int

    @property
    def targets(self):
        if self.name == "rayleigh_benard":
            return rb.PRIMARY_TARGETS
        if self.name == "active_matter":
            return (
                "kinetic_energy",
                "enstrophy",
                "nematic_order",
                "nematic_gradient_energy",
            )
        return (
            "cross_stream_kinetic_energy",
            "enstrophy",
            "tracer_variance",
            "tracer_gradient_energy",
        )

    def regime_values(self, parameters):
        values = np.asarray(
            [parameters[key] for key in self.parameters], dtype=np.float64
        )
        if not np.isfinite(values).all() or (
            self.log_parameters and np.any(values <= 0)
        ):
            raise ValueError(f"invalid {self.name} regime: {parameters}")
        return np.log10(values) if self.log_parameters else values

    def fields(self, raw, velocity_channels, channels=None):
        if (
            raw.ndim != 5
            or raw.shape[1] != self.channels
            or not torch.isfinite(raw).all()
        ):
            raise ValueError(f"invalid raw {self.name} fields: {tuple(raw.shape)}")
        if self.name == "rayleigh_benard":
            return rb.target_fields(raw)
        u, v = (raw[:, index].double() for index in velocity_channels)
        omega = periodic_derivative(v, -2, self.lengths[0]) - periodic_derivative(
            u, -1, self.lengths[1]
        )
        fields = {"enstrophy": omega.square()}
        if self.name == "active_matter":
            names = channels or [
                "concentration",
                "velocity_x",
                "velocity_y",
                "D_xx",
                "D_xy",
                "D_yx",
                "D_yy",
            ]
            c = raw[:, names.index("concentration")].double()
            if torch.any(c <= 0):
                raise ValueError("nematic order requires positive concentration")
            q = torch.stack(
                [
                    raw[:, names.index("D_" + axis)].double() / c
                    - (0.5 if axis in ("xx", "yy") else 0.0)
                    for axis in ("xx", "xy", "yx", "yy")
                ],
                dim=1,
            )
            fields.update(
                kinetic_energy=0.5 * (u.square() + v.square()),
                nematic_order=(2 * q.square().sum(1)).sqrt(),
                nematic_gradient_energy=self.gradient_squared(q).sum(1),
            )
        else:
            tracer = raw[:, channels.index("tracer") if channels else 0].double()
            fields.update(
                cross_stream_kinetic_energy=0.5 * v.square(),
                tracer_variance=(tracer - tracer.mean((-2, -1), keepdim=True)).square(),
                tracer_gradient_energy=self.gradient_squared(tracer),
            )
        return fields

    def gradient_squared(self, field):
        return sum(
            periodic_derivative(field, axis, length).square()
            for axis, length in zip((-2, -1), self.lengths)
        )

    def reduce(self, field, patch=None):
        if self.name == "rayleigh_benard":
            return (
                rb.volume_mean(field)
                if patch is None
                else rb.weighted_patch_mean(field, patch)
            )
        if patch is None:
            return field.mean(dim=(1, 2, 3))
        b, t, x, y = field.shape
        pt, px, py = patch
        if t % pt or x % px or y % py:
            raise ValueError("physical field is not divisible by the token patch")
        return (
            field.reshape(b, t // pt, pt, x // px, px, y // py, py)
            .mean(dim=(2, 4, 6))
            .reshape(b, -1)
        )


SYSTEMS = {
    "rayleigh_benard": System(
        "rayleigh_benard", ("Rayleigh", "Prandtl"), True, (4.0, 1.0), 4
    ),
    "active_matter": System(
        "active_matter", ("alpha", "zeta"), False, (10.0, 10.0), 11
    ),
    "shear_flow": System("shear_flow", ("Reynolds", "Schmidt"), True, (1.0, 2.0), 4),
}


def periodic_derivative(field, axis, length):
    """Fourier derivative on an endpoint-excluded periodic physical axis."""
    n = field.shape[axis]
    spectrum = torch.fft.rfft(field.double(), dim=axis)
    k = (
        2
        * torch.pi
        * torch.fft.rfftfreq(n, d=length / n, dtype=torch.float64, device=field.device)
    )
    shape = [1] * field.ndim
    shape[axis] = len(k)
    return torch.fft.irfft(spectrum * (1j * k.reshape(shape)), n=n, dim=axis)

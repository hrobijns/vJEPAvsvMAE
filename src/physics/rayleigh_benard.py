"""Canonical, geometry-aware Rayleigh--Benard probe targets.

Inputs are raw physical fields with layout ``(B, C, T, X, Y)`` and channel
order ``[buoyancy, pressure, velocity_x, velocity_y]``.  Derivatives are
evaluated in float64 on the periodic-x / Chebyshev-Gauss-y grid.
"""

from __future__ import annotations

import torch

from src.physics import rb_derivatives as R
from src.physics.quadrature import quad_weights

PRIMARY_TARGETS = (
    "enstrophy",
    "buoyancy_gradient_energy",
    "convective_flux",
    "pressure_gradient_magnitude",
    "buoyancy_laplacian_magnitude",
)


def _as_float64(raw_clip: torch.Tensor) -> torch.Tensor:
    if raw_clip.ndim != 5 or raw_clip.shape[1] != 4:
        raise ValueError(f"expected raw clip (B,4,T,X,Y), got {tuple(raw_clip.shape)}")
    if raw_clip.shape[-2:] != (R.RB_NX, R.RB_NY):
        raise ValueError(
            f"expected RB grid ({R.RB_NX},{R.RB_NY}), got {tuple(raw_clip.shape[-2:])}"
        )
    return raw_clip.to(torch.float64)


def volume_mean(field: torch.Tensor) -> torch.Tensor:
    """Physical mean over every non-batch dimension of ``(..., X, Y)``.

    X and time are uniform.  Y uses Fejer weights for the Chebyshev-Gauss
    nodes; an arithmetic mean would over-weight the wall clusters.
    """
    if field.ndim < 3 or field.shape[-1] != R.RB_NY:
        raise ValueError(f"expected (...,X,{R.RB_NY}), got {tuple(field.shape)}")
    weights = torch.as_tensor(
        quad_weights(R.RB_NY, R.RB_LY), dtype=field.dtype, device=field.device
    )
    integrated_y = (field * weights).sum(dim=-1) / R.RB_LY
    dims = tuple(range(1, integrated_y.ndim))
    return integrated_y.mean(dim=dims)


def weighted_patch_mean(
    field: torch.Tensor, patch: tuple[int, int, int]
) -> torch.Tensor:
    """Map ``(B,T,X,Y)`` fields to ViT token targets in time-major order."""
    if field.ndim != 4:
        raise ValueError(f"expected (B,T,X,Y), got {tuple(field.shape)}")
    pt, px, py = patch
    b, t, x, y = field.shape
    if t % pt or x % px or y % py:
        raise ValueError(f"field {(t, x, y)} not divisible by patch {patch}")
    nt, nx, ny = t // pt, x // px, y // py
    values = field.reshape(b, nt, pt, nx, px, ny, py)
    weights = torch.as_tensor(
        quad_weights(y, R.RB_LY), dtype=field.dtype, device=field.device
    )
    weights = weights.reshape(ny, py)
    weights = weights / weights.sum(dim=1, keepdim=True)
    values = (values * weights.reshape(1, 1, 1, 1, 1, ny, py)).sum(dim=-1)
    values = values.mean(dim=(2, 4))
    return values.reshape(b, nt * nx * ny)


def target_fields(raw_clip: torch.Tensor) -> dict[str, torch.Tensor]:
    raw = _as_float64(raw_clip)
    buoyancy, pressure, u, v = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

    omega = R.curl(u, v)

    return {
        "enstrophy": omega.square(),
        "buoyancy_gradient_energy": R.grad_sq(buoyancy),
        "convective_flux": v * buoyancy,
        "pressure_gradient_magnitude": R.grad_sq(pressure).clamp_min(0).sqrt(),
        "buoyancy_laplacian_magnitude": R.laplacian(buoyancy).abs(),
    }


def pooled_targets(raw_clip: torch.Tensor) -> dict[str, torch.Tensor]:
    fields = target_fields(raw_clip)
    return {name: volume_mean(fields[name]) for name in PRIMARY_TARGETS}


def token_targets(
    raw_clip: torch.Tensor,
    patch: tuple[int, int, int] = (2, 16, 16),
) -> dict[str, torch.Tensor]:
    fields = target_fields(raw_clip)
    return {name: weighted_patch_mean(fields[name], patch) for name in PRIMARY_TARGETS}

"""Canonical, geometry-aware Rayleigh--Benard probe targets.

Inputs are raw physical fields with layout ``(B, C, T, X, Y)`` and channel
order ``[buoyancy, pressure, velocity_x, velocity_y]``.  Derivatives are
evaluated in float64 on the periodic-x / Chebyshev-Gauss-y grid.  This module
is intentionally independent of the legacy image-grid targets in
``analyze_encoders.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from scripts import rb_derivatives as R
from scripts.rb_quadrature import quad_weights

TARGET_SCHEMA_VERSION = "rb-physics-v2"

PRIMARY_TARGETS = (
    "enstrophy",
    "buoyancy_gradient_energy",
    "convective_flux",
    "pressure_gradient_magnitude",
    "buoyancy_laplacian_magnitude",
    "deformation_energy",
)
TOKEN_ONLY_TARGETS = ("vorticity_signed", "okubo_weiss_signed")


def _as_float64(raw_clip: torch.Tensor) -> torch.Tensor:
    if raw_clip.ndim != 5 or raw_clip.shape[1] != 4:
        raise ValueError(f"expected raw clip (B,4,T,X,Y), got {tuple(raw_clip.shape)}")
    if raw_clip.shape[-2:] != (R.RB_NX, R.RB_NY):
        raise ValueError(
            f"expected RB grid ({R.RB_NX},{R.RB_NY}), got {tuple(raw_clip.shape[-2:])}"
        )
    return raw_clip.to(torch.float64)


def buoyancy_fluctuation(buoyancy: torch.Tensor) -> torch.Tensor:
    """Remove the instantaneous horizontal mean profile from buoyancy."""
    return buoyancy - buoyancy.mean(dim=-2, keepdim=True)


def volume_mean(field: torch.Tensor) -> torch.Tensor:
    """Physical mean over every non-batch dimension of ``(..., X, Y)``.

    X and time are uniform.  Y uses Fejer weights for the Chebyshev-Gauss
    nodes; an arithmetic mean would over-weight the wall clusters.
    """
    if field.ndim < 3 or field.shape[-1] != R.RB_NY:
        raise ValueError(f"expected (...,X,{R.RB_NY}), got {tuple(field.shape)}")
    weights = torch.as_tensor(quad_weights(R.RB_NY, R.RB_LY), dtype=field.dtype, device=field.device)
    integrated_y = (field * weights).sum(dim=-1) / R.RB_LY
    dims = tuple(range(1, integrated_y.ndim))
    return integrated_y.mean(dim=dims)


def weighted_patch_mean(field: torch.Tensor, patch: tuple[int, int, int]) -> torch.Tensor:
    """Map ``(B,T,X,Y)`` fields to ViT token targets in time-major order."""
    if field.ndim != 4:
        raise ValueError(f"expected (B,T,X,Y), got {tuple(field.shape)}")
    pt, px, py = patch
    b, t, x, y = field.shape
    if t % pt or x % px or y % py:
        raise ValueError(f"field {(t,x,y)} not divisible by patch {patch}")
    nt, nx, ny = t // pt, x // px, y // py
    values = field.reshape(b, nt, pt, nx, px, ny, py)
    weights = torch.as_tensor(quad_weights(y, R.RB_LY), dtype=field.dtype, device=field.device)
    weights = weights.reshape(ny, py)
    weights = weights / weights.sum(dim=1, keepdim=True)
    values = (values * weights.reshape(1, 1, 1, 1, 1, ny, py)).sum(dim=-1)
    values = values.mean(dim=(2, 4))
    return values.reshape(b, nt * nx * ny)


def target_fields(raw_clip: torch.Tensor, *, fluctuation_buoyancy: bool = False) -> dict[str, torch.Tensor]:
    raw = _as_float64(raw_clip)
    buoyancy, pressure, u, v = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    if fluctuation_buoyancy:
        buoyancy = buoyancy_fluctuation(buoyancy)

    ux, uy = R.grad(u)
    vx, vy = R.grad(v)
    omega = vx - uy
    normal_strain = ux - vy
    shear_strain = uy + vx
    deformation = normal_strain.square() + shear_strain.square()
    ow = deformation - omega.square()

    return {
        "enstrophy": omega.square(),
        "buoyancy_gradient_energy": R.grad_sq(buoyancy),
        "convective_flux": v * buoyancy,
        "pressure_gradient_magnitude": R.grad_sq(pressure).clamp_min(0).sqrt(),
        "buoyancy_laplacian_magnitude": R.laplacian(buoyancy).abs(),
        "deformation_energy": deformation,
        "vorticity_signed": omega,
        "okubo_weiss_signed": ow,
    }


def pooled_targets(raw_clip: torch.Tensor, *, fluctuation_buoyancy: bool = False) -> dict[str, torch.Tensor]:
    fields = target_fields(raw_clip, fluctuation_buoyancy=fluctuation_buoyancy)
    return {name: volume_mean(fields[name]) for name in PRIMARY_TARGETS}


def token_targets(
    raw_clip: torch.Tensor,
    patch: tuple[int, int, int] = (2, 16, 16),
    *,
    fluctuation_buoyancy: bool = False,
) -> dict[str, torch.Tensor]:
    fields = target_fields(raw_clip, fluctuation_buoyancy=fluctuation_buoyancy)
    return {
        name: weighted_patch_mean(fields[name], patch)
        for name in PRIMARY_TARGETS + TOKEN_ONLY_TARGETS
    }


def detect_onset_from_energy(
    energy: np.ndarray,
    *,
    threshold: float = 0.5,
    tail_fraction: float = 0.25,
    smooth: int = 5,
    sustain: int = 5,
) -> int:
    """First sustained crossing of half the late-time kinetic-energy plateau."""
    values = np.asarray(energy, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("energy must be a non-empty one-dimensional array")
    if smooth < 1 or sustain < 1:
        raise ValueError("smooth and sustain must be positive")
    if smooth > 1:
        pad_left = smooth // 2
        pad_right = smooth - 1 - pad_left
        padded = np.pad(values, (pad_left, pad_right), mode="edge")
        values = np.array([np.median(padded[i : i + smooth]) for i in range(values.size)])
    n_tail = max(1, int(np.ceil(values.size * tail_fraction)))
    plateau = float(np.median(values[-n_tail:]))
    if not np.isfinite(plateau) or plateau <= 0:
        return 0
    cutoff = threshold * plateau
    above = values >= cutoff
    for idx in range(0, max(0, len(above) - sustain + 1)):
        if bool(np.all(above[idx : idx + sustain])):
            return idx
    raise ValueError("trajectory never reaches a sustained developed-flow plateau")


def kinetic_energy_series(raw_trajectory: torch.Tensor) -> np.ndarray:
    """Volume-weighted kinetic energy for a raw ``(4,T,X,Y)`` trajectory."""
    if raw_trajectory.ndim != 4 or raw_trajectory.shape[0] != 4:
        raise ValueError(f"expected (4,T,X,Y), got {tuple(raw_trajectory.shape)}")
    raw = raw_trajectory.to(torch.float64)
    energy = 0.5 * (raw[2].square() + raw[3].square())
    weights = torch.as_tensor(quad_weights(R.RB_NY, R.RB_LY), dtype=energy.dtype, device=energy.device)
    return ((energy * weights).sum(dim=-1).mean(dim=-1) / R.RB_LY).cpu().numpy()

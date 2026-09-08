"""RB derivatives on the corrected Fourier-x / Chebyshev-Gauss-y grid.

The published HDF5 coordinate arrays include periodic endpoints and uniformly
spaced y values inconsistent with the simulation fields. The conductive profile
and divergence checks establish the geometry below. Never differentiate encoder
inputs: these operators take raw physical fields, with spatial axes (x, y).
"""

import functools
import torch

RB_NX, RB_LX = 512, 4.0
RB_NY, RB_LY = 128, 1.0
RB_DX = RB_LX / RB_NX


def cheb_gauss_nodes(
    n: int = RB_NY, length: float = RB_LY, dtype=torch.float64
) -> torch.Tensor:
    """Chebyshev-Gauss nodes on [0, length]: y_i = L*(1 - cos(pi*(i+1/2)/n))/2.

    Gauss, not Gauss-Lobatto: the walls themselves are not sampled (the first
    node sits 3.8e-5 from the wall), which is why velocity there is ~2e-8
    rather than exactly 0. Gauss-Lobatto fits the observed profile 17x worse.
    """
    i = torch.arange(n, dtype=dtype)
    return 0.5 * length * (1 - torch.cos(torch.pi * (i + 0.5) / n))


@functools.lru_cache(maxsize=8)
def _cheb_diffmat(n: int, length: float) -> torch.Tensor:
    """Barycentric differentiation matrix on Chebyshev-Gauss nodes, float64.

    Spectrally accurate: exact to ~1e-12 on smooth test functions (self_test).
    Barycentric weights for Gauss nodes are w_j = (-1)^j sin((2j+1)pi/2n),
    which are O(1) -- computing them from the node products instead would
    overflow at n=128.
    """
    i = torch.arange(n, dtype=torch.float64)
    y = cheb_gauss_nodes(n, length)
    w = (-1.0) ** i * torch.sin((2 * i + 1) * torch.pi / (2 * n))
    dy = y[:, None] - y[None, :]
    dy.fill_diagonal_(1.0)
    d = (w[None, :] / w[:, None]) / dy
    d.fill_diagonal_(0.0)
    d.diagonal().copy_(-d.sum(dim=1))  # negative-sum trick: rows annihilate constants
    return d


def _check_shape(f: torch.Tensor) -> None:
    if f.shape[-2] != RB_NX or f.shape[-1] != RB_NY:
        raise ValueError(
            f"expected (..., {RB_NX}, {RB_NY}) with x on axis -2 and y on axis -1, "
            f"got {tuple(f.shape)}. This module is rayleigh_benard-specific; the axis "
            f"order is not interchangeable (x is periodic, y has walls)."
        )


def ddx(f: torch.Tensor) -> torch.Tensor:
    """d/dx along axis -2: spectral, exploiting exact periodicity."""
    _check_shape(f)
    n = f.shape[-2]
    fh = torch.fft.rfft(f.double(), dim=-2)
    k = (
        2
        * torch.pi
        * torch.fft.rfftfreq(n, d=RB_LX / n, dtype=torch.float64, device=f.device)
    )
    shape = [1] * f.dim()
    shape[-2] = k.numel()
    out = torch.fft.irfft(fh * (1j * k.reshape(shape)), n=n, dim=-2)
    return out.to(f.dtype)


def ddy(f: torch.Tensor) -> torch.Tensor:
    """d/dy along axis -1: Chebyshev spectral differentiation, non-periodic."""
    _check_shape(f)
    d = _cheb_diffmat(f.shape[-1], RB_LY).to(f.device)
    return (f.double() @ d.transpose(0, 1)).to(f.dtype)


def d2dx2(f: torch.Tensor) -> torch.Tensor:
    _check_shape(f)
    n = f.shape[-2]
    fh = torch.fft.rfft(f.double(), dim=-2)
    k = (
        2
        * torch.pi
        * torch.fft.rfftfreq(n, d=RB_LX / n, dtype=torch.float64, device=f.device)
    )
    shape = [1] * f.dim()
    shape[-2] = k.numel()
    return torch.fft.irfft(fh * (-(k**2).reshape(shape)), n=n, dim=-2).to(f.dtype)


def d2dy2(f: torch.Tensor) -> torch.Tensor:
    return ddy(ddy(f))


def laplacian(f: torch.Tensor) -> torch.Tensor:
    return d2dx2(f) + d2dy2(f)


def grad(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ddx(f), ddy(f)


def grad_sq(f: torch.Tensor) -> torch.Tensor:
    gx, gy = grad(f)
    return gx**2 + gy**2


def curl(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vorticity du_y/dx - du_x/dy. `u` is the x-component (channel 2), `v` the
    y-component (channel 3)."""
    return ddx(v) - ddy(u)


def divergence(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return ddx(u) + ddy(v)

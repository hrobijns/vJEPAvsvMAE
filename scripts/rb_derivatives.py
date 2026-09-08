"""Spatial derivatives on rayleigh_benard's ACTUAL grid. One utility, used by every target.

THE SHIPPED COORDINATE ARRAYS ARE WRONG -- DO NOT READ THEM
-----------------------------------------------------------
The Well's rayleigh_benard HDF5 files carry `dimensions/x` and `dimensions/y`
arrays that do not describe the data:

  * `dimensions/y` reports a UNIFORM grid (dy = 1/127 at every point). The data
    is on CHEBYSHEV-GAUSS nodes, whose spacing varies by a factor of 40.7 across
    the layer.
  * `dimensions/x` reports 0 -> 4 inclusive (dx = 4/511), i.e. a duplicated
    endpoint. The true grid is periodic with 512 points and dx = 4/512; using
    the duplicated-endpoint convention leaves a residual divergence of 3.4e-2
    instead of ~1e-6.

Both were established from the physics, not from metadata (see self_test):

  1. The t=0 conductive buoyancy profile is linear in PHYSICAL y. Inverting it
     recovers the vertical grid, and it matches Chebyshev-Gauss to R^2 =
     0.99999999 with slope exactly -0.2 and intercept exactly +0.2, against
     R^2 = 0.9855 for a uniform grid.
  2. Velocity vanishes at the first and last vertical node (3e-7 of the
     interior mean) and not at the ends of the other axis -- fixing which axis
     carries the no-slip walls.
  3. Under this grid and axis assignment, and only under it, the velocity field
     is divergence-free to 1.9e-6 at Ra=1e6.

So the constants below are hard-coded on purpose. Reading the files' own
coordinate arrays is what produced an earlier, wrong conclusion that the
dataset was not divergence-free.

AXIS CONVENTION
---------------
Arrays are (..., X, Y) with X = axis -2 = 512 points, periodic, dx = 4/512;
Y = axis -1 = 128 points, Chebyshev-Gauss on [0, 1], no-slip walls. This is the
memmap's own layout: (n_traj, C, T, H, W) has H = 512 = x and W = 128 = y.

NOTE this is the opposite of `grad2d`/`curl2d`/`divergence2d` in
analyze_encoders.py, which take d/dx along axis -1 and d/dy along axis -2, use
uniform spacing vertically, and wrap periodically across the no-slip walls.
Those are retained unchanged so existing results stay reproducible; see
scripts/audit_rb_targets.py for a quantified old-vs-corrected comparison.

WORK FROM RAW PHYSICAL FIELDS
-----------------------------
The memmap stores fields z-scored per channel, and the two velocity components
carry DIFFERENT standard deviations (0.2077 and 0.1795). Normalised velocity is
therefore not solenoidal even though the physical velocity is -- the normalised
divergence is off by ~16% by construction. Every target here is computed after
`denormalize()`, on physical fields.
"""

import functools

import torch

# --- grid, hard-coded (see module docstring) --------------------------------
RB_NX, RB_LX = 512, 4.0          # x: periodic, dx = LX / NX, no duplicated endpoint
RB_NY, RB_LY = 128, 1.0          # y: Chebyshev-Gauss on [0, LY], walls not sampled
RB_DX = RB_LX / RB_NX

# The Well's z-score stats for rayleigh_benard (stats.yaml), channel order
# [buoyancy, pressure, velocity_x, velocity_y]. ZScoreNormalization uses
# `mean`/`std` (not `rms`), clipped below at 1e-4 -- none of these are clipped.
RB_MEAN = (3.7085e-01, 7.8417e-03, 2.1672e-05, 1.0063e-13)
RB_STD = (2.4928e-01, 1.4932e-01, 2.0773e-01, 1.7953e-01)


def cheb_gauss_nodes(n: int = RB_NY, length: float = RB_LY,
                      dtype=torch.float64) -> torch.Tensor:
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
    d.diagonal().copy_(-d.sum(dim=1))     # negative-sum trick: rows annihilate constants
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
    k = 2 * torch.pi * torch.fft.rfftfreq(n, d=RB_LX / n, dtype=torch.float64)
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
    k = 2 * torch.pi * torch.fft.rfftfreq(n, d=RB_LX / n, dtype=torch.float64)
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


def okubo_weiss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """normal-strain^2 + shear-strain^2 - vorticity^2."""
    ux, uy = grad(u)
    vx, vy = grad(v)
    return (ux - vy) ** 2 + (vx + uy) ** 2 - (vx - uy) ** 2


def denormalize(clip: torch.Tensor) -> torch.Tensor:
    """Undo The Well's per-channel z-score: raw = normalised * std + mean.

    clip: (..., C, T, X, Y) with C in [buoyancy, pressure, u_x, u_y] order.
    Required before differentiating: the two velocity components are scaled by
    different constants, so normalised velocity is not solenoidal (see module
    docstring).
    """
    if clip.shape[-4] != len(RB_MEAN):
        raise ValueError(f"expected {len(RB_MEAN)} channels on axis -4, got {clip.shape[-4]}")
    mean = torch.tensor(RB_MEAN, dtype=clip.dtype, device=clip.device).reshape(-1, 1, 1, 1)
    std = torch.tensor(RB_STD, dtype=clip.dtype, device=clip.device).reshape(-1, 1, 1, 1)
    return clip * std + mean


# ---------------------------------------------------------------------------
# Regression tests grounded in physics, not in "does the function run"
# ---------------------------------------------------------------------------

def _rel(a: torch.Tensor, *terms: torch.Tensor) -> float:
    """||a|| / rms of the term magnitudes -- a scale-free residual."""
    scale = torch.sqrt(torch.stack([(t**2).mean() for t in terms]).mean())
    return float(torch.sqrt((a**2).mean()) / (scale + 1e-300))


def self_test(data_root: str | None = None, dataset: str = "rayleigh_benard") -> bool:
    """Analytic checks always; physics checks against real data when --data-root
    is given. The physics checks are the ones that matter: each is also run
    against the OLD operators as a negative control, so a test that both
    versions pass would be visibly worthless."""
    import numpy as np
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
        ok = ok and bool(cond)

    x = (torch.arange(RB_NX, dtype=torch.float64) * RB_DX).reshape(-1, 1)
    y = cheb_gauss_nodes().reshape(1, -1)

    print("1. spectral accuracy of each operator on analytic fields")
    for kx in (1, 4, 16):
        f = torch.sin(2 * torch.pi * kx * x / RB_LX) * torch.ones_like(y)
        d = 2 * torch.pi * kx / RB_LX * torch.cos(2 * torch.pi * kx * x / RB_LX) * torch.ones_like(y)
        check(f"ddx exact for wavenumber {kx}", _rel(ddx(f) - d, d) < 1e-12,
              f"rel err {_rel(ddx(f)-d, d):.2e}")
    for name, f, d in [("sin(2*pi*y)", torch.sin(2 * torch.pi * y), 2 * torch.pi * torch.cos(2 * torch.pi * y)),
                        ("exp(y)", torch.exp(y), torch.exp(y)),
                        ("y**5", y**5, 5 * y**4)]:
        f = f * torch.ones_like(x); d = d * torch.ones_like(x)
        check(f"ddy spectral for {name}", _rel(ddy(f) - d, d) < 1e-9, f"rel err {_rel(ddy(f)-d, d):.2e}")

    print("2. vector-calculus identities on an analytic divergence-free field")
    psi = torch.sin(2 * torch.pi * x / RB_LX) * torch.sin(torch.pi * y)
    u, v = ddy(psi), -ddx(psi)
    check("div of a streamfunction field vanishes", _rel(divergence(u, v), ddx(u), ddy(v)) < 1e-10,
          f"{_rel(divergence(u,v), ddx(u), ddy(v)):.2e}")
    check("curl(u,v) == -laplacian(psi)", _rel(curl(u, v) + laplacian(psi), curl(u, v)) < 1e-9,
          f"{_rel(curl(u,v)+laplacian(psi), curl(u,v)):.2e}")
    check("curl of a gradient vanishes", _rel(curl(ddx(psi), ddy(psi)), ddx(psi)) < 1e-9)

    print("3. the recovered grid IS Chebyshev-Gauss, not uniform or Lobatto")
    i = np.arange(RB_NY)
    nodes = cheb_gauss_nodes().numpy()
    lob = 0.5 * (1 - np.cos(np.pi * i / (RB_NY - 1)))
    check("differs materially from a uniform grid", np.abs(nodes - i / (RB_NY - 1)).max() > 0.05,
          f"max diff {np.abs(nodes - i/(RB_NY-1)).max():.3f}")
    check("differs from Gauss-Lobatto", np.abs(nodes - lob).max() > 1e-4)
    check("spacing ratio ~40x (uniform would be 1)",
          38 < np.diff(nodes).max() / np.diff(nodes).min() < 44,
          f"{np.diff(nodes).max()/np.diff(nodes).min():.1f}x")

    if data_root is None:
        print("\n(skipping physics checks -- pass --data-root to run them against real data)")
        print("\nself-test: ALL PASS" if ok else "\nself-test: FAILURES ABOVE")
        return ok

    from pathlib import Path
    from scripts.analyze_encoders import curl2d as old_curl, divergence2d as old_div
    mm = np.load(Path(data_root) / "memmap" / dataset / "test.npy", mmap_mode="r")
    recs_path = Path(data_root) / "memmap" / dataset / "test.regime.json"
    import json
    recs = json.loads(recs_path.read_text()) if recs_path.exists() else None

    print("\n4. PHYSICS: incompressibility of the physical velocity field")
    # lowest available Rayleigh: best-resolved, so the cleanest test of the operators
    low = 0
    if recs is not None:
        ras = [r["Rayleigh"] for r in recs]
        low = int(np.argmin(ras))
        print(f"   using trajectory {low} (Rayleigh {ras[low]:.0e}, the best-resolved regime)")
    clip = torch.from_numpy(np.array(mm[low, :, 80:84])).float()
    raw = denormalize(clip)
    u, v = raw[2], raw[3]
    r_new = _rel(divergence(u, v), ddx(u), ddy(v))

    # What residual can fp16 STORAGE alone produce? preprocess_memmap.py writes
    # float16, and divergence is a near-total cancellation of two large terms,
    # which the Chebyshev operator further amplifies near the walls (node
    # spacing 3e-4 there, so d/dy carries a ~3e3 gain). Quantise an analytic
    # divergence-free field the same way to get the floor this test can reach;
    # anything at that floor means the OPERATOR is exact and the storage is the
    # limit, which is the real claim being made.
    psi = (torch.sin(2 * torch.pi * x / RB_LX) * torch.sin(torch.pi * y)).expand(RB_NX, RB_NY)
    ua, va = ddy(psi), -ddx(psi)
    scale = float(ua.abs().max())
    uq = (ua / scale).half().float() * scale
    vq = (va / scale).half().float() * scale
    floor = _rel(divergence(uq, vq), ddx(uq), ddy(vq))
    check("residual is at the fp16-storage floor (operator itself is exact)",
          r_new < max(5 * floor, 1e-3),
          f"measured {r_new:.3e} vs fp16 floor {floor:.3e}")

    # NEGATIVE CONTROL -- the old operators must fail, or this test proves nothing.
    # Scored against ITS OWN terms: old_div uses unit grid spacing, so comparing
    # it to properly-scaled terms would flatter it by a factor of ~1/dx.
    o_t1 = (torch.roll(u, -1, -1) - torch.roll(u, 1, -1)) / 2
    o_t2 = (torch.roll(v, -1, -2) - torch.roll(v, 1, -2)) / 2
    r_old = _rel(old_div(u, v), o_t1, o_t2)
    check("negative control: OLD operators fail the same test", r_old > 20 * r_new,
          f"old {r_old:.3f} vs corrected {r_new:.3e}")
    # and on NORMALISED velocity, which is not solenoidal BY CONSTRUCTION.
    # With sx != sy, div(u/sx, v/sy) = D*(1/sx - 1/sy) where D = du/dx = -dv/dy,
    # while the terms have magnitude ~D/mean(sx,sy) -- so the relative residual
    # should land at |sx - sy| / mean(sx, sy), independent of the flow. Asserting
    # that predicted VALUE (not merely "it is large") is what makes this a real
    # check on the denormalize() step rather than a vague smell test.
    sx, sy = RB_STD[2], RB_STD[3]
    predicted = abs(sx - sy) / (0.5 * (sx + sy))
    r_norm = _rel(divergence(clip[2], clip[3]), ddx(clip[2]), ddy(clip[3]))
    check("negative control: normalised velocity is non-solenoidal by exactly the "
          "sigma mismatch", abs(r_norm - predicted) < 0.25 * predicted,
          f"measured {r_norm:.4f} vs predicted |sx-sy|/mean = {predicted:.4f}")
    check("...and that is far above the corrected residual", r_norm > 5 * r_new,
          f"{r_norm:.3f} vs {r_new:.3e}")

    # Strongest form of the check, when the raw HDF5 files are also present:
    # float32 fields straight from disk, no fp16 round-trip.
    raw_dir = Path(data_root) / "datasets" / dataset / "data" / "test"
    files = sorted(raw_dir.glob("*.hdf5")) if raw_dir.is_dir() else []
    if files:
        import h5py, re
        pick = min(files, key=lambda f: float(re.search(r"Rayleigh_([0-9eE.+-]+)_", f.name).group(1)))
        with h5py.File(pick, "r") as fh:
            vel = torch.from_numpy(fh["t1_fields/velocity"][0, 150:152].astype("float64"))
        ur, vr = vel[..., 0], vel[..., 1]
        r_raw = _rel(divergence(ur, vr), ddx(ur), ddy(vr))
        check(f"raw float64 fields ({pick.name.split('_Prandtl')[0].split('_')[-1]}): "
              f"||div u||/||terms|| < 1e-4", r_raw < 1e-4, f"{r_raw:.3e}")

    print("\n5. PHYSICS: the t=0 conductive buoyancy profile is linear in Chebyshev y")
    b0 = denormalize(torch.from_numpy(np.array(mm[low, :, 0:1])).float())[0, 0]
    prof = b0.mean(dim=0).double().numpy()
    nodes = cheb_gauss_nodes().numpy()
    A = np.vstack([nodes, np.ones_like(nodes)]).T
    coef, *_ = np.linalg.lstsq(A, prof, rcond=None)
    r2 = 1 - ((prof - A @ coef) ** 2).sum() / ((prof - prof.mean()) ** 2).sum()
    check("linear in Chebyshev y with R^2 > 0.9999", r2 > 0.9999, f"R^2 = {r2:.8f}")
    check("slope ~= -0.2 (imposed temperature drop)", abs(coef[0] + 0.2) < 5e-3, f"{coef[0]:+.5f}")
    check("intercept ~= +0.2 (hot wall value)", abs(coef[1] - 0.2) < 5e-3, f"{coef[1]:+.5f}")
    Au = np.vstack([np.arange(RB_NY) / (RB_NY - 1), np.ones(RB_NY)]).T
    cu, *_ = np.linalg.lstsq(Au, prof, rcond=None)
    r2u = 1 - ((prof - Au @ cu) ** 2).sum() / ((prof - prof.mean()) ** 2).sum()
    check("negative control: a UNIFORM grid fits the profile worse", r2u < r2 - 1e-3,
          f"uniform R^2 = {r2u:.6f} vs Chebyshev {r2:.8f}")

    print("\n6. PHYSICS: no-slip at the vertical walls, and only there")
    dev = denormalize(torch.from_numpy(np.array(mm[low, :, 80:84])).float())
    for c, lbl in ((2, "u_x"), (3, "u_y")):
        f = dev[c]
        wall = max(f[..., 0].abs().mean(), f[..., -1].abs().mean()) / f.abs().mean()
        check(f"{lbl} vanishes at the y walls", wall < 1e-2, f"ratio {wall:.2e}")
    xends = max(dev[2][..., 0, :].abs().mean(), dev[2][..., -1, :].abs().mean()) / dev[2].abs().mean()
    check("negative control: u_x is NOT pinned at the x boundaries (periodic)", xends > 0.1,
          f"ratio {xends:.2f}")

    print("\nself-test: ALL PASS" if ok else "\nself-test: FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=None,
                     help="run the physics checks against a preprocessed memmap")
    ap.add_argument("--dataset", default="rayleigh_benard")
    a = ap.parse_args()
    raise SystemExit(0 if self_test(a.data_root, a.dataset) else 1)

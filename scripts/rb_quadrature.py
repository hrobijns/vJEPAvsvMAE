"""Chebyshev-Gauss vertical quadrature for the RB y grid.

The dataset's y nodes are Chebyshev-Gauss (ROOTS) points -- established by
physics reconstruction, not by trusting the shipped coordinate metadata:

    theta_i = pi*(i+0.5)/n ,  y_i = 0.5*Ly*(1 - cos(theta_i)) ,  i = 0..n-1

An arithmetic mean over these nodes is NOT a vertical integral: the spacing
varies by ~50x between wall and centre, so a plain mean massively over-weights
the boundary layers.

Two INDEPENDENT constructions of the weights for  int_0^Ly f(y) dy:

  (1) Fejer's first rule -- the classical quadrature attached to the Chebyshev
      roots grid, with weights available in closed form:
          w_i = (2/n) [ 1 - 2 * sum_{m=1}^{floor(n/2)} cos(2 m theta_i)/(4m^2-1) ]
      on [-1,1]. Sum(w) = 2 exactly, because sum_i cos(2 m theta_i) = 0 for
      1 <= m <= n-1 by discrete orthogonality.

  (2) Moment matching in the Chebyshev basis -- build V[i,k] = T_k(x_i), use the
      analytic moments  int_{-1}^{1} T_k dx = 0 (k odd), 2/(1-k^2) (k even),
      and solve  w = V^{-T} m.

These share no code path; agreement to round-off is the check.
"""
import functools
import numpy as np
import torch

RB_NY, RB_LY = 128, 1.0


def cheb_gauss_nodes(n=RB_NY, length=RB_LY):
    i = np.arange(n)
    return 0.5 * length * (1.0 - np.cos(np.pi * (i + 0.5) / n))


def _fejer1_weights(n):
    """Fejer's first rule on [-1,1]; sum = 2."""
    i = np.arange(n)
    th = np.pi * (i + 0.5) / n
    w = np.ones(n)
    for m in range(1, n // 2 + 1):
        w -= 2.0 * np.cos(2.0 * m * th) / (4.0 * m * m - 1.0)
    return (2.0 / n) * w


def _moment_weights(n):
    """Independent route: Chebyshev Vandermonde + analytic moments."""
    i = np.arange(n)
    x = -np.cos(np.pi * (i + 0.5) / n)
    V = np.cos(np.outer(np.arccos(np.clip(x, -1, 1)), np.arange(n)))   # V[i,k]=T_k(x_i)
    m = np.zeros(n)
    for k in range(n):
        m[k] = 0.0 if k % 2 else 2.0 / (1.0 - k * k) if k != 1 else 0.0
    m[0] = 2.0
    return np.linalg.solve(V.T, m)


@functools.lru_cache(maxsize=8)
def quad_weights(n=RB_NY, length=RB_LY, method="fejer"):
    """Weights for int_0^length f(y) dy over the Chebyshev-Gauss nodes."""
    w = _fejer1_weights(n) if method == "fejer" else _moment_weights(n)
    return 0.5 * length * w          # chain rule dx -> dy


def vertical_integral(field, weights=None, axis=-1):
    """int_0^Ly field dy along `axis` (default the wall-normal axis -1)."""
    if weights is None:
        weights = quad_weights()
    w = torch.as_tensor(weights, dtype=field.dtype)
    shape = [1] * field.dim(); shape[axis] = w.numel()
    return (field * w.reshape(shape)).sum(dim=axis)


def self_test(verbose=True):
    ok = True
    def chk(name, cond, detail=""):
        nonlocal ok; ok &= bool(cond)
        if verbose: print(f"  {'PASS' if cond else 'FAIL'} {name}  {detail}")
    n = RB_NY
    wf = quad_weights(n, 1.0, "fejer")
    wm = quad_weights(n, 1.0, "moment")
    y = cheb_gauss_nodes(n, 1.0)

    chk("weights sum to 1 on [0,1] (Fejer)", abs(wf.sum() - 1.0) < 1e-14,
        f"[sum-1 = {wf.sum()-1.0:.3e}]")
    chk("weights sum to 1 on [0,1] (moment)", abs(wm.sum() - 1.0) < 1e-12,
        f"[sum-1 = {wm.sum()-1.0:.3e}]")
    chk("two independent constructions agree",
        np.abs(wf - wm).max() < 1e-11, f"[max|diff| = {np.abs(wf-wm).max():.3e}]")
    chk("all weights positive", (wf > 0).all(), f"[min = {wf.min():.3e}]")

    for name, f, exact in (
        ("f=1        ", np.ones_like(y),        1.0),
        ("f=y        ", y,                      0.5),
        ("f=y^2      ", y**2,                   1.0/3),
        ("f=y^5      ", y**5,                   1.0/6),
        ("f=y^11     ", y**11,                  1.0/12),
        ("f=exp(y)   ", np.exp(y),              np.e - 1.0),
        ("f=sin(3piy)", np.sin(3*np.pi*y),      2.0/(3*np.pi)),
        ("f=1/(1+y)  ", 1.0/(1.0+y),            np.log(2.0)),
    ):
        got = float((wf * f).sum()); err = abs(got - exact) / max(abs(exact), 1e-30)
        chk(f"integrate {name}", err < 1e-12, f"[rel err {err:.2e}]")

    # Negative control: the unweighted mean is NOT an integral.
    # NB f=y is deliberately excluded here -- the Chebyshev-Gauss nodes are
    # symmetric about the midpoint, so the plain mean integrates ANY linear
    # function exactly by symmetry. That is a property of the grid, not evidence
    # the naive mean is acceptable; controls must use functions symmetry cannot
    # rescue.
    for name, f, exact in (("f=y^2", y**2, 1.0/3), ("f=exp(y)", np.exp(y), np.e-1.0),
                           ("f=sin(3piy)", np.sin(3*np.pi*y), 2.0/(3*np.pi))):
        naive = float(f.mean())
        chk(f"negative control: plain mean of {name} is wrong",
            abs(naive - exact) / abs(exact) > 1e-3,
            f"[mean={naive:.6f} vs exact={exact:.6f}, rel err {abs(naive-exact)/abs(exact):.2e}]")

    # scaling
    w2 = quad_weights(n, 2.0, "fejer")
    chk("weights scale with domain length", abs(w2.sum() - 2.0) < 1e-13,
        f"[sum = {w2.sum():.12f}]")
    if verbose:
        print(f"\n  spacing: dy[0]={np.diff(y)[0]:.3e}  dy[mid]={np.diff(y)[n//2]:.3e}  "
              f"ratio {np.diff(y)[n//2]/np.diff(y)[0]:.1f}x")
        print(f"  weights: w[0]={wf[0]:.3e}  w[mid]={wf[n//2]:.3e}  "
              f"ratio {wf[n//2]/wf[0]:.1f}x   (uniform mean would be {1.0/n:.3e} everywhere)")
        print(f"\n  self-test: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)

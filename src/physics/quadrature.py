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
    V = np.cos(np.outer(np.arccos(np.clip(x, -1, 1)), np.arange(n)))  # V[i,k]=T_k(x_i)
    m = np.zeros(n)
    for k in range(n):
        m[k] = 0.0 if k % 2 else 2.0 / (1.0 - k * k) if k != 1 else 0.0
    m[0] = 2.0
    return np.linalg.solve(V.T, m)


@functools.lru_cache(maxsize=8)
def quad_weights(n=RB_NY, length=RB_LY, method="fejer"):
    """Weights for int_0^length f(y) dy over the Chebyshev-Gauss nodes."""
    if method not in ("fejer", "moment"):
        raise ValueError(f"unknown quadrature method: {method}")
    w = _fejer1_weights(n) if method == "fejer" else _moment_weights(n)
    return 0.5 * length * w  # chain rule dx -> dy

"""Small vectorized root-finding / extremum helpers shared by the eclipse code.

These are numerical methods, not cited physics: bisection and three-point
parabolic refinement.  They are written to take a *vectorized* objective
``f(t_array) -> array`` so many roots (all contacts of an observer, all
eclipses of a century) are refined together with one array evaluation per
iteration -- the same pattern the SPICE/ERFA layer is vectorized around.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Objective = Callable[[np.ndarray], np.ndarray]


def bisect(f: Objective, t_lo, t_hi, iterations: int = 30) -> np.ndarray:
    """Bisect ``f`` to a sign change inside each bracket ``[t_lo, t_hi]`` (arrays).

    ``f(t_lo)`` and ``f(t_hi)`` must have opposite signs element-wise; the
    returned array has the same shape as the brackets.  30 iterations shrink a
    30-second bracket to ~30 ns.
    """
    t_lo = np.array(t_lo, dtype=float)
    t_hi = np.array(t_hi, dtype=float)
    f_lo = f(t_lo)
    for _ in range(iterations):
        mid = 0.5 * (t_lo + t_hi)
        f_mid = f(mid)
        same = np.sign(f_mid) == np.sign(f_lo)
        t_lo = np.where(same, mid, t_lo)
        f_lo = np.where(same, f_mid, f_lo)
        t_hi = np.where(same, t_hi, mid)
    return 0.5 * (t_lo + t_hi)


def sign_changes(t: np.ndarray, f: np.ndarray) -> list[tuple[int, bool]]:
    """Indices ``i`` where ``f`` changes sign between ``t[i]`` and ``t[i+1]``.

    Returns ``(i, rising)`` pairs; an exact zero at ``t[i]`` counts as a crossing
    at ``i`` with the direction taken from the next sample.
    """
    out = []
    for i in range(len(t) - 1):
        if f[i] == 0.0:
            out.append((i, bool(f[i + 1] > 0)))
        elif f[i] * f[i + 1] < 0:
            out.append((i, bool(f[i + 1] > f[i])))
    return out


def parabolic_minimum(f: Objective, t0, half_width, iterations: int = 10) -> np.ndarray:
    """Refine minima of ``f`` near each ``t0`` (arrays) by repeated parabolic fits.

    Each iteration evaluates ``f`` at ``t0 - h, t0, t0 + h`` (one call on the
    stacked array), moves ``t0`` to the parabola's vertex (clipped to the
    bracket) and quarters ``h``; ten iterations from a 2-hour ``h`` reach
    ~7 ms.  Assumes ``f`` is smooth and the true minimum lies within
    ``+/- half_width`` of the starting point.
    """
    t0 = np.array(t0, dtype=float)
    h = np.full_like(t0, float(half_width))
    for _ in range(iterations):
        pts = np.stack([t0 - h, t0, t0 + h])
        y = f(pts.ravel()).reshape(3, -1)
        denom = y[0] - 2.0 * y[1] + y[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            step = 0.5 * h * (y[0] - y[2]) / denom
        step = np.where(denom > 0, np.clip(step, -h, h), 0.0)
        t0 = t0 + step
        h = h / 4.0
    return t0

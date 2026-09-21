"""Delta-T (TT - UT1) model for epochs outside the IERS / leap-second era.

Within the IERS Bulletin A table (1973 - ~1 yr ahead) the code derives TT - UT1
exactly from the SPICE leap-second kernel plus the measured ``UT1-UTC``
(:mod:`app.eop`).  Outside it neither exists: UTC is undefined before 1972 and
future leap seconds are unknown, so SPICE would silently hold the first/last
leap-second count (ET-UTC = 41.18 s in 1900, where the true delta-T is -2.8 s).

This module provides the polynomial delta-T expressions of Espenak & Meeus
[Espenak] (Five Millennium Canon, sec. 2.5, "Polynomial expressions for
delta-T"; also NASA eclipse web site, ``SEhelp/deltatpoly2004.html``), fitted
to the historical record of Morrison & Stephenson (2004) and extended by a
parabola with the long-term tidal acceleration.  They are the delta-T used by
the Canon itself, so historical eclipse geometry is consistent with it.

Units: input decimal year (e.g. 1919.41); output delta-T in **seconds**.
"""

from __future__ import annotations

import numpy as np


def delta_t_seconds(year: float | np.ndarray) -> float | np.ndarray:
    """Delta-T = TT - UT1 [s] at decimal ``year`` (Espenak & Meeus polynomials [Espenak]).

    Piecewise polynomial in ``y``; the segment boundaries and coefficients are
    those of [Espenak] sec. 2.5 (also NASA ``deltatpoly2004``).  Continuous at the
    boundaries to ~1 s.  Vectorized over ``year``.
    """
    y = np.asarray(year, dtype=float)
    out = np.empty_like(y)

    def seg(lo, hi, f):
        mask = (y >= lo) & (y < hi)
        if mask.any():
            out[mask] = f(y[mask])

    u = lambda yy: (yy - 1820.0) / 100.0  # noqa: E731  parabola argument
    seg(-np.inf, -500.0, lambda yy: -20.0 + 32.0 * u(yy) ** 2)
    seg(-500.0, 500.0, lambda yy: np.polyval(
        [0.0090316521, 0.022174192, -0.1798452, -5.952053, 33.78311, -1014.41, 10583.6],
        yy / 100.0))
    seg(500.0, 1600.0, lambda yy: np.polyval(
        [0.0083572073, -0.005050998, -0.8503463, 0.319781, 71.23472, -556.01, 1574.2],
        (yy - 1000.0) / 100.0))
    seg(1600.0, 1700.0, lambda yy: np.polyval(
        [1.0 / 7129.0, -0.01532, -0.9808, 120.0], yy - 1600.0))
    seg(1700.0, 1800.0, lambda yy: np.polyval(
        [-1.0 / 1174000.0, 0.00013336, -0.0059285, 0.1603, 8.83], yy - 1700.0))
    seg(1800.0, 1860.0, lambda yy: np.polyval(
        [0.000000000875, -0.0000001699, 0.0000121272, -0.00037436, 0.0041116,
         0.0068612, -0.332447, 13.72], yy - 1800.0))
    seg(1860.0, 1900.0, lambda yy: np.polyval(
        [1.0 / 233174.0, -0.0004473624, 0.01680668, -0.251754, 0.5737, 7.62], yy - 1860.0))
    seg(1900.0, 1920.0, lambda yy: np.polyval(
        [-0.000197, 0.0061966, -0.0598939, 1.494119, -2.79], yy - 1900.0))
    seg(1920.0, 1941.0, lambda yy: np.polyval(
        [0.0020936, -0.076100, 0.84493, 21.20], yy - 1920.0))
    seg(1941.0, 1961.0, lambda yy: np.polyval(
        [1.0 / 2547.0, -1.0 / 233.0, 0.407, 29.07], yy - 1950.0))
    seg(1961.0, 1986.0, lambda yy: np.polyval(
        [-1.0 / 718.0, -1.0 / 260.0, 1.067, 45.45], yy - 1975.0))
    seg(1986.0, 2005.0, lambda yy: np.polyval(
        [0.00002373599, 0.000651814, 0.0017275, -0.060374, 0.3345, 63.86], yy - 2000.0))
    seg(2005.0, 2050.0, lambda yy: np.polyval([0.005589, 0.32217, 62.92], yy - 2000.0))
    seg(2050.0, 2150.0, lambda yy: -20.0 + 32.0 * u(yy) ** 2 - 0.5628 * (2150.0 - yy))
    seg(2150.0, np.inf, lambda yy: -20.0 + 32.0 * u(yy) ** 2)

    return float(out) if out.ndim == 0 else out


def decimal_year_from_jd(jd: float | np.ndarray) -> float | np.ndarray:
    """Decimal year of a Julian date (any uniform scale; the model is smooth).

    ``2000.0 + (JD - 2451545.0) / 365.25``: J2000.0 is 2000 Jan 1.5, i.e. year
    2000.0 to the precision that matters for a polynomial spanning centuries.
    """
    return 2000.0 + (np.asarray(jd, dtype=float) - 2451545.0) / 365.25

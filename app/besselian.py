"""Besselian element polynomials for a solar eclipse.

The elements are sampled at a few epochs bracketing a reference time ``T0`` and
each is fit with a low-order polynomial in ``t`` = hours from ``T0`` (the usual
tabular form, e.g. Fred Espenak's eclipse bulletins).  Sampling and geometry
come from :mod:`app.ephemeris` (SPICE / JPL DE440).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ephemeris import (
    DEFAULT_EARTH_FRAME,
    BesselianInstant,
    besselian_instant,
    load_kernels,
    utc_to_et,
)

# Polynomial degree used for each element (Espenak convention: x, y cubic;
# d, l1, l2 quadratic; mu linear).  tan_f1 / tan_f2 are treated as constants.
_DEGREES = {"x": 3, "y": 3, "d": 2, "l1": 2, "l2": 2, "mu": 1}


@dataclass
class BesselianPolynomials:
    """Polynomial coefficients (increasing powers of ``t`` in hours from T0)."""

    t0_utc: str
    x: list[float]
    y: list[float]
    d: list[float]
    l1: list[float]
    l2: list[float]
    mu: list[float]
    tan_f1: float
    tan_f2: float


@dataclass
class BesselianModel:
    t0_utc: str
    earth_frame: str = DEFAULT_EARTH_FRAME
    half_window_hours: float = 2.0
    step_hours: float = 1.0
    samples: list[BesselianInstant] = field(default_factory=list, repr=False)
    polynomials: BesselianPolynomials | None = None

    def __post_init__(self) -> None:
        load_kernels()
        et0 = utc_to_et(self.t0_utc)
        offsets = np.arange(
            -self.half_window_hours,
            self.half_window_hours + self.step_hours / 2,
            self.step_hours,
        )
        self.samples = [
            besselian_instant(et0 + off * 3600.0, self.earth_frame) for off in offsets
        ]
        self.polynomials = self._fit(offsets)

    def _fit(self, offsets: np.ndarray) -> BesselianPolynomials:
        def col(name: str) -> np.ndarray:
            return np.array([getattr(s, name) for s in self.samples], dtype=float)

        # mu is an angle: unwrap so the linear fit is not corrupted by a
        # 360-degree wrap inside the sampling window.
        mu = np.degrees(np.unwrap(np.radians(col("mu"))))

        def fit(values: np.ndarray, deg: int) -> list[float]:
            return np.polynomial.polynomial.polyfit(offsets, values, deg).tolist()

        return BesselianPolynomials(
            t0_utc=self.t0_utc,
            x=fit(col("x"), _DEGREES["x"]),
            y=fit(col("y"), _DEGREES["y"]),
            d=fit(col("d"), _DEGREES["d"]),
            l1=fit(col("l1"), _DEGREES["l1"]),
            l2=fit(col("l2"), _DEGREES["l2"]),
            mu=fit(mu, _DEGREES["mu"]),
            tan_f1=float(np.mean(col("tan_f1"))),
            tan_f2=float(np.mean(col("tan_f2"))),
        )

    def evaluate(self, t_hours: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate every element from the polynomial fit at ``t_hours`` (hours from T0).

        ``x, y, l1, l2`` are in Earth radii; ``d, mu`` in degrees; ``tan_f1,
        tan_f2`` are dimensionless (broadcast to the shape of ``t_hours``).

        This is the tabular polynomial product (Espenak convention) and is valid
        only **within** the sampling window ``[-half_window_hours,
        +half_window_hours]``; it extrapolates (and degrades) outside it.  For the
        central line and for contact bracketing, which may run past the window,
        use :meth:`evaluate_direct` instead (item A2).
        """
        p = self.polynomials
        assert p is not None
        t = np.asarray(t_hours, dtype=float)
        pv = np.polynomial.polynomial.polyval
        return {
            "x": pv(t, p.x),
            "y": pv(t, p.y),
            "d": pv(t, p.d),
            "mu": pv(t, p.mu),
            "l1": pv(t, p.l1),
            "l2": pv(t, p.l2),
            "tan_f1": np.full_like(t, p.tan_f1),
            "tan_f2": np.full_like(t, p.tan_f2),
        }

    def evaluate_direct(self, t_hours: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate every element DIRECTLY from the ephemeris at each ``t_hours``.

        Same units and dict shape as :meth:`evaluate`, but each instant is
        recomputed with :func:`app.ephemeris.besselian_instant` rather than read
        off the polynomial fit.  It is therefore exact at every instant and does
        **not** degrade outside the sampling window, so it is the correct choice
        for the central line and for bracketing contacts C1/C4 that may fall
        beyond ``+/-half_window_hours`` (review item A2).  The polynomial
        :meth:`evaluate` remains the ``/besselian`` tabular deliverable.

        ``t_hours`` is treated as a 1-D sequence; results are 1-D arrays.
        """
        et0 = utc_to_et(self.t0_utc)
        t = np.atleast_1d(np.asarray(t_hours, dtype=float))
        instants = [
            besselian_instant(et0 + ti * 3600.0, self.earth_frame) for ti in t
        ]
        # mu is an angle: unwrap so a 360-degree wrap inside the range does not
        # leave a discontinuity (harmless for the modular use in geography, but
        # keeps the array continuous and consistent with evaluate()).
        mu = np.degrees(np.unwrap([np.radians(bi.mu) for bi in instants]))
        return {
            "x": np.array([bi.x for bi in instants]),
            "y": np.array([bi.y for bi in instants]),
            "d": np.array([bi.d for bi in instants]),
            "mu": mu,
            "l1": np.array([bi.l1 for bi in instants]),
            "l2": np.array([bi.l2 for bi in instants]),
            "tan_f1": np.array([bi.tan_f1 for bi in instants]),
            "tan_f2": np.array([bi.tan_f2 for bi in instants]),
        }

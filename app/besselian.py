"""Besselian element polynomials for a solar eclipse.

The elements are sampled at a few epochs bracketing a reference time ``T0`` and
each is fit with a low-order polynomial in ``t`` = hours from ``T0`` (the usual
tabular form, e.g. Fred Espenak's eclipse bulletins).  Sampling and geometry
come from :mod:`app.ephemeris` (SPICE / a JPL DE ephemeris; the GitHub mirror
ships DE432s, the NAIF source DE440).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .ephemeris import (
    DEFAULT_EARTH_FRAME,
    BesselianInstant,
    besselian_instant,
    besselian_instants,
    load_kernels,
    utc_to_et,
)

# Polynomial degree used for each element (Espenak convention: x, y cubic;
# d, l1, l2 quadratic; mu linear).  tan_f1 / tan_f2 are treated as constants.
_DEGREES = {"x": 3, "y": 3, "d": 2, "l1": 2, "l2": 2, "mu": 1}

# Earth-orientation frames in preference order: ITRS (ERFA + IERS EOP; needs no
# binary PCK) first, then the binary-PCK ITRF93, the PCK-free TOD, and the coarse
# IAU_EARTH fallback.  Which are usable depends on the furnished kernels.
_FRAME_PREFERENCE = ("ITRS", "ITRF93", "TOD", "IAU_EARTH")


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


def normalize_utc(epoch: str) -> str:
    """Validate an epoch and return it as naive ISO-8601 UTC (``YYYY-MM-DDTHH:MM:SS[.fff]``).

    Accepts what ``datetime.fromisoformat`` accepts plus a trailing ``Z``; an
    explicit offset is converted to UTC.  This is the single epoch parser: SPICE
    reads the normalized string and :func:`app.geography.format_clock` formats
    it, so the two can never disagree on what is a valid epoch (a SPICE-only
    format such as ``2024 APR 08`` used to pass model construction and then
    crash the clock formatter with a 500).  Raises ``ValueError``.
    """
    s = epoch.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="milliseconds" if dt.microsecond else "seconds")


@dataclass
class BesselianModel:
    t0_utc: str
    earth_frame: str = DEFAULT_EARTH_FRAME
    half_window_hours: float = 2.0
    step_hours: float = 1.0
    samples: list[BesselianInstant] = field(default_factory=list, repr=False)
    polynomials: BesselianPolynomials | None = None
    et0: float = field(init=False, repr=False)  # TDB seconds past J2000 of t0_utc

    def __post_init__(self) -> None:
        self.t0_utc = normalize_utc(self.t0_utc)
        load_kernels()
        et0 = utc_to_et(self.t0_utc)
        self.et0 = et0
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

        ``t_hours`` is treated as a 1-D sequence; results are 1-D arrays.  One
        vectorized :func:`~app.ephemeris.besselian_instants` call, so a dense
        grid costs little more than a single instant.
        """
        t = np.atleast_1d(np.asarray(t_hours, dtype=float))
        e = besselian_instants(self.et0 + t * 3600.0, self.earth_frame)
        # mu is an angle: unwrap so a 360-degree wrap inside the range does not
        # leave a discontinuity (harmless for the modular use in geography, but
        # keeps the array continuous and consistent with evaluate()).
        e["mu"] = np.degrees(np.unwrap(np.radians(e["mu"])))
        return e


def best_earth_frame(frames: tuple[str, ...] = _FRAME_PREFERENCE) -> str:
    """Return the first Earth-orientation frame the furnished kernels support (item R5).

    Probes :func:`~app.ephemeris.besselian_instant` at J2000 (covered by every DE
    ephemeris used here) and returns the first frame in preference order that does
    not raise a SPICE error (e.g. ITRF93 needs a binary Earth PCK the mirror omits).
    Raises ``RuntimeError`` if none work.
    """
    import spiceypy

    load_kernels()
    for frame in frames:
        try:
            besselian_instant(0.0, frame)  # et = 0 -> J2000
            return frame
        except spiceypy.utils.exceptions.SpiceSPKINSUFFDATA:
            raise  # ephemeris coverage, not a frame problem: no frame can fix it
        except spiceypy.utils.exceptions.SpiceyError:
            spiceypy.reset()
    raise RuntimeError("no usable Earth-orientation frame in the furnished kernels")


def build_model_best_frame(
    t0_utc: str,
    half_window_hours: float = 2.0,
    step_hours: float = 1.0,
    frames: tuple[str, ...] = _FRAME_PREFERENCE,
) -> tuple[BesselianModel, str]:
    """Build a :class:`BesselianModel` with the best available frame (item R5).

    Tries each frame in preference order and returns ``(model, frame)`` for the
    first that the furnished kernels support.  Factors the frame-fallback loop out
    of the tests.  Raises ``RuntimeError`` if none work.  An epoch outside the
    SPK's coverage (``SpiceSPKINSUFFDATA``, e.g. 1919 with the mirror's DE432s,
    1949-2050) is re-raised as is: it is not a frame failure, and callers
    (the pre-IERS reference tests) skip on it.
    """
    import spiceypy

    for frame in frames:
        try:
            model = BesselianModel(
                t0_utc=t0_utc,
                earth_frame=frame,
                half_window_hours=half_window_hours,
                step_hours=step_hours,
            )
            return model, frame
        except spiceypy.utils.exceptions.SpiceSPKINSUFFDATA:
            raise  # ephemeris coverage, not a frame problem: no frame can fix it
        except spiceypy.utils.exceptions.SpiceyError:
            spiceypy.reset()
    raise RuntimeError("no usable Earth-orientation frame in the furnished kernels")

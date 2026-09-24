"""An eclipse model: the reference epoch T0 and Earth frame the core evaluates at.

:class:`BesselianModel` is a handle, not a computation: it validates the
epoch, converts it to TDB (``et0``) and asks the core for the tabular
polynomial fit that ``/besselian`` publishes.  Every other computation takes
``(et0, earth_frame, half_window_hours)`` from it and runs in the core
(:mod:`app.core`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import _eclipse
import numpy as np

from .core import DEFAULT_EARTH_FRAME, SpiceError, load_kernels

# Earth-orientation frames in preference order: ITRS (ERFA + IERS EOP; needs no
# binary PCK) first, then the binary-PCK ITRF93, the PCK-free TOD, and the coarse
# IAU_EARTH fallback.  Which are usable depends on the furnished kernels.
_FRAME_PREFERENCE = ("ITRS", "ITRF93", "TOD", "IAU_EARTH")


@dataclass
class BesselianPolynomials:
    """Polynomial coefficients (increasing powers of ``t`` in hours from T0).

    ``x, y`` cubic; ``d, l1, l2`` quadratic; ``mu`` linear; ``tan_f1, tan_f2``
    constants (the Espenak / NASA bulletin form).  Units: ``x, y, l1, l2`` in
    Earth equatorial radii, ``d, mu`` in degrees.
    """

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
    reads the normalized string and :func:`app.formatting.format_clock` formats
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
    """An eclipse at ``t0_utc`` in ``earth_frame``.

    ``et0`` is T0 in TDB seconds past J2000 (``_eclipse.utc_to_et``: UTC inside
    the IERS era, UT1 + the [Espenak] delta-T model outside it).
    ``half_window_hours`` / ``step_hours`` set the polynomial fit's sampling
    (``_eclipse.fit_polynomials``) and the initial search window of the local
    circumstances. The fitted cubic needs at least 4 samples, so a window
    shorter than 1.5 steps is sampled at ``2 * half_window_hours / 3`` instead
    (4 samples; ``step_hours`` holds the step used, item C5).
    """

    t0_utc: str
    earth_frame: str = DEFAULT_EARTH_FRAME
    half_window_hours: float = 2.0
    step_hours: float = 1.0
    polynomials: BesselianPolynomials | None = None
    et0: float = field(init=False, repr=False)  # TDB seconds past J2000 of t0_utc

    def __post_init__(self) -> None:
        self.t0_utc = normalize_utc(self.t0_utc)
        load_kernels()
        self.et0 = _eclipse.utc_to_et(self.t0_utc)
        self.step_hours = min(float(self.step_hours), 2.0 * float(self.half_window_hours) / 3.0)
        p = _eclipse.fit_polynomials(self.et0, self.earth_frame, float(self.half_window_hours),
                                     float(self.step_hours))
        self.polynomials = BesselianPolynomials(t0_utc=self.t0_utc, **p)

    def evaluate(self, t_hours) -> dict[str, np.ndarray]:
        """The published polynomials evaluated at ``t_hours`` (hours from T0).

        This is how a reader of the tabular product uses it (Horner in ``t``),
        valid **within** the sampling window only.  For any computation use
        :meth:`evaluate_direct`, which is exact at every instant.
        """
        p = self.polynomials
        assert p is not None
        t = np.asarray(t_hours, dtype=float)
        pv = np.polynomial.polynomial.polyval
        out = {k: pv(t, getattr(p, k)) for k in ("x", "y", "d", "mu", "l1", "l2")}
        out["tan_f1"] = np.full_like(t, p.tan_f1)
        out["tan_f2"] = np.full_like(t, p.tan_f2)
        return out

    def evaluate_direct(self, t_hours, k1: float = _eclipse.K_PENUMBRA,
                        k2: float = _eclipse.K_UMBRA) -> dict[str, np.ndarray]:
        """The elements computed at each ``t_hours`` (hours from T0) by the core
        (``_eclipse.elements_direct``: ``x, y, z, l1, l2`` [Earth radii], ``d,
        mu`` [deg], ``mu`` unwrapped, ``tan_f1, tan_f2``).  ``k1`` / ``k2``: the
        cones' lunar radii [Earth radii], default the [Espenak] pair."""
        t = np.ascontiguousarray(np.atleast_1d(np.asarray(t_hours, dtype=float)))
        return _eclipse.elements_direct(self.et0, self.earth_frame, t, float(k1), float(k2))


def _is_coverage_error(exc: SpiceError) -> bool:
    return "SPKINSUFFDATA" in exc.short_message


def best_earth_frame(frames: tuple[str, ...] = _FRAME_PREFERENCE) -> str:
    """Return the first Earth-orientation frame the furnished kernels support (item R5).

    Probes the elements at J2000 (covered by every DE ephemeris used here) and
    returns the first frame in preference order that does not raise a SPICE
    error (e.g. ITRF93 needs a binary Earth PCK the mirror omits).  An
    ephemeris-coverage error is re-raised.  ``RuntimeError`` if none work.
    """
    load_kernels()
    for frame in frames:
        try:
            _eclipse.besselian_instants(np.array([0.0]), frame)
            return frame
        except SpiceError as exc:
            if _is_coverage_error(exc):
                raise  # ephemeris coverage, not a frame problem: no frame can fix it
    raise RuntimeError("no usable Earth-orientation frame in the furnished kernels")


def build_model_best_frame(
    t0_utc: str,
    half_window_hours: float = 2.0,
    step_hours: float = 1.0,
    frames: tuple[str, ...] = _FRAME_PREFERENCE,
) -> tuple[BesselianModel, str]:
    """Build a :class:`BesselianModel` with the best available frame (item R5).

    Returns ``(model, frame)`` for the first frame in preference order that the
    furnished kernels support.  An epoch outside the SPK's coverage
    (``SPICE(SPKINSUFFDATA)``, e.g. 1919 with the mirror's DE432s, 1949-2050)
    is re-raised as is: callers (the pre-IERS reference tests) skip on it.
    """
    for frame in frames:
        try:
            model = BesselianModel(t0_utc=t0_utc, earth_frame=frame,
                                   half_window_hours=half_window_hours, step_hours=step_hours)
            return model, frame
        except SpiceError as exc:
            if _is_coverage_error(exc):
                raise
    raise RuntimeError("no usable Earth-orientation frame in the furnished kernels")

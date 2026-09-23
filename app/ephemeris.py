"""SPICE-backed ephemeris layer for solar-eclipse Besselian elements.

Why SPICE (and not astropy / a lunar theory)
---------------------------------------------
A solar eclipse needs the Sun *and* the Moon expressed in one consistent
geocentric framework, as **apparent** positions (corrected for light-time and
stellar aberration) and referred to a high-precision Earth-rotation model.
SPICE + a JPL DE ephemeris (DE440 from NAIF, DE432s from the GitHub mirror) gives
all of that from a single toolkit.  The two DE sources are interchangeable at
our accuracy ceiling: across the three reference eclipses their apparent
geocentric Moon agrees to ~2-3 mm and Sun to ~0.2 km, an effect of ~3 mm on the
fundamental-plane elements -- far below the ~km lunar-limb frontier.  In detail:

* ``spkpos(..., abcorr="LT+S", ...)`` returns apparent positions (light-time +
  stellar aberration) -- exactly what the Besselian construction assumes.

Four Earth-orientation frames are supported (``earth_frame``):

* ``"ITRS"`` (default) -- full IAU 2006/2000A celestial-to-terrestrial transform
  (ERFA ``c2t06a``) with IERS Earth-orientation parameters (polar motion +
  UT1-UTC from :mod:`app.eop`).  Most accurate, and needs only PyPI data (no
  binary PCK).
* ``"TOD"``   -- true equator & equinox of date (ERFA ``pnm06a`` + ``gst06a``),
  no polar motion.  Differs from ITRS only by the EOP terms.
* ``"ITRF93"`` -- SPICE Earth body-fixed frame from the high-precision binary
  Earth PCK (``earth_latest_high_prec.bpc``); equivalent accuracy to ITRS.
  Verified interchangeable with ITRS across the three reference eclipses --
  x, y agree to < 0.1 m and d, mu to < 0.01 arcsec -- so the default ITRS frame
  loses nothing by needing no binary PCK (guarded by
  ``tests/test_frame_consistency.py``).
* ``"IAU_EARTH"`` -- SPICE low-precision analytic rotation; coarse fallback.

How ``mu`` and the fundamental-plane coordinates stay consistent
---------------------------------------------------------------
* ``x, y, z`` ([ES92] eq. 8.322-6, Explanatory Supplement to the Astronomical
  Almanac) depend only on ``(alpha - a)`` and on the declinations ``delta`` (Moon) and
  ``d`` (axis); a common Greenwich rotation of all right ascensions leaves them
  unchanged, so they are identical in an Earth-fixed frame or in true-of-date.
* ``mu`` (Greenwich hour angle of the axis) ``= GST - a_of_date``.  For a SPICE
  body-fixed frame the axis longitude already equals ``a_of_date - GST`` so
  ``mu = -a``; equivalently, using ``mu = GST - a`` with ``GST = 0`` for the
  body-fixed vectors and ``GST = GAST`` for true-of-date unifies both cases.

Time scales and delta-T
-----------------------
The input epoch is a **UTC** string.  Inside the IERS era (the Bulletin A table,
1973 to ~1 year ahead) ``str2et`` converts it to TDB through the leap-second
kernel, and UT1 = UTC + (UT1-UTC) from :mod:`app.eop`; delta-T is therefore the
measured value.  Outside that era UTC is undefined (before 1972) or leap seconds
are unknown (future), and SPICE would silently hold the first/last leap-second
count -- e.g. ET-UTC = 41.18 s in 1900 against a true delta-T of -2.8 s.  There
the epoch string is read as **UT1** and TT = UT1 + delta-T from the [Espenak]
polynomial model (:mod:`app.deltat`), with polar motion set to zero.  The same
rule is applied in both directions so :func:`utc_to_et` and the Earth-rotation
angle stay consistent; see :func:`earth_rotation_times`.

Published Besselian tables quote ``mu`` as the *ephemeris hour angle* (Earth
rotation evaluated as if TT were UT); ours is the true Greenwich hour angle, so
``mu_published = mu_ours + delta-T * 1.002738 * 15"/s`` [ES92] sec. 8.36 (see
``tests/test_besselian_integration.py``).

Native backend
--------------
With ``ECLIPSE_BACKEND=native`` (:mod:`app.native`) the public functions below
that compute -- :func:`utc_to_et`, :func:`et_to_utc`,
:func:`earth_rotation_times`, :func:`besselian_instants`,
:func:`sub_solar_points` (roadmap phase 1) and :func:`axis_separation` (phase
4) -- dispatch to the C++ port in ``core/``.  The Python bodies stay as the
oracle; ``tests/test_native.py`` holds the two to the roadmap's parity
tolerances.

Thread safety
-------------
CSPICE keeps one global kernel pool and is **not** thread-safe.  Every function
here that touches SPICE takes :data:`SPICE_LOCK` (a re-entrant lock), so callers
running in a thread pool (FastAPI ``def`` endpoints) are serialized around the
SPICE calls while the pure-NumPy geometry stays parallel.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import erfa
import numpy as np
import spiceypy as spice

from . import native
from .constants import EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM
from .deltat import decimal_year_from_jd, delta_t_seconds
from .eop import eop, iers_mjd_range

# Julian date of J2000.0, for two-part TT/UT1 dates passed to ERFA.
_J2000_JD = 2451545.0
_MJD_OFFSET = 2400000.5  # JD - MJD

# Ratio of the lunar radius to Earth's equatorial radius.  The standard eclipse
# predictions (Fred Espenak; Explanatory Supplement) use TWO values: a larger
# constant for the penumbra and a smaller one for the umbra/antumbra (the umbral
# value is reduced to allow for the mean effect of the lunar limb profile /
# valleys).  Using a single value biases the umbral width.
K_PENUMBRA = 0.2725076  # penumbral contacts (IAU mean lunar radius)  [Espenak]
K_UMBRA = 0.2722810     # umbral / antumbral contacts                 [Espenak]

# Default Earth-orientation frame.  "ITRS" (ERFA IAU 2006/2000A + IERS EOP) is the
# most accurate and needs no binary PCK -- only PyPI data -- so it is the default.
# "TOD" drops the EOP correction; "ITRF93" uses the SPICE binary Earth PCK;
# "IAU_EARTH" is a coarse fallback.  Override with $SPICE_EARTH_FRAME.
DEFAULT_EARTH_FRAME = os.environ.get("SPICE_EARTH_FRAME", "ITRS")
EARTH_FRAMES = ("ITRS", "TOD", "ITRF93", "IAU_EARTH")  # every supported earth_frame

# Serializes every CSPICE call (global, non-thread-safe kernel pool); see the
# module docstring.  Re-entrant so nested helpers can take it freely.
SPICE_LOCK = threading.RLock()

_KERNELS_LOADED = False


def default_metakernel() -> Path:
    return Path(__file__).resolve().parent.parent / "kernels" / "eclipse.tm"


def load_kernels(metakernel: str | os.PathLike | None = None) -> None:
    """Furnish the SPICE kernel pool (idempotent).

    Resolution order: explicit argument, then ``$SPICE_METAKERNEL``, then the
    repo default ``kernels/eclipse.tm``.
    """
    global _KERNELS_LOADED
    with SPICE_LOCK:
        if _KERNELS_LOADED:
            return
        mk = Path(metakernel or os.environ.get("SPICE_METAKERNEL") or default_metakernel())
        if not mk.exists():
            raise FileNotFoundError(
                f"SPICE metakernel not found at {mk}. Run `python -m kernels.bootstrap` "
                "to download the required kernels (needs network access to NAIF)."
            )
        spice.furnsh(str(mk))
        if native.is_native():
            native.furnish(str(mk))  # the extension has its own kernel pool
        _KERNELS_LOADED = True


def unload_kernels() -> None:
    global _KERNELS_LOADED
    with SPICE_LOCK:
        spice.kclear()
        if native.is_native():
            native.kclear()
        _KERNELS_LOADED = False


def earth_equatorial_radius_km() -> float:
    """Fundamental-plane unit radius [km]: the cited WGS-84 semi-major axis.

    This is the single Earth radius that both scales the geocentric Sun/Moon
    vectors here and defines the reduction ellipsoid in :mod:`app.geography`, so
    the fundamental plane and the geographic reduction stay consistent (item A1).
    Previously this returned the SPICE PCK value (``a_e = 6378.1366 km``), which
    differed from the WGS-84 ``a = 6378.137 km`` used by the reduction by ~0.4 m.
    """
    return EARTH_EQUATORIAL_RADIUS_KM


def sun_radius_km() -> float:
    """Solar radius [km]: :data:`app.constants.SUN_RADIUS_KM`, 696 000 km (IAU 1976).

    This is the radius the [Espenak] predictions and the ``K_PENUMBRA``/``K_UMBRA``
    lunar radii are paired with; it must not be swapped for the IAU 2015 nominal
    695 700 km without re-deriving the k's, since the shadow-cone angles
    ``sin f = (d_s +/- k) / G`` [ES92] eq. 8.323-1 depend on the pair.  It is
    therefore a constant rather than ``bodvrd("SUN", "RADII")``: NAIF's
    pck00011 carries 695 700 km (review item W1).
    """
    return SUN_RADIUS_KM


# --- Time scales ----------------------------------------------------------------

def _in_iers_era(mjd) -> np.ndarray:
    """True where ``mjd`` (any scale; the boundary tolerance is a day) has IERS EOP."""
    lo, hi = iers_mjd_range()
    m = np.asarray(mjd, dtype=float)
    return (m >= lo) & (m <= hi)


def utc_to_et(utc: str) -> float:
    """Convert an epoch string to ephemeris time (TDB seconds past J2000).

    Inside the IERS era the string is UTC and ``str2et`` applies the leap-second
    kernel.  Outside it the string is read as UT1 and TT = UT1 + delta-T from
    the [Espenak] polynomial (module docstring, "Time scales and delta-T");
    TDB - TT (< 1.7 ms) is neglected there.
    """
    if native.is_native():
        return native.module().utc_to_et(utc)
    with SPICE_LOCK:
        formal = spice.tparse(utc)[0]  # seconds past J2000 on the leap-second-free calendar
        mjd = _J2000_JD - _MJD_OFFSET + formal / 86400.0
        if _in_iers_era(mjd):
            return float(spice.str2et(utc))
    return float(formal + delta_t_seconds(decimal_year_from_jd(formal / 86400.0 + _J2000_JD)))


def et_to_utc(et: float) -> str:
    """Inverse of :func:`utc_to_et`: ISO ``YYYY-MM-DDTHH:MM:SS`` (whole seconds).

    Inside the IERS era this is UTC via ``et2utc``; outside it the string is
    UT1 = TT - delta-T(model), formatted on the leap-second-free calendar, so a
    round trip through :func:`utc_to_et` is consistent to the rounding.
    """
    et = float(et)
    if native.is_native():
        return native.module().et_to_utc(et)
    with SPICE_LOCK:
        if _in_iers_era(_J2000_JD - _MJD_OFFSET + et / 86400.0):
            return spice.et2utc(et, "ISOC", 0)
        ut1 = et - float(delta_t_seconds(decimal_year_from_jd(_J2000_JD + et / 86400.0)))
        return spice.timout(ut1, "YYYY-MM-DDTHR:MN:SC ::TDB ::RND")


def earth_rotation_times(et):
    """Time arguments for Earth orientation at TDB ``et`` (scalar or 1-D array).

    Returns ``(tt2, ut1_frac, xp, yp)``: TT and UT1 as fractional days past
    J2000 (two-part JD second halves, first half ``_J2000_JD``) and polar motion
    in radians.  Inside the IERS era UT1 = UTC + (UT1-UTC) with UTC from the
    leap-second kernel [IERS2010]; outside it UT1 = TT - delta-T([Espenak]
    polynomial) and xp = yp = 0.  TDB is used for TT (<= 1.7 ms, sub-mas; A4).
    """
    et = np.atleast_1d(np.asarray(et, dtype=float))
    if native.is_native():
        return native.module().earth_rotation_times(et)
    tt2 = et / 86400.0
    with SPICE_LOCK:
        delta_et = np.array([spice.deltet(e, "ET") for e in et])  # ET - UTC [s]
    utc2 = (et - delta_et) / 86400.0
    mjd_utc = _J2000_JD - _MJD_OFFSET + utc2

    in_era = _in_iers_era(mjd_utc)
    xp, yp, dut1 = eop(mjd_utc)
    ut1_frac = np.where(in_era, utc2 + dut1 / 86400.0, np.nan)
    if not in_era.all():
        dt_model = delta_t_seconds(decimal_year_from_jd(_J2000_JD + tt2))
        ut1_frac = np.where(in_era, ut1_frac, tt2 - dt_model / 86400.0)
        xp = np.where(in_era, xp, 0.0)
        yp = np.where(in_era, yp, 0.0)
    return tt2, ut1_frac, xp, yp


def tt_minus_ut1(et) -> np.ndarray:
    """Delta-T = TT - UT1 [s] as actually used at ``et`` (measured or modelled)."""
    tt2, ut1, _, _ = earth_rotation_times(et)
    return (tt2 - ut1) * 86400.0


# --- Geometry -------------------------------------------------------------------

def _rotate(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply per-row 3x3 matrices ``r`` (n, 3, 3) to vectors ``v`` (n, 3).

    Written as an elementwise product summed over the last axis rather than
    ``np.einsum``/``np.matmul`` so the three-term accumulation order is fixed
    (``(r0 v0 + r1 v1) + r2 v2``, no FMA) -- the same order as ERFA's ``eraRxp``
    and the C++ core, giving bit-identical vectors.  ``einsum`` dispatches to
    SIMD/FMA kernels that vary by CPU and differ from this by ~1 ulp
    (~6e-11 km at the Moon), which is numerically irrelevant but would make
    the oracle machine-dependent.
    """
    return (r * v[:, None, :]).sum(axis=-1)


def _reclat(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``spice.reclat``: (radius, longitude, latitude) of (..., 3) vectors."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.linalg.norm(v, axis=-1), np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))


@dataclass(frozen=True)
class BesselianInstant:
    """Besselian elements at a single instant.

    Angles ``d`` and ``mu`` are in **degrees**; ``x, y, l1, l2`` are in Earth
    equatorial radii; ``tan_f1, tan_f2`` are dimensionless.
    """

    x: float
    y: float
    d: float
    mu: float
    l1: float
    l2: float
    tan_f1: float
    tan_f2: float


class _FundamentalXYZ(NamedTuple):
    """The Moon's fundamental-plane coordinates and the shadow-axis direction."""

    x: np.ndarray       # [Earth radii]
    y: np.ndarray       # [Earth radii]
    z: np.ndarray       # [Earth radii], toward the Sun
    a: np.ndarray       # axis longitude (right ascension) in the vectors' frame [rad]
    d: np.ndarray       # axis declination [rad]
    g_dist: np.ndarray  # Sun-Moon distance [Earth radii]


def _fundamental_xyz(moon: np.ndarray, sun: np.ndarray) -> _FundamentalXYZ:
    """Fundamental-plane coordinates of the Moon from geocentric Moon/Sun vectors.

    ``moon``/``sun`` are ``(n, 3)`` apparent geocentric vectors in Earth radii,
    in ANY common equatorial frame.  The shadow-axis direction is from the Moon
    toward the Sun (``S - M``), matching the Explanatory Supplement sign
    convention [ES92] ch. 8: its longitude is ``a``, its latitude the axis
    declination ``d`` and its length the Sun-Moon distance of the shadow-cone
    geometry.  ``x, y, z`` are [ES92] eq. 8.322-6.  Shared by
    :func:`besselian_instants` (Earth-fixed / true-of-date vectors) and
    :func:`axis_separation` (un-rotated J2000 vectors): one formula, one home.
    """
    # Moon geocentric spherical coordinates.
    r_moon, alpha, delta = _reclat(moon)

    # Shadow-axis direction: from the Moon toward the Sun (S - M), matching the
    # Explanatory Supplement sign convention [ES92] ch. 8.  Its longitude is ``a``
    # and its latitude is the axis declination ``d``; its length is the Sun-Moon
    # distance used by the shadow-cone geometry.
    g_dist, a, d = _reclat(sun - moon)

    # Fundamental-plane coordinates of the Moon [ES92] eq. 8.322-6.
    ha = alpha - a
    x = r_moon * np.cos(delta) * np.sin(ha)
    y = r_moon * (np.sin(delta) * np.cos(d) - np.cos(delta) * np.sin(d) * np.cos(ha))
    z = r_moon * (np.sin(delta) * np.sin(d) + np.cos(delta) * np.cos(d) * np.cos(ha))
    return _FundamentalXYZ(x=x, y=y, z=z, a=a, d=d, g_dist=g_dist)


def _geocentric_vectors(et: np.ndarray, earth_frame: str):
    """Apparent geocentric Moon and Sun (Earth radii) plus the sidereal offset.

    ``et`` is a 1-D array of TDB seconds.  Returns ``(moon, sun, gast)`` with
    ``moon``/``sun`` of shape (n, 3) and ``gast`` of shape (n,): 0 for an
    Earth-fixed frame (longitudes are already Earth-fixed) and the Greenwich
    apparent sidereal time (radians) for the ``"TOD"`` true-of-date frame;
    either way ``mu = gast - a`` gives the axis' Greenwich hour angle.
    """
    a_e = earth_equatorial_radius_km()
    n = len(et)

    if earth_frame in ("ITRS", "TOD"):
        with SPICE_LOCK:
            moon, _ = spice.spkpos("MOON", et, "J2000", "LT+S", "EARTH")
            sun, _ = spice.spkpos("SUN", et, "J2000", "LT+S", "EARTH")
        moon = np.asarray(moon).reshape(n, 3)
        sun = np.asarray(sun).reshape(n, 3)
        tt2, ut1_frac, xp, yp = earth_rotation_times(et)

        if earth_frame == "ITRS":
            # ERFA c2t06a: the full IAU 2006/2000A celestial-to-terrestrial rotation
            # matrix [SOFA] (Wallace & Capitaine 2006), driven by IERS polar motion
            # (xp, yp) and UT1-UTC [IERS2010] from :mod:`app.eop`.  Most accurate
            # Earth-fixed frame; needs no binary PCK.
            rc2t = erfa.c2t06a(_J2000_JD, tt2, _J2000_JD, ut1_frac, xp, yp)  # (n, 3, 3)
            moon = _rotate(rc2t, moon)
            sun = _rotate(rc2t, sun)
            return moon / a_e, sun / a_e, np.zeros(n)

        # TOD: true equator & equinox of date, GAST from UT1 (no polar motion).
        # ERFA pnm06a = IAU 2006/2000A precession-nutation matrix; gst06a = Greenwich
        # apparent sidereal time, both [SOFA] (Wallace & Capitaine 2006).
        rbpn = erfa.pnm06a(_J2000_JD, tt2)  # GCRS -> true-of-date, (n, 3, 3)
        moon = _rotate(rbpn, moon)
        sun = _rotate(rbpn, sun)
        gast = np.asarray(erfa.gst06a(_J2000_JD, ut1_frac, _J2000_JD, tt2), dtype=float)
        return moon / a_e, sun / a_e, gast.reshape(n)

    with SPICE_LOCK:
        moon, _ = spice.spkpos("MOON", et, earth_frame, "LT+S", "EARTH")
        sun, _ = spice.spkpos("SUN", et, earth_frame, "LT+S", "EARTH")
    return (np.asarray(moon).reshape(n, 3) / a_e, np.asarray(sun).reshape(n, 3) / a_e,
            np.zeros(n))


def besselian_instants(et, earth_frame: str = DEFAULT_EARTH_FRAME, k1: float = K_PENUMBRA,
                       k2: float = K_UMBRA) -> dict[str, np.ndarray]:
    """Besselian elements at each TDB ``et`` of a 1-D array, as arrays.

    ``k1``/``k2`` are the lunar radii [Earth equatorial radii] of the penumbral
    and umbral cones; the defaults are the [Espenak] pair (``K_PENUMBRA``,
    ``K_UMBRA``: the umbral one reduced for the mean limb valleys).  The limb
    profile mode passes the LOLA reference sphere for both (``app.limb.K_REF``),
    the valleys then coming from the DEM (docs/LIMB_PROFILE.md sec. 3.3).

    Returns a dict with keys ``x, y, d, mu, l1, l2, tan_f1, tan_f2`` (same units
    as :class:`BesselianInstant`: ``d, mu`` in degrees, lengths in Earth radii)
    plus ``z``, the Moon's distance along the axis toward the Sun [ES92] eq.
    8.322-6 (positive at a solar eclipse; negative at full moon, when the Moon
    sits on the far side of the Earth on the same line -- the catalog uses it
    to reject lunar-eclipse geometry), each an array of ``len(et)``.  This is
    the vectorized core; the whole computation is a few array-valued SPICE/ERFA
    calls, so evaluating a thousand instants costs about the same as one did.
    ``mu`` is returned wrapped to (-180, 180]; callers that fit or difference it
    should ``np.unwrap``.
    """
    et = np.atleast_1d(np.asarray(et, dtype=float))
    if native.is_native():
        return native.module().besselian_instants(et, earth_frame, float(k1), float(k2))
    a_e = earth_equatorial_radius_km()

    moon, sun, gast = _geocentric_vectors(et, earth_frame)

    # Fundamental-plane coordinates of the Moon and the axis direction
    # [ES92] eq. 8.322-6 (shared with axis_separation).
    x, y, z, a, d, g_dist = _fundamental_xyz(moon, sun)

    mu = gast - a  # Greenwich hour angle of the axis [ES92] ch. 8

    # Penumbral (f1) and umbral (f2) shadow cones [ES92] eq. 8.323-1, 8.323-6,
    # 8.323-7, with distinct penumbral/umbral lunar radii k1/k2 [Espenak].
    d_s = sun_radius_km() / a_e  # solar radius in Earth radii
    sin_f1 = (d_s + k1) / g_dist
    sin_f2 = (d_s - k2) / g_dist
    tan_f1 = np.tan(np.arcsin(sin_f1))
    tan_f2 = np.tan(np.arcsin(sin_f2))

    l1 = (z + k1 / sin_f1) * tan_f1
    l2 = (z - k2 / sin_f2) * tan_f2

    return {
        "x": x,
        "y": y,
        "z": z,
        "d": np.degrees(d),
        "mu": (np.degrees(mu) + 180.0) % 360.0 - 180.0,
        "l1": l1,
        "l2": l2,
        "tan_f1": tan_f1,
        "tan_f2": tan_f2,
    }


def besselian_instant(et: float, earth_frame: str = DEFAULT_EARTH_FRAME) -> BesselianInstant:
    """Compute the Besselian elements at ephemeris time ``et`` (scalar form).

    All quantities are geocentric and apparent (``LT+S``); ``earth_frame`` selects
    the Earth-orientation model (``"ITRS"`` (default), ``"TOD"``, ``"ITRF93"`` or
    ``"IAU_EARTH"``) -- see the module docstring.  Thin wrapper over
    :func:`besselian_instants`.
    """
    e = besselian_instants(np.array([float(et)]), earth_frame)
    return BesselianInstant(**{k: float(v[0]) for k, v in e.items() if k != "z"})


def axis_separation(et) -> tuple[np.ndarray, np.ndarray]:
    """``(rho, z)`` at each TDB ``et`` of a 1-D array: the Moon's cylindrical
    coordinates about the shadow axis, frame-free.

    ``rho = hypot(x, y)`` is the shadow axis' distance from the Earth's centre in
    the fundamental plane and ``z`` the Moon's distance along the axis toward
    the Sun, both in Earth equatorial radii, from the same [ES92] eq. 8.322-6
    expressions as :func:`besselian_instants` (:func:`_fundamental_xyz`) applied
    to the un-rotated apparent (``LT+S``) J2000 geocentric vectors.  The
    Earth-orientation frames of :func:`besselian_instants` (ITRS / TOD / ITRF93 /
    IAU_EARTH) all *rotate* those same two vectors by one common rotation, and
    ``x, y, z`` are the Moon's coordinates in the axis-aligned frame built from
    the Sun-Moon direction and the Moon direction alone, so they are invariant
    under any rotation of the reference frame; only ``mu`` (and ``d``) depend
    on it.  Hence ``rho`` and ``z`` here equal the ``hypot(x, y)`` and ``z`` of
    :func:`besselian_instants` in every frame to rounding, with no ERFA
    nutation and no EOP -- this is the catalog's scan objective
    (:mod:`app.catalog`), which reads nothing else.

    Dispatches to the native core under ``ECLIPSE_BACKEND=native`` (phase 4).
    """
    et = np.atleast_1d(np.asarray(et, dtype=float))
    if native.is_native():
        return native.module().axis_separation(et)
    a_e = earth_equatorial_radius_km()
    n = len(et)
    with SPICE_LOCK:
        moon, _ = spice.spkpos("MOON", et, "J2000", "LT+S", "EARTH")
        sun, _ = spice.spkpos("SUN", et, "J2000", "LT+S", "EARTH")
    moon = np.asarray(moon).reshape(n, 3) / a_e
    sun = np.asarray(sun).reshape(n, 3) / a_e
    f = _fundamental_xyz(moon, sun)
    return np.hypot(f.x, f.y), f.z


def _earth_pole_j2000(et: np.ndarray, earth_frame: str) -> np.ndarray:
    """(n, 3) unit vector of ``earth_frame``'s +z (the pole the elements' ``d``
    and the fundamental plane's north ``y`` refer to), in J2000.

    It is row 2 of the J2000 -> ``earth_frame`` matrix: ERFA ``c2t06a`` (ITRS)
    or ``pnm06a`` (TOD; GAST turns about this pole and leaves it fixed) [SOFA],
    exactly as :func:`_geocentric_vectors` builds them, or SPICE ``pxform``
    for a SPICE frame.
    """
    if earth_frame in ("ITRS", "TOD"):
        tt2, ut1_frac, xp, yp = earth_rotation_times(et)
        if earth_frame == "ITRS":
            r = erfa.c2t06a(_J2000_JD, tt2, _J2000_JD, ut1_frac, xp, yp)
        else:
            r = erfa.pnm06a(_J2000_JD, tt2)
        return np.ascontiguousarray(np.asarray(r).reshape(len(et), 3, 3)[:, 2, :])
    with SPICE_LOCK:
        return np.array([spice.pxform("J2000", earth_frame, float(t))[2] for t in et])


def _dot3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise dot product of (n, 3) arrays, summed left to right."""
    return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) + a[:, 2] * b[:, 2]


def limb_axes(et, earth_frame: str = DEFAULT_EARTH_FRAME,
              moon_frame: str = "MOON_ME") -> tuple[np.ndarray, np.ndarray]:
    """Fundamental-plane axes in the Moon's body frame at each TDB ``et``.

    Returns ``(axes, distance_km)``.  ``distance_km`` (n,) is the Moon's
    distance from the fundamental plane along the axis, ``moon . z^`` [km]
    (= the element ``z`` times the Earth's radius, [ES92] eq. 8.322-6): the
    viewing distance of the perspective silhouette (``app.limb.silhouette``).
    ``axes`` is an ``(n, 3, 3)`` array whose rows are the unit vectors ``x^``
    (east), ``y^`` (north), ``z^`` (shadow axis, toward the Sun) of the
    fundamental plane [ES92] 8.322, expressed in ``moon_frame`` (``MOON_ME`` =
    the LOLA DEM's frame, from ``moon_de440_*.tf`` + ``moon_pa_de440_*.bpc``;
    ``IAU_MOON`` works with a text PCK alone).  ``z^`` is the apparent (LT+S)
    Moon-to-Sun direction, as for the elements; ``y^`` is ``earth_frame``'s
    pole projected onto the plane, ``x^ = y^ x z^`` -- the same axes as
    :func:`_fundamental_xyz`'s ``(a, d)`` construction, built rotation-free.
    The Moon's orientation is taken at the light-time-corrected epoch
    ``et - lt``.  Stellar aberration tilts ``z^`` by <= 20.5" relative to the
    geometric ray direction, which moves a limb point at most ~20 m; ignored.
    ``docs/LIMB_PROFILE.md`` sec. 3.1-3.2.  Dispatches to the native core.
    """
    et = np.atleast_1d(np.asarray(et, dtype=float))
    if native.is_native():
        return native.module().limb_axes(et, earth_frame, moon_frame)
    n = len(et)
    with SPICE_LOCK:
        moon, lt = spice.spkpos("MOON", et, "J2000", "LT+S", "EARTH")
        sun, _ = spice.spkpos("SUN", et, "J2000", "LT+S", "EARTH")
        lt = np.atleast_1d(np.asarray(lt, dtype=float))
        rot = np.array([spice.pxform("J2000", moon_frame, float(t) - float(tau))
                        for t, tau in zip(et, lt, strict=True)])
    moon = np.asarray(moon).reshape(n, 3)
    sun = np.asarray(sun).reshape(n, 3)
    w = sun - moon
    zh = w / np.sqrt(_dot3(w, w))[:, None]
    pole = _earth_pole_j2000(et, earth_frame)
    yv = pole - _dot3(pole, zh)[:, None] * zh
    yh = yv / np.sqrt(_dot3(yv, yv))[:, None]
    xh = np.stack([yh[:, 1] * zh[:, 2] - yh[:, 2] * zh[:, 1],
                   yh[:, 2] * zh[:, 0] - yh[:, 0] * zh[:, 2],
                   yh[:, 0] * zh[:, 1] - yh[:, 1] * zh[:, 0]], axis=1)
    out = np.empty((n, 3, 3))
    for k, v in enumerate((xh, yh, zh)):
        for r in range(3):
            out[:, k, r] = _dot3(rot[:, r, :], v)
    return out, _dot3(moon, zh)


def sub_solar_points(et, earth_frame: str = DEFAULT_EARTH_FRAME) -> tuple[np.ndarray, np.ndarray]:
    """Geographic (lon, lat) arrays [deg] of the sub-solar point at each ``et``.

    Latitude is the Sun's declination in the Earth-fixed frame, which equals the
    geodetic latitude of the point where the Sun is at the zenith (the zenith is
    the ellipsoid normal); longitude is east-positive in (-180, 180].
    """
    et = np.atleast_1d(np.asarray(et, dtype=float))
    if native.is_native():
        return native.module().sub_solar_points(et, earth_frame)
    _moon, sun, gast = _geocentric_vectors(et, earth_frame)
    _r, lon, lat = _reclat(sun)
    lon_deg = (np.degrees(lon - gast) + 180.0) % 360.0 - 180.0
    return lon_deg, np.degrees(lat)


def sub_solar_point(et: float, earth_frame: str = DEFAULT_EARTH_FRAME) -> tuple[float, float]:
    """Scalar form of :func:`sub_solar_points`: ``(lon, lat)`` in degrees."""
    lon, lat = sub_solar_points(np.array([float(et)]), earth_frame)
    return float(lon[0]), float(lat[0])

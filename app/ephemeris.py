"""SPICE-backed ephemeris layer for solar-eclipse Besselian elements.

Why SPICE (and not astropy / a lunar theory)
---------------------------------------------
A solar eclipse needs the Sun *and* the Moon expressed in one consistent
geocentric framework, as **apparent** positions (corrected for light-time and
stellar aberration) and referred to a high-precision Earth-rotation model.
SPICE + JPL DE440 gives all of that from a single toolkit:

* ``spkpos(..., abcorr="LT+S", ...)`` returns apparent positions (light-time +
  stellar aberration) -- exactly what the Besselian construction assumes.

Four Earth-orientation frames are supported (``earth_frame``):

* ``"ITRS"`` (default) -- full IAU 2006/2000A celestial-to-terrestrial transform
  (ERFA ``c2t06a``) with IERS Earth-orientation parameters (polar motion +
  UT1-UTC from :mod:`app.eop`).  Most accurate, and needs only PyPI data (no
  binary PCK).
* ``"TOD"``   -- true equator & equinox of date (ERFA ``pnm06a`` + ``gst06a``),
  UT1 approximated by UTC (no EOP).  Differs from ITRS only by the EOP terms.
* ``"ITRF93"`` -- SPICE Earth body-fixed frame from the high-precision binary
  Earth PCK (``earth_latest_high_prec.bpc``); equivalent accuracy to ITRS.
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

The input epoch is **UTC**; ``str2et`` converts it to TDB, so delta-T is handled
by construction rather than as an afterthought.  (Note: the dominant historical
error here was not the frame but the parametric->geodetic latitude conversion in
``geography.py`` -- the frame contributes only a few km.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import erfa
import numpy as np
import spiceypy as spice

from .constants import EARTH_EQUATORIAL_RADIUS_KM
from .eop import eop

# Julian date of J2000.0, for two-part TT/UT1 dates passed to ERFA.
_J2000_JD = 2451545.0

# Ratio of the lunar radius to Earth's equatorial radius.  The standard eclipse
# predictions (Fred Espenak; Explanatory Supplement) use TWO values: a larger
# constant for the penumbra and a smaller one for the umbra/antumbra (the umbral
# value is reduced to allow for the mean effect of the lunar limb profile /
# valleys).  Using a single value biases the umbral width.
K_PENUMBRA = 0.2725076  # penumbral contacts (IAU mean lunar radius)
K_UMBRA = 0.2722810     # umbral / antumbral contacts (Espenak)

# Default Earth-orientation frame.  "ITRS" (ERFA IAU 2006/2000A + IERS EOP) is the
# most accurate and needs no binary PCK -- only PyPI data -- so it is the default.
# "TOD" drops the EOP correction; "ITRF93" uses the SPICE binary Earth PCK;
# "IAU_EARTH" is a coarse fallback.  Override with $SPICE_EARTH_FRAME.
DEFAULT_EARTH_FRAME = os.environ.get("SPICE_EARTH_FRAME", "ITRS")

_KERNELS_LOADED = False


def default_metakernel() -> Path:
    return Path(__file__).resolve().parent.parent / "kernels" / "eclipse.tm"


def load_kernels(metakernel: str | os.PathLike | None = None) -> None:
    """Furnish the SPICE kernel pool (idempotent).

    Resolution order: explicit argument, then ``$SPICE_METAKERNEL``, then the
    repo default ``kernels/eclipse.tm``.
    """
    global _KERNELS_LOADED
    if _KERNELS_LOADED:
        return
    mk = Path(metakernel or os.environ.get("SPICE_METAKERNEL") or default_metakernel())
    if not mk.exists():
        raise FileNotFoundError(
            f"SPICE metakernel not found at {mk}. Run `python -m kernels.bootstrap` "
            "to download the required kernels (needs network access to NAIF)."
        )
    spice.furnsh(str(mk))
    _KERNELS_LOADED = True


def unload_kernels() -> None:
    global _KERNELS_LOADED
    spice.kclear()
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


@lru_cache(maxsize=1)
def sun_radius_km() -> float:
    return float(spice.bodvrd("SUN", "RADII", 3)[1][0])


def utc_to_et(utc: str) -> float:
    """Convert a UTC time string to ephemeris time (TDB seconds past J2000)."""
    return spice.str2et(utc)


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


def _geocentric_vectors(et: float, earth_frame: str):
    """Apparent geocentric Moon and Sun (Earth radii) plus the sidereal offset.

    Returns ``(moon, sun, gast)`` where ``gast`` is 0 for a SPICE body-fixed
    frame (longitudes are already Earth-fixed) and the Greenwich apparent
    sidereal time (radians) for the ``"TOD"`` true-of-date frame; either way
    ``mu = gast - a`` gives the axis' Greenwich hour angle.
    """
    a_e = earth_equatorial_radius_km()

    if earth_frame in ("ITRS", "TOD"):
        moon, _ = spice.spkpos("MOON", et, "J2000", "LT+S", "EARTH")
        sun, _ = spice.spkpos("SUN", et, "J2000", "LT+S", "EARTH")
        # TDB is used where ERFA expects TT: they differ by <= 1.7 ms, i.e. sub-mas
        # in Earth orientation, well below our accuracy ceiling (item A4).
        tt2 = et / 86400.0  # TT ~ TDB to < 2 ms
        utc_jd = _J2000_JD + (et - spice.deltet(et, "ET")) / 86400.0

        if earth_frame == "ITRS":
            # ERFA c2t06a: the full IAU 2006/2000A celestial-to-terrestrial rotation
            # matrix [SOFA] (Wallace & Capitaine 2006), driven by IERS polar motion
            # (xp, yp) and UT1-UTC [IERS2010] from :mod:`app.eop`.  Most accurate
            # Earth-fixed frame; needs no binary PCK.
            xp, yp, dut1 = eop(utc_jd - 2400000.5)
            ut1_frac = (utc_jd - _J2000_JD) + dut1 / 86400.0
            rc2t = erfa.c2t06a(_J2000_JD, tt2, _J2000_JD, ut1_frac, xp, yp)
            return (rc2t @ np.asarray(moon)) / a_e, (rc2t @ np.asarray(sun)) / a_e, 0.0

        # TOD: true equator & equinox of date, GAST with UT1 ~ UTC (no EOP).
        # ERFA pnm06a = IAU 2006/2000A precession-nutation matrix; gst06a = Greenwich
        # apparent sidereal time, both [SOFA] (Wallace & Capitaine 2006).
        rbpn = erfa.pnm06a(_J2000_JD, tt2)  # GCRS -> true-of-date
        moon = rbpn @ np.asarray(moon)
        sun = rbpn @ np.asarray(sun)
        gast = float(erfa.gst06a(utc_jd, 0.0, _J2000_JD, tt2))
        return moon / a_e, sun / a_e, gast

    moon, _ = spice.spkpos("MOON", et, earth_frame, "LT+S", "EARTH")
    sun, _ = spice.spkpos("SUN", et, earth_frame, "LT+S", "EARTH")
    return np.asarray(moon) / a_e, np.asarray(sun) / a_e, 0.0


def besselian_instant(et: float, earth_frame: str = DEFAULT_EARTH_FRAME) -> BesselianInstant:
    """Compute the Besselian elements at ephemeris time ``et``.

    All quantities are geocentric and apparent (``LT+S``); ``earth_frame`` selects
    the Earth-orientation model (``"ITRF93"``, ``"TOD"`` or ``"IAU_EARTH"``) -- see
    the module docstring.
    """
    a_e = earth_equatorial_radius_km()

    moon, sun, gast = _geocentric_vectors(et, earth_frame)

    # Moon geocentric spherical coordinates.
    r_moon, alpha, delta = spice.reclat(moon)

    # Shadow-axis direction: from the Moon toward the Sun (S - M), matching the
    # Explanatory Supplement sign convention [ES92] ch. 8.  Its longitude is ``a``
    # and its latitude is the axis declination ``d``; its length is the Sun-Moon
    # distance used by the shadow-cone geometry.
    axis = sun - moon
    g_dist, a, d = spice.reclat(axis)

    mu = gast - a  # Greenwich hour angle of the axis [ES92] ch. 8

    # Fundamental-plane coordinates of the Moon [ES92] eq. 8.322-6.
    ha = alpha - a
    x = r_moon * np.cos(delta) * np.sin(ha)
    y = r_moon * (np.sin(delta) * np.cos(d) - np.cos(delta) * np.sin(d) * np.cos(ha))
    z = r_moon * (np.sin(delta) * np.sin(d) + np.cos(delta) * np.cos(d) * np.cos(ha))

    # Penumbral (f1) and umbral (f2) shadow cones [ES92] eq. 8.323-1, 8.323-6,
    # 8.323-7, with distinct penumbral/umbral lunar radii k1/k2 [Espenak].
    d_s = sun_radius_km() / a_e  # solar radius in Earth radii
    sin_f1 = (d_s + K_PENUMBRA) / g_dist
    sin_f2 = (d_s - K_UMBRA) / g_dist
    tan_f1 = float(np.tan(np.arcsin(sin_f1)))
    tan_f2 = float(np.tan(np.arcsin(sin_f2)))

    c1 = z + K_PENUMBRA / sin_f1
    c2 = z - K_UMBRA / sin_f2
    l1 = c1 * tan_f1
    l2 = c2 * tan_f2

    return BesselianInstant(
        x=float(x),
        y=float(y),
        d=float(np.degrees(d)),
        mu=float(np.degrees(mu)),
        l1=float(l1),
        l2=float(l2),
        tan_f1=tan_f1,
        tan_f2=tan_f2,
    )


def sub_solar_point(et: float, earth_frame: str = DEFAULT_EARTH_FRAME) -> tuple[float, float]:
    """Geographic (lon, lat) in degrees of the point where the Sun is at zenith.

    Latitude is the Sun's declination-of-date (which equals the geodetic latitude
    of the sub-solar point); longitude is east-positive in (-180, 180].  Used to
    place the frontend's sunlight and render the day/night terminator.
    """
    _moon, sun, gast = _geocentric_vectors(et, earth_frame)
    _r, lon, lat = spice.reclat(sun)
    lon_deg = (np.degrees(lon - gast) + 180.0) % 360.0 - 180.0
    return float(lon_deg), float(np.degrees(lat))

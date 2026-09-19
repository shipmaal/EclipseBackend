"""SPICE-backed ephemeris layer for solar-eclipse Besselian elements.

Why SPICE (and not astropy / a lunar theory)
---------------------------------------------
A solar eclipse needs the Sun *and* the Moon expressed in one consistent
geocentric framework, as **apparent** positions (corrected for light-time and
stellar aberration) and referred to a high-precision Earth-rotation model.
SPICE + JPL DE440 gives all of that from a single toolkit:

* ``spkpos(..., abcorr="LT+S", ...)`` returns apparent positions (light-time +
  stellar aberration) -- exactly what the Besselian construction assumes.
* Evaluating positions in the Earth body-fixed frame **ITRF93** (from the
  high-precision binary Earth PCK) folds precession, nutation and Earth
  rotation -- including UT1 -- into the frame itself.

Frame choice removes two classic eclipse bugs
---------------------------------------------
By working directly in the Earth-fixed frame we never form Greenwich sidereal
time by hand, and we never apply a separate delta-T term:

* The fundamental-plane coordinates ``x, y, z`` (Explanatory Supplement to the
  Astronomical Almanac, eq. 8.322-6) depend only on ``(alpha - a)`` and on the
  declinations ``delta`` (Moon) and ``d`` (axis).  A change of frame that
  rotates every right ascension by the same Greenwich sidereal angle leaves
  ``(alpha - a)`` and both latitudes unchanged, so ``x, y, z`` are identical
  whether computed in true-of-date or in the Earth-fixed frame.
* The Besselian ``mu`` (Greenwich hour angle of the shadow axis) is, by
  definition, ``GST - a_of_date``.  In the Earth-fixed frame the axis longitude
  is ``a_ef = a_of_date - GST``, so ``mu = -a_ef`` -- read straight off the
  rotated axis vector, with no sidereal-time or delta-T arithmetic.

The input epoch is therefore taken as **UTC**; ``str2et`` converts it to
TDB and the binary Earth PCK supplies the matching orientation, so delta-T is
handled by construction rather than as an afterthought.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import spiceypy as spice

# Ratio of the mean lunar radius to Earth's equatorial radius.  This is the
# value adopted for umbral *and* penumbral contacts in the standard eclipse
# predictions (Fred Espenak; Explanatory Supplement).  It is deliberately a
# fixed constant rather than a body radius pulled from a kernel.
K_MOON = 0.2725076

# Default Earth body-fixed frame.  ITRF93 needs the high-precision binary Earth
# PCK; IAU_EARTH (low-precision analytic rotation) is only a coarse fallback.
DEFAULT_EARTH_FRAME = "ITRF93"

_KERNELS_LOADED = False


def default_metakernel() -> Path:
    return Path(__file__).resolve().parent.parent / "kernels" / "eclipse.tm"


def load_kernels(metakernel: Optional[str | os.PathLike] = None) -> None:
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


@lru_cache(maxsize=1)
def earth_equatorial_radius_km() -> float:
    return float(spice.bodvrd("EARTH", "RADII", 3)[1][0])


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


def besselian_instant(et: float, earth_frame: str = DEFAULT_EARTH_FRAME) -> BesselianInstant:
    """Compute the Besselian elements at ephemeris time ``et``.

    All quantities are geocentric and apparent (``LT+S``); the Sun and Moon are
    evaluated in ``earth_frame`` (Earth body-fixed) so that ``mu`` and the
    fundamental-plane coordinates are internally consistent -- see the module
    docstring.
    """
    a_e = earth_equatorial_radius_km()

    moon_km, _ = spice.spkpos("MOON", et, earth_frame, "LT+S", "EARTH")
    sun_km, _ = spice.spkpos("SUN", et, earth_frame, "LT+S", "EARTH")

    moon = np.asarray(moon_km) / a_e  # Earth radii, Earth-fixed
    sun = np.asarray(sun_km) / a_e

    # Moon geocentric spherical coordinates (Earth-fixed).
    r_moon, alpha, delta = spice.reclat(moon)

    # Shadow-axis direction: from the Moon toward the Sun (S - M), matching the
    # Explanatory Supplement sign convention.  Its longitude is ``a`` and its
    # latitude is the axis declination ``d``; its length is the Sun-Moon
    # distance used by the shadow-cone geometry.
    axis = sun - moon
    g_dist, a, d = spice.reclat(axis)

    mu = -a  # Greenwich hour angle of the axis (= GST - a_of_date)

    # Fundamental-plane coordinates of the Moon (Explanatory Supplement 8.322-6).
    ha = alpha - a
    x = r_moon * np.cos(delta) * np.sin(ha)
    y = r_moon * (np.sin(delta) * np.cos(d) - np.cos(delta) * np.sin(d) * np.cos(ha))
    z = r_moon * (np.sin(delta) * np.sin(d) + np.cos(delta) * np.cos(d) * np.cos(ha))

    # Penumbral (f1) and umbral (f2) shadow cones (Explanatory Supplement
    # 8.323-1, 8.323-6, 8.323-7).
    d_s = sun_radius_km() / a_e  # solar radius in Earth radii
    sin_f1 = (d_s + K_MOON) / g_dist
    sin_f2 = (d_s - K_MOON) / g_dist
    tan_f1 = float(np.tan(np.arcsin(sin_f1)))
    tan_f2 = float(np.tan(np.arcsin(sin_f2)))

    c1 = z + K_MOON / sin_f1
    c2 = z - K_MOON / sin_f2
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

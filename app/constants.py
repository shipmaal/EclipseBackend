"""Physical and geodetic constants, cited in one place.

Centralizing these fixes the earlier duplication/inconsistency (review item A1):
the fundamental-plane unit radius that scales the geocentric Sun/Moon vectors in
:mod:`app.ephemeris` and the reference ellipsoid used by the geographic reduction
in :mod:`app.geography` are now the *same* cited WGS-84 value, and the ellipsoid's
``b`` and ``e^2`` are derived from the defining flattening rather than rounded.

See ``CLAUDE.md`` conventions §2 (reuse, don't re-derive) and its References keys
([WGS84], [ES92]).
"""

from __future__ import annotations

# --- WGS-84 reference ellipsoid ----------------------------------------------
# Defining parameters of WGS-84 [WGS84] (NIMA TR8350.2, 3rd ed., 2000): the
# semi-major axis ``a`` and the inverse flattening ``1/f``.  Everything else is
# DERIVED from these so the ellipsoid is internally consistent and exact -- the
# earlier code hard-coded a rounded ``b = 6356.752 km`` alongside ``a``, which
# over-specifies the ellipsoid and leaves ``b``/``e^2`` slightly off (item A1/A5).
WGS84_A_KM = 6378.137               # semi-major axis a [km]  [WGS84]
WGS84_INV_FLATTENING = 298.257223563  # inverse flattening 1/f  [WGS84]
WGS84_F = 1.0 / WGS84_INV_FLATTENING  # flattening f = (a - b) / a
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)  # semi-minor axis b = a(1 - f) [km]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)       # first eccentricity squared e^2 = f(2 - f)

# --- Fundamental-plane unit radius -------------------------------------------
# The Besselian fundamental-plane coordinates (x, y, l1, l2, ...) are expressed
# in units of Earth's equatorial radius [ES92] ch. 8.  The geographic reduction
# (:func:`app.geography.fund_to_geo`) is carried out on the WGS-84 ellipsoid
# normalized to unit semi-major axis, so the SAME radius must scale the
# geocentric Sun/Moon vectors in :mod:`app.ephemeris`.  Using the WGS-84 ``a``
# keeps the fundamental plane and the reduction ellipsoid consistent; previously
# the SPICE PCK value ``a_e = 6378.1366 km`` scaled the vectors while the
# reduction used WGS-84 ``a = 6378.137 km`` -- a ~0.4 m mismatch (item A1).
EARTH_EQUATORIAL_RADIUS_KM = WGS84_A_KM

# --- Mean radius for short great-circle offsets ------------------------------
# IUGG arithmetic mean radius R_1 = (2a + b) / 3, used only by the haversine /
# destination helpers for the short (<~600 km) perpendicular offsets when
# root-finding the shadow-edge limits; the ellipsoidal reduction itself uses the
# full WGS-84 ellipsoid above.
EARTH_MEAN_RADIUS_KM = (2.0 * WGS84_A_KM + WGS84_B_KM) / 3.0

# --- Solar radius --------------------------------------------------------------
# The IAU (1976) solar radius, 696 000 km, the value the [Espenak] predictions
# pair with the lunar radii k1/k2 (``app.ephemeris.K_PENUMBRA``/``K_UMBRA``) in
# the shadow-cone angles ``sin f = (d_s +/- k) / G`` [ES92] eq. 8.323-1.  It is
# a constant, NOT read from the loaded PCK: NAIF's pck00011 carries the IAU 2015
# nominal 695 700 km, and reading it made the cones -- and every umbral width --
# depend on which kernel set was loaded (review item W1: tan f 0.04% low, l2
# ~0.7 km too negative, totals ~1% wide).  Espenak's published tan f1 / tan f2
# imply 695 982 - 695 995 km for six reference eclipses (rounding of the 7th
# decimal), i.e. this value.
SUN_RADIUS_KM = 696000.0

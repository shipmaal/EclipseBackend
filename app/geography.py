"""Geometry helpers: fundamental-plane -> geographic reduction and time formatting.

Given the Besselian ``x, y`` (shadow-axis position on the fundamental plane, in
Earth equatorial radii) and the axis orientation ``d, mu``, find the geographic
point where the axis pierces the Earth ellipsoid (the eclipse central line).
Follows the Explanatory Supplement to the Astronomical Almanac [ES92] sec. 8.33;
the parametric <-> geodetic latitude relation is Meeus, *Astronomical Algorithms*
[Meeus98] ch. 11 ("The Earth's Globe", eq. 11.1).

Two correctness fixes over the earlier astropy version:

* The frame rotation now uses the correct combination
  ``sin(phi) = eta1 cos d1 + zeta1 sin d1`` (the old code had the wrong sign and
  swapped components, putting the central line ~15 deg off in latitude).
* The auxiliary latitude produced by the reduction is the **parametric
  (reduced)** latitude, so it is converted to geodetic with
  ``tan(phi_geodetic) = tan(phi_parametric) / (1 - f)``.  (Verified against the
  published 2024-04-08 central line: this drops the residual to <1 km, whereas
  the intuitive ``/(1 - e^2)`` double-counts the flattening and leaves ~11 km.)
  This also replaces the old Bowring call that returned a bare float and then
  crashed on ``.to(u.deg)``.

Everything is pure NumPy in radians.  Ellipsoid: WGS-84.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from . import native
from .constants import (
    EARTH_MEAN_RADIUS_KM,
    WGS84_A_KM,
    WGS84_E2,
    WGS84_F,
)

if TYPE_CHECKING:
    from .besselian import BesselianModel

# WGS-84 ellipsoid (cited + derived in app.constants; item A1/R2).
_A_KM = WGS84_A_KM  # semi-major axis [km]
_F = WGS84_F        # flattening
_E2 = WGS84_E2      # first eccentricity squared


def _native_args(*values) -> tuple[tuple[int, ...], list[np.ndarray]]:
    """Broadcast ``values`` as float and flatten each for the native core.

    The ``_eclipse`` array functions take equal-length, C-contiguous 1-D float64
    arrays and do no broadcasting of their own; broadcast views are zero-stride
    and read-only, so each is copied contiguous before ravelling.  Returns the
    common broadcast shape (``()`` for all-scalar input) and the flat arrays;
    the caller reshapes the results back with :func:`_native_shape`.
    """
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in values))
    return arrays[0].shape, [np.ascontiguousarray(a).ravel() for a in arrays]


def _native_shape(flat: np.ndarray, shape: tuple[int, ...]):
    """Restore a flat native result to ``shape``; ``()`` gives a NumPy scalar
    (``np.float64``), as the NumPy expressions in the Python bodies produce."""
    return flat.reshape(shape)[()]


class _ReductionAux(NamedTuple):
    """Ellipsoid-reduction auxiliaries at the axis declination ``d`` [ES92] 8.331."""

    rho1: float       # sqrt(1 - e^2 cos^2 d)
    rho2: float       # sqrt(1 - e^2 sin^2 d)
    sin_d1: float     # sin d1 = sin d / rho1
    cos_d1: float     # cos d1 = sqrt(1 - e^2) cos d / rho1
    sin_d1_d2: float  # sin(d1 - d2) = e^2 sin d cos d / (rho1 rho2)
    cos_d1_d2: float  # cos(d1 - d2) = sqrt(1 - e^2) / (rho1 rho2)


def _reduction_aux(d: float) -> _ReductionAux:
    """Auxiliary quantities of the ellipsoid reduction at axis declination ``d`` (radians).

    The reduction carries the WGS-84 flattening into the fundamental plane through
    the auxiliary declinations ``d1`` (and ``d1 - d2``) and the radii ``rho1``,
    ``rho2`` (Explanatory Supplement to the Astronomical Almanac, [ES92] eq. 8.331).
    Factored out of :func:`fund_to_geo`, :func:`geo_to_fund` and
    :func:`shadow_radii`, which all need some subset of these (item R1).
    """
    cos_d, sin_d = np.cos(d), np.sin(d)
    root = np.sqrt(1.0 - _E2)
    rho1 = np.sqrt(1.0 - _E2 * cos_d**2)
    rho2 = np.sqrt(1.0 - _E2 * sin_d**2)
    return _ReductionAux(
        rho1=rho1,
        rho2=rho2,
        sin_d1=sin_d / rho1,
        cos_d1=root * cos_d / rho1,
        sin_d1_d2=_E2 * sin_d * cos_d / (rho1 * rho2),
        cos_d1_d2=root / (rho1 * rho2),
    )


def fund_to_geo(x: float, y: float, d_deg: float, mu_deg: float) -> tuple[float, float]:
    """Convert fundamental-plane coordinates to geographic (lon, lat) in degrees.

    ``x, y`` are in Earth equatorial radii; ``d_deg`` and ``mu_deg`` are the axis
    declination and Greenwich hour angle in degrees.  Longitude is returned in
    (-180, 180], east positive.  Raises ``ValueError`` when the axis misses the
    Earth (``x**2 + y1**2 > 1``) -- i.e. no central line at this instant.

    Ellipsoid reduction on the auxiliary sphere [ES92] eq. 8.331-8.334; the
    resulting parametric (reduced) latitude is converted to geodetic latitude
    with ``tan(phi) = tan(phi1)/(1 - f)`` [Meeus98] ch. 11, eq. 11.1.

    Geometric only: no atmospheric refraction (matters for a very low Sun) and
    the mean lunar limb is folded into the umbral radius k2 (item A3).
    """
    aux = _reduction_aux(np.radians(d_deg))

    eta1 = y / aux.rho1
    disc = 1.0 - x**2 - eta1**2
    if disc < 0.0:
        raise ValueError("shadow axis does not intersect the Earth at this instant")
    zeta1 = np.sqrt(disc)

    # Fundamental-plane -> geocentric direction (auxiliary sphere, latitude d1)
    # [ES92] eq. 8.333.
    sin_phi1 = eta1 * aux.cos_d1 + zeta1 * aux.sin_d1
    theta = np.arctan2(x, zeta1 * aux.cos_d1 - eta1 * aux.sin_d1)  # hour angle E of axis meridian
    phi1 = np.arcsin(sin_phi1)  # parametric (reduced) latitude

    # Parametric -> geodetic latitude [Meeus98] ch. 11, eq. 11.1.
    phi = np.arctan(np.tan(phi1) / (1.0 - _F))

    lon = (np.degrees(theta) - mu_deg + 180.0) % 360.0 - 180.0
    lat = np.degrees(phi)
    return float(lon), float(lat)


def fund_to_geo_v(x, y, d_deg, mu_deg) -> tuple[np.ndarray, np.ndarray]:
    """Array form of :func:`fund_to_geo` (same math, [ES92] 8.331-8.334).

    Broadcasts over its arguments and returns ``(lon, lat)`` arrays in degrees,
    with ``NaN`` where the axis misses the Earth instead of raising.
    Dispatches to the native core under ``ECLIPSE_BACKEND=native``.
    """
    if native.is_native():
        shape, flat = _native_args(x, y, d_deg, mu_deg)
        lon, lat = native.module().fund_to_geo(*flat)
        return _native_shape(lon, shape), _native_shape(lat, shape)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu_deg = np.asarray(mu_deg, dtype=float)
    aux = _reduction_aux(np.radians(np.asarray(d_deg, dtype=float)))

    eta1 = y / aux.rho1
    disc = 1.0 - x**2 - eta1**2
    zeta1 = np.sqrt(np.where(disc >= 0.0, disc, np.nan))

    sin_phi1 = eta1 * aux.cos_d1 + zeta1 * aux.sin_d1
    theta = np.arctan2(x, zeta1 * aux.cos_d1 - eta1 * aux.sin_d1)
    phi1 = np.arcsin(np.clip(sin_phi1, -1.0, 1.0))
    phi = np.arctan(np.tan(phi1) / (1.0 - _F))

    lon = (np.degrees(theta) - mu_deg + 180.0) % 360.0 - 180.0
    return lon, np.degrees(phi)


def geo_to_fund(lat_deg: float, lon_deg: float, d_deg: float, mu_deg: float,
                height_m: float = 0.0):
    """Forward transform: geographic (lat, lon, height) -> fundamental-plane (xi, eta, zeta).

    Inverse of :func:`fund_to_geo` for a point on the WGS-84 ellipsoid.  ``xi, eta``
    are the point's fundamental-plane coordinates (Earth equatorial radii) and
    ``zeta`` its distance along the shadow axis; used for shadow-edge / limit
    computations and local circumstances.

    On the ellipsoid (``height_m = 0``) this is the inverse ellipsoid reduction
    [ES92] eq. 8.331 (via :func:`_reduction_aux`), with the geodetic->parametric
    latitude ``tan(beta) = (1 - f) tan(phi)`` [Meeus98] ch. 11, eq. 11.1 --
    an exact rotation of the ellipsoid point ``(cos beta cos lon, cos beta
    sin lon, (b/a) sin beta)`` into the fundamental-plane axes.

    A point at height ``H`` [m] above the ellipsoid (along the geodetic normal)
    has the geocentric coordinates of [Meeus98] ch. 11 ("Geocentric rectangular
    coordinates of an observer"; Meeus writes H / 6378140 m, here H / a WGS-84)

        rho sin phi' = (b/a) sin beta + (H/a) sin phi
        rho cos phi' =       cos beta + (H/a) cos phi,

    i.e. the ellipsoid point plus ``H/a`` times the unit normal
    ``(cos phi cos lon, cos phi sin lon, sin phi)``.  The rotation into the
    fundamental plane is linear, so the height adds the rotated normal, with
    the standard spherical form of [MeeusSE], [NASA-LC]
    (``xi = rho cos phi' sin theta``, ``eta = rho sin phi' cos d - rho cos phi'
    cos theta sin d``, ``zeta = rho sin phi' sin d + rho cos phi' cos theta
    cos d``) applied to the normal alone:

        xi   += (H/a) cos phi sin theta
        eta  += (H/a) (sin phi cos d - cos phi cos theta sin d)
        zeta += (H/a) (sin phi sin d + cos phi cos theta cos d).

    This is the same projection NASA's local-circumstances calculator applies
    to the whole ``rho sin phi'``, ``rho cos phi'`` [NASA-LC] (``readdata`` /
    ``timelocdependent``); split into the ellipsoid and height parts here so
    the sea-level part stays the [ES92] 8.331 reduction.  Exact, not a
    first-order term (``tests/test_geography.py`` checks it against geodetic
    -> ECEF -> rotated axes; the two forms agree to 4e-16).  When every height is zero the
    term is skipped, so the sea-level result is bit-identical to the reduction
    alone.

    Array-native: every argument may be a scalar or an array and the usual NumPy
    broadcasting applies (e.g. observers of shape ``(P, 1)`` against elements of
    shape ``(N,)`` give ``(P, N)`` results).  Scalars in give Python floats out.
    Dispatches to the native core under ``ECLIPSE_BACKEND=native``.
    """
    if native.is_native():
        shape, flat = _native_args(lat_deg, lon_deg, d_deg, mu_deg, height_m)
        xi, eta, zeta = native.module().geo_to_fund(*flat)
        if shape == ():
            return float(xi[0]), float(eta[0]), float(zeta[0])
        return xi.reshape(shape), eta.reshape(shape), zeta.reshape(shape)
    lat_deg = np.asarray(lat_deg, dtype=float)
    lon_deg = np.asarray(lon_deg, dtype=float)
    mu_deg = np.asarray(mu_deg, dtype=float)
    height_m = np.asarray(height_m, dtype=float)
    d = np.radians(np.asarray(d_deg, dtype=float))
    aux = _reduction_aux(d)

    beta = np.arctan((1.0 - _F) * np.tan(np.radians(lat_deg)))  # parametric latitude
    theta = np.radians(lon_deg + mu_deg)  # hour angle east of the axis meridian
    cb, sb = np.cos(beta), np.sin(beta)

    xi = cb * np.sin(theta)
    eta1 = sb * aux.cos_d1 - cb * np.cos(theta) * aux.sin_d1
    zeta1 = sb * aux.sin_d1 + cb * np.cos(theta) * aux.cos_d1
    eta = eta1 * aux.rho1
    zeta = aux.rho2 * (zeta1 * aux.cos_d1_d2 - eta1 * aux.sin_d1_d2)
    if np.any(height_m != 0.0):
        # Height along the geodetic normal [Meeus98] ch. 11, rotated
        # into the fundamental plane [MeeusSE], [NASA-LC] (docstring).
        h = height_m / (_A_KM * 1000.0)  # H/a
        phi = np.radians(lat_deg)
        cp, sp = np.cos(phi), np.sin(phi)
        cos_d, sin_d = np.cos(d), np.sin(d)
        xi = xi + h * (cp * np.sin(theta))
        eta = eta + h * (sp * cos_d - cp * np.cos(theta) * sin_d)
        zeta = zeta + h * (sp * sin_d + cp * np.cos(theta) * cos_d)
    if xi.ndim == 0 and eta.ndim == 0 and zeta.ndim == 0:
        return float(xi), float(eta), float(zeta)
    return xi, eta, zeta


def shadow_radii(
    x: float, y: float, d_deg: float, l1: float, l2: float, tan_f1: float, tan_f2: float
) -> tuple[float, float, bool]:
    """Approximate penumbra/umbra shadow radii on the ground, in km.

    Returns ``(penumbra_km, umbra_km, is_total)``.  ``umbra_km`` is the absolute
    radius; ``is_total`` is True for a total eclipse (umbra reaches the ground)
    and False for annular.  Uses the shadow-cone radii reduced to the observer's
    distance below the fundamental plane, ``L = l - zeta*tan f`` [ES92] eq. 8.353;
    adequate for visualization, not for precise limit computation (use
    :func:`shadow_edge_limits` for the true limits/width).
    Dispatches to the native core under ``ECLIPSE_BACKEND=native``.
    """
    if native.is_native():
        return native.module().shadow_radii(
            float(x), float(y), float(d_deg), float(l1), float(l2), float(tan_f1), float(tan_f2)
        )
    eta1 = y / _reduction_aux(np.radians(d_deg)).rho1
    disc = 1.0 - x**2 - eta1**2
    if disc < 0.0:
        return 0.0, 0.0, False
    zeta = np.sqrt(disc)

    L1 = l1 - zeta * tan_f1  # penumbra radius at the observer (Earth radii)
    L2 = l2 - zeta * tan_f2  # umbra radius; negative => total, positive => annular
    return float(abs(L1) * _A_KM), float(abs(L2) * _A_KM), bool(L2 < 0.0)


_EARTH_MEAN_KM = EARTH_MEAN_RADIUS_KM  # mean radius for short great-circle offsets


# Great-circle helpers (standard spherical trigonometry): destination point,
# haversine distance and initial bearing on a sphere of radius _EARTH_MEAN_KM.
# Used only for the short (<~600 km) perpendicular offsets when root-finding the
# shadow-edge limits; the reduction itself is on the full WGS-84 ellipsoid.
def _destination(lat_deg, lon_deg, bearing_deg, dist_km):
    """Great-circle destination from a point given bearing and distance."""
    ad = dist_km / _EARTH_MEAN_KM
    lat1, brg = np.radians(lat_deg), np.radians(bearing_deg)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(ad) + np.cos(lat1) * np.sin(ad) * np.cos(brg))
    lon2 = np.radians(lon_deg) + np.arctan2(
        np.sin(brg) * np.sin(ad) * np.cos(lat1), np.cos(ad) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), (np.degrees(lon2) + 180.0) % 360.0 - 180.0


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * _EARTH_MEAN_KM * np.arcsin(np.sqrt(a))


def geodesic_km(lat1, lon1, lat2, lon2):
    """Distance [km] between two points on the WGS-84 ellipsoid (geodetic degrees).

    H. Andoyer's formula with its first-order flattening correction, as given in
    [Meeus98] ch. 11 ("Distance between two points on the Earth's surface"):
    ``F, G, lambda`` the half-sum / half-differences, ``S, C``, ``tan w =
    sqrt(S / C)``, ``R = sqrt(S C) / w``, ``D = 2 w a``, ``H1 = (3R - 1) / 2C``,
    ``H2 = (3R + 1) / 2S`` and ``s = D (1 + f H1 sin^2 F cos^2 G - f H2 cos^2 F
    sin^2 G)``; accurate to O(f^2) (tens of metres at path-width scale).  This is
    the path-width metric: the mean-radius sphere of :func:`_haversine_km`
    understates a polar width by ~0.4% (2021-12-04: 416.1 vs 417.8 km) and
    overstates an equatorial one by ~0.5% (review item W2).  Undefined (NaN)
    for coincident points.
    """
    fh = np.radians((lat1 + lat2) / 2.0)
    gh = np.radians((lat1 - lat2) / 2.0)
    lh = np.radians(((lon1 - lon2 + 180.0) % 360.0 - 180.0) / 2.0)
    sin_g, cos_g = np.sin(gh), np.cos(gh)
    sin_f, cos_f = np.sin(fh), np.cos(fh)
    sin_l, cos_l = np.sin(lh), np.cos(lh)
    s = sin_g * sin_g * (cos_l * cos_l) + cos_f * cos_f * (sin_l * sin_l)
    c = cos_g * cos_g * (cos_l * cos_l) + sin_f * sin_f * (sin_l * sin_l)
    w = np.arctan(np.sqrt(s / c))
    r = np.sqrt(s * c) / w
    dist = 2.0 * w * _A_KM
    h1 = (3.0 * r - 1.0) / (2.0 * c)
    h2 = (3.0 * r + 1.0) / (2.0 * s)
    return dist * (1.0 + _F * h1 * (sin_f * sin_f) * (cos_g * cos_g)
                   - _F * h2 * (cos_f * cos_f) * (sin_g * sin_g))


def bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing (degrees) from point 1 to point 2.

    Broadcasts over its arguments; dispatches to the native core under
    ``ECLIPSE_BACKEND=native``.
    """
    if native.is_native():
        shape, flat = _native_args(lat1, lon1, lat2, lon2)
        return _native_shape(native.module().bearing(*flat), shape)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


# Path-limit envelope (review item W2): with element rates, a point is inside
# the path when it is inside the shadow at the instant the shadow axis passes
# closest to it, found by Newton steps on the squared axis separation with
# central finite differences of step ENVELOPE_H [h], the step clamped to
# +/- ENVELOPE_MAX_TAU_H [h] from the track instant. (numerical)
ENVELOPE_H = 1e-3
ENVELOPE_ITERATIONS = 3
ENVELOPE_MAX_TAU_H = 0.25


def shadow_edge_limits_v(x, y, d_deg, mu_deg, l, tan_f, path_bearing_deg, max_km=600.0,
                         sunlit_only=True, rates=None):
    """Shadow-edge points and widths for arrays of track instants (vectorized).

    Array form of :func:`shadow_edge_limits`: every argument is a 1-D array of
    the same length ``n`` (or a scalar).  Returns ``(north_lat, north_lon,
    south_lat, south_lon, width_km)`` arrays; entries are ``NaN`` (width ``0``)
    where the central point is outside the shadow or the edge is not found
    within ``max_km``.  The width is the ellipsoidal distance between the two
    points (:func:`geodesic_km`).  The bisection runs on all ``2n`` edge searches at once,
    so the cost is 50 array evaluations regardless of ``n``.

    ``sunlit_only`` treats points beyond the terminator (``zeta <= 0``, Sun
    geometrically below the horizon) as outside the shadow, so a penumbral
    limit that would fall on the night side is clipped to the terminator --
    the boundary of where the partial eclipse is actually *seen*.  For the
    umbra (``max_km`` ~600) it never engages.

    ``rates``, when given, is ``(dx, dy, dd_deg, dmu_deg, dl)``: the per-hour
    rates of ``x, y, d, mu, l`` at each instant (:func:`element_rates`).  The
    limits are then the path's -- the envelope of the shadow over time -- not
    the shadow's outline at the one instant: a point counts as inside when it
    is inside at the instant of the axis' closest approach to it, with the
    elements linear in time about the track instant and the Earth's rotation
    exact through :func:`geo_to_fund`.  The two agree to < 0.4 km for a
    high-Sun path but not for a grazing one (2021-12-04: 412.0 km at one
    instant, 418.5 km path width; review item W2).  Without ``rates`` the
    outline at the instant is returned (the penumbral limits of
    ``/central-line`` use that).
    Dispatches to the native core under ``ECLIPSE_BACKEND=native``.
    """
    x, y, d_deg, mu_deg, l, tan_f, brg = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v, dtype=float))
          for v in (x, y, d_deg, mu_deg, l, tan_f, path_bearing_deg))
    )
    if rates is not None:
        rx, ry, rd, rmu, rl = (np.ascontiguousarray(np.broadcast_to(
            np.asarray(v, dtype=float), x.shape)) for v in rates)
    if native.is_native():
        extra = {} if rates is None else {"rates": (rx, ry, rd, rmu, rl)}
        return native.module().shadow_edge_limits(
            *(np.ascontiguousarray(v) for v in (x, y, d_deg, mu_deg, l, tan_f, brg)),
            max_km=float(max_km), sunlit_only=bool(sunlit_only), **extra,
        )
    n = len(x)
    lon0, lat0 = fund_to_geo_v(x, y, d_deg, mu_deg)

    # Residual for the two perpendicular searches stacked as (2, n): edge
    # condition |axis separation| - |l - zeta tan f| = 0 [ES92] / [MeeusSE].
    bearings = np.stack([(brg - 90.0) % 360.0, (brg + 90.0) % 360.0])
    xx, yy, dd, mm, ll, tf = (np.broadcast_to(v, (2, n)) for v in (x, y, d_deg, mu_deg, l, tan_f))

    def instant_residual(lat, lon):
        xi, eta, zeta = geo_to_fund(lat, lon, dd, mm)
        r = np.hypot(xi - xx, eta - yy) - np.abs(ll - zeta * tf)
        return np.where(zeta <= 0.0, 1.0, r) if sunlit_only else r

    if rates is not None:
        rxx, ryy, rdd, rmm, rll = (np.broadcast_to(v, (2, n)) for v in (rx, ry, rd, rmu, rl))

    def separation2(lat, lon, tau):
        # Squared axis separation and zeta at tau [h] from the track instant,
        # elements linear in tau; [ES92] eq. 8.331 via geo_to_fund.
        xi, eta, zeta = geo_to_fund(lat, lon, dd + rdd * tau, mm + rmm * tau)
        u = (xx + rxx * tau) - xi
        v = (yy + ryy * tau) - eta
        return u * u + v * v, zeta

    def envelope_residual(lat, lon):
        h = ENVELOPE_H
        tau = np.zeros(np.shape(lat))
        for _ in range(ENVELOPE_ITERATIONS):
            s_m = separation2(lat, lon, tau - h)[0]
            s_0 = separation2(lat, lon, tau)[0]
            s_p = separation2(lat, lon, tau + h)[0]
            g = (s_p - s_m) / (2.0 * h)
            c = (s_p - 2.0 * s_0 + s_m) / (h * h)
            tau = np.where(c > 0.0,
                           np.clip(tau - g / c, -ENVELOPE_MAX_TAU_H, ENVELOPE_MAX_TAU_H), tau)
        s2, zeta = separation2(lat, lon, tau)
        r = np.sqrt(s2) - np.abs((ll + rll * tau) - zeta * tf)
        return np.where(zeta <= 0.0, 1.0, r) if sunlit_only else r

    residual = instant_residual if rates is None else envelope_residual

    inside = residual(lat0, lon0) < 0.0  # central point inside this shadow (n,)
    s_lo = np.zeros((2, n))
    s_hi = np.full((2, n), float(max_km))
    found = (residual(*_destination(lat0, lon0, bearings, s_hi)) >= 0.0) & inside  # (2, n)
    for _ in range(50):
        s_mid = 0.5 * (s_lo + s_hi)
        r = residual(*_destination(lat0, lon0, bearings, s_mid))
        s_lo = np.where(r < 0.0, s_mid, s_lo)
        s_hi = np.where(r < 0.0, s_hi, s_mid)
    lat_e, lon_e = _destination(lat0, lon0, bearings, 0.5 * (s_lo + s_hi))

    ok = found[0] & found[1]
    lat_e = np.where(ok, lat_e, np.nan)
    lon_e = np.where(ok, lon_e, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):  # NaN points where not ok
        width = np.where(ok, geodesic_km(lat_e[0], lon_e[0], lat_e[1], lon_e[1]), 0.0)
    first_is_north = lat_e[0] >= lat_e[1]
    n_lat = np.where(first_is_north, lat_e[0], lat_e[1])
    n_lon = np.where(first_is_north, lon_e[0], lon_e[1])
    s_lat = np.where(first_is_north, lat_e[1], lat_e[0])
    s_lon = np.where(first_is_north, lon_e[1], lon_e[0])
    return n_lat, n_lon, s_lat, s_lon, width


def shadow_edge_limits(x, y, d_deg, mu_deg, l, tan_f, path_bearing_deg, max_km=600.0,
                       rates=None):
    """Find the two shadow-edge points perpendicular to the track, and the width.

    ``l``/``tan_f`` are the fundamental-plane radius and cone tangent of the shadow
    (umbra: l2, tan_f2; penumbra: l1, tan_f1).  Marches perpendicular to
    ``path_bearing_deg`` from the central point and root-finds where the point
    crosses the shadow-cone edge on the WGS-84 ellipsoid.  Returns
    ``(north_point, south_point, width_km)`` with points as (lat, lon), or
    ``(None, None, 0.0)`` if the central point is outside the shadow (no width).

    The edge condition -- separation from the shadow axis equals the cone radius
    ``|l - zeta*tan f|`` reduced to the ground -- is the umbral-limit definition
    of [ES92] / [MeeusSE]; solving it by bisection here is a numerical method, not
    a cited closed form.  Geometric only (no refraction; mean lunar limb), item A3.
    ``rates`` (per-hour ``dx, dy, dd, dmu, dl``) makes them the path limits, the
    envelope over time (see :func:`shadow_edge_limits_v`).
    Scalar wrapper over :func:`shadow_edge_limits_v`.
    """
    n_lat, n_lon, s_lat, s_lon, width = shadow_edge_limits_v(
        x, y, d_deg, mu_deg, l, tan_f, path_bearing_deg, max_km, rates=rates
    )
    if np.isnan(n_lat[0]):
        return None, None, 0.0
    return (float(n_lat[0]), float(n_lon[0])), (float(s_lat[0]), float(s_lon[0])), float(width[0])


# Global (whole-Earth) contacts: the instants the penumbra / umbra first and
# last touch the Earth's outline.  In the fundamental plane the outline is the
# auxiliary circle of unit radius in (x, y/rho1) [ES92] eq. 8.331, so the
# external contacts are where the axis distance equals 1 + (cone radius) and
# the internal umbral contacts (whole umbra on the Earth) where it equals
# 1 - |l2| [ES92] sec. 8.34.  Treating the cone radius as unscaled by rho1 is
# the standard bulletin approximation (a few seconds).
GLOBAL_CONTACTS = ("P1", "U1", "U2", "U3", "U4", "P4")


def global_contacts(model: BesselianModel, half_window_hours: float = 5.0) -> dict[str, float]:
    """Times (hours from T0) of P1/P4 (penumbra) and U1-U4 (umbra) global contacts.

    Direct evaluation on a 1-minute grid over ``+/- half_window_hours`` (a solar
    eclipse lasts at most ~6 h on Earth), roots bracketed by sign change and
    bisected with one vectorized evaluation per iteration.  Missing contacts
    (no umbra on Earth, or the window too narrow) are omitted from the dict.
    Dispatches to the native core under ``ECLIPSE_BACKEND=native`` (the
    elements are re-evaluated there from the model's ``et0`` / frame).
    """
    if native.is_native():
        return dict(native.module().global_contacts(
            model.et0, model.earth_frame, float(half_window_hours)
        ))
    from .numerics import bisect, sign_changes

    def rho_and_radii(t):
        e = model.evaluate_direct(t)
        aux = _reduction_aux(np.radians(e["d"]))
        rho = np.hypot(e["x"], e["y"] / aux.rho1)
        return rho, e["l1"], np.abs(e["l2"])

    t = np.arange(-half_window_hours, half_window_hours + 1e-9, 1.0 / 60.0)
    rho, l1, l2 = rho_and_radii(t)
    conditions = {
        "P": rho - (1.0 + l1),   # penumbra external contacts P1 (falling), P4 (rising)
        "UE": rho - (1.0 + l2),  # umbra external contacts U1, U4
        "UI": rho - (1.0 - l2),  # umbra internal contacts U2, U3
    }
    names = {"P": ("P1", "P4"), "UE": ("U1", "U4"), "UI": ("U2", "U3")}
    brackets: list[tuple[str, int]] = []
    for key, f in conditions.items():
        crossings = sign_changes(t, f)
        falling = [i for i, rising in crossings if not rising]
        rising_ = [i for i, rising in crossings if rising]
        if falling:
            brackets.append((names[key][0], min(falling)))
        if rising_:
            brackets.append((names[key][1], max(rising_)))
    if not brackets:
        return {}

    kinds = np.array([b[0][0] + ("I" if b[0][1] in "23" else "E") for b in brackets])

    def f_all(tt):
        r, a1, a2 = rho_and_radii(tt)
        return np.select(
            [kinds == "PE", kinds == "UE", kinds == "UI"],
            [r - (1.0 + a1), r - (1.0 + a2), r - (1.0 - a2)],
        )

    idx = np.array([b[1] for b in brackets])
    roots = bisect(f_all, t[idx], t[idx + 1])
    return {name: float(tc) for (name, _), tc in zip(brackets, roots, strict=True)}


class TrackPoint(NamedTuple):
    """One point of the central-line track (index into the source element arrays)."""

    i: int          # index into the element arrays returned alongside
    t_hours: float  # offset from T0 [hours]
    lat: float      # geodetic latitude [deg]
    lon: float      # longitude, east positive, (-180, 180] [deg]
    bearing: float  # along-track great-circle bearing [deg]


def central_track(
    model: BesselianModel, t_hours: np.ndarray
) -> tuple[dict[str, np.ndarray], list[TrackPoint]]:
    """Central-line geographic track over ``t_hours`` (hours from T0).

    Evaluates the elements directly at each instant (item A2), reduces each to the
    geographic point where the shadow axis pierces the WGS-84 ellipsoid
    (:func:`fund_to_geo`), skipping instants where the axis misses the Earth, and
    tags each surviving point with the along-track great-circle bearing from its
    neighbours.  Returns ``(elems, points)`` where ``elems`` is the direct
    element evaluation (so the caller can read ``elems[key][pt.i]`` for shadow
    limits/width without re-evaluating) and ``points`` is a list of
    :class:`TrackPoint`.  Shared by ``/central-line`` and the integration tests
    (item R3).
    """
    t = np.atleast_1d(np.asarray(t_hours, dtype=float))
    elems = model.evaluate_direct(t)

    lon_all, lat_all = fund_to_geo_v(elems["x"], elems["y"], elems["d"], elems["mu"])
    valid: list[tuple[int, float, float, float]] = [  # (i, t, lat, lon)
        (i, float(t[i]), float(lat_all[i]), float(lon_all[i]))
        for i in np.flatnonzero(~np.isnan(lat_all))
    ]

    points: list[TrackPoint] = []
    n = len(valid)
    for k, (i, ti, lat, lon) in enumerate(valid):
        prev = valid[max(0, k - 1)]
        nxt = valid[min(n - 1, k + 1)]
        brg = bearing(prev[2], prev[3], nxt[2], nxt[3]) if n > 1 else 0.0
        points.append(TrackPoint(i=i, t_hours=ti, lat=lat, lon=lon, bearing=brg))
    return elems, points


# Half-step [h] of the central differences in :func:`element_rates`: one
# minute, over which the elements are polynomial to far below the rates' use.
RATE_DT_H = 1.0 / 60.0


def element_rates(model: BesselianModel, t_hours) -> dict[str, np.ndarray]:
    """Per-hour rates of ``x, y, d, mu, l1, l2`` at ``t_hours`` (hours from T0).

    Central differences over +/- :data:`RATE_DT_H` of the direct element
    evaluation (one vectorized call); ``d``/``mu`` in degrees per hour, the
    lengths in Earth radii per hour.  The ``mu`` difference is wrapped to
    (-180, 180] before dividing.  These are the ``rates`` of the path-limit
    envelope in :func:`shadow_edge_limits_v`. (numerical)
    """
    t = np.atleast_1d(np.asarray(t_hours, dtype=float))
    n = len(t)
    e = model.evaluate_direct(np.concatenate([t - RATE_DT_H, t + RATE_DT_H]))
    out = {k: (e[k][n:] - e[k][:n]) / (2.0 * RATE_DT_H) for k in ("x", "y", "d", "l1", "l2")}
    dmu = (e["mu"][n:] - e["mu"][:n] + 180.0) % 360.0 - 180.0
    out["mu"] = dmu / (2.0 * RATE_DT_H)
    return out


def dec_to_hms(t_hours: float, t0_hours: float = 0.0) -> tuple[int, int, float]:
    """Decimal hours (offset by ``t0_hours``) -> (hour, minute, second)."""
    decimal_hour = t0_hours + t_hours
    hour = int(decimal_hour)
    minute_f = (decimal_hour - hour) * 60.0
    minute = int(minute_f)
    second = (minute_f - minute) * 60.0
    if round(second) == 60:
        minute += 1
        second = 0.0
    if minute == 60:
        hour += 1
        minute = 0
    return hour, minute, second


def format_offset(t_hours: float) -> str:
    """Signed offset from T0 as ``+HH:MM:SS.s`` / ``-HH:MM:SS.s`` (item R4)."""
    sign = "-" if t_hours < 0 else "+"
    h, m, s = dec_to_hms(abs(t_hours))
    return f"{sign}{h:02d}:{m:02d}:{s:04.1f}"


def format_clock(t0_utc: str, t_hours: float) -> str:
    """Absolute UTC wall-clock ``HH:MM:SS`` at ``t_hours`` from ISO ``t0_utc`` (item R4).

    Central home for time formatting (was ``circumstances._clock``); ``datetime``
    handles date rollover across midnight.
    """
    return (
        datetime.fromisoformat(t0_utc).replace(tzinfo=timezone.utc)
        + timedelta(hours=t_hours)
    ).strftime("%H:%M:%S")

"""Geometry helpers: fundamental-plane -> geographic reduction and time formatting.

Given the Besselian ``x, y`` (shadow-axis position on the fundamental plane, in
Earth equatorial radii) and the axis orientation ``d, mu``, find the geographic
point where the axis pierces the Earth ellipsoid (the eclipse central line).
Follows the Explanatory Supplement to the Astronomical Almanac (sec. 8.33) /
Meeus, *Astronomical Algorithms* ch. 54.

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

import numpy as np

from .constants import (
    EARTH_MEAN_RADIUS_KM,
    WGS84_A_KM,
    WGS84_E2,
    WGS84_F,
)

# WGS-84 ellipsoid (cited + derived in app.constants; item A1/R2).
_A_KM = WGS84_A_KM  # semi-major axis [km]
_F = WGS84_F        # flattening
_E2 = WGS84_E2      # first eccentricity squared


def fund_to_geo(x: float, y: float, d_deg: float, mu_deg: float) -> tuple[float, float]:
    """Convert fundamental-plane coordinates to geographic (lon, lat) in degrees.

    ``x, y`` are in Earth equatorial radii; ``d_deg`` and ``mu_deg`` are the axis
    declination and Greenwich hour angle in degrees.  Longitude is returned in
    (-180, 180], east positive.  Raises ``ValueError`` when the axis misses the
    Earth (``x**2 + y1**2 > 1``) -- i.e. no central line at this instant.
    """
    d = np.radians(d_deg)

    rho1 = np.sqrt(1.0 - _E2 * np.cos(d) ** 2)
    sin_d1 = np.sin(d) / rho1
    cos_d1 = np.sqrt(1.0 - _E2) * np.cos(d) / rho1

    eta1 = y / rho1
    disc = 1.0 - x**2 - eta1**2
    if disc < 0.0:
        raise ValueError("shadow axis does not intersect the Earth at this instant")
    zeta1 = np.sqrt(disc)

    # Fundamental-plane -> geocentric direction (auxiliary sphere, latitude d1).
    sin_phi1 = eta1 * cos_d1 + zeta1 * sin_d1
    theta = np.arctan2(x, zeta1 * cos_d1 - eta1 * sin_d1)  # hour angle E of axis meridian
    phi1 = np.arcsin(sin_phi1)  # parametric (reduced) latitude

    # Parametric -> geodetic latitude.
    phi = np.arctan(np.tan(phi1) / (1.0 - _F))

    lon = (np.degrees(theta) - mu_deg + 180.0) % 360.0 - 180.0
    lat = np.degrees(phi)
    return float(lon), float(lat)


def geo_to_fund(lat_deg: float, lon_deg: float, d_deg: float, mu_deg: float):
    """Forward transform: geographic (lat, lon) -> fundamental-plane (xi, eta, zeta).

    Inverse of :func:`fund_to_geo` for a point on the WGS-84 ellipsoid.  ``xi, eta``
    are the point's fundamental-plane coordinates (Earth radii) and ``zeta`` its
    distance along the shadow axis; used for shadow-edge / limit computations.
    """
    d = np.radians(d_deg)
    rho1 = np.sqrt(1.0 - _E2 * np.cos(d) ** 2)
    rho2 = np.sqrt(1.0 - _E2 * np.sin(d) ** 2)
    sin_d1 = np.sin(d) / rho1
    cos_d1 = np.sqrt(1.0 - _E2) * np.cos(d) / rho1
    sin_d1_d2 = _E2 * np.sin(d) * np.cos(d) / (rho1 * rho2)
    cos_d1_d2 = np.sqrt(1.0 - _E2) / (rho1 * rho2)

    beta = np.arctan((1.0 - _F) * np.tan(np.radians(lat_deg)))  # parametric latitude
    theta = np.radians(lon_deg + mu_deg)  # hour angle east of the axis meridian
    cb, sb = np.cos(beta), np.sin(beta)

    xi = cb * np.sin(theta)
    eta1 = sb * cos_d1 - cb * np.cos(theta) * sin_d1
    zeta1 = sb * sin_d1 + cb * np.cos(theta) * cos_d1
    eta = eta1 * rho1
    zeta = rho2 * (zeta1 * cos_d1_d2 - eta1 * sin_d1_d2)
    return float(xi), float(eta), float(zeta)


def shadow_radii(
    x: float, y: float, d_deg: float, l1: float, l2: float, tan_f1: float, tan_f2: float
) -> tuple[float, float, bool]:
    """Approximate penumbra/umbra shadow radii on the ground, in km.

    Returns ``(penumbra_km, umbra_km, is_total)``.  ``umbra_km`` is the absolute
    radius; ``is_total`` is True for a total eclipse (umbra reaches the ground)
    and False for annular.  Uses the shadow-cone radii reduced to the observer's
    distance below the fundamental plane (Explanatory Supplement); adequate for
    visualization, not for precise limit computation.
    """
    d = np.radians(d_deg)
    rho1 = np.sqrt(1.0 - _E2 * np.cos(d) ** 2)
    eta1 = y / rho1
    disc = 1.0 - x**2 - eta1**2
    if disc < 0.0:
        return 0.0, 0.0, False
    zeta = np.sqrt(disc)

    L1 = l1 - zeta * tan_f1  # penumbra radius at the observer (Earth radii)
    L2 = l2 - zeta * tan_f2  # umbra radius; negative => total, positive => annular
    return float(abs(L1) * _A_KM), float(abs(L2) * _A_KM), bool(L2 < 0.0)


_EARTH_MEAN_KM = EARTH_MEAN_RADIUS_KM  # mean radius for short great-circle offsets


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


def bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing (degrees) from point 1 to point 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def shadow_edge_limits(x, y, d_deg, mu_deg, l, tan_f, path_bearing_deg, max_km=600.0):
    """Find the two shadow-edge points perpendicular to the track, and the width.

    ``l``/``tan_f`` are the fundamental-plane radius and cone tangent of the shadow
    (umbra: l2, tan_f2; penumbra: l1, tan_f1).  Marches perpendicular to
    ``path_bearing_deg`` from the central point and root-finds where the point
    crosses the shadow-cone edge on the WGS-84 ellipsoid.  Returns
    ``(north_point, south_point, width_km)`` with points as (lat, lon), or
    ``(None, None, 0.0)`` if the central point is outside the shadow (no width).
    """
    lon0, lat0 = fund_to_geo(x, y, d_deg, mu_deg)

    def residual(lat, lon):
        xi, eta, zeta = geo_to_fund(lat, lon, d_deg, mu_deg)
        perp = np.hypot(xi - x, eta - y)              # distance from the shadow axis
        radius = abs(l - zeta * tan_f)                 # cone radius at this depth
        return perp - radius

    if residual(lat0, lon0) > 0:
        return None, None, 0.0  # central point not inside this shadow

    def find_edge(brg):
        # bisection for the outward crossing residual = 0
        s_lo, s_hi = 0.0, max_km
        if residual(*_destination(lat0, lon0, brg, s_hi)) < 0:
            return None  # still inside at max range
        for _ in range(50):
            s_mid = 0.5 * (s_lo + s_hi)
            if residual(*_destination(lat0, lon0, brg, s_mid)) < 0:
                s_lo = s_mid
            else:
                s_hi = s_mid
        return _destination(lat0, lon0, brg, 0.5 * (s_lo + s_hi))

    e1 = find_edge((path_bearing_deg - 90.0) % 360.0)
    e2 = find_edge((path_bearing_deg + 90.0) % 360.0)
    if e1 is None or e2 is None:
        return None, None, 0.0

    width = _haversine_km(e1[0], e1[1], e2[0], e2[1])
    north, south = (e1, e2) if e1[0] >= e2[0] else (e2, e1)
    return north, south, float(width)


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

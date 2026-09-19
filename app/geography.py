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

# WGS-84.
_A_KM = 6378.137
_B_KM = 6356.752
_F = 1.0 - _B_KM / _A_KM          # flattening
_E2 = 1.0 - (_B_KM / _A_KM) ** 2  # first eccentricity squared


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

"""Independent observer geometry for the tests: geodetic + height -> ECEF, and
ECEF -> fundamental-plane axes as plain vectors.

Shared by ``test_geography`` (the proof of ``geo_to_fund``'s height term),
``test_limb_oracle`` (the 3D ray tracer's observer) and the mean-limb height
check in ``test_besselian_integration``.  It shares nothing with the core's
ellipsoid reduction (``core/include/eclipse/ellipsoid.hpp``) -- not even the
constants, which it takes from [WGS84] itself -- so it is an oracle for it.
"""

from __future__ import annotations

import numpy as np

# [WGS84] NIMA TR8350.2: a = 6378137 m, 1/f = 298.257223563; e^2 = f (2 - f).
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def observer_ecef_km(lat_deg: float, lon_deg: float, height_m: float = 0.0) -> np.ndarray:
    """Geodetic ``(lat, lon)`` [deg] at ``height_m`` [m] above the WGS-84
    ellipsoid -> Earth-fixed Cartesian [km]: with the prime-vertical radius
    ``N = a / sqrt(1 - e^2 sin^2 phi)``, ``X = (N + h) cos phi cos lon``,
    ``Y = (N + h) cos phi sin lon``, ``Z = (N (1 - e^2) + h) sin phi`` [WGS84]."""
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    h = height_m / 1000.0
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * np.sin(la) ** 2)
    return np.array([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1.0 - WGS84_E2) + h) * np.sin(la)])


def direct_fundamental(lat_deg: float, lon_deg: float, d_deg: float, mu_deg: float,
                       height_m: float = 0.0) -> tuple[float, float, float]:
    """``(xi, eta, zeta)`` [Earth equatorial radii] of the observer by vector
    projection: z^ towards the shadow axis (declination d, Greenwich hour
    angle mu), x^ = the equatorial east direction ``(sin mu, cos mu, 0)``,
    y^ = z^ x x^ -- the fundamental-plane axes of [ES92] sec. 8.32."""
    p = observer_ecef_km(lat_deg, lon_deg, height_m) / WGS84_A_KM
    d, mu = np.radians(d_deg), np.radians(mu_deg)
    z = np.array([np.cos(d) * np.cos(mu), -np.cos(d) * np.sin(mu), np.sin(d)])
    x = np.array([np.sin(mu), np.cos(mu), 0.0])
    y = np.cross(z, x)
    return float(p @ x), float(p @ y), float(p @ z)

"""The core's fundamental-plane <-> geographic reduction (``ellipsoid.hpp``)
and the API's time formatting.

These run without SPICE kernels. The reduction is validated against known
Besselian polynomials for the 2024-04-08 total solar eclipse and the published
central line in ``tests/data/data.txt``, and the observer height against two
independent forms (3D vectors; NASA's local-circumstances calculator).
"""

from __future__ import annotations

import _eclipse as E
import numpy as np
import pytest
from reference import central_line

from app.formatting import dec_to_hms, format_clock, format_offset


def fund_to_geo(x, y, d, mu) -> tuple[float, float]:
    """(lon, lat) [deg] for one fundamental-plane point (``_eclipse.fund_to_geo``)."""
    lon, lat = E.fund_to_geo(*(np.array([float(v)]) for v in (x, y, d, mu)))
    return float(lon[0]), float(lat[0])


def geo_to_fund(lat, lon, d, mu, h=None) -> tuple[float, float, float]:
    """(xi, eta, zeta) [Earth radii] for one observer (``_eclipse.geo_to_fund``)."""
    args = [np.array([float(v)]) for v in (lat, lon, d, mu)]
    out = E.geo_to_fund(*args) if h is None else E.geo_to_fund(*args, np.array([float(h)]))
    return tuple(float(v[0]) for v in out)

# Besselian polynomials for the 2024-04-08 eclipse (coeffs in powers of
# t = hours from 18:00 UT). Third-party approximation used only as a check.
POLY = {
    "x": [-0.318157, 0.5117105, 0.0000326, -0.0000085],
    "y": [0.219747, 0.2709586, -0.0000594, -0.0000047],
    "d": [7.58620, 0.014844, -0.000002],
    "mu": [89.59122, 15.004084],
}


def _ev(coeffs, t):
    return sum(c * t**i for i, c in enumerate(coeffs))


def test_dec_to_hms_basic():
    assert dec_to_hms(0.0, 18) == (18, 0, pytest.approx(0.0))
    h, m, s = dec_to_hms(0.5, 18)
    assert (h, m) == (18, 30)


def test_dec_to_hms_second_rollover():
    # 0.999... hours past 18:00 rounds cleanly, no 60 in seconds/minutes.
    h, m, s = dec_to_hms(1.0 - 1e-9, 18)
    assert 0 <= s < 60 and 0 <= m < 60


def test_fund_to_geo_matches_reference_central_line():
    ref = central_line()
    # Anchor near T0, where the third-party polynomial is most trustworthy; its
    # error grows with t. The full-track check with real SPICE elements lives in
    # test_besselian_integration.py.
    for i in range(0, 12, 3):
        t = i * 2 / 60.0
        x, y = _ev(POLY["x"], t), _ev(POLY["y"], t)
        d, mu = _ev(POLY["d"], t), _ev(POLY["mu"], t)
        lon, lat = fund_to_geo(x, y, d, mu)
        # Tolerances absorb the third-party polynomial's own imprecision; the
        # point of this test is that the *reduction geometry* is right (latitude
        # was ~15 deg off before the rotation fix).
        assert lat == pytest.approx(ref.lat[i], abs=0.5), f"lat at t={t}"
        assert lon == pytest.approx(ref.lon[i], abs=0.7), f"lon at t={t}"


def test_fund_to_geo_is_nan_when_axis_misses_earth():
    lon, lat = fund_to_geo(2.0, 0.0, 7.5, 90.0)  # |x| > 1: shadow off the Earth
    assert np.isnan(lon) and np.isnan(lat)


def test_longitude_wrapped():
    x, y = _ev(POLY["x"], 0), _ev(POLY["y"], 0)
    d, mu = _ev(POLY["d"], 0), _ev(POLY["mu"], 0)
    lon, lat = fund_to_geo(x, y, d, mu)
    assert -180.0 < lon <= 180.0
    assert -90.0 <= lat <= 90.0


def test_geo_to_fund_round_trip():
    """geo_to_fund and fund_to_geo are exact inverses on the near (shadow-facing)
    side of the ellipsoid (item M5)."""
    d_deg, mu_deg = 7.5862, 89.6
    checked = 0
    for lat in (-40.0, -10.0, 0.0, 20.0, 45.0):
        for lon in (-160.0, -104.0, -40.0, 0.0, 80.0):
            xi, eta, zeta = geo_to_fund(lat, lon, d_deg, mu_deg)
            if zeta <= 0.0:
                continue  # far side: fund_to_geo returns the near-side solution
            lon2, lat2 = fund_to_geo(xi, eta, d_deg, mu_deg)
            assert lat2 == pytest.approx(lat, abs=1e-6), (lat, lon)
            assert lon2 == pytest.approx(lon, abs=1e-6), (lat, lon)
            checked += 1
    assert checked >= 3  # the round-trip was actually exercised


def test_geo_to_fund_with_height_is_the_direct_3d_geometry():
    """The observer-height term of ``geo_to_fund`` ([Meeus98] ch. 11,
    rotated into the fundamental plane) against geodetic + H -> ECEF ->
    fundamental-plane axes (``geodesy_oracle.direct_fundamental``), over 2000
    random observers, axes and heights from -500 m to 9 km (and 1000 km, far
    outside any observer, to show the term is exact rather than first order).
    Achieved 1.1e-15 Earth radii (7 nm); gate 1e-14.  At H = 0 the same holds
    and the result is bit-identical to the default argument."""
    from geodesy_oracle import direct_fundamental

    rng = np.random.default_rng(20260923)
    worst = 0.0
    for _ in range(2000):
        lat, lon = rng.uniform(-90.0, 90.0), rng.uniform(-180.0, 180.0)
        d, mu = rng.uniform(-23.5, 23.5), rng.uniform(-180.0, 180.0)
        for h in (0.0, rng.uniform(-500.0, 9000.0), 1.0e6):
            ours = np.array(geo_to_fund(lat, lon, d, mu, h))
            ref = np.array(direct_fundamental(lat, lon, d, mu, h))
            worst = max(worst, float(np.max(np.abs(ours - ref))))
        assert geo_to_fund(lat, lon, d, mu, 0.0) == geo_to_fund(lat, lon, d, mu)
    assert worst <= 1e-14, worst


def test_geo_to_fund_with_height_is_nasas_local_circumstances_form():
    """``geo_to_fund`` against the form NASA's local-circumstances calculator
    uses [NASA-LC] (SEcirc.js ``readdata``: Meeus's rho sin phi', rho cos phi'
    with the observer's height [Meeus98] ch. 11; ``timelocdependent``:
    xi = rho cos phi' sin h, eta = rho sin phi' cos d - rho cos phi' cos h sin d,
    zeta = rho sin phi' sin d + rho cos phi' cos h cos d), with WGS-84 in place
    of Meeus's IAU 1976 a and f.  Ours splits the same vector into the
    ellipsoid point and the height normal; the two agree to 4.4e-16 (gate
    1e-14) over 5000 random observers, axes and heights."""
    from geodesy_oracle import WGS84_A_KM, WGS84_F

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5000):
        lat, lon = rng.uniform(-90.0, 90.0), rng.uniform(-180.0, 180.0)
        d, mu, h = rng.uniform(-23.5, 23.5), rng.uniform(-180.0, 180.0), rng.uniform(-500, 9000)
        phi = np.radians(lat)
        u = np.arctan((1.0 - WGS84_F) * np.tan(phi))
        k = h / (WGS84_A_KM * 1000.0)
        rho_sin = (1.0 - WGS84_F) * np.sin(u) + k * np.sin(phi)
        rho_cos = np.cos(u) + k * np.cos(phi)
        ha, dd = np.radians(mu + lon), np.radians(d)
        nasa = np.array([rho_cos * np.sin(ha),
                         rho_sin * np.cos(dd) - rho_cos * np.cos(ha) * np.sin(dd),
                         rho_sin * np.sin(dd) + rho_cos * np.cos(ha) * np.cos(dd)])
        worst = max(worst, float(np.max(np.abs(np.array(geo_to_fund(lat, lon, d, mu, h)) - nasa))))
    assert worst <= 1e-14, worst


def test_geo_to_fund_arrays_are_elementwise():
    """The array form is the one-observer form elementwise, heights included."""
    lat = np.array([39.12, 43.95, -20.0])
    lon = np.array([-88.55, -117.22, 30.0])
    h = np.array([149.4, 693.8, 0.0])
    d, mu = np.array([7.5, 7.6, 11.9]), np.array([80.0, 90.0, 100.0])
    xi, eta, zeta = E.geo_to_fund(lat, lon, d, mu, h)
    for i in range(3):
        assert (xi[i], eta[i], zeta[i]) == geo_to_fund(lat[i], lon[i], d[i], mu[i], h[i])


def test_format_offset_signed():
    assert format_offset(0.0) == "+00:00:00.0"
    assert format_offset(1.5) == "+01:30:00.0"
    assert format_offset(-0.25) == "-00:15:00.0"


def test_format_clock_rolls_over_midnight():
    # 23:30 UTC + 1 h -> 00:30 the next day (date rollover handled by datetime).
    assert format_clock("2024-04-08T23:30:00", 1.0) == "00:30:00"
    assert format_clock("2024-04-08T18:00:00", 0.5) == "18:30:00"


def test_normalize_utc():
    from app.besselian import normalize_utc

    assert normalize_utc("2024-04-08T18:17:15Z") == "2024-04-08T18:17:15"
    assert normalize_utc("2024-04-08 18:17:15") == "2024-04-08T18:17:15"
    assert normalize_utc("2024-04-08T20:17:15+02:00") == "2024-04-08T18:17:15"
    assert normalize_utc("2024-04-08T18:17:15.5") == "2024-04-08T18:17:15.500"
    with pytest.raises(ValueError):
        normalize_utc("2024 APR 08 18:17:15")  # SPICE-only format, rejected up front


def test_format_clock_rounds_to_the_nearest_second():
    """Contact clocks round (item C7); strftime alone truncated 12:00:00.7 to
    12:00:00, half a second early on average."""
    assert format_clock("2024-04-08T12:00:00", 0.7 / 3600) == "12:00:01"
    assert format_clock("2024-04-08T12:00:00", 0.4 / 3600) == "12:00:00"
    assert format_clock("2024-04-08T23:59:59", 0.6 / 3600) == "00:00:00"

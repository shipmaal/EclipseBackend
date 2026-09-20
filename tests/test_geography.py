"""Pure-function tests for the fundamental-plane -> geographic reduction.

These run without SPICE kernels. The reduction is validated against known
Besselian polynomials for the 2024-04-08 total solar eclipse and the published
central line in ``app/data.txt``.
"""

from __future__ import annotations

import pytest

from app.geography import dec_to_hms, fund_to_geo
from app.reference import central_line

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


def test_fund_to_geo_raises_when_axis_misses_earth():
    with pytest.raises(ValueError):
        fund_to_geo(2.0, 0.0, 7.5, 90.0)  # |x| > 1: shadow off the Earth


def test_longitude_wrapped():
    x, y = _ev(POLY["x"], 0), _ev(POLY["y"], 0)
    d, mu = _ev(POLY["d"], 0), _ev(POLY["mu"], 0)
    lon, lat = fund_to_geo(x, y, d, mu)
    assert -180.0 < lon <= 180.0
    assert -90.0 <= lat <= 90.0

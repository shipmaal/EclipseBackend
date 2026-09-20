"""End-to-end validation against published references for two eclipses:
2024-04-08 (full central-line table) and 2017-08-21 (published Besselian
elements + greatest-eclipse point).

Requires the SPICE kernels (run ``python -m kernels.bootstrap``); the whole
module is skipped automatically when they are absent, so the rest of the suite
still runs in CI without network access.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ephemeris import default_metakernel
from app.reference import central_line

pytestmark = pytest.mark.skipif(
    not default_metakernel().exists(),
    reason="SPICE kernels not downloaded (run `python -m kernels.bootstrap`)",
)

T0 = "2024-04-08T18:00:00"


def _build_model(t0):
    """Build a model with the best available Earth-orientation frame (ITRS first,
    then ITRF93, TOD, IAU_EARTH). Thin wrapper over app.besselian for the tests."""
    from app.besselian import build_model_best_frame

    return build_model_best_frame(t0, half_window_hours=2.0)


def test_central_line_matches_published_track():
    from app.geography import fund_to_geo

    ref = central_line()
    model, _frame = _build_model(T0)

    # Reference rows are every 2 minutes starting at T0 (18:00 UT).
    t = np.arange(len(ref)) * 2 / 60.0
    elems = model.evaluate(t)

    dlat, dlon = [], []
    for i in range(len(ref)):
        lon, lat = fund_to_geo(elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i])
        dlat.append(abs(lat - ref.lat[i]))
        dlon.append(abs(lon - ref.lon[i]))

    # Verified against the published track. With the parametric->geodetic fix the
    # latitude residual is sub-km; the longitude residual (~a few km) is limited by
    # the reference table's own resolution and UT1~UTC in the TOD frame.
    assert max(dlat) < 0.03, f"max latitude error {max(dlat):.3f} deg"
    assert max(dlon) < 0.15, f"max longitude error {max(dlon):.3f} deg"


# --- Second eclipse: 2017-08-21, independent reference (Espenak / NASA) ------
# Published Besselian elements at t0 = 2017 Aug 21 18.000 TDT and the
# greatest-eclipse circumstances (18:25:30 UT). Source: F. Espenak, elements as
# distributed in andrmoel/astronomy-bundle-php (VSOP87/ELP2000).
_2017_PUBLISHED = {"x": -0.1295710, "y": 0.4854160, "d": 11.8669596}
_2017_GREATEST = {"utc": "2017-08-21T18:25:30", "lat": 37.0, "lon": -87.7, "width_km": 114.7, "is_total": True, "duration_s": 160, "magnitude": 1.0306}


def _usable_frame():
    from app.besselian import best_earth_frame

    return best_earth_frame()


def test_2017_elements_match_published():
    """x, y, d at 18:00 TDT match Espenak's published coefficients.

    (mu is intentionally not compared: its value depends on the delta-T /
    ephemeris-hour-angle convention, which differs from ours by ~ delta-T of Earth
    rotation. The end-to-end test below exercises mu via the geography.)
    """
    import spiceypy

    from app.ephemeris import besselian_instant

    frame = _usable_frame()
    et = spiceypy.str2et("2017-08-21 18:00:00 TDT")  # elements are tabulated vs TDT
    bi = besselian_instant(et, frame)
    for key, pub in _2017_PUBLISHED.items():
        assert getattr(bi, key) == pytest.approx(pub, abs=1e-3), key


# 2024-04-08, independent Espenak elements + greatest-eclipse circumstances.
_2024_PUBLISHED = {"x": -0.3182440, "y": 0.2197640, "d": 7.5862002}
_2024_GREATEST = {"utc": "2024-04-08T18:17:15", "lat": 25.3, "lon": -104.1, "width_km": 197.5, "is_total": True, "duration_s": 268, "magnitude": 1.0566}


def _check_greatest(g):
    from app.geography import central_track, shadow_edge_limits, shadow_radii

    model, _frame = _build_model(g["utc"])
    dt = 0.02
    # central_track reduces each instant to (lat, lon) with the along-track
    # bearing from its neighbours -- the same path /central-line uses (item R3).
    elems, track = central_track(model, np.array([-dt, 0.0, dt]))
    tp = next(p for p in track if abs(p.t_hours) < 1e-9)  # central instant t=0
    i = tp.i
    north, south, width = shadow_edge_limits(
        elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i],
        elems["l2"][i], elems["tan_f2"][i], tp.bearing,
    )
    _pen, _umb, is_total = shadow_radii(
        elems["x"][i], elems["y"][i], elems["d"][i],
        elems["l1"][i], elems["l2"][i], elems["tan_f1"][i], elems["tan_f2"][i],
    )
    # Published lat/lon are rounded to 0.1 deg; width matches to ~1 km.
    assert tp.lat == pytest.approx(g["lat"], abs=0.12)
    assert tp.lon == pytest.approx(g["lon"], abs=0.12)
    assert north is not None and north[0] > south[0]
    assert width == pytest.approx(g["width_km"], abs=2.0)
    assert is_total is g["is_total"]


def test_2017_greatest_eclipse_point():
    _check_greatest(_2017_GREATEST)


def test_2024_elements_match_published():
    import spiceypy

    from app.ephemeris import besselian_instant

    frame = _usable_frame()
    et = spiceypy.str2et("2024-04-08 18:00:00 TDT")
    bi = besselian_instant(et, frame)
    for key, pub in _2024_PUBLISHED.items():
        assert getattr(bi, key) == pytest.approx(pub, abs=1e-3), key


def test_2024_greatest_eclipse_point():
    _check_greatest(_2024_GREATEST)


# 2023-10-14 ANNULAR eclipse: exercises is_total=False and the antumbral (l2 > 0)
# width path. Independent Espenak elements + greatest-eclipse circumstances.
_2023_PUBLISHED = {"x": 0.1696580, "y": 0.3348590, "d": -8.2441902}
_2023_GREATEST = {"utc": "2023-10-14T17:59:27", "lat": 11.4, "lon": -83.1, "width_km": 187.4, "is_total": False, "duration_s": 317, "magnitude": 0.9520}


def test_2023_annular_elements_match_published():
    import spiceypy

    from app.ephemeris import besselian_instant

    frame = _usable_frame()
    et = spiceypy.str2et("2023-10-14 18:00:00 TDT")
    bi = besselian_instant(et, frame)
    for key, pub in _2023_PUBLISHED.items():
        assert getattr(bi, key) == pytest.approx(pub, abs=1e-3), key


def test_2023_annular_greatest_eclipse_point():
    _check_greatest(_2023_GREATEST)


# --- Local circumstances at the greatest-eclipse point ----------------------
def _check_circumstances(g):
    from app.circumstances import local_circumstances
    from app.geography import fund_to_geo

    model, _frame = _build_model(g["utc"])
    e = model.evaluate(np.array([0.0]))  # central point at maximum
    lon, lat = fund_to_geo(e["x"][0], e["y"][0], e["d"][0], e["mu"][0])
    c = local_circumstances(model, lat, lon)

    assert c["eclipse"] is True
    assert c["type"] == ("total" if g["is_total"] else "annular")
    assert c["central_duration_s"] == pytest.approx(g["duration_s"], abs=3.0)
    assert c["magnitude"] == pytest.approx(g["magnitude"], abs=0.003)


def test_circumstances_2017():
    _check_circumstances(_2017_GREATEST)


def test_circumstances_2024():
    _check_circumstances(_2024_GREATEST)


def test_circumstances_2023_annular():
    _check_circumstances(_2023_GREATEST)


def test_circumstances_partial_observer():
    """A location off the central path sees a partial eclipse (no central phase)."""
    from app.besselian import BesselianModel
    from app.circumstances import local_circumstances

    model = BesselianModel(t0_utc="2024-04-08T18:17:15", half_window_hours=2.5)
    c = local_circumstances(model, 40.71, -74.01)  # New York City
    assert c["eclipse"] is True
    assert c["type"] == "partial"
    assert "central_duration_s" not in c
    assert 0.85 < c["magnitude"] < 0.95  # NYC saw ~90% of the Sun's diameter covered

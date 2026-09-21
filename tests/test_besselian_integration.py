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

    # Measured (ITRS, DE432s): latitude max 0.0069 deg (0.8 km); longitude max
    # 0.109 deg (7.9 km) at the very end of the track, mean -0.029 deg.  The
    # longitude pattern is a consistent 2-4 s implied *time* offset against the
    # table (a delta-T / lunar-radius assumption of its producer), not a frame
    # error.  data.txt's exact provenance (and the delta-T it assumed) is not
    # recorded; the tolerances carry ~40% margin over the measured residuals.
    assert max(dlat) < 0.03, f"max latitude error {max(dlat):.3f} deg"
    assert max(dlon) < 0.15, f"max longitude error {max(dlon):.3f} deg"


# --- Second eclipse: 2017-08-21, independent reference (Espenak / NASA) ------
# Published Besselian elements at t0 = 2017 Aug 21 18.000 TDT and the
# greatest-eclipse circumstances (18:25:30 UT).  Source: F. Espenak, NASA Eclipse
# Web Site, https://eclipse.gsfc.nasa.gov/SEgoogle/SEgoogle2001/SE2017Aug21Tgoogle.html
# (the values here were transcribed via the copy in andrmoel/astronomy-bundle-php;
# our x, y agree with them to ~1e-6, i.e. they are Espenak's).  2024:
# .../SE2024Apr08Tgoogle.html; 2023: .../SE2023Oct14Agoogle.html.
_2017_PUBLISHED = {"x": -0.1295710, "y": 0.4854160, "d": 11.8669596}
_2017_GREATEST = {"utc": "2017-08-21T18:25:30", "lat": 37.0, "lon": -87.7, "width_km": 114.7, "is_total": True, "duration_s": 160, "magnitude": 1.0306}


def _usable_frame():
    from app.besselian import best_earth_frame

    return best_earth_frame()


def test_2017_elements_match_published():
    """x, y, d at 18:00 TDT match Espenak's published coefficients.

    (mu is compared separately in ``test_2024_mu_matches_published_with_delta_t``:
    the published value is the ephemeris hour angle, ours the true one.)
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


_2024_PUBLISHED_MU = 89.591217  # Espenak mu0 at 18:00 TDT (ephemeris hour angle)


def test_2024_mu_matches_published_with_delta_t():
    """Published mu is the *ephemeris hour angle*: Earth rotation evaluated as if
    TT were UT.  Ours is the true Greenwich hour angle (UT1 from the leap-second
    kernel + IERS UT1-UTC), so mu_pub = mu_ours + delta-T * 1.002738 * 15"/s
    [ES92] sec. 8.36.  Measured agreement after the correction: 7e-6 deg (this
    test previously skipped mu, mistaking the convention for an error)."""
    import spiceypy

    from app.ephemeris import besselian_instant, tt_minus_ut1

    et = spiceypy.str2et("2024-04-08 18:00:00 TDT")
    bi = besselian_instant(et, _usable_frame())
    delta_t = float(tt_minus_ut1(et)[0])
    assert delta_t == pytest.approx(69.2, abs=0.2)  # measured (LSK + IERS) delta-T
    mu_eph = bi.mu % 360.0 + delta_t * 1.002738 * 15.0 / 3600.0
    assert mu_eph == pytest.approx(_2024_PUBLISHED_MU, abs=1e-4)


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
    assert c["below_horizon"] == []
    assert c["C1_alt"] > 0 and c["C4_alt"] > 0


# --- Horizon handling (review item 1) ---------------------------------------
def test_circumstances_night_side_observer_sees_nothing():
    """The shadow-cone geometry continues through the Earth, so a night-side
    observer is geometrically "inside the penumbra"; before the horizon check
    India reported a magnitude-0.93 partial eclipse with the Sun at -62 deg."""
    from app.besselian import BesselianModel
    from app.circumstances import local_circumstances

    model = BesselianModel(t0_utc="2024-04-08T18:17:15", half_window_hours=2.5)
    c = local_circumstances(model, 20.0, 77.0)  # central India, local night
    assert c["eclipse"] is False
    assert set(c["below_horizon"]) == {"C1", "max", "C4"}


def test_circumstances_sunset_eclipse_flags_contacts():
    """W Ireland, 2024-04-08: C1 with the Sun up, maximum and C4 after sunset."""
    from app.besselian import BesselianModel
    from app.circumstances import local_circumstances

    model = BesselianModel(t0_utc="2024-04-08T18:17:15", half_window_hours=2.5)
    c = local_circumstances(model, 53.3, -9.0)
    assert c["eclipse"] is True and c["type"] == "partial"
    assert c["C1_alt"] > 0 and c["C4_alt"] < 0 and c["sun_alt"] < 0
    assert c["below_horizon"] == ["max", "C4"]


# --- Grid (vectorized) parity with the scalar path --------------------------
def test_circumstances_grid_matches_scalar():
    from app.besselian import BesselianModel
    from app.circumstances import circumstances_grid, local_circumstances

    model = BesselianModel(t0_utc="2024-04-08T18:17:15", half_window_hours=2.5)
    sites = [(25.3, -104.1), (40.71, -74.01), (53.3, -9.0), (20.0, 77.0), (-30.0, -60.0)]
    g = circumstances_grid(model, [s[0] for s in sites], [s[1] for s in sites], step_minutes=0.5)
    for k, (lat, lon) in enumerate(sites):
        c = local_circumstances(model, lat, lon)
        if not c.get("eclipse"):
            if "below_horizon" in c:
                # Night side: the grid keeps the geometric magnitude (so a map can
                # shade it) and reports the horizon through ``visible``.
                assert g["magnitude"][k] > 0 and not g["visible"][k], (lat, lon)
            else:
                assert g["magnitude"][k] == 0.0, (lat, lon)
            continue
        # Grid samples every 30 s without refinement: magnitude flat at max -> 1e-3.
        assert g["magnitude"][k] == pytest.approx(c["magnitude"], abs=2e-3), (lat, lon)
        # Grid "visible" is Sun-up *at maximum*; the scalar path also counts an
        # eclipse whose C1 alone is above the horizon (the sunset site).
        assert g["visible"][k] == (c["eclipse"] and "max" not in c["below_horizon"]), (lat, lon)
    assert bool(g["central"][0]) is True   # greatest-eclipse point
    assert bool(g["central"][1]) is False  # NYC


# --- Delta-T outside the IERS era (review item 2) ----------------------------
def test_delta_t_model_agrees_with_measured_inside_era():
    """Independent cross-check: the [Espenak] polynomial vs the leap-second kernel
    + IERS UT1-UTC, at epochs where both exist.  (After ~2005 the polynomial is
    known to over-predict -- 73.9 vs 69.2 s in 2024 -- which is exactly why the
    measured value is used inside the era.)"""
    from app.deltat import delta_t_seconds
    from app.ephemeris import load_kernels, tt_minus_ut1, utc_to_et

    load_kernels()
    for year in (1975, 1990, 2000):
        et = utc_to_et(f"{year}-07-01T00:00:00")
        measured = float(tt_minus_ut1(et)[0])
        assert delta_t_seconds(year + 0.5) == pytest.approx(measured, abs=1.0), year


def test_delta_t_used_outside_era_is_the_model_not_the_frozen_leap_count():
    """1919: SPICE's leap-second table would give ET-UTC = 41.2 s; the true
    delta-T is ~21 s [Meeus98] Table 10.A.  Both utc_to_et and the Earth-rotation
    angle must use the model there."""
    import spiceypy

    from app.ephemeris import load_kernels, tt_minus_ut1, utc_to_et

    load_kernels()
    et = utc_to_et("1919-05-29T13:08:00")
    assert float(tt_minus_ut1(et)[0]) == pytest.approx(21.0, abs=0.5)
    # The string was read as UT1: TT = UT1 + delta-T.
    assert et - spiceypy.tparse("1919-05-29T13:08:00")[0] == pytest.approx(21.0, abs=0.5)


def test_1919_eddington_eclipse_smoke():
    """1919-05-29 (the Eddington eclipse): total, greatest eclipse in the
    equatorial Atlantic off West Africa, ~6m51s.  A smoke test of the
    pre-IERS-era path (delta-T model; no EOP), with deliberately loose bounds --
    we hold no digitized Canon reference for it."""
    import spiceypy

    from app.besselian import BesselianModel
    from app.circumstances import local_circumstances
    from app.geography import central_track

    try:
        model = BesselianModel(t0_utc="1919-05-29T13:08:00", half_window_hours=2.5)
    except spiceypy.utils.exceptions.SpiceSPKINSUFFDATA:
        pytest.skip("ephemeris does not cover 1919 (mirror DE432s spans 1949-2050; use NAIF DE440s)")
    _elems, track = central_track(model, np.array([0.0]))
    assert track, "no central line at greatest eclipse"
    tp = track[0]
    assert -5.0 < tp.lat < 12.0 and -30.0 < tp.lon < -5.0
    c = local_circumstances(model, tp.lat, tp.lon)
    assert c["type"] == "total"
    assert c["central_duration_s"] == pytest.approx(411.0, abs=20.0)


"""End-to-end validation against published references: 2024-04-08 (full
central-line table) and, against F. Espenak's NASA values [Espenak], seven
eclipses that between them cover total / annular / hybrid, a polar grazing
path, and both delta-T regimes (IERS-measured 2017-2024; the polynomial model
for 1868 and 1919): published Besselian elements, mu, greatest-eclipse point,
path width, central duration and magnitude.

Requires the SPICE kernels (run ``python -m kernels.bootstrap``); the whole
module is skipped automatically when they are absent, so the rest of the suite
still runs in CI without network access.
"""

from __future__ import annotations

from datetime import datetime

import _eclipse as E
import numpy as np
import pytest
from reference import central_line

from app.core import SpiceError, default_band_path, default_metakernel, load_kernels

pytestmark = pytest.mark.skipif(
    not default_metakernel().exists(),
    reason="SPICE kernels not downloaded (run `python -m kernels.bootstrap`)",
)

T0 = "2024-04-08T18:00:00"

_NO_COVERAGE = ("ephemeris does not cover this epoch (mirror DE432s spans 1949-2050; "
                "use NAIF DE440s, 1849-2150)")


def _instant(et: float, frame: str) -> dict[str, float]:
    """The elements at one TDB instant (``_eclipse.besselian_instants``)."""
    return {k: float(v[0]) for k, v in E.besselian_instants(np.array([et]), frame).items()}


def _fund_to_geo(x, y, d, mu) -> tuple[float, float]:
    """(lon, lat) [deg] where the shadow axis meets the ellipsoid (the core's reduction)."""
    lon, lat = E.fund_to_geo(*(np.array([float(v)]) for v in (x, y, d, mu)))
    return float(lon[0]), float(lat[0])


def _tdt(s: str) -> float:
    """TDB seconds past J2000 of a TDT epoch string (published elements' time scale)."""
    load_kernels()
    return E.str_to_et(s + " TDT")


def _build_model(t0):
    """Build a model with the best available Earth-orientation frame (ITRS first,
    then ITRF93, TOD, IAU_EARTH). Thin wrapper over app.besselian for the tests."""
    from app.besselian import build_model_best_frame

    return build_model_best_frame(t0, half_window_hours=2.0)


def test_central_line_matches_published_track():
    ref = central_line()
    model, _frame = _build_model(T0)

    # Reference rows are every 2 minutes starting at T0 (18:00 UT).
    t = np.arange(len(ref)) * 2 / 60.0
    elems = model.evaluate(t)

    dlat, dlon = [], []
    for i in range(len(ref)):
        lon, lat = _fund_to_geo(elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i])
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
    bi = _instant(_tdt("2017-08-21 18:00:00"), _usable_frame())  # tabulated vs TDT
    for key, pub in _2017_PUBLISHED.items():
        assert bi[key] == pytest.approx(pub, abs=1e-3), key


# 2024-04-08, independent Espenak elements + greatest-eclipse circumstances.
_2024_PUBLISHED = {"x": -0.3182440, "y": 0.2197640, "d": 7.5862002}
_2024_GREATEST = {"utc": "2024-04-08T18:17:15", "lat": 25.3, "lon": -104.1, "width_km": 197.5, "is_total": True, "duration_s": 268, "magnitude": 1.0566}


def _t0_utc(g):
    """UTC epoch of greatest eclipse for case ``g``: its ``utc``, or its published
    ``td`` (TDT) converted with *our* delta-T.  The Canon's own UT is TD minus
    the Canon's delta-T (73.4 s in 2023, vs 69.2 s measured), so for the newer
    cases the TD instant is the unambiguous anchor: ~4 s moves a grazing polar
    shadow by several km."""
    if "utc" in g:
        return g["utc"]
    return E.et_to_utc(_tdt(g["td"]))


def _greatest_model(g):
    try:
        return _build_model(_t0_utc(g))
    except SpiceError as exc:
        if "SPKINSUFFDATA" not in exc.short_message:
            raise
        pytest.skip(_NO_COVERAGE)


def _greatest_edges(g):
    """The ``central_line`` point at greatest eclipse (t = 0), as a dict of floats.

    ``_eclipse.central_line`` over [-dt, 0, +dt] -- the same path /central-line
    uses (item R3): the along-track bearing from the neighbours, the umbral
    limits as the time envelope (element rates) and the width between them.
    """
    model, frame = _greatest_model(g)
    dt = 0.02
    c = E.central_line(model.et0, frame, np.array([-dt, 0.0, dt]))
    k = int(np.flatnonzero(np.abs(c["t_hours"]) < 1e-9)[0])  # central instant t=0
    return {key: v[k].item() for key, v in c.items()}


def _check_greatest(g):
    p = _greatest_edges(g)
    # Published lat/lon are rounded to 0.1 deg (measured |dlat|, |dlon| <= 0.05
    # deg over all seven cases); the width is test_path_width_at_greatest_eclipse.
    assert p["lat"] == pytest.approx(g["lat"], abs=0.12)
    assert p["lon"] == pytest.approx(g["lon"], abs=0.12)
    assert not np.isnan(p["north_lat"]) and p["north_lat"] > p["south_lat"]
    assert p["is_total"] is g["is_total"]


def test_2017_greatest_eclipse_point():
    _check_greatest(_2017_GREATEST)


def test_2024_elements_match_published():
    bi = _instant(_tdt("2024-04-08 18:00:00"), _usable_frame())  # tabulated vs TDT
    for key, pub in _2024_PUBLISHED.items():
        assert bi[key] == pytest.approx(pub, abs=1e-3), key


def test_2024_greatest_eclipse_point():
    _check_greatest(_2024_GREATEST)


_2024_PUBLISHED_MU = 89.591217  # Espenak mu0 at 18:00 TDT (ephemeris hour angle)


def test_2024_mu_matches_published_with_delta_t():
    """Published mu is the *ephemeris hour angle*: Earth rotation evaluated as if
    TT were UT.  Ours is the true Greenwich hour angle (UT1 from the leap-second
    kernel + IERS UT1-UTC), so mu_pub = mu_ours + delta-T * 1.002738 * 15"/s
    [ES92] sec. 8.36.  Measured agreement after the correction: 7e-6 deg (this
    test previously skipped mu, mistaking the convention for an error)."""
    et = _tdt("2024-04-08 18:00:00")
    bi = _instant(et, _usable_frame())
    delta_t = float(E.tt_minus_ut1(np.array([et]))[0])
    assert delta_t == pytest.approx(69.2, abs=0.2)  # measured (LSK + IERS) delta-T
    mu_eph = bi["mu"] % 360.0 + delta_t * 1.002738 * 15.0 / 3600.0
    assert mu_eph == pytest.approx(_2024_PUBLISHED_MU, abs=1e-4)


# 2023-10-14 ANNULAR eclipse: exercises is_total=False and the antumbral (l2 > 0)
# width path. Independent Espenak elements + greatest-eclipse circumstances.
_2023_PUBLISHED = {"x": 0.1696580, "y": 0.3348590, "d": -8.2441902}
_2023_GREATEST = {"utc": "2023-10-14T17:59:27", "lat": 11.4, "lon": -83.1, "width_km": 187.4, "is_total": False, "duration_s": 317, "magnitude": 0.9520}


def test_2023_annular_elements_match_published():
    bi = _instant(_tdt("2023-10-14 18:00:00"), _usable_frame())
    for key, pub in _2023_PUBLISHED.items():
        assert bi[key] == pytest.approx(pub, abs=1e-3), key


def test_2023_annular_greatest_eclipse_point():
    _check_greatest(_2023_GREATEST)


# --- Local circumstances at the greatest-eclipse point ----------------------
def _check_circumstances(g):
    from app.circumstances import local_circumstances

    model, _frame = _greatest_model(g)
    e = model.evaluate(np.array([0.0]))  # central point at maximum
    lon, lat = _fund_to_geo(e["x"][0], e["y"][0], e["d"][0], e["mu"][0])
    c = local_circumstances(model, lat, lon)

    assert c["eclipse"] is True
    assert c["type"] == ("total" if g["is_total"] else "annular")
    # Measured over all seven cases: duration -0.4 .. +0.4 s (published to 1 s),
    # magnitude within 1e-4 -- since the solar radius stopped following the
    # PCK (item W1; it was +1.7 .. +2.4 s / +0.0005 with pck00011's 695 700 km).
    assert c["central_duration_s"] == pytest.approx(g["duration_s"], abs=1.0)
    assert c["magnitude"] == pytest.approx(g["magnitude"], abs=1e-3)


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


# --- More reference eclipses [Espenak] ----------------------------------------
# Source for each: F. Espenak, NASA Eclipse Web Site, Besselian-elements page
# https://eclipse.gsfc.nasa.gov/SEsearch/SEdata.php?Ecl=YYYYMMDD (the
# Five Millennium Canon's elements, VSOP87/ELP2000-82): the polynomial
# elements' constant terms at t0 (TDT), the instant of greatest eclipse (TDT),
# and the circumstances there (lat/lon to 0.1 deg, path width to 0.1 km,
# central duration to 1 s, magnitude).  What each case adds:
#   2023-04-20 H -- a hybrid (l2 crosses 0 along the path; total at greatest).
#   2021-12-04 T -- a polar, grazing path (gamma -0.95, Sun 17 deg high).
#   1919-05-29 T -- pre-IERS: delta-T from the polynomial model (Canon 21.0 s).
#   1868-08-18 T -- pre-IERS, a century earlier (Canon delta-T 2.2 s).
_REF_ECLIPSES = {
    "2023-04-20H": {
        "t0": "2023-04-20 04:00:00", "x": 0.0268500, "y": -0.4273660, "d": 11.4117899,
        "mu": 240.242935, "td": "2023-04-20 04:17:56", "lat": -9.6, "lon": 125.8,
        "width_km": 49.0, "is_total": True, "duration_s": 76, "magnitude": 1.0132,
    },
    "2021-12-04T": {
        "t0": "2021-12-04 08:00:00", "x": 0.0252090, "y": -0.9836530, "d": -22.2747192,
        "mu": 302.452179, "td": "2021-12-04 07:34:38", "lat": -76.8, "lon": -46.2,
        "width_km": 418.7, "is_total": True, "duration_s": 114, "magnitude": 1.0367,
    },
    "1919-05-29T": {
        "t0": "1919-05-29 13:00:00", "x": -0.0658300, "y": -0.3007040, "d": 21.5041408,
        "mu": 15.729970, "td": "1919-05-29 13:08:55", "lat": 4.4, "lon": -16.7,
        "width_km": 244.4, "is_total": True, "duration_s": 411, "magnitude": 1.0719,
    },
    "1868-08-18T": {
        "t0": "1868-08-18 05:00:00", "x": -0.1255510, "y": -0.0143720, "d": 13.0380297,
        "mu": 254.097061, "td": "1868-08-18 05:12:10", "lat": 10.6, "lon": 102.2,
        "width_km": 245.1, "is_total": True, "duration_s": 407, "magnitude": 1.0756,
    },
}


def _elements_at_t0(g):
    et = _tdt(g["t0"])  # published elements are tabulated vs TDT
    try:
        return et, _instant(et, _usable_frame())
    except SpiceError as exc:
        if "SPKINSUFFDATA" not in exc.short_message:
            raise
        pytest.skip(_NO_COVERAGE)


@pytest.mark.parametrize("name", sorted(_REF_ECLIPSES))
def test_reference_elements_match_published(name):
    """x, y, d at t0 (TDT).  Measured: |dx|, |dy| <= 1e-5, |dd| <= 9.3e-5 deg."""
    g = _REF_ECLIPSES[name]
    _et, bi = _elements_at_t0(g)
    for key in ("x", "y", "d"):
        assert bi[key] == pytest.approx(g[key], abs=1e-3), key


@pytest.mark.parametrize("name", sorted(_REF_ECLIPSES))
def test_reference_mu_matches_published_with_delta_t(name):
    """mu_published = mu_ours + delta-T * 1.002738 * 15"/s [ES92] sec. 8.36 (see
    test_2024_mu_matches_published_with_delta_t), here with delta-T measured
    (2021, 2023) or from the model (1868, 1919).  Measured: <= 1.9e-5 deg."""
    g = _REF_ECLIPSES[name]
    et, bi = _elements_at_t0(g)
    delta_t = float(E.tt_minus_ut1(np.array([et]))[0])
    mu_eph = (bi["mu"] % 360.0 + delta_t * 1.002738 * 15.0 / 3600.0) % 360.0
    assert mu_eph == pytest.approx(g["mu"], abs=1e-4)


@pytest.mark.parametrize("name", sorted(_REF_ECLIPSES))
def test_reference_greatest_eclipse_point(name):
    _check_greatest(_REF_ECLIPSES[name])


@pytest.mark.parametrize("name", sorted(_REF_ECLIPSES))
def test_reference_circumstances(name):
    """Duration and magnitude at the greatest-eclipse point.  Measured: duration
    within +/-0.4 s, magnitude within 1e-4 (see _check_circumstances)."""
    _check_circumstances(_REF_ECLIPSES[name])


# Every case with a published path width at greatest eclipse: the umbral path
# width as the envelope of the shadow over time, measured on the WGS-84
# ellipsoid (items W1/W2, docs/CODE_REVIEW_FOLLOWUPS.md section 6).  Measured
# (ours - Espenak, km): 2017 -0.01, 2023-10 annular +0.12, 2024 -0.03,
# 2023-04 hybrid -0.10, 1919 -0.10, 1868 -0.12, and the polar grazing
# 2021-12-04 -0.86 (0.2%).  Before W1/W2 these were +1.64, -1.63, +1.37,
# +1.51, +2.70, +2.30 and -6.71.
_WIDTH_CASES = {
    "2017-08-21T": _2017_GREATEST,
    "2023-10-14A": _2023_GREATEST,
    "2024-04-08T": _2024_GREATEST,
    **_REF_ECLIPSES,
}


@pytest.mark.parametrize("name", sorted(_WIDTH_CASES))
def test_path_width_at_greatest_eclipse(name):
    g = _WIDTH_CASES[name]
    width = _greatest_edges(g)["width_km"]
    assert width == pytest.approx(g["width_km"], abs=1.0)


# --- Delta-T outside the IERS era (review item 2) ----------------------------
def test_delta_t_model_agrees_with_measured_inside_era():
    """Independent cross-check: the [Espenak] polynomial vs the leap-second kernel
    + IERS UT1-UTC, at epochs where both exist.  (After ~2005 the polynomial is
    known to over-predict -- 73.9 vs 69.2 s in 2024 -- which is exactly why the
    measured value is used inside the era.)"""
    load_kernels()
    for year in (1975, 1990, 2000):
        et = E.utc_to_et(f"{year}-07-01T00:00:00")
        measured = float(E.tt_minus_ut1(np.array([et]))[0])
        model = float(E.delta_t_seconds(np.array([year + 0.5]))[0])
        assert model == pytest.approx(measured, abs=1.0), year


def test_delta_t_used_outside_era_is_the_model_not_the_frozen_leap_count():
    """1919: SPICE's leap-second table would give ET-UTC = 41.2 s; the true
    delta-T is ~21 s [Meeus98] Table 10.A.  Both utc_to_et and the Earth-rotation
    angle must use the model there."""
    load_kernels()
    et = E.utc_to_et("1919-05-29T13:08:00")
    assert float(E.tt_minus_ut1(np.array([et]))[0]) == pytest.approx(21.0, abs=0.5)
    # The string was read as UT1: TT = UT1 + delta-T, with the epoch's seconds
    # past J2000 on the leap-second-free calendar.
    formal = (datetime(1919, 5, 29, 13, 8) - datetime(2000, 1, 1, 12)).total_seconds()
    assert et - formal == pytest.approx(21.0, abs=0.5)


@pytest.mark.skipif(not default_metakernel().exists(), reason="SPICE kernels not downloaded")
def test_best_frame_reraises_ephemeris_coverage_gaps():
    """An epoch outside the loaded SPK is a coverage error, not "no usable
    frame": it must reach the caller as SPICE(SPKINSUFFDATA) so the pre-IERS
    reference tests skip on the mirror's DE432s (1949-2050) instead of failing
    (CI full-validation, 1868/1919 cases)."""
    from app.besselian import build_model_best_frame

    with pytest.raises(SpiceError, match="SPKINSUFFDATA"):
        build_model_best_frame("1700-06-01T12:00:00", half_window_hours=2.0)


# ------------------------------------------------ limb-profile contacts, 2024
# [EB2024] F. Espenak & J. Anderson, *Eclipse Bulletin: Total Solar Eclipse
# of 2024 April 08*, Table 2-3 (sample page, eclipsewise.com/pubs/images/
# EB2024-sample2.pdf): "These tables include the effect of the Moon's profile
# on the contact times" (book page).  C2/C3 printed in local daylight time
# (IL CDT = UT-5, IN EDT = UT-4, Evansville CDT), here in UT; coordinates as
# printed (0.01 deg).  The book is all rights reserved: a cited subset only.
#
# Heights (not in the Bulletin, which has no elevation column): the ground
# height H is SRTM90 [SRTM] at the printed coordinates (opentopodata.org
# ``srtm90m``; orthometric, above the EGM96 geoid); N is the EGM96 geoid
# undulation there [EGM96] (GeographicLib's ``egm96-5`` grid, bilinear;
# docs/LIMB_VALIDATION_SOURCES.md).  The observer's height above the WGS-84
# ellipsoid is h = H + N [EGM96].
#   name, lat, lon [deg], C2, C3 [UT], T0 (near the site's maximum) [UT], H, N [m]
_EB2024_MID = [
    ("Carbondale", 37.73, -89.22, "18:59:16.6", "19:03:25.8", "19:01:21", 130, -29.8),
    ("Herrin", 37.80, -89.03, "18:59:38.0", "19:03:47.0", "19:01:43", 129, -29.9),
    ("Mount Vernon", 38.32, -88.92, "19:00:34.8", "19:04:15.3", "19:02:26", 155, -30.8),
    ("Evansville", 37.97, -87.58, "19:02:35.0", "19:05:38.0", "19:04:07", 108, -31.6),
    ("Vincennes", 38.68, -87.53, "19:02:51.9", "19:06:57.6", "19:04:55", 130, -32.4),
    ("Terre Haute", 39.47, -87.42, "19:04:22.9", "19:07:19.3", "19:05:51", 142, -33.3),
    ("Jasper", 38.40, -86.93, "19:03:55.5", "19:07:09.8", "19:05:33", 156, -33.4),
    ("Bedford", 38.87, -86.48, "19:04:46.6", "19:08:30.9", "19:06:41", 217, -33.9),
    ("Bloomington", 39.17, -86.53, "19:04:50.8", "19:08:53.8", "19:06:53", 240, -33.7),
    ("Indianapolis", 39.77, -86.15, "19:06:05.2", "19:09:52.1", "19:08:00", 219, -34.3),
    ("Columbus", 39.22, -85.92, "19:05:55.2", "19:09:42.3", "19:07:50", 197, -34.7),
    ("Anderson", 40.17, -85.68, "19:07:14.4", "19:10:45.3", "19:09:02", 271, -35.0),
    ("Muncie", 40.20, -85.38, "19:07:37.1", "19:11:19.6", "19:09:29", 289, -34.8),
    ("Richmond", 39.83, -84.90, "19:07:55.4", "19:11:45.9", "19:09:52", 281, -34.5),
]
# Within ~3 km of the northern limit (NASA SVS limb-corrected path; the
# Bulletin's FDCL column, 0.981 and 0.968): 1.1 and 2.9 km.
_EB2024_NEAR = [
    ("Effingham", 39.12, -88.55, "19:03:25.2", "19:03:50.9", "19:03:49", 182, -32.6),
    ("Crawfordsville", 40.03, -86.90, "19:06:38.6", "19:07:18.1", "19:07:08", 246, -34.1),
]


def _limb_ready() -> bool:
    if not default_band_path().exists():
        return False
    load_kernels()
    try:
        E.limb_axes(np.array([0.0]), "ITRS", "MOON_ME")  # needs the lunar kernels
    except SpiceError:
        return False
    return True


def _clock_s(hms: str) -> float:
    """``HH:MM:SS(.s)`` -> seconds of the day."""
    return sum(float(v) * k for v, k in zip(hms.split(":"), (3600, 60, 1), strict=True))


def _eb_residuals(site, limb_mode, with_height=True):
    """(ours - [EB2024]) for C2 and C3 [s], for the observer at its height
    above the ellipsoid (H + N) or, with ``with_height=False``, at sea level."""
    from app.besselian import BesselianModel
    from app.circumstances import local_raw

    _name, lat, lon, c2, c3, t0, h_geoid, n_geoid = site
    model = BesselianModel(t0_utc=f"2024-04-08T{t0}")
    height = h_geoid + n_geoid if with_height else 0.0
    raw = local_raw(model, lat, lon, limb_mode, height)
    assert raw.central, site
    base = _clock_s(t0)
    return [base + c * 3600.0 - _clock_s(ref) for c, ref in ((raw.c2, c2), (raw.c3, c3))]


def test_limb_profile_contacts_match_eclipse_bulletin_2024():
    """Limb-profile C2/C3 against [EB2024] at 14 mid-path sites (22-93 km from
    a limit), each observer at its height above the ellipsoid (H + N above).
    Achieved: median 0.39 s, max 1.10 s over the 28 contacts (at sea level
    0.37 / 1.15 s), against median 0.96 s / max 3.28 s for the mean limb (k2)
    -- the profile is what brings us to the Bulletin.  Over all 36 mid-path
    Bulletin sites elevation takes the profile from median 0.35 / p90 0.68 /
    max 1.15 s to 0.30 / 0.61 / 1.10 s (docs/LIMB_PROFILE.md sec. 9.10); the
    remaining site-to-site scatter (+-0.5 s) is not elevation."""
    if not _limb_ready():
        pytest.skip("limb band / lunar kernels not installed (python -m kernels.bootstrap --limb)")
    prof = np.abs([r for s in _EB2024_MID for r in _eb_residuals(s, "profile")])
    mean = np.abs([r for s in _EB2024_MID for r in _eb_residuals(s, "mean")])
    assert np.max(prof) <= 1.2, prof
    assert np.median(prof) <= 0.4, np.median(prof)
    assert np.median(prof) < 0.5 * np.median(mean), (np.median(prof), np.median(mean))


_ROUNDING_XFAIL = pytest.mark.xfail(strict=True, reason=(
    "1.1 km inside the northern limit the contacts move ~1000 s of duration per "
    "degree of latitude, and the Bulletin's coordinates are rounded to 0.01 deg: "
    "across that +-0.005 deg box our C2 residual spans about -1 to +14 s. At the "
    "printed point: +6.0 / -3.5 s with the site's 149 m (+5.5 / -3.4 s at sea "
    "level), so elevation does not explain it (docs/LIMB_PROFILE.md sec. 9.10)"))


@pytest.mark.parametrize("site", [
    pytest.param(_EB2024_NEAR[0], id="Effingham", marks=_ROUNDING_XFAIL),
    pytest.param(_EB2024_NEAR[1], id="Crawfordsville"),  # 2.9 km: -0.44 / -0.02 s
])
def test_limb_profile_contacts_near_the_limit_eb2024(site):
    if not _limb_ready():
        pytest.skip("limb band / lunar kernels not installed")
    assert np.max(np.abs(_eb_residuals(site, "profile"))) <= 1.2


# [Irwin21] J. Irwin et al. (2021), arXiv:2107.09416, Table 3: a site near
# Vale OR, ~1.5 km inside the 2017 southern limit, 43 57' 10.9" N,
# 117 13' 09.8" W, h = 711 m (taken as orthometric; + EGM96 N = -17.2 m, as
# above), and four independent limb-corrected predictions of C2 / C3 (UT).
_VALE = (43.953028, -117.219389, 711.0 - 17.2)
_VALE_C2 = ("17:25:31.6", "17:25:33.6", "17:25:32.9", "17:25:34.3")
_VALE_C3 = ("17:26:07.7", "17:26:06.9", "17:26:07.0", "17:26:06.9")


def test_limb_profile_contacts_at_vale_match_irwin_2021():
    """Profile-mode C2/C3 at the Vale OR site against the four predictions of
    [Irwin21] Table 3 (Solar Eclipse Maestro, Occult x2, Irwin et al.; they
    span 2.7 s at C2).  Coordinates there are to 0.1 arcsec (3 m), so, unlike
    the Bulletin's, they resolve a near-limit site, and the site's 694 m
    matter: at sea level our C2 is 17:25:41.3, 7 s after the latest of the
    four.  With the height: C2 17:25:34.5 (0.2 s after the latest, Irwin et
    al.'s own 34.3), C3 17:26:07.0 (inside 06.9-07.7).  Gate: each contact
    within 0.5 s of the four predictions' range."""
    if not _limb_ready():
        pytest.skip("limb band / lunar kernels not installed")
    from app.besselian import BesselianModel
    from app.circumstances import local_raw

    model = BesselianModel(t0_utc="2017-08-21T17:25:50")
    raw = local_raw(model, *_VALE[:2], "profile", _VALE[2])
    assert raw.central
    for ours_h, refs in ((raw.c2, _VALE_C2), (raw.c3, _VALE_C3)):
        ours = _clock_s("17:25:50") + ours_h * 3600.0
        ref = [_clock_s(r) for r in refs]
        assert min(ref) - 0.5 <= ours <= max(ref) + 0.5, (ours, ref)


def test_mean_limb_contacts_at_height_match_the_direct_3d_geometry():
    """Mean-limb contacts for an observer 694 m up (Vale OR, 2017) against
    geometry that shares nothing with ``geo_to_fund``: at each contact the
    observer, placed by geodetic + h -> ECEF -> fundamental-plane axes
    (``geodesy_oracle.direct_fundamental``), is on its shadow cone
    (``m = L1'`` for C1/C4, ``m = |L2'|`` for C2/C3, [ES92] eq. 8.353) to
    1e-10 Earth radii (0.6 mm), and the sea-level contacts are not (the height
    moves the observer by ~1e-4).  Second, independent physics: a raised
    observer sees what a sea-level one does at the foot of its line of sight,
    shifted H cot(alt) away from the Sun (0.69 km here): C2/C3 agree with that
    point's to 0.03 s (gate 0.1 s; totality 36.6 s vs 33.8 s at sea level)."""
    from geodesy_oracle import direct_fundamental

    from app.besselian import BesselianModel
    from app.circumstances import local_raw

    lat, lon, h = _VALE
    model = BesselianModel(t0_utc="2017-08-21T17:25:50")
    raw = local_raw(model, lat, lon, "mean", h)
    sea = local_raw(model, lat, lon, "mean")
    assert raw.central and sea.central

    def cone_residual(t_hours, central, height):
        e = model.evaluate_direct(np.array([t_hours]))
        xi, eta, zeta = direct_fundamental(lat, lon, e["d"][0], e["mu"][0], height)
        m = np.hypot(xi - e["x"][0], eta - e["y"][0])
        if central:
            return m - abs(e["l2"][0] - zeta * e["tan_f2"][0])
        return m - (e["l1"][0] - zeta * e["tan_f1"][0])

    for name, central in (("c1", False), ("c4", False), ("c2", True), ("c3", True)):
        assert abs(cone_residual(getattr(raw, name), central, h)) <= 1e-10, name
        assert abs(cone_residual(getattr(sea, name), central, h)) > 1e-7, name

    alt, az = raw.alt_deg[1], raw.az_deg[1]
    la2, lo2 = E.destination(lat, lon, az + 180.0, h / 1000.0 / np.tan(np.radians(alt)))
    foot = local_raw(model, float(la2), float(lo2), "mean")
    assert abs(raw.c2 - foot.c2) * 3600.0 <= 0.1
    assert abs(raw.c3 - foot.c3) * 3600.0 <= 0.1

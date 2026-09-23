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


def _t0_utc(g):
    """UTC epoch of greatest eclipse for case ``g``: its ``utc``, or its published
    ``td`` (TDT) converted with *our* delta-T.  The Canon's own UT is TD minus
    the Canon's delta-T (73.4 s in 2023, vs 69.2 s measured), so for the newer
    cases the TD instant is the unambiguous anchor: ~4 s moves a grazing polar
    shadow by several km."""
    if "utc" in g:
        return g["utc"]
    import spiceypy

    from app.ephemeris import et_to_utc, load_kernels

    load_kernels()
    return et_to_utc(spiceypy.str2et(g["td"] + " TDT"))


def _greatest_model(g):
    import spiceypy

    try:
        return _build_model(_t0_utc(g))
    except spiceypy.utils.exceptions.SpiceSPKINSUFFDATA:
        pytest.skip("ephemeris does not cover this epoch (mirror DE432s spans 1949-2050; "
                    "use NAIF DE440s, 1849-2150)")


def _greatest_edges(g):
    """(track point at greatest eclipse, north, south, width_km, is_total)."""
    from app.geography import central_track, element_rates, shadow_edge_limits, shadow_radii

    model, _frame = _greatest_model(g)
    dt = 0.02
    # central_track reduces each instant to (lat, lon) with the along-track
    # bearing from its neighbours -- the same path /central-line uses (item R3).
    elems, track = central_track(model, np.array([-dt, 0.0, dt]))
    tp = next(p for p in track if abs(p.t_hours) < 1e-9)  # central instant t=0
    i = tp.i
    r = element_rates(model, np.array([0.0]))
    north, south, width = shadow_edge_limits(
        elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i],
        elems["l2"][i], elems["tan_f2"][i], tp.bearing,
        rates=(r["x"], r["y"], r["d"], r["mu"], r["l2"]),
    )
    _pen, _umb, is_total = shadow_radii(
        elems["x"][i], elems["y"][i], elems["d"][i],
        elems["l1"][i], elems["l2"][i], elems["tan_f1"][i], elems["tan_f2"][i],
    )
    return tp, north, south, width, is_total


def _check_greatest(g):
    tp, north, south, _width, is_total = _greatest_edges(g)
    # Published lat/lon are rounded to 0.1 deg (measured |dlat|, |dlon| <= 0.05
    # deg over all seven cases); the width is test_path_width_at_greatest_eclipse.
    assert tp.lat == pytest.approx(g["lat"], abs=0.12)
    assert tp.lon == pytest.approx(g["lon"], abs=0.12)
    assert north is not None and north[0] > south[0]
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

    model, _frame = _greatest_model(g)
    e = model.evaluate(np.array([0.0]))  # central point at maximum
    lon, lat = fund_to_geo(e["x"][0], e["y"][0], e["d"][0], e["mu"][0])
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
    import spiceypy

    from app.ephemeris import besselian_instant, load_kernels

    load_kernels()
    et = spiceypy.str2et(g["t0"] + " TDT")  # published elements are tabulated vs TDT
    try:
        return et, besselian_instant(et, _usable_frame())
    except spiceypy.utils.exceptions.SpiceSPKINSUFFDATA:
        pytest.skip("ephemeris does not cover this epoch (mirror DE432s spans 1949-2050; "
                    "use NAIF DE440s, 1849-2150)")


@pytest.mark.parametrize("name", sorted(_REF_ECLIPSES))
def test_reference_elements_match_published(name):
    """x, y, d at t0 (TDT).  Measured: |dx|, |dy| <= 1e-5, |dd| <= 9.3e-5 deg."""
    g = _REF_ECLIPSES[name]
    _et, bi = _elements_at_t0(g)
    for key in ("x", "y", "d"):
        assert getattr(bi, key) == pytest.approx(g[key], abs=1e-3), key


@pytest.mark.parametrize("name", sorted(_REF_ECLIPSES))
def test_reference_mu_matches_published_with_delta_t(name):
    """mu_published = mu_ours + delta-T * 1.002738 * 15"/s [ES92] sec. 8.36 (see
    test_2024_mu_matches_published_with_delta_t), here with delta-T measured
    (2021, 2023) or from the model (1868, 1919).  Measured: <= 1.9e-5 deg."""
    from app.ephemeris import tt_minus_ut1

    g = _REF_ECLIPSES[name]
    et, bi = _elements_at_t0(g)
    delta_t = float(tt_minus_ut1(et)[0])
    mu_eph = (bi.mu % 360.0 + delta_t * 1.002738 * 15.0 / 3600.0) % 360.0
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
    _tp, _north, _south, width, _is_total = _greatest_edges(g)
    assert width == pytest.approx(g["width_km"], abs=1.0)


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


@pytest.mark.skipif(not default_metakernel().exists(), reason="SPICE kernels not downloaded")
def test_best_frame_reraises_ephemeris_coverage_gaps():
    """An epoch outside the loaded SPK is a coverage error, not "no usable
    frame": it must reach the caller as SpiceSPKINSUFFDATA so the pre-IERS
    reference tests skip on the mirror's DE432s (1949-2050) instead of failing
    (CI full-validation, 1868/1919 cases)."""
    import spiceypy

    from app.besselian import build_model_best_frame

    with pytest.raises(spiceypy.utils.exceptions.SpiceSPKINSUFFDATA):
        build_model_best_frame("1700-06-01T12:00:00", half_window_hours=2.0)


# ------------------------------------------------ limb-profile contacts, 2024
# [EB2024] F. Espenak & J. Anderson, *Eclipse Bulletin: Total Solar Eclipse
# of 2024 April 08*, Table 2-3 (sample page, eclipsewise.com/pubs/images/
# EB2024-sample2.pdf): "These tables include the effect of the Moon's profile
# on the contact times" (book page).  C2/C3 printed in local daylight time
# (IL CDT = UT-5, IN EDT = UT-4, Evansville CDT), here in UT; coordinates as
# printed (0.01 deg).  The book is all rights reserved: a cited subset only.
#   name, lat, lon [deg], C2, C3 [UT], T0 (near the site's maximum) [UT]
_EB2024_MID = [
    ("Carbondale", 37.73, -89.22, "18:59:16.6", "19:03:25.8", "19:01:21"),
    ("Herrin", 37.80, -89.03, "18:59:38.0", "19:03:47.0", "19:01:43"),
    ("Mount Vernon", 38.32, -88.92, "19:00:34.8", "19:04:15.3", "19:02:26"),
    ("Evansville", 37.97, -87.58, "19:02:35.0", "19:05:38.0", "19:04:07"),
    ("Vincennes", 38.68, -87.53, "19:02:51.9", "19:06:57.6", "19:04:55"),
    ("Terre Haute", 39.47, -87.42, "19:04:22.9", "19:07:19.3", "19:05:51"),
    ("Jasper", 38.40, -86.93, "19:03:55.5", "19:07:09.8", "19:05:33"),
    ("Bedford", 38.87, -86.48, "19:04:46.6", "19:08:30.9", "19:06:41"),
    ("Bloomington", 39.17, -86.53, "19:04:50.8", "19:08:53.8", "19:06:53"),
    ("Indianapolis", 39.77, -86.15, "19:06:05.2", "19:09:52.1", "19:08:00"),
    ("Columbus", 39.22, -85.92, "19:05:55.2", "19:09:42.3", "19:07:50"),
    ("Anderson", 40.17, -85.68, "19:07:14.4", "19:10:45.3", "19:09:02"),
    ("Muncie", 40.20, -85.38, "19:07:37.1", "19:11:19.6", "19:09:29"),
    ("Richmond", 39.83, -84.90, "19:07:55.4", "19:11:45.9", "19:09:52"),
]
# Within ~3 km of the limit (NASA SVS limb-corrected path): 1.1 and 2.9 km.
_EB2024_NEAR = [
    ("Effingham", 39.12, -88.55, "19:03:25.2", "19:03:50.9", "19:03:49"),
    ("Crawfordsville", 40.03, -86.90, "19:06:38.6", "19:07:18.1", "19:07:08"),
]


def _limb_ready() -> bool:
    import spiceypy

    from app import limb
    from app.ephemeris import load_kernels

    if not limb.default_band_path().exists():
        return False
    load_kernels()
    try:
        spiceypy.pxform("J2000", "MOON_ME", 0.0)
    except spiceypy.utils.exceptions.SpiceyError:
        return False
    return True


def _eb_residuals(site, limb_mode):
    """(ours - [EB2024]) for C2 and C3 [s] at a sea-level observer."""
    from app.besselian import BesselianModel
    from app.circumstances import local_raw

    _name, lat, lon, c2, c3, t0 = site
    model = BesselianModel(t0_utc=f"2024-04-08T{t0}")
    raw = local_raw(model, lat, lon, limb_mode)
    assert raw.central, site
    base = sum(float(v) * k for v, k in zip(t0.split(":"), (3600, 60, 1), strict=True))
    out = []
    for c, ref in ((raw.c2, c2), (raw.c3, c3)):
        ref_s = sum(float(v) * k for v, k in zip(ref.split(":"), (3600, 60, 1), strict=True))
        out.append(base + c * 3600.0 - ref_s)
    return out


def test_limb_profile_contacts_match_eclipse_bulletin_2024():
    """Limb-profile C2/C3 against [EB2024] at 14 mid-path sites (22-93 km from
    a limit).  Achieved (sea-level observers; the sites are 130-270 m up and
    elevation is not modelled yet): median 0.37 s, max 1.15 s over the 28
    contacts, against median 0.82 s / max 3.14 s for the mean limb (k2) --
    the profile is what brings us to the Bulletin.  Over all 36 mid-path
    Bulletin sites: median 0.35 s, p90 0.68 s (docs/LIMB_PROFILE.md sec. 9.7).
    The 0.3 s gate of docs sec. 5.4 waits on observer elevation."""
    if not _limb_ready():
        pytest.skip("limb band / lunar kernels not installed (python -m kernels.bootstrap --limb)")
    prof = np.abs([r for s in _EB2024_MID for r in _eb_residuals(s, "profile")])
    mean = np.abs([r for s in _EB2024_MID for r in _eb_residuals(s, "mean")])
    assert np.max(prof) <= 1.2, prof
    assert np.median(prof) <= 0.4, np.median(prof)
    assert np.median(prof) < 0.5 * np.median(mean), (np.median(prof), np.median(mean))


_ELEVATION_XFAIL = pytest.mark.xfail(strict=True, reason=(
    "1.1 km from the limit the contacts are seconds-sensitive to the observer's "
    "elevation (182 m here), which is not modelled: +5.5 / -3.4 s against [EB2024] "
    "(docs/LIMB_PROFILE.md sec. 9.7)"))


@pytest.mark.parametrize("site", [
    pytest.param(_EB2024_NEAR[0], id="Effingham", marks=_ELEVATION_XFAIL),
    pytest.param(_EB2024_NEAR[1], id="Crawfordsville"),  # 2.9 km: -0.88 / +0.19 s
])
def test_limb_profile_contacts_near_the_limit_eb2024(site):
    if not _limb_ready():
        pytest.skip("limb band / lunar kernels not installed")
    assert np.max(np.abs(_eb_residuals(site, "profile"))) <= 1.2

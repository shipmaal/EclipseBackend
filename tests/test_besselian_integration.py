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
    """Prefer ITRF93 (binary Earth PCK); fall back to the pyerfa TOD frame, which
    needs no binary PCK, and finally to IAU_EARTH."""
    import spiceypy

    from app.besselian import BesselianModel

    for frame in ("ITRF93", "TOD", "IAU_EARTH"):
        try:
            return BesselianModel(t0_utc=t0, earth_frame=frame, half_window_hours=2.0), frame
        except spiceypy.utils.exceptions.SpiceyError:
            spiceypy.reset()
    raise RuntimeError("no usable Earth-orientation frame in the furnished kernels")


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
_2017_GREATEST = {"utc": "2017-08-21T18:25:30", "lat": 37.0, "lon": -87.7, "width_km": 114.7}


def _usable_frame():
    import spiceypy

    from app.ephemeris import besselian_instant, utc_to_et, load_kernels

    load_kernels()
    et = utc_to_et("2017-08-21T18:00:00")
    for frame in ("ITRF93", "TOD", "IAU_EARTH"):
        try:
            besselian_instant(et, frame)
            return frame
        except spiceypy.utils.exceptions.SpiceyError:
            spiceypy.reset()
    raise RuntimeError("no usable Earth-orientation frame")


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


def test_2017_greatest_eclipse_point():
    """Central line at greatest eclipse matches the published lat/lon and width."""
    from app.geography import fund_to_geo, shadow_radii

    model, _frame = _build_model(_2017_GREATEST["utc"])
    e = model.evaluate(np.array([0.0]))
    lon, lat = fund_to_geo(e["x"][0], e["y"][0], e["d"][0], e["mu"][0])
    _pen, umb, is_total = shadow_radii(
        e["x"][0], e["y"][0], e["d"][0], e["l1"][0], e["l2"][0], e["tan_f1"][0], e["tan_f2"][0]
    )

    assert lat == pytest.approx(_2017_GREATEST["lat"], abs=0.1)
    assert lon == pytest.approx(_2017_GREATEST["lon"], abs=0.1)
    assert is_total
    # Path width within ~10 km (our single k differs from Espenak's k1/k2 split).
    assert (2 * umb) == pytest.approx(_2017_GREATEST["width_km"], abs=10.0)

"""Eclipse catalog + global contacts against the Espenak / NASA canon (needs kernels)."""

from __future__ import annotations

import pytest

from app.ephemeris import default_metakernel

pytestmark = pytest.mark.skipif(
    not default_metakernel().exists(),
    reason="SPICE kernels not downloaded (run `python -m kernels.bootstrap`)",
)

# Every solar eclipse 2019-2024 [Espenak] (NASA Eclipse Web Site decade tables):
# date, Canon type.  The lunar eclipses of the same years share the coarse
# rho-minimum and must NOT appear (z > 0 filter).
_CANON_2019_2024 = [
    ("2019-01-06", "partial"), ("2019-07-02", "total"), ("2019-12-26", "annular"),
    ("2020-06-21", "annular"), ("2020-12-14", "total"),
    ("2021-06-10", "annular"), ("2021-12-04", "total"),
    ("2022-04-30", "partial"), ("2022-10-25", "partial"),
    ("2023-04-20", "hybrid"), ("2023-10-14", "annular"),
    ("2024-04-08", "total"), ("2024-10-02", "annular"),
]

# Greatest-eclipse rows [Espenak]: UT instant, gamma, magnitude, point, duration, width.
_GREATEST = {
    "2017-08-21": dict(utc="18:25:30", gamma=0.4367, mag=1.0306, lat=37.0, lon=-87.7, dur=160, width=114.7),
    "2023-10-14": dict(utc="17:59:27", gamma=0.3753, mag=0.9520, lat=11.4, lon=-83.1, dur=317, width=187.4),
    "2024-04-08": dict(utc="18:17:15", gamma=0.3431, mag=1.0566, lat=25.3, lon=-104.1, dur=268, width=197.5),
}

# Penumbral first/last contact with the Earth [Espenak] (NASA eclipse pages), UT.
_P1_P4 = {"2017-08-21": ("15:46:48", "21:04:19"), "2024-04-08": ("15:42:15", "20:52:19")}


def _secs(hms: str) -> int:
    h, m, s = (int(v) for v in hms.split(":"))
    return h * 3600 + m * 60 + s


def test_catalog_2019_2024_matches_canon():
    from app.catalog import find_eclipses

    events = find_eclipses("2019-01-01", "2024-12-31", detail=False)
    got = [(e["greatest_utc"][:10], e["type"]) for e in events]
    assert got == _CANON_2019_2024
    assert all(e["magnitude"] > 0 for e in events)  # a negative magnitude was the lunar tell


@pytest.mark.parametrize("day", sorted(_GREATEST))
def test_catalog_greatest_eclipse_rows(day):
    """Greatest-eclipse instant to ~10 s (Espenak's delta-T vs measured), gamma to
    1e-3, magnitude to 1e-3, point to the published 0.1 deg, duration <3 s,
    width ~1 km -- the Canon row reproduced from the catalog alone."""
    from app.catalog import find_eclipses

    g = _GREATEST[day]
    (ev,) = find_eclipses(f"{day}T00:00:00", f"{day}T23:59:59", detail=True)
    assert ev["greatest_utc"][:10] == day
    assert abs(_secs(ev["greatest_utc"][11:]) - _secs(g["utc"])) <= 30
    assert ev["gamma"] == pytest.approx(g["gamma"], abs=1e-3)
    assert ev["magnitude"] == pytest.approx(g["mag"], abs=1e-3)
    assert ev["lat"] == pytest.approx(g["lat"], abs=0.12)
    assert ev["lon"] == pytest.approx(g["lon"], abs=0.12)
    assert ev["central_duration_s"] == pytest.approx(g["dur"], abs=3.0)
    assert ev["width_km"] == pytest.approx(g["width"], abs=2.0)
    assert set(ev["contacts"]) == {"P1", "U1", "U2", "U3", "U4", "P4"}


@pytest.mark.parametrize("day", sorted(_P1_P4))
def test_global_contacts_p1_p4(day):
    """P1/P4 from the auxiliary-circle approximation [ES92] sec. 8.34: within
    ~10 s of the bulletin values (measured 1-5 s)."""
    from app.besselian import BesselianModel
    from app.geography import format_clock, global_contacts

    t0 = f"{day}T{_GREATEST[day]['utc']}"
    model = BesselianModel(t0_utc=t0, half_window_hours=2.5)
    c = {k: format_clock(model.t0_utc, v) for k, v in global_contacts(model).items()}
    p1, p4 = _P1_P4[day]
    assert abs(_secs(c["P1"]) - _secs(p1)) <= 15
    assert abs(_secs(c["P4"]) - _secs(p4)) <= 15
    # Ordering: P1 < U1 < U2 < U3 < U4 < P4.
    order = [_secs(c[k]) for k in ("P1", "U1", "U2", "U3", "U4", "P4")]
    assert order == sorted(order)


def test_penumbral_limits_bound_the_partial_region():
    """Penumbral limits from the same bisection with l1/tan_f1, clipped to the
    terminator: they exist, straddle the central line, and lie ~3000+ km out."""
    import numpy as np

    from app.besselian import BesselianModel
    from app.geography import _haversine_km, central_track, shadow_edge_limits_v

    model = BesselianModel(t0_utc="2024-04-08T18:17:15")
    dt = 0.02
    elems, track = central_track(model, np.array([-dt, 0.0, dt]))  # 3 points -> a real bearing
    tp = next(p for p in track if abs(p.t_hours) < 1e-9)
    i = tp.i
    n_lat, n_lon, s_lat, s_lon, _w = shadow_edge_limits_v(
        elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i],
        elems["l1"][i], elems["tan_f1"][i], tp.bearing, max_km=10_000.0,
    )
    assert not np.isnan(n_lat[0]) and n_lat[0] > tp.lat > s_lat[0]
    assert 2500 < _haversine_km(tp.lat, tp.lon, n_lat[0], n_lon[0]) < 5000
    assert 2500 < _haversine_km(tp.lat, tp.lon, s_lat[0], s_lon[0]) < 5000

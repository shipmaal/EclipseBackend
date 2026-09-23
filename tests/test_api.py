"""HTTP-level tests of the FastAPI surface (need the SPICE kernels)."""

from __future__ import annotations

import pytest

from app.ephemeris import default_metakernel

pytestmark = pytest.mark.skipif(
    not default_metakernel().exists(),
    reason="SPICE kernels not downloaded (run `python -m kernels.bootstrap`)",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_non_iso_epoch_is_a_400_not_a_500(client):
    # SPICE would parse this, but the clock formatter would not: one parser now.
    r = client.get("/circumstances",
                   params={"epoch": "2024 APR 08 18:17:15", "lat": 25.3, "lon": -104.1})
    assert r.status_code == 400
    assert "epoch" in r.json()["detail"]


def test_z_suffix_and_offset_are_normalized(client):
    a = client.get("/besselian", params={"epoch": "2024-04-08T18:00:00Z"}).json()
    b = client.get("/besselian", params={"epoch": "2024-04-08T20:00:00+02:00"}).json()
    assert a["t0_utc"] == b["t0_utc"] == "2024-04-08T18:00:00"


def test_central_line_point_cap(client):
    r = client.get("/central-line", params={"epoch": "2024-04-08T18:17:15", "start_hours": -6,
                                            "end_hours": 6, "step_minutes": 0.01})
    assert r.status_code == 400 and "samples" in r.json()["detail"]


def test_central_line_has_limits_at_greatest_eclipse(client):
    r = client.get("/central-line", params={"epoch": "2024-04-08T18:17:15", "start_hours": -0.1,
                                            "end_hours": 0.1, "step_minutes": 6})
    assert r.status_code == 200
    pts = r.json()["central_line"]
    mid = min(pts, key=lambda p: abs(p["t_hours"]))
    assert mid["is_total"] and 190 < mid["width_km"] < 205
    assert mid["north_limit"]["lat"] > mid["south_limit"]["lat"]


def test_map_shape_and_visibility(client):
    r = client.get("/map", params={"epoch": "2024-04-08T18:17:15", "lat_step": 10, "lon_step": 10})
    assert r.status_code == 200
    m = r.json()
    assert len(m["magnitude"]) == len(m["lats"]) == 19
    assert len(m["magnitude"][0]) == len(m["lons"]) == 36
    # Greatest eclipse near (25N, 104W): the (30N, -100E) cell is central and visible.
    i, j = m["lats"].index(30.0), m["lons"].index(-100.0)
    assert m["central"][i][j] is True and m["visible"][i][j] is True and m["magnitude"][i][j] > 1.0
    # Central India: geometry says penumbra, horizon says night.
    i, j = m["lats"].index(20.0), m["lons"].index(80.0)
    assert m["visible"][i][j] is False


def test_map_cell_cap(client):
    r = client.get("/map",
                   params={"epoch": "2024-04-08T18:17:15", "lat_step": 0.1, "lon_step": 0.1})
    assert r.status_code == 400


def test_central_line_carries_contacts_and_penumbral_limits(client):
    r = client.get("/central-line", params={"epoch": "2024-04-08T18:17:15", "start_hours": -0.1,
                                            "end_hours": 0.1, "step_minutes": 6})
    body = r.json()
    assert set(body["contacts"]) == {"P1", "U1", "U2", "U3", "U4", "P4"}
    mid = min(body["central_line"], key=lambda p: abs(p["t_hours"]))
    assert mid["penumbra_north_limit"]["lat"] > mid["north_limit"]["lat"]
    assert mid["penumbra_south_limit"]["lat"] < mid["south_limit"]["lat"]


def test_eclipses_endpoint_and_range_cap(client):
    r = client.get("/eclipses",
                   params={"start": "2024-01-01", "end": "2024-12-31", "detail": False})
    assert r.status_code == 200
    assert [e["type"] for e in r.json()["eclipses"]] == ["total", "annular"]
    r = client.get("/eclipses", params={"start": "2000-01-01", "end": "2024-12-31"})
    assert r.status_code == 400 and "exceeds" in r.json()["detail"]


# ------------------------- review items C1, C3, C7, C8 (docs/CODE_REVIEW_FOLLOWUPS.md §5)
def test_map_tiny_step_is_a_400_before_any_allocation(client):
    # Previously np.arange ran first: 1e-6 allocated GBs, 1e-300 raised a 500.
    for step in (1e-6, 1e-300):
        r = client.get("/map",
                       params={"epoch": "2024-04-08T18:17:15", "lat_step": step, "lon_step": step})
        assert r.status_code == 400 and "cells" in r.json()["detail"], step


def test_central_line_subnormal_step_is_a_400(client):
    # (4 h * 60) / 1e-310 overflows to inf; int(inf) used to be a 500.
    r = client.get("/central-line",
                   params={"epoch": "2024-04-08T18:17:15", "step_minutes": 1e-310})
    assert r.status_code == 400 and "samples" in r.json()["detail"]


def test_eclipses_on_a_fresh_worker_loads_the_kernels(client):
    # The range guard converts the epochs before find_eclipses loads the kernels;
    # on a fresh worker (empty pools) that was a 400 "SPICE error".
    from app.ephemeris import load_kernels, unload_kernels

    unload_kernels()
    try:
        r = client.get("/eclipses",
                       params={"start": "2024-01-01", "end": "2024-12-31", "detail": False})
        assert r.status_code == 200, r.text
        assert r.json()["count"] == 2
    finally:
        load_kernels()


def test_eclipses_compute_value_error_is_a_500_not_invalid_epoch(monkeypatch):
    # Only epoch parsing maps to 400; a ValueError from the computation (e.g. a
    # native invariant check) is a bug and must surface as a 500.
    from fastapi.testclient import TestClient

    from app import main

    def broken(*args, **kwargs):
        raise ValueError("local_circumstances: partial roots are all one-signed")

    monkeypatch.setattr(main, "find_eclipses", broken)
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.get("/eclipses", params={"start": "2024-01-01", "end": "2024-12-31"})
    assert r.status_code == 500
    r = c.get("/eclipses", params={"start": "2024 JAN 01", "end": "2024-12-31"})
    assert r.status_code == 400 and "invalid epoch" in r.json()["detail"]


def test_circumstances_limb_parameter(client):
    """limb= selects the lunar limb model (docs/LIMB_PROFILE.md); it is echoed,
    an unknown value is a 400, and the profile mode is a 503 (not a 500) when
    its inputs are missing."""
    from app import limb

    params = {"epoch": "2024-04-08T19:08:00", "lat": 39.77, "lon": -86.15}
    r = client.get("/circumstances", params={**params, "limb": "watts"})
    assert r.status_code == 400
    mean = client.get("/circumstances", params=params).json()
    assert mean["limb"] == "mean"
    r = client.get("/circumstances", params={**params, "limb": "profile"})
    if not limb.default_band_path().exists():
        assert r.status_code == 503
        return
    if r.status_code == 503:
        pytest.skip("MOON_ME not loaded (kernels.bootstrap --limb)")
    prof = r.json()
    assert prof["limb"] == "profile" and prof["type"] == "total"
    # Indianapolis 2024: the limb shortens totality by ~3 s (229.6 -> 226.4 s,
    # docs sec. 9.6); C1/C4 and the magnitude are the mean limb's.
    assert prof["central_duration_s"] == pytest.approx(mean["central_duration_s"] - 3.2, abs=0.3)
    assert (prof["C1"], prof["C4"], prof["magnitude"]) == (mean["C1"], mean["C4"],
                                                         mean["magnitude"])

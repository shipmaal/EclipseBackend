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

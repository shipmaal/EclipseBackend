"""HTTP-level tests of the FastAPI surface (need the SPICE kernels)."""

from __future__ import annotations

import math

import pytest

from app.core import default_metakernel

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
    # on a fresh worker (an empty pool) that was a 400 "SPICE error".
    from app.core import load_kernels, unload_kernels

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
    from app.core import default_band_path

    params = {"epoch": "2024-04-08T19:08:00", "lat": 39.77, "lon": -86.15}
    r = client.get("/circumstances", params={**params, "limb": "watts"})
    assert r.status_code == 400
    mean = client.get("/circumstances", params=params).json()
    assert mean["limb"] == "mean"
    r = client.get("/circumstances", params={**params, "limb": "profile"})
    if not default_band_path().exists():
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


def test_circumstances_elev_parameter(client):
    """elev= is the observer's height above the WGS-84 ellipsoid [m], echoed
    as elev_m (0 by default); out of [-500, 9000] is a 422.  At Vale OR (2017,
    ~1.5 km inside the southern limit, [Irwin21]) its 694 m lengthen the
    mean-limb totality from 33.8 s to 36.6 s: the raised observer sees the
    shadow as the ground 0.69 km deeper in the path does."""
    params = {"epoch": "2017-08-21T17:25:50", "lat": 43.953028, "lon": -117.219389}
    sea = client.get("/circumstances", params=params).json()
    assert sea["elev_m"] == 0.0
    up = client.get("/circumstances", params={**params, "elev": 693.8}).json()
    assert up["elev_m"] == 693.8
    assert sea["central_duration_s"] == pytest.approx(33.8, abs=0.1)
    assert up["central_duration_s"] == pytest.approx(36.6, abs=0.1)
    for bad in (-501.0, 9001.0):
        assert client.get("/circumstances", params={**params, "elev": bad}).status_code == 422


@pytest.mark.parametrize("window_hours", [0.5, 1.0, 1.25])
def test_short_fit_windows_are_valid(client, window_hours):
    """Every window the API accepts can be fitted (item C5): below 1.5 h the
    sampling step shrinks to 2 hw / 3 (4 samples for the cubic). Before, these
    were 400 "invalid epoch: ... fewer than 4 samples"."""
    r = client.get("/besselian", params={"epoch": "2024-04-08T18:17:20",
                                         "window_hours": window_hours})
    assert r.status_code == 200, r.text
    x = r.json()["polynomials"]["x"]
    assert len(x) == 4 and all(math.isfinite(v) for v in x)
    r = client.get("/circumstances", params={"epoch": "2024-04-08T18:17:20", "lat": 32.0,
                                             "lon": -104.0, "window_hours": 1.0})
    assert r.status_code == 200, r.text


def test_model_value_error_is_a_500_not_invalid_epoch(monkeypatch):
    """Only epoch parsing is "invalid epoch" (item C5): a ValueError from the
    model's computation is a bug and surfaces as a 500."""
    from fastapi.testclient import TestClient

    from app import main

    def broken(*args, **kwargs):
        raise ValueError("besselian::fit_polynomials: fewer than 4 samples in the window")

    monkeypatch.setattr(main, "BesselianModel", broken)
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.get("/besselian", params={"epoch": "2024-04-08T18:17:20"})
    assert r.status_code == 500
    r = c.get("/besselian", params={"epoch": "2024 APR 08"})
    assert r.status_code == 400 and "invalid epoch" in r.json()["detail"]


def test_openmp_entry_points_hold_the_parallel_lock(client, monkeypatch):
    """/circumstances (mean and profile) and /central-line run OpenMP teams in the
    core, so they hold app.core.PARALLEL_LOCK: one team per process (item C6)."""
    import _eclipse

    from app import circumstances, main
    from app.core import PARALLEL_LOCK

    held = []

    def spy(real):
        def wrapped(*args, **kwargs):
            held.append(PARALLEL_LOCK.locked())
            return real(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(circumstances._eclipse, "local_circumstances",
                        spy(_eclipse.local_circumstances))
    monkeypatch.setattr(main._eclipse, "central_line", spy(_eclipse.central_line))
    r = client.get("/circumstances", params={"epoch": "2024-04-08T18:17:20", "lat": 32.0,
                                             "lon": -104.0})
    assert r.status_code == 200, r.text
    r = client.get("/central-line", params={"epoch": "2024-04-08T18:17:20"})
    assert r.status_code == 200, r.text
    assert held == [True, True]


def test_central_line_infinite_step_is_a_400(client):
    """step_minutes=inf passed the gt=0 check and returned an empty track (C7)."""
    r = client.get("/central-line", params={"epoch": "2024-04-08T18:00:00",
                                            "step_minutes": "inf"})
    assert r.status_code == 400

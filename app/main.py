"""FastAPI surface for the eclipse backend.

Endpoints are parameterized (no hard-coded eclipse) and do only computation --
no plotting in the request path. Positions come from SPICE / JPL DE440 via
:mod:`app.besselian`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .besselian import BesselianModel
from .circumstances import local_circumstances
from .ephemeris import DEFAULT_EARTH_FRAME, sub_solar_point, utc_to_et
from .geography import bearing, dec_to_hms, fund_to_geo, shadow_edge_limits, shadow_radii

_FRAMES = ("ITRS", "TOD", "ITRF93", "IAU_EARTH")

app = FastAPI(title="EclipseBackend", version="0.1.0")

# Allow the Three.js frontend (served from a file server or another port) to
# call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _build_model(epoch: str, window_hours: float, frame: str) -> BesselianModel:
    if frame not in _FRAMES:
        raise HTTPException(status_code=400, detail=f"frame must be one of {_FRAMES}")
    try:
        return BesselianModel(
            t0_utc=epoch, earth_frame=frame, half_window_hours=window_hours
        )
    except FileNotFoundError as exc:
        # Kernels not downloaded yet.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # SPICE errors (bad time string, missing body, ...)
        raise HTTPException(status_code=400, detail=f"SPICE error: {exc}") from exc


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/besselian")
async def besselian(
    epoch: str = Query(..., description="Reference epoch T0, UTC ISO-8601, e.g. 2024-04-08T18:00:00"),
    window_hours: float = Query(2.0, ge=0.5, le=6.0, description="Half-window sampled around T0"),
    frame: str = Query(DEFAULT_EARTH_FRAME, description="Earth-orientation frame: ITRF93, TOD or IAU_EARTH"),
) -> dict:
    """Besselian element polynomials (coefficients in powers of t = hours from T0)."""
    model = _build_model(epoch, window_hours, frame)
    return {"t0_utc": epoch, "frame": frame, "polynomials": asdict(model.polynomials)}


@app.get("/central-line")
async def central_line(
    epoch: str = Query(..., description="Reference epoch T0, UTC ISO-8601"),
    start_hours: float = Query(-2.0, description="Track start, hours from T0"),
    end_hours: float = Query(2.0, description="Track end, hours from T0"),
    step_minutes: float = Query(2.0, gt=0.0, description="Sample spacing in minutes"),
    window_hours: float = Query(2.0, ge=0.5, le=6.0),
    frame: str = Query(DEFAULT_EARTH_FRAME),
) -> dict:
    """Geographic track of the shadow axis (central line) over a time range."""
    if end_hours <= start_hours:
        raise HTTPException(status_code=400, detail="end_hours must exceed start_hours")

    model = _build_model(epoch, window_hours, frame)
    t = np.arange(start_hours, end_hours + 1e-9, step_minutes / 60.0)
    # Direct per-instant evaluation: exact along the whole track, including
    # start/end hours beyond the polynomial fit window (item A2).
    elems = model.evaluate_direct(t)

    # Pass 1: central-line geographic points (skip instants where the axis misses).
    valid = []  # (i, ti, lat, lon)
    for i, ti in enumerate(t):
        try:
            lon, lat = fund_to_geo(elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i])
        except ValueError:
            continue
        valid.append((i, float(ti), lat, lon))

    # Pass 2: per-point shadow limits + accurate width (bearing from neighbours).
    points = []
    for k, (i, ti, lat, lon) in enumerate(valid):
        prev = valid[max(0, k - 1)]
        nxt = valid[min(len(valid) - 1, k + 1)]
        brg = bearing(prev[2], prev[3], nxt[2], nxt[3]) if len(valid) > 1 else 0.0

        north, south, width = shadow_edge_limits(
            elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i],
            elems["l2"][i], elems["tan_f2"][i], brg,
        )
        pen_km, _umb_km, is_total = shadow_radii(
            elems["x"][i], elems["y"][i], elems["d"][i],
            elems["l1"][i], elems["l2"][i], elems["tan_f1"][i], elems["tan_f2"][i],
        )
        sign = "-" if ti < 0 else "+"
        h, m, s = dec_to_hms(abs(ti))
        pt = {
            "t_hours": round(ti, 4),
            "offset": f"{sign}{h:02d}:{m:02d}:{s:04.1f}",
            "lon": round(lon, 5),
            "lat": round(lat, 5),
            "penumbra_km": round(pen_km, 1),
            "is_total": is_total,
            "width_km": round(width, 2),
        }
        if north is not None:
            pt["north_limit"] = {"lat": round(north[0], 5), "lon": round(north[1], 5)}
            pt["south_limit"] = {"lat": round(south[0], 5), "lon": round(south[1], 5)}
        points.append(pt)

    # Sub-solar point at T0 drives the frontend's sunlight / day-night terminator.
    ss_lon, ss_lat = sub_solar_point(utc_to_et(epoch), frame)

    return {
        "t0_utc": epoch,
        "frame": frame,
        "count": len(points),
        "sun": {"lon": round(ss_lon, 4), "lat": round(ss_lat, 4)},
        "central_line": points,
    }


@app.get("/circumstances")
async def circumstances(
    epoch: str = Query(..., description="Reference epoch T0 (UTC), near maximum eclipse"),
    lat: float = Query(..., ge=-90.0, le=90.0, description="Observer latitude (deg)"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Observer longitude (deg, east +)"),
    window_hours: float = Query(2.5, ge=1.0, le=6.0, description="Half-window sampled around T0"),
    frame: str = Query(DEFAULT_EARTH_FRAME),
) -> dict:
    """Local eclipse circumstances at an observer: contacts, duration, magnitude.

    Returns C1-C4 contact clock times (UTC), maximum-eclipse time, magnitude,
    obscuration, Sun altitude/azimuth, and the central-phase duration when the
    observer is inside the umbra/antumbra.  ``epoch`` should be near the
    observer's maximum (e.g. the greatest-eclipse time).
    """
    model = _build_model(epoch, window_hours, frame)
    return local_circumstances(model, lat, lon)


# Serve the Three.js frontend at /ui (same origin as the API, so no CORS hop).
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="ui")

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
from .ephemeris import DEFAULT_EARTH_FRAME, sub_solar_point, utc_to_et
from .geography import dec_to_hms, fund_to_geo, shadow_radii

_FRAMES = ("ITRF93", "TOD", "IAU_EARTH")

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
    elems = model.evaluate(t)

    points = []
    for i, ti in enumerate(t):
        try:
            lon, lat = fund_to_geo(elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i])
        except ValueError:
            continue  # axis misses the Earth at this instant
        pen_km, umb_km, is_total = shadow_radii(
            elems["x"][i], elems["y"][i], elems["d"][i],
            elems["l1"][i], elems["l2"][i], elems["tan_f1"][i], elems["tan_f2"][i],
        )
        sign = "-" if ti < 0 else "+"
        h, m, s = dec_to_hms(abs(float(ti)))
        points.append({
            "t_hours": round(float(ti), 4),
            "offset": f"{sign}{h:02d}:{m:02d}:{s:04.1f}",
            "lon": round(lon, 5),
            "lat": round(lat, 5),
            "umbra_km": round(umb_km, 1),
            "penumbra_km": round(pen_km, 1),
            "is_total": is_total,
        })

    # Sub-solar point at T0 drives the frontend's sunlight / day-night terminator.
    ss_lon, ss_lat = sub_solar_point(utc_to_et(epoch), frame)

    return {
        "t0_utc": epoch,
        "frame": frame,
        "count": len(points),
        "sun": {"lon": round(ss_lon, 4), "lat": round(ss_lat, 4)},
        "central_line": points,
    }


# Serve the Three.js frontend at /ui (same origin as the API, so no CORS hop).
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="ui")

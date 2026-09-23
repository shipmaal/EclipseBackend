"""FastAPI surface for the eclipse backend.

Endpoints are parameterized (no hard-coded eclipse) and do only computation --
no plotting in the request path. Positions come from SPICE / a JPL DE ephemeris
via :mod:`app.besselian`.

Concurrency: the handlers are plain ``def`` functions, so FastAPI runs them in
its thread pool and a long computation does not stall ``/health`` or other
requests.  CSPICE is not thread-safe; every SPICE call in :mod:`app.ephemeris`
takes :data:`app.ephemeris.SPICE_LOCK`, so only the (fast) ephemeris lookups
are serialized and the NumPy geometry runs in parallel.  Do not switch these
back to ``async def`` (the whole event loop blocks) and do not bypass the lock.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import numpy as np
import spiceypy
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .besselian import BesselianModel, normalize_utc
from .catalog import find_eclipses
from .circumstances import circumstances_grid, local_circumstances
from .ephemeris import (
    DEFAULT_EARTH_FRAME,
    EARTH_FRAMES,
    load_kernels,
    sub_solar_point,
    utc_to_et,
)
from .geography import (
    central_track,
    format_clock,
    format_offset,
    global_contacts,
    shadow_edge_limits_v,
    shadow_radii,
)

# Request-size guards: the work per point is small since the geometry is
# vectorized, but the parameters are user-controlled and unbounded otherwise.
MAX_TRACK_POINTS = 5000
MAX_MAP_CELLS = 150_000
MAX_ABS_HOURS = 12.0
MAX_CATALOG_YEARS = {True: 10.0, False: 100.0}  # by `detail`


def _arange_len(start: float, stop: float, step: float) -> float:
    """``len(np.arange(start, stop, step))`` for ``step > 0``, without allocating.

    NumPy's length is ``ceil((stop - start) / step)``; returned as a float that
    is ``inf`` when the quotient overflows (a subnormal ``step``), so the size
    guards below compare it before any array exists (review item C3/C8).
    """
    n = (stop - start) / step
    return float(math.ceil(n)) if math.isfinite(n) else math.inf


class LimitPoint(TypedDict):
    """A northern/southern shadow-limit point (item M4)."""

    lat: float
    lon: float


class CentralLinePoint(TypedDict, total=False):
    """One central-line sample (item M4).

    ``north_limit``/``south_limit`` are present only when the shadow edge is found
    at that instant (central point inside the shadow).
    """

    t_hours: float
    offset: str          # signed +/-HH:MM:SS.s from T0
    lon: float
    lat: float
    penumbra_km: float
    is_total: bool
    width_km: float
    north_limit: LimitPoint            # umbral / antumbral limits
    south_limit: LimitPoint
    penumbra_north_limit: LimitPoint   # partial-eclipse region edge (clipped to the terminator)
    penumbra_south_limit: LimitPoint

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
    if frame not in EARTH_FRAMES:
        raise HTTPException(status_code=400, detail=f"frame must be one of {EARTH_FRAMES}")
    try:
        return BesselianModel(
            t0_utc=epoch, earth_frame=frame, half_window_hours=window_hours
        )
    except FileNotFoundError as exc:
        # Kernels not downloaded yet.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # normalize_utc: not an ISO-8601 epoch (the single accepted format).
        raise HTTPException(status_code=400, detail=f"invalid epoch: {exc}") from exc
    except spiceypy.utils.exceptions.SpiceyError as exc:
        # Missing body / kernel coverage (SPICE); narrowed from a blanket
        # `except Exception` so genuine bugs surface (M2).
        raise HTTPException(status_code=400, detail=f"SPICE error: {exc}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/besselian")
def besselian(
    epoch: str = Query(
        ..., description="Reference epoch T0, UTC ISO-8601, e.g. 2024-04-08T18:00:00"
    ),
    window_hours: float = Query(2.0, ge=0.5, le=6.0, description="Half-window sampled around T0"),
    frame: str = Query(
        DEFAULT_EARTH_FRAME,
        description="Earth-orientation frame: ITRS, TOD, ITRF93 or IAU_EARTH",
    ),
) -> dict:
    """Besselian element polynomials (coefficients in powers of t = hours from T0)."""
    model = _build_model(epoch, window_hours, frame)
    return {"t0_utc": model.t0_utc, "frame": frame, "polynomials": asdict(model.polynomials)}


@app.get("/central-line")
def central_line(
    epoch: str = Query(..., description="Reference epoch T0, UTC ISO-8601"),
    start_hours: float = Query(-2.0, ge=-MAX_ABS_HOURS, le=MAX_ABS_HOURS,
                               description="Track start, hours from T0"),
    end_hours: float = Query(2.0, ge=-MAX_ABS_HOURS, le=MAX_ABS_HOURS,
                             description="Track end, hours from T0"),
    step_minutes: float = Query(2.0, gt=0.0, description="Sample spacing in minutes"),
    window_hours: float = Query(2.0, ge=0.5, le=6.0),
    frame: str = Query(DEFAULT_EARTH_FRAME),
) -> dict:
    """Geographic track of the shadow axis (central line) over a time range."""
    if end_hours <= start_hours:
        raise HTTPException(status_code=400, detail="end_hours must exceed start_hours")
    n_points = _arange_len(start_hours, end_hours + 1e-9, step_minutes / 60.0)
    if n_points > MAX_TRACK_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"{n_points:.3g} samples requested; at most {MAX_TRACK_POINTS} allowed "
                   "(increase step_minutes or narrow the range)",
        )

    model = _build_model(epoch, window_hours, frame)
    t = np.arange(start_hours, end_hours + 1e-9, step_minutes / 60.0)
    # Central-line reduction with along-track bearing, from direct per-instant
    # evaluation (exact beyond the polynomial fit window; items A2/R3).
    elems, track = central_track(model, t)

    # Per-point umbral limits + accurate width (perpendicular to the along-track
    # bearing carried by each TrackPoint), all points bisected at once; then the
    # penumbral limits the same way (the partial-eclipse region, up to a quarter
    # circumference away and clipped to the terminator).
    idx = np.array([tp.i for tp in track], dtype=int)
    bearings = np.array([tp.bearing for tp in track])
    n_lat, n_lon, s_lat, s_lon, widths = shadow_edge_limits_v(
        elems["x"][idx], elems["y"][idx], elems["d"][idx], elems["mu"][idx],
        elems["l2"][idx], elems["tan_f2"][idx], bearings,
    )
    pn_lat, pn_lon, ps_lat, ps_lon, _pw = shadow_edge_limits_v(
        elems["x"][idx], elems["y"][idx], elems["d"][idx], elems["mu"][idx],
        elems["l1"][idx], elems["tan_f1"][idx], bearings, max_km=10_000.0,
    )

    points = []
    for k, tp in enumerate(track):
        i = tp.i
        pen_km, _umb_km, is_total = shadow_radii(
            elems["x"][i], elems["y"][i], elems["d"][i],
            elems["l1"][i], elems["l2"][i], elems["tan_f1"][i], elems["tan_f2"][i],
        )
        pt: CentralLinePoint = {
            "t_hours": round(tp.t_hours, 4),
            "offset": format_offset(tp.t_hours),
            "lon": round(tp.lon, 5),
            "lat": round(tp.lat, 5),
            "penumbra_km": round(pen_km, 1),
            "is_total": is_total,
            "width_km": round(float(widths[k]), 2),
        }
        if not np.isnan(n_lat[k]):
            pt["north_limit"] = {"lat": round(float(n_lat[k]), 5), "lon": round(float(n_lon[k]), 5)}
            pt["south_limit"] = {"lat": round(float(s_lat[k]), 5), "lon": round(float(s_lon[k]), 5)}
        if not np.isnan(pn_lat[k]):
            pt["penumbra_north_limit"] = {"lat": round(float(pn_lat[k]), 5),
                                          "lon": round(float(pn_lon[k]), 5)}
            pt["penumbra_south_limit"] = {"lat": round(float(ps_lat[k]), 5),
                                          "lon": round(float(ps_lon[k]), 5)}
        points.append(pt)

    # Sub-solar point at T0 drives the frontend's sunlight / day-night terminator.
    ss_lon, ss_lat = sub_solar_point(model.et0, frame)

    # Global contacts: when the penumbra (P1/P4) and umbra (U1-U4) touch the Earth.
    contacts = {name: format_clock(model.t0_utc, th) for name, th in global_contacts(model).items()}

    return {
        "t0_utc": model.t0_utc,
        "frame": frame,
        "count": len(points),
        "sun": {"lon": round(ss_lon, 4), "lat": round(ss_lat, 4)},
        "contacts": contacts,
        "central_line": points,
    }


@app.get("/circumstances")
def circumstances(
    epoch: str = Query(..., description="Reference epoch T0 (UTC), near maximum eclipse"),
    lat: float = Query(..., ge=-90.0, le=90.0, description="Observer latitude (deg)"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Observer longitude (deg, east +)"),
    window_hours: float = Query(2.5, ge=1.0, le=6.0, description="Half-window sampled around T0"),
    frame: str = Query(DEFAULT_EARTH_FRAME),
) -> dict:
    """Local eclipse circumstances at an observer: contacts, duration, magnitude.

    Returns C1-C4 contact clock times (UTC) with the Sun's altitude at each,
    maximum-eclipse time, magnitude, obscuration, Sun altitude/azimuth, the
    central-phase duration when the observer is inside the umbra/antumbra, and
    ``below_horizon`` listing the events the observer cannot see.  ``epoch``
    should be near the observer's maximum (e.g. the greatest-eclipse time).
    """
    model = _build_model(epoch, window_hours, frame)
    return local_circumstances(model, lat, lon)


@app.get("/map")
def eclipse_map(
    epoch: str = Query(..., description="Reference epoch T0 (UTC), near greatest eclipse"),
    lat_step: float = Query(2.0, gt=0.0, le=30.0, description="Grid spacing in latitude (deg)"),
    lon_step: float = Query(2.0, gt=0.0, le=30.0, description="Grid spacing in longitude (deg)"),
    window_hours: float = Query(3.0, ge=1.0, le=6.0, description="Half-window sampled around T0"),
    step_minutes: float = Query(2.0, ge=0.5, le=30.0, description="Time sampling of each cell"),
    frame: str = Query(DEFAULT_EARTH_FRAME),
) -> dict:
    """Global maximum-eclipse map: magnitude, obscuration and visibility on a lat/lon grid.

    Cells are the grid nodes ``lats x lons`` (row-major, ``values[i][j]`` at
    ``lats[i], lons[j]``).  ``magnitude`` is 0 where no eclipse occurs;
    ``visible`` is false where the Sun is below the horizon at the cell's
    maximum.  Computed by :func:`app.circumstances.circumstances_grid`.
    """
    # Size the grid before allocating it (review item C3).
    n_cells = _arange_len(-90.0, 90.0 + 1e-9, lat_step) * _arange_len(-180.0, 180.0, lon_step)
    if n_cells > MAX_MAP_CELLS:
        raise HTTPException(
            status_code=400,
            detail=f"{n_cells:.3g} cells requested; at most {MAX_MAP_CELLS} allowed",
        )
    lats = np.arange(-90.0, 90.0 + 1e-9, lat_step)
    lons = np.arange(-180.0, 180.0, lon_step)
    model = _build_model(epoch, window_hours, frame)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    g = circumstances_grid(model, la.ravel(), lo.ravel(), step_minutes=step_minutes)
    shape = la.shape
    return {
        "t0_utc": model.t0_utc,
        "frame": frame,
        "lats": np.round(lats, 4).tolist(),
        "lons": np.round(lons, 4).tolist(),
        "magnitude": np.round(g["magnitude"].reshape(shape), 3).tolist(),
        "obscuration": np.round(g["obscuration"].reshape(shape), 3).tolist(),
        "visible": g["visible"].reshape(shape).tolist(),
        "central": g["central"].reshape(shape).tolist(),
    }


@app.get("/eclipses")
def eclipses(
    start: str = Query(..., description="Range start, UTC ISO-8601"),
    end: str = Query(..., description="Range end, UTC ISO-8601"),
    detail: bool = Query(True, description="Add greatest-eclipse duration, width and contacts"),
    frame: str = Query(DEFAULT_EARTH_FRAME),
) -> dict:
    """Catalog of solar eclipses with greatest eclipse in ``[start, end]``.

    Each row: greatest-eclipse instant, type (partial/annular/total/hybrid),
    gamma, magnitude, and for central eclipses the greatest-eclipse point;
    with ``detail`` also the central duration, path width and global contacts
    P1-U4 there.  Range limit: 10 years with detail, 100 without.
    """
    if frame not in EARTH_FRAMES:
        raise HTTPException(status_code=400, detail=f"frame must be one of {EARTH_FRAMES}")
    try:
        # The range guard converts the epochs before find_eclipses would load
        # the kernels, so load them here (both pools) first (review item C1).
        load_kernels()
        et_start = utc_to_et(normalize_utc(start))
        et_end = utc_to_et(normalize_utc(end))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # normalize_utc: not an ISO-8601 epoch.  Only the parsing is caught as
        # ValueError: one from the computation below is a bug and must be a
        # 500, not "invalid epoch" (review item C7).
        raise HTTPException(status_code=400, detail=f"invalid epoch: {exc}") from exc
    except spiceypy.utils.exceptions.SpiceyError as exc:
        raise HTTPException(status_code=400, detail=f"SPICE error: {exc}") from exc
    span_years = (et_end - et_start) / 3.15576e7
    if span_years <= 0:
        raise HTTPException(status_code=400, detail="end must be after start")
    if span_years > MAX_CATALOG_YEARS[detail]:
        raise HTTPException(
            status_code=400,
            detail=f"range of {span_years:.1f} years exceeds {MAX_CATALOG_YEARS[detail]:.0f} "
                   f"(detail={detail})",
        )
    try:
        events = find_eclipses(start, end, earth_frame=frame, detail=detail)
    except spiceypy.utils.exceptions.SpiceyError as exc:
        # E.g. a range outside the SPK's coverage.
        raise HTTPException(status_code=400, detail=f"SPICE error: {exc}") from exc
    return {"start": start, "end": end, "frame": frame, "count": len(events), "eclipses": events}


# Serve the Three.js frontend at /ui (same origin as the API, so no CORS hop).
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="ui")

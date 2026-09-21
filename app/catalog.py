"""Eclipse catalog: find every solar eclipse in a date range and characterize it.

The search is the vectorized Besselian-element evaluation over a coarse time
grid: the shadow axis' distance from the Earth's centre in the fundamental
plane, ``rho = sqrt(x^2 + y^2)``, has a local minimum at every syzygy, and a
solar eclipse occurs when that minimum is less than ``1 + l1`` (the penumbra
reaches the Earth) [ES92] sec. 8.34 **and** the Moon is on the Sun's side of
the Earth (``z > 0``; at full moon the Moon lies on the same line beyond the
Earth -- lunar-eclipse geometry, which the same minimum would otherwise catch).  Each minimum is
refined by parabolic iteration to the instant of **greatest eclipse** -- the
axis' closest approach to the Earth's centre, the [Espenak] definition -- and
classified:

* ``gamma``: signed ``rho`` at greatest eclipse, positive when the axis passes
  north of the centre [Espenak];
* ``central``: the axis meets the ellipsoid (:func:`~app.geography.fund_to_geo_v`);
* ``type``: ``partial`` if the umbra never touches (``rho > 1 + |l2|``),
  otherwise ``total`` / ``annular`` by the sign of the umbral radius at the
  ground, ``L2' = l2 - zeta tan f2``, and ``hybrid`` when that sign changes
  along the central line [Espenak] (Canon type codes P/T/A/H).  At the ends of
  the central line the ground point is on the limb, ``zeta = 0`` and
  ``L2' = l2``, so a path that is total at greatest eclipse is hybrid exactly
  when ``l2 > 0`` at either end -- no sampling of the path is needed;
* ``magnitude`` at greatest eclipse: the Moon/Sun diameter ratio for a central
  eclipse, the covered diameter fraction at the closest limb point otherwise
  [Espenak].

With ``detail=True`` each eclipse also gets the greatest-eclipse local
circumstances (central duration), the path width there and the global
contacts P1-U4 from the same machinery as the ``/circumstances`` and
``/central-line`` endpoints.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from .besselian import BesselianModel, normalize_utc
from .ephemeris import DEFAULT_EARTH_FRAME, besselian_instants, et_to_utc, load_kernels, utc_to_et
from .geography import (
    central_track,
    format_clock,
    fund_to_geo_v,
    geo_to_fund,
    global_contacts,
    shadow_edge_limits,
)
from .numerics import parabolic_minimum

_COARSE_STEP_S = 2.0 * 3600.0   # rho changes by < 0.6 Earth radii per hour
_COARSE_CHUNK = 50_000          # instants per besselian_instants call
_CANDIDATE_RHO = 2.6            # coarse minimum <= true minimum + 1.2 -> keep if < 1 + l1 + margin


class EclipseEvent(TypedDict, total=False):
    """One catalogued solar eclipse (Canon-style row)."""

    greatest_utc: str        # instant of greatest eclipse, UTC (UT1 outside the IERS era)
    type: str                # partial | annular | total | hybrid
    central: bool            # the axis meets the Earth
    gamma: float             # signed axis distance from Earth's centre [Earth radii]
    magnitude: float
    lat: float               # greatest-eclipse point (central eclipses only)
    lon: float
    sun_alt: float
    central_duration_s: float  # detail only, central only
    width_km: float            # detail only, central only
    contacts: dict[str, str]   # detail only: P1, U1, U2, U3, U4, P4 as UTC HH:MM:SS


def _rho_at(et: np.ndarray, frame: str) -> np.ndarray:
    """Axis distance from the Earth's centre; +inf on the far side (full moon)."""
    e = besselian_instants(et, frame)
    return np.where(e["z"] > 0.0, np.hypot(e["x"], e["y"]), np.inf)


def _hybrid(et_greatest: float, L2p_greatest: float, frame: str) -> bool:
    """True when a path total at greatest eclipse is annular at either end.

    At the ends of the central line ``zeta = 0`` so the umbral ground radius is
    ``l2`` itself; ``l2 > 0`` there with ``L2' < 0`` at greatest eclipse means
    the umbra's tip lifts off the surface before the path ends (module
    docstring).  ``l2`` varies by ~1e-4 per hour, so it is read at the first and
    last on-Earth instants of a 5-minute sampling.
    """
    if L2p_greatest >= 0.0:
        return False  # annular at closest approach: annular throughout
    et = et_greatest + np.arange(-3.5, 3.5 + 1e-9, 5.0 / 60.0) * 3600.0
    e = besselian_instants(et, frame)
    _lon, lat = fund_to_geo_v(e["x"], e["y"], e["d"], e["mu"])
    on = np.flatnonzero(~np.isnan(lat))
    if len(on) == 0:
        return False
    return bool(max(e["l2"][on[0]], e["l2"][on[-1]]) > 0.0)


def find_eclipses(
    start_utc: str,
    end_utc: str,
    earth_frame: str = DEFAULT_EARTH_FRAME,
    detail: bool = True,
) -> list[EclipseEvent]:
    """Every solar eclipse with greatest eclipse in ``[start_utc, end_utc]``.

    Epoch strings follow :func:`~app.besselian.normalize_utc`.  Coverage is
    that of the loaded SPK (DE432s mirror: 1949-2050; DE440s: 1550-2650).
    """
    load_kernels()
    et_a = utc_to_et(normalize_utc(start_utc))
    et_b = utc_to_et(normalize_utc(end_utc))
    if et_b <= et_a:
        raise ValueError("end must be after start")

    # Coarse scan (chunked) for local minima of rho.
    et = np.arange(et_a - _COARSE_STEP_S, et_b + 2 * _COARSE_STEP_S, _COARSE_STEP_S)
    rho = np.concatenate([_rho_at(et[i:i + _COARSE_CHUNK], earth_frame)
                          for i in range(0, len(et), _COARSE_CHUNK)])
    is_min = (rho[1:-1] < rho[:-2]) & (rho[1:-1] <= rho[2:]) & (rho[1:-1] < _CANDIDATE_RHO)
    cand = et[1:-1][is_min]
    if len(cand) == 0:
        return []

    # Refine every candidate at once to the instant of greatest eclipse.
    et_g = parabolic_minimum(lambda tt: _rho_at(tt, earth_frame), cand, _COARSE_STEP_S)
    e = besselian_instants(et_g, earth_frame)
    rho_g = np.hypot(e["x"], e["y"])
    lon_g, lat_g = fund_to_geo_v(e["x"], e["y"], e["d"], e["mu"])

    events: list[EclipseEvent] = []
    for k in range(len(et_g)):
        if not (et_a <= et_g[k] <= et_b) or rho_g[k] > 1.0 + e["l1"][k]:
            continue
        central = not np.isnan(lat_g[k])
        l1, l2, tf2 = float(e["l1"][k]), float(e["l2"][k]), float(e["tan_f2"][k])
        if central:
            _xi, _eta, zeta = geo_to_fund(lat_g[k], lon_g[k], e["d"][k], e["mu"][k])
            L1p = l1 - zeta * float(e["tan_f1"][k])
            L2p = l2 - zeta * tf2
            kind = "total" if L2p < 0 else "annular"
            if _hybrid(float(et_g[k]), L2p, earth_frame):
                kind = "hybrid"
            magnitude = (L1p - L2p) / (L1p + L2p)
        elif rho_g[k] > 1.0 + abs(l2):
            kind = "partial"
            magnitude = (l1 - (rho_g[k] - 1.0)) / (l1 + l2)  # closest limb point, zeta = 0
        else:
            kind = "total" if l2 < 0 else "annular"  # non-central: umbra grazes the limb
            magnitude = (l1 - l2) / (l1 + l2)

        ev: EclipseEvent = {
            "greatest_utc": et_to_utc(float(et_g[k])),
            "type": kind,
            "central": central,
            "gamma": round(float(np.copysign(rho_g[k], e["y"][k])), 4),
            "magnitude": round(float(magnitude), 4),
        }
        if central:
            ev["lat"] = round(float(lat_g[k]), 2)
            ev["lon"] = round(float(lon_g[k]), 2)
        if detail:
            _add_detail(ev, earth_frame)
        events.append(ev)
    return events


def _add_detail(ev: EclipseEvent, frame: str) -> None:
    """Greatest-eclipse circumstances, path width and global contacts (in place)."""
    from .circumstances import local_circumstances

    model = BesselianModel(t0_utc=ev["greatest_utc"], earth_frame=frame, half_window_hours=2.5)
    ev["contacts"] = {name: format_clock(model.t0_utc, th)
                      for name, th in global_contacts(model).items()}
    if not ev["central"]:
        return
    c = local_circumstances(model, ev["lat"], ev["lon"])
    ev["sun_alt"] = c.get("sun_alt", 0.0)
    if "central_duration_s" in c:
        ev["central_duration_s"] = c["central_duration_s"]
    dt = 0.02
    elems, track = central_track(model, np.array([-dt, 0.0, dt]))
    mid = [tp for tp in track if abs(tp.t_hours) < 1e-9]
    if mid:
        tp = mid[0]
        i = tp.i
        _n, _s, width = shadow_edge_limits(
            elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i],
            elems["l2"][i], elems["tan_f2"][i], tp.bearing,
        )
        ev["width_km"] = round(width, 1)

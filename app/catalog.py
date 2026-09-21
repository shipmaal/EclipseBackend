"""Eclipse catalog: find every solar eclipse in a date range and characterize it.

The search is the vectorized evaluation, over a coarse time grid, of the
shadow axis' distance from the Earth's centre in the fundamental plane,
``rho = sqrt(x^2 + y^2)`` [ES92] eq. 8.322-6: it has a local minimum at every
syzygy, and a solar eclipse occurs when that minimum is less than ``1 + l1``
(the penumbra reaches the Earth) [ES92] sec. 8.34 **and** the Moon is on the
Sun's side of the Earth (``z > 0``; at full moon the Moon lies on the same line
beyond the Earth -- lunar-eclipse geometry, which the same minimum would
otherwise catch).  ``rho`` and ``z`` are invariant under any rotation of the
Earth-orientation frame, so the scan and the refinement read them from
:func:`~app.ephemeris.axis_separation` (the same eq. 8.322-6 on the un-rotated
J2000 vectors: no nutation, no EOP); only the elements *at* greatest eclipse and
everything after them use the configured ``earth_frame``.  Each minimum is
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

Layout (the phase-3 pattern of :mod:`app.circumstances`): the numbers are
:func:`_catalog_raw` (a list of :class:`_EventRaw`), the dict shape is
:func:`_format_event`, and :func:`find_eclipses` composes them.  The native
core's ``catalog.hpp`` mirrors the raw layer function for function.
"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import numpy as np

from .besselian import BesselianModel, normalize_utc
from .ephemeris import (
    DEFAULT_EARTH_FRAME,
    axis_separation,
    besselian_instants,
    et_to_utc,
    load_kernels,
    utc_to_et,
)
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
_COARSE_CHUNK = 50_000          # instants per axis_separation call
_CANDIDATE_RHO = 2.6            # coarse minimum <= true minimum + 1.2 -> keep if < 1 + l1 + margin
_HYBRID_HALF_SPAN_H = 3.5       # _hybrid samples +/- this around greatest eclipse [h]
_HYBRID_STEP_H = 5.0 / 60.0     # ... every 5 minutes
_DETAIL_HALF_WINDOW_H = 2.5     # BesselianModel fit half-window of the detail model [h]
_TRACK_DT_H = 0.02              # central_track offsets +/- this give the width's bearing [h]


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


class _EventRaw(NamedTuple):
    """One eclipse before any rounding / formatting (:func:`_format_event` does that).

    ``lat``/``lon`` are the UNROUNDED greatest-eclipse point [deg] (NaN when not
    central); ``gamma`` the signed ``rho``; ``et0`` the detail model's epoch
    [TDB s] (``utc_to_et(et_to_utc(et_g))``: the whole-second greatest-eclipse
    string re-parsed, exactly as the model is built), ``contacts`` the global
    contacts as ``(name, t_hours)`` pairs in :func:`~app.geography.global_contacts`'
    insertion order, ``local`` the :class:`~app.circumstances._LocalRaw` at the
    greatest point rounded to 0.01 deg (the point the row reports) and
    ``width_km`` the umbral path width there -- ``None`` each where absent
    (no detail; not central; no on-Earth track point).  The native core's
    ``catalog::Event`` / ``catalog::Detail`` hold exactly these fields.
    """

    et_g: float
    kind: str
    central: bool
    gamma: float
    magnitude: float
    lat: float
    lon: float
    et0: float | None
    contacts: list[tuple[str, float]] | None
    local: object | None      # circumstances._LocalRaw
    width_km: float | None


def _rho_at(et: np.ndarray) -> np.ndarray:
    """Axis distance from the Earth's centre [Earth radii]; +inf on the far side (full moon).

    Frame-free (:func:`~app.ephemeris.axis_separation`): the scan objective and
    the refinement objective, both of which read only ``rho`` and ``z``.

    Conditioning note (phase 4).  This objective replaced ``hypot(x, y)`` of the
    ITRS elements; the two agree to 4e-14 (rotation invariance, rounding only),
    but ``rho`` is quadratic and flat at its minimum, so a change ``delta`` in
    the objective moves the refined instant by ``~sqrt(2 delta / rho'')`` with
    ``rho'' ~ 8e-8`` Earth radii/s^2: ~1 ms.  Measured over the 114 eclipses of
    2000-2050: |d et_g| <= 3.4 ms (median 0.45 ms).  Every 2019-2024 canon row
    (detail on and off) is unchanged; over 2000-2100 exactly one whole-second
    ``greatest_utc`` flipped (2000-07-31 02:13:03.4997 -> 03.5005, a 0.75 ms
    shift across the rounding boundary).  The greatest-eclipse instant is thus
    determined only to the millisecond from double-precision ``rho`` by any
    implementation, and a parity gate on the raw ``et_g`` must allow ~1e-2 s
    (rows equal up to such a boundary flip), not 1e-6 s.
    """
    rho, z = axis_separation(et)
    return np.where(z > 0.0, rho, np.inf)


def _coarse_grid(et_a: float, et_b: float) -> np.ndarray:
    """The coarse scan instants: ``arange(et_a - step, et_b + 2 step, step)`` [TDB s]."""
    return np.arange(et_a - _COARSE_STEP_S, et_b + 2 * _COARSE_STEP_S, _COARSE_STEP_S)


def _local_minima(rho: np.ndarray) -> np.ndarray:
    """Indices ``i`` (``1 <= i <= n - 2``) of the coarse local minima of ``rho``.

    Strictly below the previous sample, at or below the next one (a plateau's
    first sample wins), and below ``_CANDIDATE_RHO``; ``inf`` (far-side)
    samples never qualify.  Grid order, never sorted.
    """
    is_min = (rho[1:-1] < rho[:-2]) & (rho[1:-1] <= rho[2:]) & (rho[1:-1] < _CANDIDATE_RHO)
    return np.flatnonzero(is_min) + 1


def _scan_candidates(et_a: float, et_b: float) -> np.ndarray:
    """Coarse-grid instants [TDB s] of every candidate minimum (chunked evaluation)."""
    et = _coarse_grid(et_a, et_b)
    rho = np.concatenate([_rho_at(et[i:i + _COARSE_CHUNK])
                          for i in range(0, len(et), _COARSE_CHUNK)])
    return et[_local_minima(rho)]


def _classify(rho_g: float, l1: float, l2: float, tan_f1: float, tan_f2: float,
              central: bool, zeta: float) -> tuple[str, float, float]:
    """``(kind, magnitude, L2')`` at greatest eclipse, BEFORE the hybrid test.

    Central: ``L1' = l1 - zeta tan f1``, ``L2' = l2 - zeta tan f2`` at the
    greatest-eclipse ground point, ``total`` when ``L2' < 0`` else ``annular``,
    magnitude ``(L1' - L2') / (L1' + L2')`` [Espenak].  Non-central: ``partial``
    when ``rho > 1 + |l2|`` with magnitude ``(l1 - (rho - 1)) / (l1 + l2)`` at
    the closest limb point (``zeta = 0``); otherwise the umbra grazes the limb
    and the kind follows the sign of ``l2`` with magnitude
    ``(l1 - l2) / (l1 + l2)``.  ``L2'`` is NaN when not central (``zeta`` is
    then unused).
    """
    if central:
        L1p = l1 - zeta * tan_f1
        L2p = l2 - zeta * tan_f2
        kind = "total" if L2p < 0 else "annular"
        magnitude = (L1p - L2p) / (L1p + L2p)
        return kind, magnitude, L2p
    if rho_g > 1.0 + abs(l2):
        return "partial", (l1 - (rho_g - 1.0)) / (l1 + l2), float("nan")
    kind = "total" if l2 < 0 else "annular"  # non-central: umbra grazes the limb
    return kind, (l1 - l2) / (l1 + l2), float("nan")


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
    et = et_greatest + np.arange(-_HYBRID_HALF_SPAN_H, _HYBRID_HALF_SPAN_H + 1e-9,
                                 _HYBRID_STEP_H) * 3600.0
    e = besselian_instants(et, frame)
    _lon, lat = fund_to_geo_v(e["x"], e["y"], e["d"], e["mu"])
    on = np.flatnonzero(~np.isnan(lat))
    if len(on) == 0:
        return False
    return bool(max(e["l2"][on[0]], e["l2"][on[-1]]) > 0.0)


def _catalog_raw(et_a: float, et_b: float, frame: str, detail: bool) -> list[_EventRaw]:
    """Every eclipse with greatest eclipse in ``[et_a, et_b]`` [TDB s], as raw numbers.

    Coarse scan (:func:`_scan_candidates`), parabolic refinement of every
    candidate at once (:func:`~app.numerics.parabolic_minimum` on
    :func:`_rho_at`), the elements at greatest eclipse in ``frame``,
    :func:`_classify` / :func:`_hybrid`, and :func:`_detail_raw` when asked.
    Candidates outside the range or with ``rho > 1 + l1`` are dropped.
    """
    cand = _scan_candidates(et_a, et_b)
    if len(cand) == 0:
        return []

    # Refine every candidate at once to the instant of greatest eclipse.
    et_g = parabolic_minimum(_rho_at, cand, _COARSE_STEP_S)
    e = besselian_instants(et_g, frame)
    rho_g = np.hypot(e["x"], e["y"])
    lon_g, lat_g = fund_to_geo_v(e["x"], e["y"], e["d"], e["mu"])

    events: list[_EventRaw] = []
    for k in range(len(et_g)):
        if not (et_a <= et_g[k] <= et_b) or rho_g[k] > 1.0 + e["l1"][k]:
            continue
        central = not np.isnan(lat_g[k])
        l1, l2, tf2 = float(e["l1"][k]), float(e["l2"][k]), float(e["tan_f2"][k])
        zeta = float("nan")
        if central:
            _xi, _eta, zeta = geo_to_fund(lat_g[k], lon_g[k], e["d"][k], e["mu"][k])
        kind, magnitude, L2p = _classify(rho_g[k], l1, l2, float(e["tan_f1"][k]), tf2,
                                         central, zeta)
        if central and _hybrid(float(et_g[k]), L2p, frame):
            kind = "hybrid"
        raw = _EventRaw(
            et_g=float(et_g[k]),
            kind=kind,
            central=central,
            gamma=float(np.copysign(rho_g[k], e["y"][k])),
            magnitude=float(magnitude),
            lat=float(lat_g[k]),
            lon=float(lon_g[k]),
            et0=None, contacts=None, local=None, width_km=None,
        )
        if detail:
            raw = _detail_raw(raw, frame)
        events.append(raw)
    return events


def _detail_raw(raw: _EventRaw, frame: str) -> _EventRaw:
    """Greatest-eclipse circumstances, path width and global contacts (numbers only).

    The model is built from the whole-second greatest-eclipse string, exactly
    as the row reports it; the local circumstances are taken at the point
    rounded to 0.01 deg (the point the row reports); the width is the umbral
    :func:`~app.geography.shadow_edge_limits` at the ``t = 0`` point of a
    three-point :func:`~app.geography.central_track` (its along-track bearing).
    """
    from .circumstances import local_raw

    model = BesselianModel(t0_utc=et_to_utc(raw.et_g), earth_frame=frame,
                           half_window_hours=_DETAIL_HALF_WINDOW_H)
    contacts = list(global_contacts(model).items())
    if not raw.central:
        return raw._replace(et0=model.et0, contacts=contacts)
    local = local_raw(model, round(raw.lat, 2), round(raw.lon, 2))
    width = None
    elems, track = central_track(model, np.array([-_TRACK_DT_H, 0.0, _TRACK_DT_H]))
    mid = [tp for tp in track if abs(tp.t_hours) < 1e-9]
    if mid:
        tp = mid[0]
        i = tp.i
        _n, _s, width = shadow_edge_limits(
            elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i],
            elems["l2"][i], elems["tan_f2"][i], tp.bearing,
        )
    return raw._replace(et0=model.et0, contacts=contacts, local=local, width_km=width)


def _format_event(raw: _EventRaw, detail: bool) -> EclipseEvent:
    """Assemble the :class:`EclipseEvent` row from an :class:`_EventRaw`.

    All rounding, clock formatting and the varying dict shape live here (the
    numbers come from :func:`_catalog_raw`, or from the native core).
    """
    from .circumstances import _format_local

    greatest_utc = et_to_utc(raw.et_g)
    ev: EclipseEvent = {
        "greatest_utc": greatest_utc,
        "type": raw.kind,
        "central": raw.central,
        "gamma": round(raw.gamma, 4),
        "magnitude": round(raw.magnitude, 4),
    }
    if raw.central:
        ev["lat"] = round(raw.lat, 2)
        ev["lon"] = round(raw.lon, 2)
    if not detail:
        return ev
    assert raw.contacts is not None
    ev["contacts"] = {name: format_clock(greatest_utc, th) for name, th in raw.contacts}
    if not raw.central:
        return ev
    assert raw.local is not None
    c = _format_local(greatest_utc, ev["lat"], ev["lon"], raw.local)
    ev["sun_alt"] = c.get("sun_alt", 0.0)
    if "central_duration_s" in c:
        ev["central_duration_s"] = c["central_duration_s"]
    if raw.width_km is not None:
        ev["width_km"] = round(raw.width_km, 1)
    return ev


def find_eclipses(
    start_utc: str,
    end_utc: str,
    earth_frame: str = DEFAULT_EARTH_FRAME,
    detail: bool = True,
) -> list[EclipseEvent]:
    """Every solar eclipse with greatest eclipse in ``[start_utc, end_utc]``.

    Epoch strings follow :func:`~app.besselian.normalize_utc`.  Coverage is
    that of the loaded SPK (DE432s mirror: 1949-2050; DE440s: 1550-2650).
    The numbers are :func:`_catalog_raw`; the row shape is :func:`_format_event`.
    """
    load_kernels()
    et_a = utc_to_et(normalize_utc(start_utc))
    et_b = utc_to_et(normalize_utc(end_utc))
    if et_b <= et_a:
        raise ValueError("end must be after start")
    return [_format_event(raw, detail) for raw in _catalog_raw(et_a, et_b, earth_frame, detail)]

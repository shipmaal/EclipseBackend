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
the core's ``axis_separation`` (the same eq. 8.322-6 on the un-rotated J2000
vectors: no nutation, no EOP); only the elements *at* greatest eclipse and
everything after them use the configured ``earth_frame``.  Each minimum is
refined by parabolic iteration to the instant of **greatest eclipse** -- the
axis' closest approach to the Earth's centre, the [Espenak] definition -- and
classified:

* ``gamma``: signed ``rho`` at greatest eclipse, positive when the axis passes
  north of the centre [Espenak];
* ``central``: the axis meets the ellipsoid (``ellipsoid::fund_to_geo``);
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

The core computes all of it (``core/src/catalog.cpp``; the per-event work is
OpenMP-parallel); this module formats its :class:`_EventRaw` rows.
"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import _eclipse

from .besselian import normalize_utc
from .circumstances import _format_local, _LocalRaw
from .core import DEFAULT_EARTH_FRAME, PARALLEL_LOCK, load_kernels
from .formatting import format_clock


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
    contacts as ``(name, t_hours)`` pairs in the core's order (P1, P4, U1, U4,
    U2, U3), ``local`` the :class:`~app.circumstances._LocalRaw` at the greatest
    point rounded to 0.01 deg (the point the row reports) and
    ``width_km`` the umbral path width there -- ``None`` each where absent
    (no detail; not central; no on-Earth track point).  The core's
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


def _format_event(raw: _EventRaw, detail: bool) -> EclipseEvent:
    """Assemble the :class:`EclipseEvent` row from an :class:`_EventRaw`.

    All rounding, clock formatting and the varying dict shape live here; the
    numbers come from the core.
    """
    greatest_utc = _eclipse.et_to_utc(raw.et_g)
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
    that of the loaded SPK (DE432s mirror: 1949-2050; DE440s: 1849-2150).
    """
    load_kernels()
    et_a = _eclipse.utc_to_et(normalize_utc(start_utc))
    et_b = _eclipse.utc_to_et(normalize_utc(end_utc))
    if et_b <= et_a:
        raise ValueError("end must be after start")
    with PARALLEL_LOCK:  # one OpenMP team per process (app.core.PARALLEL_LOCK)
        events = _eclipse.find_eclipses(et_a, et_b, earth_frame, detail)
    return [_format_event(_event_raw(t), detail) for t in events]


def _event_raw(t: tuple) -> _EventRaw:
    """An :class:`_EventRaw` from the core's tuple (``local`` as a :class:`_LocalRaw`)."""
    raw = _EventRaw(*t)
    return raw if raw.local is None else raw._replace(local=_LocalRaw(*raw.local))

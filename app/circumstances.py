"""Local circumstances of an eclipse for one observer, or a grid of them.

The core computes them (``core/src/circumstances.cpp``; method: [ES92] sec.
8.35, eq. 8.353-8.354, [MeeusSE]): for an observer at geodetic ``lat``/``lon``
and ``height_m`` above the WGS-84 ellipsoid, the partial contacts C1/C4, the
central contacts C2/C3 and their duration, the time and magnitude of maximum
eclipse [Espenak], the obscuration and the Sun's altitude/azimuth, with every
event checked against the rise/set altitude ``h0 = -0 deg 50'`` [Meeus98]
ch. 15.  ``limb="mean"`` (the default) is the mean lunar limb folded into k2
[Espenak]; ``limb="profile"`` takes C2/C3 (and whether there is a central
phase at all) from the LRO LOLA limb profile (docs/LIMB_PROFILE.md).

This module formats the core's numbers (:class:`_LocalRaw`) into the API's
dicts.
"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import _eclipse
import numpy as np

from .core import PARALLEL_LOCK, ensure_limb_band
from .formatting import format_clock

# Lunar limb models (docs/LIMB_PROFILE.md): the mean limb in k2, or the profile.
LIMB_MODES = ("mean", "profile")


class LocalCircumstances(TypedDict, total=False):
    """Local eclipse circumstances for one observer (item M4).

    ``total=False`` because the shape varies: an observer who sees no eclipse gets
    only ``lat``/``lon``/``eclipse`` (plus ``below_horizon`` when the geometry
    would give one but the Sun is down throughout); the central-phase keys
    ``C2``/``C3``/``central_duration_s`` are present only when the observer
    enters the umbra/antumbra.  Times are UTC ``HH:MM:SS`` strings; ``*_alt``
    are Sun altitudes [deg] at those instants.
    """

    lat: float
    lon: float
    eclipse: bool
    type: str            # "partial", "total" or "annular"
    magnitude: float
    obscuration: float   # covered fraction of the Sun's area
    max_time: str
    sun_alt: float       # at maximum
    sun_az: float        # at maximum
    C1: str              # first partial contact
    C4: str              # last partial contact
    C1_alt: float
    C4_alt: float
    C2: str              # second contact (central phase only)
    C3: str              # third contact (central phase only)
    C2_alt: float
    C3_alt: float
    central_duration_s: float  # totality/annularity duration (central phase only)
    below_horizon: list[str]   # events ("C1", "C2", "max", "C3", "C4") with Sun below h0
    limb: str                  # lunar limb model: "mean" (k2) or "profile" (LOLA)
    elev_m: float              # observer height above the WGS-84 ellipsoid [m]


class _LocalRaw(NamedTuple):
    """The raw numbers of :func:`local_circumstances` before any rounding / formatting.

    ``geometric`` is False when fewer than two partial roots were found or the
    peak magnitude is ``<= 0``; every other field is then a placeholder (NaN /
    False).  Otherwise ``central`` says the observer has a central phase (C2 and C3; NaN when
    not).  Times are hours from T0.  ``alt_deg``/``az_deg``/``below`` hold five
    entries in the event order C1, max, C4, C2, C3 (NaN / False where absent);
    ``eclipse`` is False when every present event is below the horizon.  The core's
    ``LocalRaw`` (``core/include/eclipse/circumstances.hpp``) has exactly these
    fields in this order.
    """

    geometric: bool
    central: bool
    c1: float
    c4: float
    c2: float
    c3: float
    t_max: float
    magnitude: float
    obscuration: float
    L2_x: float
    alt_deg: tuple[float, float, float, float, float]
    az_deg: tuple[float, float, float, float, float]
    below: tuple[bool, bool, bool, bool, bool]
    eclipse: bool



def _format_local(t0_utc: str, lat: float, lon: float, raw: _LocalRaw) -> LocalCircumstances:
    """Assemble the :class:`LocalCircumstances` dict from a :class:`_LocalRaw`.

    All rounding, clock formatting and the varying dict shape live here; the
    numbers come from the core (:func:`local_raw`).
    """
    if not raw.geometric:
        no_eclipse: LocalCircumstances = {"lat": lat, "lon": lon, "eclipse": False}
        return no_eclipse

    events = ["C1", "max", "C4"] + (["C2", "C3"] if raw.central else [])
    below = [name for name, b in zip(events, raw.below, strict=False) if b]
    if not raw.eclipse:
        return {"lat": lat, "lon": lon, "eclipse": False, "below_horizon": below}

    alts, azs = raw.alt_deg, raw.az_deg
    result: LocalCircumstances = {
        "lat": lat,
        "lon": lon,
        "eclipse": True,
        "type": "partial",  # upgraded to total/annular below if the observer enters the umbra
        "magnitude": round(float(raw.magnitude), 4),
        "obscuration": round(float(raw.obscuration), 4),
        "max_time": format_clock(t0_utc, raw.t_max),
        "sun_alt": round(float(alts[1]), 1),
        "sun_az": round(float(azs[1]), 1),
        "C1": format_clock(t0_utc, raw.c1),
        "C4": format_clock(t0_utc, raw.c4),
        "C1_alt": round(float(alts[0]), 1),
        "C4_alt": round(float(alts[2]), 1),
        "below_horizon": below,
    }

    if raw.central:
        c2, c3 = raw.c2, raw.c3
        result["type"] = "total" if raw.L2_x < 0 else "annular"
        result["C2"] = format_clock(t0_utc, c2)
        result["C3"] = format_clock(t0_utc, c3)
        result["C2_alt"] = round(float(alts[3]), 1)
        result["C3_alt"] = round(float(alts[4]), 1)
        result["central_duration_s"] = round((c3 - c2) * 3600.0, 1)

    return result


def _check_limb(limb: str) -> None:
    if limb not in LIMB_MODES:
        raise ValueError(f"limb must be one of {LIMB_MODES}, not {limb!r}")


def local_raw(model, lat: float, lon: float, limb: str = "mean",
              height_m: float = 0.0) -> _LocalRaw:
    """The core's :class:`_LocalRaw` numbers at (lat [deg], lon [deg]) for
    ``model`` (its ``et0``, frame and fit half-window), ``height_m`` [m]
    above the WGS-84 ellipsoid.  ``limb="profile"`` needs the limb band and
    lunar kernels (``kernels.bootstrap --limb``; ``FileNotFoundError`` / a
    SPICE error otherwise)."""
    _check_limb(limb)
    if limb == "profile":
        ensure_limb_band()
    # The core runs OpenMP teams here too (the element loop for the sampled
    # window; the silhouette of a cold profile node): one team per process
    # (``PARALLEL_LOCK``, item C6), which also stops concurrent requests from
    # building the same cold node twice.
    with PARALLEL_LOCK:
        raw = _eclipse.local_circumstances(
            model.et0, model.earth_frame, float(model.half_window_hours), float(lat),
            float(lon), limb == "profile", float(height_m),
        )
    return _LocalRaw(*raw)


def local_circumstances(model, lat: float, lon: float, limb: str = "mean",
                        height_m: float = 0.0) -> LocalCircumstances:
    """The eclipse circumstances at (lat, lon) as the ``/circumstances`` dict.

    T0 should be near the observer's maximum eclipse.  Events with the Sun
    below the horizon are listed in ``below_horizon``; if every event is,
    ``eclipse`` is ``False``.  ``limb`` and ``height_m`` are echoed (``limb``,
    ``elev_m``).
    """
    out = _format_local(model.t0_utc, lat, lon, local_raw(model, lat, lon, limb, height_m))
    out["limb"] = limb
    out["elev_m"] = float(height_m)
    return out


def circumstances_grid(model, lats, lons, step_minutes: float = 2.0) -> dict[str, np.ndarray]:
    """Maximum-eclipse circumstances for observers at sea level (the ``/map`` grid).

    ``lats``/``lons`` are 1-D arrays [deg] of the same length ``P``; every
    observer is sampled over the model's ``+/- half_window_hours`` every
    ``step_minutes``, so that window must cover the eclipse globally (~+/-3 h
    around greatest eclipse).  Returns length-``P`` arrays ``magnitude`` (0
    where no eclipse), ``obscuration``, ``t_max_hours``, ``sun_alt`` [deg] at
    maximum, ``visible`` (Sun above ``_eclipse.HORIZON_ALT_DEG`` at maximum)
    and ``central`` (inside the umbra/antumbra at maximum).  The observer loop
    is OpenMP-parallel in the core (one team per process: ``PARALLEL_LOCK``).
    """
    lats = np.ascontiguousarray(np.atleast_1d(np.asarray(lats, dtype=float)))
    lons = np.ascontiguousarray(np.atleast_1d(np.asarray(lons, dtype=float)))
    with PARALLEL_LOCK:
        return _eclipse.circumstances_grid(model.et0, model.earth_frame,
                                           float(model.half_window_hours), lats, lons,
                                           float(step_minutes))

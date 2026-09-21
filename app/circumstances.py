"""Local circumstances of an eclipse for a specific observer (or a grid of them).

From the Besselian elements (as a :class:`~app.besselian.BesselianModel`) and an
observer's geographic position, compute what that observer sees: the partial
contact times C1/C4, the total/annular contacts C2/C3 and their duration, the
time and magnitude of maximum eclipse, the obscuration (covered fraction of the
Sun's area) and the Sun's altitude/azimuth.

Method ([ES92] sec. 8.35, eq. 8.353-8.354; [MeeusSE]): the observer's
fundamental-plane position (xi, eta, zeta) gives the separation ``m`` from the
shadow axis; the penumbral/umbral cone radii reduced to the observer are
``L1' = l1 - zeta*tan_f1`` and ``L2' = l2 - zeta*tan_f2``.  Contacts are where
``m`` equals ``L1'`` (partial) or ``|L2'|`` (central); the eclipse magnitude is
``(L1' - m)/(L1' + L2')`` [Espenak].

Horizon: the cone geometry is valid on the whole ellipsoid, including the
night side (zeta < 0), where it describes the shadow cone continued *through*
the Earth.  Contacts are therefore always computed geometrically and then
checked against the Sun's altitude: any contact (or the maximum) with the Sun
below the rise/set altitude ``h0 = -0 deg 50'`` [Meeus98] ch. 15 is listed in
``below_horizon``, and an observer for whom C1, maximum and C4 are all below it
sees no eclipse at all (``eclipse: False``).

Geometric only otherwise: no atmospheric refraction on the contacts themselves
and the mean lunar limb (folded into k2), so grazing/limb-profile effects are
not modelled (item A3).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from .ephemeris import sub_solar_points
from .geography import bearing, format_clock, geo_to_fund

# Sun rise/set altitude: refraction 34' + solar semidiameter 16' [Meeus98] ch. 15.
HORIZON_ALT_DEG = -0.8333


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


def _series(model, lat, lon, t):
    """Arrays ``(m, L1', L2', magnitude)`` over offsets ``t`` (hours from T0).

    Direct per-instant evaluation (:meth:`~app.besselian.BesselianModel.
    evaluate_direct`, item A2) and the array-native :func:`geo_to_fund`, so
    ``lat``/``lon`` may be scalars (1-D results over ``t``) or column arrays of
    shape ``(P, 1)`` (results of shape ``(P, N)`` for a grid of observers).
    """
    e = model.evaluate_direct(t)
    xi, eta, zeta = geo_to_fund(lat, lon, e["d"], e["mu"])
    m = np.hypot(xi - e["x"], eta - e["y"])
    L1p = e["l1"] - zeta * e["tan_f1"]
    L2p = e["l2"] - zeta * e["tan_f2"]
    mag = (L1p - m) / (L1p + L2p)
    return m, L1p, L2p, mag


def _roots(t, f):
    """Linear-interpolated zero crossings: list of (t_cross, rising?, i)."""
    out = []
    for i in range(len(t) - 1):
        if f[i] == 0.0:
            out.append((float(t[i]), f[i + 1] > 0, i))
        elif f[i] * f[i + 1] < 0:
            tc = t[i] - f[i] * (t[i + 1] - t[i]) / (f[i + 1] - f[i])
            out.append((float(tc), f[i + 1] > f[i], i))
    return out


def _overlap_area(r, R, d):
    """Area of intersection of two circles (radii r<=R, centre distance d).

    Standard circular-segment (lens) area; used for the obscuration below.
    Array-native (``np.where`` on the disjoint / contained cases).
    """
    r, R, d = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (r, R, d)))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.clip((d * d + r * r - R * R) / (2 * d * r), -1, 1)
        b = np.clip((d * d + R * R - r * r) / (2 * d * R), -1, 1)
        tri = 0.5 * np.sqrt(np.maximum(0.0, (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)))
        lens = r * r * np.arccos(a) + R * R * np.arccos(b) - tri
    area = np.where(d >= r + R, 0.0, np.where(d <= R - r, np.pi * r * r, lens))
    return float(area) if area.ndim == 0 else area


def _obscuration(L1p, L2p, m):
    """Fraction of the Sun's area covered, from the shadow radii + separation.

    Obscuration is the covered fraction of the Sun's *area* [Espenak] (distinct
    from magnitude, a diameter ratio): the Moon/Sun disks are two circles whose
    overlap area (:func:`_overlap_area`) is divided by the Sun's area.  The disk
    radius ratio and centre separation come from the reduced cone radii L1'/L2'.
    Array-native.
    """
    L1p, L2p, m = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (L1p, L2p, m)))
    denom = L1p + L2p
    safe = np.where(denom > 0, denom, 1.0)
    q = (L1p - L2p) / safe           # Moon/Sun apparent radius ratio
    sep = 2.0 * m / safe             # centre separation in units of Sun radius
    r = np.minimum(q, 1.0)
    R = np.maximum(q, 1.0)
    obs = np.where(denom > 0, _overlap_area(r, R, sep) / np.pi, 0.0)  # Sun area = pi * 1^2
    return float(obs) if obs.ndim == 0 else obs


def _sun_altaz(model, lat, lon, t_hours):
    """Sun altitude/azimuth [deg] at ``t_hours`` (scalar or array) for the observer.

    Geodetic altitude is exact here: the sub-solar latitude equals the Sun's
    declination and the ellipsoid normal is the zenith, so the spherical
    ``sin h = sin phi sin delta + cos phi cos delta cos H`` holds with geodetic
    ``phi`` (parallax of the Sun, < 9", neglected).  ``lat``/``lon`` may be
    column arrays ``(P, 1)`` against ``t_hours`` of shape ``(N,)``.
    """
    t = np.atleast_1d(np.asarray(t_hours, dtype=float))
    ss_lon, ss_lat = sub_solar_points(model.et0 + t * 3600.0, model.earth_frame)
    p1, p2 = np.radians(lat), np.radians(ss_lat)
    dl = np.radians(ss_lon - lon)
    cos_c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    alt = 90.0 - np.degrees(np.arccos(np.clip(cos_c, -1, 1)))
    az = bearing(lat, lon, ss_lat, ss_lon)
    return alt, az


# Contact bracketing: the sampling half-window is expanded (up to this cap) until
# the observer is outside the penumbra at both ends, so C1/C4 are bracketed even
# when the partial phase runs past the model's fit window (item A2).  A partial
# eclipse lasts at most a few hours at any one site, so 6 h is a safe ceiling.
_MAX_HALF_WINDOW_HOURS = 6.0
_GRID_STEP_HOURS = 0.5 / 60.0  # 30-second grid
_REFINE_ITERATIONS = 30        # bisection: 30 s / 2^30 -> sub-microsecond


def _bracketed_series(model, lat, lon):
    """Sample (t, m, L1', L2', mag) on a 30-second grid wide enough to contain C1..C4.

    Starts at the model's fit half-window and widens (direct evaluation stays
    exact outside the fit window) until the observer is outside the penumbra at
    both ends, or the cap is reached.  Returns ``(t, m, L1p, L2p, mag)``.
    """
    hw = model.half_window_hours
    while True:
        t = np.arange(-hw, hw + 1e-9, _GRID_STEP_HOURS)
        m, L1p, L2p, mag = _series(model, lat, lon, t)
        outside = m - L1p  # > 0 when the observer is outside the penumbra
        edges_clear = outside[0] > 0 and outside[-1] > 0
        if edges_clear or hw >= _MAX_HALF_WINDOW_HOURS:
            return t, m, L1p, L2p, mag
        hw = min(hw * 1.5, _MAX_HALF_WINDOW_HOURS)


def _refine_contacts(model, lat, lon, t_lo, t_hi, central):
    """Bisect the contact conditions to machine precision inside 30-s brackets.

    ``t_lo``/``t_hi`` are arrays of bracket ends (hours) and ``central`` a bool
    array selecting ``m - |L2'|`` (C2/C3) over ``m - L1'`` (C1/C4).  All
    contacts are refined together, one vectorized :func:`_series` per iteration.
    """
    t_lo = np.array(t_lo, dtype=float)
    t_hi = np.array(t_hi, dtype=float)
    central = np.asarray(central, dtype=bool)

    def f(t):
        m, L1p, L2p, _ = _series(model, lat, lon, t)
        return np.where(central, m - np.abs(L2p), m - L1p)

    f_lo = f(t_lo)
    for _ in range(_REFINE_ITERATIONS):
        t_mid = 0.5 * (t_lo + t_hi)
        f_mid = f(t_mid)
        same = np.sign(f_mid) == np.sign(f_lo)
        t_lo = np.where(same, t_mid, t_lo)
        f_lo = np.where(same, f_mid, f_lo)
        t_hi = np.where(same, t_hi, t_mid)
    return 0.5 * (t_lo + t_hi)


def _refine_maximum(model, lat, lon, t, mag, imax):
    """Parabolic refinement of the time of maximum through the three grid points."""
    if 0 < imax < len(t) - 1:
        y0, y1, y2 = mag[imax - 1], mag[imax], mag[imax + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0.0:
            return float(t[imax] + 0.5 * _GRID_STEP_HOURS * (y0 - y2) / denom)
    return float(t[imax])


def local_circumstances(model, lat: float, lon: float) -> LocalCircumstances:
    """Compute the eclipse circumstances at (lat, lon).

    T0 should be near the observer's maximum eclipse (e.g. the greatest-eclipse
    time).  The sampling window starts at the model's fit window and is expanded
    as needed so the partial contacts C1/C4 are always bracketed, even for an
    observer whose partial phase extends beyond it (item A2).  Contacts and the
    maximum are refined below the 30-s grid (bisection / parabolic).  Events
    with the Sun below the horizon are reported in ``below_horizon``; if every
    event is, ``eclipse`` is ``False`` (module docstring, "Horizon").
    """
    t, m, L1p, L2p, mag = _bracketed_series(model, lat, lon)

    partial = _roots(t, m - L1p)
    if len(partial) < 2 or mag.max() <= 0:
        no_eclipse: LocalCircumstances = {"lat": lat, "lon": lon, "eclipse": False}
        return no_eclipse

    tc1, _, i1 = min((r for r in partial if not r[1]), key=lambda r: r[0])
    tc4, _, i4 = max((r for r in partial if r[1]), key=lambda r: r[0])

    imax = int(np.argmax(mag))
    # Central phase: observer inside the umbra/antumbra at maximum.
    central_phase = bool(m[imax] < abs(L2p[imax]))
    brackets = [(t[i1], t[i1 + 1], False), (t[i4], t[i4 + 1], False)]
    if central_phase:
        central = _roots(t, m - np.abs(L2p))
        enters = [r for r in central if not r[1] and r[0] <= t[imax]]
        exits = [r for r in central if r[1] and r[0] >= t[imax]]
        if enters and exits:
            _, _, i2 = max(enters, key=lambda r: r[0])
            _, _, i3 = min(exits, key=lambda r: r[0])
            brackets += [(t[i2], t[i2 + 1], True), (t[i3], t[i3 + 1], True)]
        else:
            central_phase = False
    times = _refine_contacts(model, lat, lon, [b[0] for b in brackets],
                             [b[1] for b in brackets], [b[2] for b in brackets])
    c1, c4 = float(times[0]), float(times[1])
    tmax = _refine_maximum(model, lat, lon, t, mag, imax)

    # Quantities at the refined maximum.
    m_x, L1_x, L2_x, _ = (float(v[0]) for v in _series(model, lat, lon, np.array([tmax])))
    # Eclipse magnitude: for total/annular it is the Moon/Sun apparent diameter
    # ratio (Espenak convention); for a partial eclipse it is the fraction of the
    # Sun's diameter covered.
    if central_phase:
        magnitude = (L1_x - L2_x) / (L1_x + L2_x)
    else:
        magnitude = (L1_x - m_x) / (L1_x + L2_x)

    events = ["C1", "max", "C4"] + (["C2", "C3"] if central_phase else [])
    ev_times = [c1, tmax, c4] + ([float(times[2]), float(times[3])] if central_phase else [])
    alts, azs = _sun_altaz(model, lat, lon, np.array(ev_times))
    below = [name for name, a in zip(events, alts, strict=True) if a <= HORIZON_ALT_DEG]
    if len(below) == len(events):
        # The cone geometry continues through the Earth; a night-side observer
        # is "inside" it but sees nothing.
        return {"lat": lat, "lon": lon, "eclipse": False, "below_horizon": below}

    result: LocalCircumstances = {
        "lat": lat,
        "lon": lon,
        "eclipse": True,
        "type": "partial",  # upgraded to total/annular below if the observer enters the umbra
        "magnitude": round(float(magnitude), 4),
        "obscuration": round(float(_obscuration(L1_x, L2_x, m_x)), 4),
        "max_time": format_clock(model.t0_utc, tmax),
        "sun_alt": round(float(alts[1]), 1),
        "sun_az": round(float(azs[1]), 1),
        "C1": format_clock(model.t0_utc, c1),
        "C4": format_clock(model.t0_utc, c4),
        "C1_alt": round(float(alts[0]), 1),
        "C4_alt": round(float(alts[2]), 1),
        "below_horizon": below,
    }

    if central_phase:
        c2, c3 = float(times[2]), float(times[3])
        result["type"] = "total" if L2_x < 0 else "annular"
        result["C2"] = format_clock(model.t0_utc, c2)
        result["C3"] = format_clock(model.t0_utc, c3)
        result["C2_alt"] = round(float(alts[3]), 1)
        result["C3_alt"] = round(float(alts[4]), 1)
        result["central_duration_s"] = round((c3 - c2) * 3600.0, 1)

    return result


def circumstances_grid(
    model,
    lats: np.ndarray,
    lons: np.ndarray,
    step_minutes: float = 2.0,
    chunk: int = 4096,
) -> dict[str, np.ndarray]:
    """Maximum-eclipse circumstances for a set of observers, vectorized.

    ``lats``/``lons`` are 1-D arrays (degrees) of the same length ``P``.  Every
    observer is sampled over ``[-half_window_hours, +half_window_hours]`` of the
    model at ``step_minutes`` (2 min is ample for a magnitude/obscuration map:
    the magnitude curve is flat at its maximum), so the model's window must
    cover the eclipse globally (~+/-3 h around greatest eclipse).  Work is done in
    chunks of ``chunk`` observers to bound memory at ``chunk x N`` doubles.

    Returns arrays of length ``P``: ``magnitude`` (0 where no eclipse),
    ``obscuration``, ``t_max_hours``, ``sun_alt`` at maximum, ``visible`` (Sun
    above ``HORIZON_ALT_DEG`` at maximum) and ``central`` (inside the
    umbra/antumbra at maximum).  Same geometry as :func:`local_circumstances`,
    minus the contact times and refinement.
    """
    lats = np.atleast_1d(np.asarray(lats, dtype=float))
    lons = np.atleast_1d(np.asarray(lons, dtype=float))
    hw = model.half_window_hours
    t = np.arange(-hw, hw + 1e-9, step_minutes / 60.0)
    ss_lon, ss_lat = sub_solar_points(model.et0 + t * 3600.0, model.earth_frame)

    P = len(lats)
    out = {k: np.zeros(P) for k in ("magnitude", "obscuration", "t_max_hours", "sun_alt")}
    out["visible"] = np.zeros(P, dtype=bool)
    out["central"] = np.zeros(P, dtype=bool)

    for s in range(0, P, chunk):
        la = lats[s:s + chunk, None]
        lo = lons[s:s + chunk, None]
        m, L1p, L2p, mag = _series(model, la, lo, t)          # (p, N)
        imax = np.argmax(mag, axis=1)
        rows = np.arange(len(la))
        mag_x = mag[rows, imax]
        m_x, L1_x, L2_x = m[rows, imax], L1p[rows, imax], L2p[rows, imax]
        has = mag_x > 0
        central = has & (m_x < np.abs(L2_x))
        # Central magnitude is the diameter ratio [Espenak], as in local_circumstances.
        magnitude = np.where(central, (L1_x - L2_x) / (L1_x + L2_x), mag_x)

        p1, p2 = np.radians(la), np.radians(ss_lat[imax])[:, None]
        dl = np.radians(ss_lon[imax])[:, None] - np.radians(lo)
        cos_c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
        alt = (90.0 - np.degrees(np.arccos(np.clip(cos_c, -1, 1))))[:, 0]

        sl = slice(s, s + len(la))
        out["magnitude"][sl] = np.where(has, magnitude, 0.0)
        out["obscuration"][sl] = np.where(has, _obscuration(L1_x, L2_x, m_x), 0.0)
        out["t_max_hours"][sl] = t[imax]
        out["sun_alt"][sl] = alt
        out["visible"][sl] = has & (alt > HORIZON_ALT_DEG)
        out["central"][sl] = central
    return out

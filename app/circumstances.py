"""Local circumstances of an eclipse for a specific observer.

From the Besselian elements (as a :class:`~app.besselian.BesselianModel`) and an
observer's geographic position, compute what that observer sees: the partial
contact times C1/C4, the total/annular contacts C2/C3 and their duration, the
time and magnitude of maximum eclipse, the obscuration (covered fraction of the
Sun's area) and the Sun's altitude/azimuth.

Method (Explanatory Supplement to the Astronomical Almanac): the observer's
fundamental-plane position (xi, eta, zeta) gives the separation ``m`` from the
shadow axis; the penumbral/umbral cone radii reduced to the observer are
``L1' = l1 - zeta*tan_f1`` and ``L2' = l2 - zeta*tan_f2``.  Contacts are where
``m`` equals ``L1'`` (partial) or ``|L2'|`` (central); the eclipse magnitude is
``(L1' - m)/(L1' + L2')``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from .ephemeris import sub_solar_point, utc_to_et
from .geography import _EARTH_MEAN_KM, bearing, geo_to_fund


def _series(model, lat, lon, t):
    """Return arrays (m, L1', L2', magnitude) over offsets ``t`` (hours from T0)."""
    e = model.evaluate(t)
    n = len(t)
    m = np.empty(n)
    L1p = np.empty(n)
    L2p = np.empty(n)
    for i in range(n):
        xi, eta, zeta = geo_to_fund(lat, lon, e["d"][i], e["mu"][i])
        m[i] = np.hypot(xi - e["x"][i], eta - e["y"][i])
        L1p[i] = e["l1"][i] - zeta * e["tan_f1"][i]
        L2p[i] = e["l2"][i] - zeta * e["tan_f2"][i]
    mag = (L1p - m) / (L1p + L2p)
    return m, L1p, L2p, mag


def _roots(t, f):
    """Linear-interpolated zero crossings: list of (t_cross, rising?)."""
    out = []
    for i in range(len(t) - 1):
        if f[i] == 0.0:
            out.append((float(t[i]), f[i + 1] > 0))
        elif f[i] * f[i + 1] < 0:
            tc = t[i] - f[i] * (t[i + 1] - t[i]) / (f[i + 1] - f[i])
            out.append((float(tc), f[i + 1] > f[i]))
    return out


def _overlap_area(r, R, d):
    """Area of intersection of two circles (radii r<=R, centre distance d)."""
    if d >= r + R:
        return 0.0
    if d <= R - r:
        return np.pi * r * r
    a = np.clip((d * d + r * r - R * R) / (2 * d * r), -1, 1)
    b = np.clip((d * d + R * R - r * r) / (2 * d * R), -1, 1)
    tri = 0.5 * np.sqrt(max(0.0, (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)))
    return r * r * np.arccos(a) + R * R * np.arccos(b) - tri


def _obscuration(L1p, L2p, m):
    """Fraction of the Sun's area covered, from the shadow radii + separation."""
    denom = L1p + L2p
    if denom <= 0:
        return 0.0
    q = (L1p - L2p) / denom           # Moon/Sun apparent radius ratio
    sep = 2.0 * m / denom             # centre separation in units of Sun radius
    r, R = (q, 1.0) if q <= 1.0 else (1.0, q)
    return float(_overlap_area(r, R, sep) / np.pi)  # Sun area = pi * 1^2


def _sun_altaz(model, lat, lon, t_hours):
    et = utc_to_et(model.t0_utc) + t_hours * 3600.0
    ss_lon, ss_lat = sub_solar_point(et, model.earth_frame)
    # central angle observer -> sub-solar point
    p1, p2 = np.radians(lat), np.radians(ss_lat)
    dl = np.radians(ss_lon - lon)
    cos_c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    alt = 90.0 - np.degrees(np.arccos(np.clip(cos_c, -1, 1)))
    az = bearing(lat, lon, ss_lat, ss_lon)
    return float(alt), float(az)


def _clock(model, t_hours):
    return (
        datetime.fromisoformat(model.t0_utc).replace(tzinfo=timezone.utc)
        + timedelta(hours=t_hours)
    ).strftime("%H:%M:%S")


def local_circumstances(model, lat: float, lon: float) -> dict:
    """Compute the eclipse circumstances at (lat, lon).

    The sampled window is the model's fit window around T0, so T0 should be near
    the observer's maximum eclipse (e.g. the greatest-eclipse time).
    """
    tw = model.half_window_hours
    t = np.arange(-tw, tw + 1e-9, 0.5 / 60.0)  # 30-second grid
    m, L1p, L2p, mag = _series(model, lat, lon, t)

    partial = _roots(t, m - L1p)
    if len(partial) < 2 or mag.max() <= 0:
        return {"lat": lat, "lon": lon, "eclipse": False}

    c1 = min(tc for tc, rising in partial if not rising)
    c4 = max(tc for tc, rising in partial if rising)

    imax = int(np.argmax(mag))
    tmax = float(t[imax])
    alt, az = _sun_altaz(model, lat, lon, tmax)

    # Central phase: observer inside the umbra/antumbra at maximum.
    central_phase = m[imax] < abs(L2p[imax])
    # Eclipse magnitude: for total/annular it is the Moon/Sun apparent diameter
    # ratio (Espenak convention); for a partial eclipse it is the fraction of the
    # Sun's diameter covered.
    if central_phase:
        magnitude = (L1p[imax] - L2p[imax]) / (L1p[imax] + L2p[imax])
    else:
        magnitude = (L1p[imax] - m[imax]) / (L1p[imax] + L2p[imax])

    result = {
        "lat": lat,
        "lon": lon,
        "eclipse": True,
        "type": "partial",  # upgraded to total/annular below if the observer enters the umbra
        "magnitude": round(float(magnitude), 4),
        "obscuration": round(_obscuration(L1p[imax], L2p[imax], m[imax]), 4),
        "max_time": _clock(model, tmax),
        "sun_alt": round(alt, 1),
        "sun_az": round(az, 1),
        "C1": _clock(model, c1),
        "C4": _clock(model, c4),
    }

    if central_phase:
        central = _roots(t, m - np.abs(L2p))
        enters = [tc for tc, rising in central if not rising and tc <= tmax]
        exits = [tc for tc, rising in central if rising and tc >= tmax]
        if enters and exits:
            c2, c3 = max(enters), min(exits)
            result["type"] = "total" if L2p[imax] < 0 else "annular"
            result["C2"] = _clock(model, c2)
            result["C3"] = _clock(model, c3)
            result["central_duration_s"] = round((c3 - c2) * 3600.0, 1)

    return result

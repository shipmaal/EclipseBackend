"""Lunar limb profile from the LRO LOLA DEM (``docs/LIMB_PROFILE.md``).

The Moon's silhouette seen along the shadow axis, as a radius offset from the
LOLA reference sphere at each position angle in the fundamental plane.  An
observer in or near the umbra looks at the Moon along that axis to within
~0.02 deg, so for every umbral quantity the profile depends on time only
(design note sec. 3.1).

Conventions (units in every signature):

* ``psi`` is the angle in the fundamental plane measured from ``x^`` (east)
  toward ``y^`` (north), radians in ``(-pi, pi]`` -- the plane of
  :func:`app.ephemeris.limb_axes`, NOT the sky position angle (north through
  east).  Using the elements' own axes avoids any sky-mirror convention.
* A profile is ``N_BINS`` values of ``delta_rho`` [km]: the silhouette radius
  minus ``R_REF_KM`` in bins of ``2 pi / N_BINS``; bin ``k`` covers
  ``psi in [-pi + k w, -pi + (k + 1) w)`` and is represented at its centre.

The DEM comes as a limb-band file cut by :mod:`kernels.limb_band`
(``kernels.bootstrap --limb``); ``$ECLIPSE_LIMB_BAND`` overrides its path.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import native
from .ephemeris import DEFAULT_EARTH_FRAME, limb_axes

# LOLA reference sphere: LDEM label OFFSET = 1737400 m, the radius the heights
# are measured from, about the Moon's centre of mass [LOLA].
R_REF_KM = 1737.4

# Profile resolution: 0.05 deg of position angle, ~1.5 km along the limb
# (design note sec. 3.2; resolution study sec. 9).
N_BINS = 7200

# A peak of height h shows over the limb from alpha behind it while
# cos(alpha) > R / (R + h): 6.2 deg for h = 10 km.  The view axis must stay
# within BAND_DEG - VISIBLE_DEG of the band's centre (+x of MOON_ME).
VISIBLE_DEG = 8.0

# Points projecting more than this below R_REF_KM cannot be on the silhouette
# (the lowest limb point is ~9 km down); dropping them first only saves time
# -- every bin keeps a point above it, which silhouette() checks.
FLOOR_KM = 20.0

# More empty bins than this fraction (before :func:`_fill_empty`) is an error.
MAX_EMPTY_FRACTION = 0.1

DEFAULT_BAND_FILE = "lola_ldem16_limb20.bin"


class LimbBand(NamedTuple):
    """A limb-band DEM ready for projection (all arrays read-only).

    ``runs`` (n, 3) int32 of ``(line, first_sample, count)`` and ``dn`` (int16)
    are the file's; ``line_idx``/``col_idx`` expand the runs per point;
    ``right``/``down`` are each point's east (``j + 1``, wrapping at 360 deg)
    and southern (``i + 1``) grid neighbour's index, ``-1`` if not in the band;
    ``cos_lat``/``sin_lat`` are per DEM line and ``cos_lon``/``sin_lon`` per
    sample, at pixel centres; ``radius = offset_km + dn * scale_km`` [km].
    """

    header: dict
    runs: np.ndarray
    dn: np.ndarray
    line_idx: np.ndarray
    col_idx: np.ndarray
    right: np.ndarray
    down: np.ndarray
    cos_lat: np.ndarray
    sin_lat: np.ndarray
    cos_lon: np.ndarray
    sin_lon: np.ndarray
    offset_km: float
    scale_km: float
    band_deg: float


def default_band_path() -> Path:
    env = os.environ.get("ECLIPSE_LIMB_BAND")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "kernels" / DEFAULT_BAND_FILE


def band_from_file(bf) -> LimbBand:
    """Build a :class:`LimbBand` from a :class:`kernels.limb_band.BandFile`.

    Pixel centres follow the PDS map projection of the LDEM label (see
    :mod:`kernels.limb_band`): latitude ``(LINE_PROJECTION_OFFSET - i) / ppd``,
    east longitude ``(j - SAMPLE_PROJECTION_OFFSET) / ppd + CENTER_LONGITUDE``.
    """
    h = bf.header
    ppd = float(h["map_resolution"])
    lat = np.radians((float(h["line_projection_offset"]) - np.arange(int(h["lines"]))) / ppd)
    lon = np.radians((np.arange(int(h["line_samples"])) - float(h["sample_projection_offset"]))
                     / ppd + float(h["center_longitude"]))
    runs = np.ascontiguousarray(bf.runs, dtype=np.int32)
    counts = runs[:, 2].astype(np.int64)
    line_idx = np.repeat(runs[:, 0], counts)
    starts = np.repeat(runs[:, 1] - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    col_idx = starts + np.arange(int(counts.sum()))
    right, down = _neighbours(runs, int(h["line_samples"]), int(counts.sum()))
    arrays = dict(
        right=right, down=down,
        runs=runs,
        dn=np.ascontiguousarray(bf.dn, dtype=np.int16),
        line_idx=line_idx.astype(np.int32),
        col_idx=col_idx.astype(np.int32),
        cos_lat=np.cos(lat), sin_lat=np.sin(lat), cos_lon=np.cos(lon), sin_lon=np.sin(lon),
    )
    for a in arrays.values():
        a.setflags(write=False)
    return LimbBand(header=h, offset_km=float(h["offset"]) / 1000.0,
                    scale_km=float(h["scaling_factor"]) / 1000.0,
                    band_deg=float(h["band_deg"]), **arrays)


def _neighbours(runs: np.ndarray, samples: int, n_points: int):
    """Per-point indices of the east and southern grid neighbours (``-1``: none)."""
    right = np.full(n_points, -1, dtype=np.int32)
    down = np.full(n_points, -1, dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(runs[:, 2].astype(np.int64))[:-1]))
    by_line: dict[int, np.ndarray] = {}
    for (line, first, count), off in zip(runs, offsets, strict=True):
        row = by_line.setdefault(int(line), np.full(samples, -1, dtype=np.int64))
        row[first:first + count] = off + np.arange(count)
    for line, row in by_line.items():
        have = row >= 0
        right[row[have]] = np.roll(row, -1)[have]
        below = by_line.get(line + 1)
        if below is not None:
            down[row[have]] = below[have]
    return right, down


@lru_cache(maxsize=4)
def load_band(path: str | None = None) -> LimbBand:
    """The limb band at ``path`` (default :func:`default_band_path`), cached.

    Raises ``FileNotFoundError`` when absent (``kernels.bootstrap --limb``).
    :func:`silhouette` installs it into the native core on first use.
    """
    from kernels import limb_band

    p = Path(path) if path else default_band_path()
    if not p.exists():
        raise FileNotFoundError(
            f"lunar limb band not found at {p}. Run `python -m kernels.bootstrap --limb`.")
    return band_from_file(limb_band.read(p))


# The band the native core holds (it keeps one); :func:`silhouette` installs
# the band it is given when it is not this one.
_native_band: LimbBand | None = None


def install_native(band: LimbBand) -> None:
    """Copy ``band`` into the native core (``_eclipse.set_limb_band``)."""
    global _native_band
    native.module().set_limb_band(
        band.runs[:, 0].copy(), band.runs[:, 1].copy(), band.runs[:, 2].copy(), band.dn,
        band.right, band.down, band.cos_lat, band.sin_lat, band.cos_lon, band.sin_lon,
        band.offset_km, band.scale_km, band.band_deg)
    _native_band = band


def silhouette(band: LimbBand, axes, n_bins: int = N_BINS) -> np.ndarray:
    """``delta_rho`` [km] per bin: the DEM's silhouette seen along ``axes[2]``.

    ``axes`` is one ``(3, 3)`` block of :func:`app.ephemeris.limb_axes` (rows
    ``x^, y^, z^`` in the DEM's frame).  Each band point ``p = r n`` (``n``
    from the pixel-centre latitude/longitude, ``r = offset + DN * scale``) is
    projected onto the plane, ``q = (p . x^, p . y^)``; a bin's radius is the
    largest ``|q|`` of the points falling in it and of the grid edges crossing
    it (:func:`_edge_crossings`) -- the orthographic silhouette of the
    piecewise-linear DEM surface (design note sec. 3.2), the view distance
    being ~220 lunar radii.  Only points above ``R_REF_KM - FLOOR_KM`` and
    edges between two of them take part.  Bins still empty are filled by
    :func:`_fill_empty`.  Raises ``ValueError`` if the view axis
    leaves the band's coverage or too many bins are empty.  Dispatches to
    the native core.
    """
    axes = np.asarray(axes, dtype=float).reshape(3, 3)
    cover = np.cos(np.radians(band.band_deg - VISIBLE_DEG))
    if abs(axes[2, 0]) < cover:
        raise ValueError(
            f"view axis is {np.degrees(np.arccos(min(1.0, abs(axes[2, 0])))):.2f} deg from the "
            f"limb band's centre; the band covers {band.band_deg - VISIBLE_DEG:g} deg")
    if native.is_native():
        with native.PARALLEL_LOCK:  # one OpenMP team per process (review item C9)
            if _native_band is not band:
                install_native(band)
            return native.module().limb_silhouette(np.ascontiguousarray(axes), int(n_bins))
    x, y = axes[0], axes[1]
    r = band.offset_km + band.dn * band.scale_km
    cl = band.cos_lat[band.line_idx]
    px = r * (cl * band.cos_lon[band.col_idx])
    py = r * (cl * band.sin_lon[band.col_idx])
    pz = r * band.sin_lat[band.line_idx]
    qx = (px * x[0] + py * x[1]) + pz * x[2]
    qy = (px * y[0] + py * y[1]) + pz * y[2]
    s = np.sqrt(qx * qx + qy * qy)  # not np.hypot: SIMD, differs from libm by 1 ulp
    floor_km = R_REF_KM - FLOOR_KM
    near = s > floor_km
    # Continuous bin coordinate: bin k spans u in [k, k + 1).
    u = np.full(len(s), np.nan)
    u[near] = (_atan2(qy[near], qx[near]) + np.pi) * (n_bins / (2.0 * np.pi))
    rho = np.full(n_bins, -np.inf)
    k = np.minimum(np.floor(u[near]).astype(np.int64), n_bins - 1)
    np.maximum.at(rho, k, s[near])
    for nb in (band.right, band.down):
        a = np.flatnonzero((nb >= 0) & near)
        b = nb[a]
        a, b = a[near[b]], b[near[b]]
        _edge_crossings(rho, u[a], u[b], s[a], s[b], n_bins)
    return _fill_empty(rho) - R_REF_KM


def _atan2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``atan2`` elementwise through the C library (``math.atan2``), not
    ``np.arctan2``: NumPy's SIMD kernel differs from libm by 1 ulp on ~7 % of
    inputs, and a 1-ulp shift of a bin coordinate is amplified by steep grid
    edges (``frac = (bnd - ua) / du``), so parity with the native core, which
    calls libm, would not be bit-level (CLAUDE.md "port, don't improve")."""
    return np.fromiter(map(math.atan2, y.tolist(), x.tolist()), dtype=float, count=len(y))


def _edge_crossings(rho, ua, ub, sa, sb, n_bins: int) -> None:
    """Max-accumulate into ``rho`` the radius of each projected DEM grid edge
    where it crosses a bin boundary (integer ``u``), linear in ``u`` between
    its end points.  The DEM is sampled every ~1.9 km along the limb (LDEM_16)
    and a bin is ~1.5 km, so end points alone leave bins holding only points
    from behind the limb; the edges make the silhouette that of the
    piecewise-linear surface (docs/LIMB_PROFILE.md sec. 9).  Edges are taken
    the short way round the -pi/pi seam.
    """
    du = ub - ua
    du = np.where(du > n_bins / 2, du - n_bins, np.where(du <= -n_bins / 2, du + n_bins, du))
    ub = ua + du
    first = np.floor(np.minimum(ua, ub)) + 1.0
    last = np.floor(np.maximum(ua, ub))
    span = (last - first).astype(np.int64)
    if len(span) == 0 or span.max() < 0:
        return
    for m in range(int(span.max()) + 1):
        on = span >= m
        bnd = first[on] + m
        frac = (bnd - ua[on]) / du[on]
        rad = sa[on] + (sb[on] - sa[on]) * frac
        ib = bnd.astype(np.int64)
        np.maximum.at(rho, (ib - 1) % n_bins, rad)
        np.maximum.at(rho, ib % n_bins, rad)


def _fill_empty(rho: np.ndarray) -> np.ndarray:
    """Fill bins no DEM point fell in by periodic linear interpolation between
    the nearest filled bins (``a + (b - a) * (m / L)``, ``m`` steps into a gap
    of ``L``).  A view aligned with the DEM grid's rows can leave isolated
    bins empty (the rows project to a comb of angles); more than
    ``MAX_EMPTY_FRACTION`` empty means the bins are too fine for the DEM.
    """
    empty = ~np.isfinite(rho)
    if not empty.any():
        return rho
    n = len(rho)
    if empty.sum() > MAX_EMPTY_FRACTION * n:
        raise ValueError("limb profile has too many empty bins: n_bins too fine for this DEM")
    out = rho.copy()
    filled = np.flatnonzero(~empty)
    for a_i, b_i in zip(filled, np.roll(filled, -1), strict=True):
        gap = (b_i - a_i) % n
        if gap <= 1:
            continue
        a, b = rho[a_i], rho[b_i]
        for m in range(1, gap):
            out[(a_i + m) % n] = a + (b - a) * (m / gap)
    return out


def delta_rho_at(profile, psi) -> np.ndarray:
    """Linear interpolation of ``profile`` (bin centres, periodic) at ``psi`` [rad]."""
    profile = np.asarray(profile, dtype=float)
    n = len(profile)
    u = (np.asarray(psi, dtype=float) + np.pi) * (n / (2.0 * np.pi)) - 0.5
    k0 = np.floor(u)
    w = u - k0
    i0 = k0.astype(np.int64) % n
    i1 = (i0 + 1) % n
    return profile[i0] + w * (profile[i1] - profile[i0])


def limb_profiles(et, earth_frame: str = DEFAULT_EARTH_FRAME, moon_frame: str = "MOON_ME",
                  n_bins: int = N_BINS, band: LimbBand | None = None) -> np.ndarray:
    """``(n, n_bins)`` limb profiles [km] at each TDB ``et`` (see :func:`silhouette`)."""
    band = band if band is not None else load_band()
    axes = limb_axes(et, earth_frame, moon_frame)
    return np.stack([silhouette(band, a, n_bins) for a in axes])

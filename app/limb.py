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

# One lunar radius for both cones in profile mode: the LOLA sphere in Earth
# equatorial radii ([LOLA] / [WGS84] a); the valleys come from the DEM, not a
# reduced k (docs/LIMB_PROFILE.md sec. 3.3).
K_REF = R_REF_KM / 6378.137

# Time lattice of cached profiles [s of TDB]: nodes at multiples of 300 s,
# linear in time between them (<= 22 m, docs sec. 9.3).
LATTICE_S = 300.0

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
    source: str = ""  # the band file (resolved path); "" for an in-memory band


def default_band_path() -> Path:
    env = os.environ.get("ECLIPSE_LIMB_BAND")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "kernels" / DEFAULT_BAND_FILE


def band_from_file(bf, source: str = "") -> LimbBand:
    """Build a :class:`LimbBand` from a :class:`kernels.limb_band.BandFile`.

    Pixel centres follow the PDS map projection of the LDEM label (see
    :mod:`kernels.limb_band`): latitude ``(LINE_PROJECTION_OFFSET - i) / ppd``,
    east longitude ``(j - SAMPLE_PROJECTION_OFFSET) / ppd + CENTER_LONGITUDE``.
    Their cos/sin go through libm (:func:`_libm`), as the native core computes
    them when it reads the band file itself (``limb::load_band_file``).
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
        cos_lat=_libm(math.cos, lat), sin_lat=_libm(math.sin, lat),
        cos_lon=_libm(math.cos, lon), sin_lon=_libm(math.sin, lon),
    )
    for a in arrays.values():
        a.setflags(write=False)
    return LimbBand(header=h, offset_km=float(h["offset"]) / 1000.0,
                    scale_km=float(h["scaling_factor"]) / 1000.0,
                    band_deg=float(h["band_deg"]), source=source, **arrays)


def _libm(fn, x: np.ndarray) -> np.ndarray:
    """``fn`` (``math.cos`` / ``math.sin``) elementwise through the C library:
    NumPy's SIMD cos/sin can differ from libm by 1 ulp, and the native core
    builds the same tables with libm (see :func:`_atan2`)."""
    return np.fromiter(map(fn, x.tolist()), dtype=float, count=len(x))


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


def _resolve(path: str | Path | None) -> Path:
    """The band file for ``path`` (default :func:`default_band_path`), resolved;
    ``FileNotFoundError`` when absent (``kernels.bootstrap --limb``)."""
    p = Path(path) if path else default_band_path()
    if not p.exists():
        raise FileNotFoundError(
            f"lunar limb band not found at {p}. Run `python -m kernels.bootstrap --limb`.")
    return p.resolve()


@lru_cache(maxsize=4)
def load_band(path: str | None = None) -> LimbBand:
    """The limb band at ``path`` (default :func:`default_band_path`) as NumPy
    arrays, cached -- the Python oracle's copy.  The native core reads the file
    itself (:func:`ensure_native_band`), so under the native backend nothing
    needs to call this.  Raises ``FileNotFoundError`` when absent.
    """
    from kernels import limb_band

    p = _resolve(path)
    return band_from_file(limb_band.read(p), source=str(p))


# The in-memory (``set_limb_band``) band the native core holds, if any; a band
# loaded from a file is identified by the core's ``limb_band_source()``.
_native_band: LimbBand | None = None


def install_native(band: LimbBand) -> None:
    """Make the native core hold ``band``: by path when it came from a file
    (``_eclipse.load_limb_band``: the core reads the file and owns the data),
    else by copying its arrays (``_eclipse.set_limb_band``, synthetic bands)."""
    global _native_band
    mod = native.module()
    if band.source:
        mod.load_limb_band(band.source)
        _native_band = None
    else:
        mod.set_limb_band(
            band.runs[:, 0].copy(), band.runs[:, 1].copy(), band.runs[:, 2].copy(), band.dn,
            band.cos_lat, band.sin_lat, band.cos_lon, band.sin_lon,
            band.offset_km, band.scale_km, band.band_deg)
        _native_band = band


def _native_holds(band: LimbBand) -> bool:
    if band.source:
        return native.module().limb_band_source() == band.source
    return _native_band is band


def ensure_native_band(path: str | None = None) -> str:
    """Make sure the native core holds the band file at ``path`` (default
    :func:`default_band_path`), loading it there if not; returns the resolved
    path.  No NumPy copy of the band is built (for ``LDEM_64`` that is ~2 GB
    of Python arrays saved; docs/LIMB_PROFILE.md sec. 9.11)."""
    global _native_band
    p = str(_resolve(path))
    with native.PARALLEL_LOCK:
        mod = native.module()
        if mod.limb_band_source() != p:
            mod.load_limb_band(p)
            _native_band = None
    return p


def silhouette(band: LimbBand, axes, n_bins: int = N_BINS,
               distance_km: float = math.inf) -> np.ndarray:
    """``delta_rho`` [km] per bin: the DEM's silhouette seen along ``axes[2]``.

    ``axes`` is one ``(3, 3)`` block of :func:`app.ephemeris.limb_axes` (rows
    ``x^, y^, z^`` in the DEM's frame).  Each band point ``p = r n`` (``n``
    from the pixel-centre latitude/longitude, ``r = offset + DN * scale``) is
    projected onto the plane, ``q = (p . x^, p . y^)``, and seen in
    perspective from ``distance_km`` along ``-z^`` (:func:`app.ephemeris.
    limb_axes`): its apparent radius, in km at that distance, is
    ``|q| / (1 + w / D)`` with ``w = p . z^`` (a point in front of the limb
    plane looks larger by ``R^2 sin(alpha) / D``, up to ~0.9 km at 6 deg --
    the orthographic silhouette was 0.2-0.6 s off the 3D oracle, sec. 9.6).
    A bin's radius is the largest of the points falling in it and of the grid
    edges crossing it (:func:`_edge_crossings`): the silhouette of the
    piecewise-linear DEM surface (design note sec. 3.2).  ``delta_rho`` is
    measured from the LOLA sphere's own apparent radius at that distance,
    ``R / sqrt(1 - (R / D)^2)`` (the tangent cone, i.e. what the cone
    elements' ``k`` describes), so a smooth sphere gives 0 for any ``D``;
    ``D = inf`` is the orthographic silhouette.  Only points above
    ``R_REF_KM - FLOOR_KM`` and edges between two of them take part.  Bins
    still empty are filled by :func:`_fill_empty`.  Raises ``ValueError`` if the view axis
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
            if not _native_holds(band):
                install_native(band)
            return native.module().limb_silhouette(np.ascontiguousarray(axes), int(n_bins),
                                                   float(distance_km))
    x, y, z = axes[0], axes[1], axes[2]
    r = band.offset_km + band.dn * band.scale_km
    cl = band.cos_lat[band.line_idx]
    px = r * (cl * band.cos_lon[band.col_idx])
    py = r * (cl * band.sin_lon[band.col_idx])
    pz = r * band.sin_lat[band.line_idx]
    qx = (px * x[0] + py * x[1]) + pz * x[2]
    qy = (px * y[0] + py * y[1]) + pz * y[2]
    w = (px * z[0] + py * z[1]) + pz * z[2]
    # Perspective from distance D along -z^; not np.hypot: SIMD, 1 ulp off libm.
    s = np.sqrt(qx * qx + qy * qy) / (1.0 + w / distance_km)
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
    return _fill_empty(rho) - sphere_radius(distance_km)


def sphere_radius(distance_km: float) -> float:
    """Apparent radius of the LOLA sphere seen from ``distance_km``, in km at
    that distance: ``D tan(asin(R / D)) = R / sqrt(1 - (R / D)^2)`` (the
    tangent cone of a sphere); ``R_REF_KM`` for ``D = inf``."""
    return R_REF_KM / math.sqrt(1.0 - (R_REF_KM / distance_km) * (R_REF_KM / distance_km))


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
    """``(n, n_bins)`` limb profiles [km] at each TDB ``et`` (see :func:`silhouette`),
    in perspective from the Moon's distance to the fundamental plane.  With no
    ``band`` under the native backend, the core's own copy of the default band
    file is used (:func:`ensure_native_band`)."""
    axes, dist = limb_axes(et, earth_frame, moon_frame)
    if band is None and native.is_native():
        ensure_native_band()
        with native.PARALLEL_LOCK:
            return np.stack([native.module().limb_silhouette(np.ascontiguousarray(a), int(n_bins),
                                                             float(d))
                             for a, d in zip(axes, dist, strict=True)])
    band = band if band is not None else load_band()
    return np.stack([silhouette(band, a, n_bins, float(d)) for a, d in zip(axes, dist,
                                                                           strict=True)])


# ------------------------------------------------------------ time lattice


@lru_cache(maxsize=256)
def _node_profile(node: int, earth_frame: str, moon_frame: str, band_path: str) -> np.ndarray:
    """The profile at lattice node ``node`` (TDB ``node * LATTICE_S``), cached."""
    band = load_band(band_path)
    axes, dist = limb_axes(np.array([node * LATTICE_S]), earth_frame, moon_frame)
    prof = silhouette(band, axes[0], N_BINS, float(dist[0]))
    prof.setflags(write=False)
    return prof


def profiles_at(et, earth_frame: str = DEFAULT_EARTH_FRAME, moon_frame: str = "MOON_ME",
                band_path: str | None = None) -> np.ndarray:
    """``(n, N_BINS)`` profiles [km] at TDB ``et``, linear in time between the
    cached ``LATTICE_S`` nodes: ``p0 + w (p1 - p0)``, ``w = q - floor(q)``,
    ``q = et / LATTICE_S`` (docs/LIMB_PROFILE.md sec. 3.2, 9.3)."""
    path = str(band_path or default_band_path())
    et = np.atleast_1d(np.asarray(et, dtype=float))
    out = np.empty((len(et), N_BINS))
    for i, t in enumerate(et):
        q = t / LATTICE_S
        k0 = math.floor(q)
        w = q - k0
        p0 = _node_profile(k0, earth_frame, moon_frame, path)
        p1 = _node_profile(k0 + 1, earth_frame, moon_frame, path)
        out[i] = p0 + w * (p1 - p0)
    return out


# ------------------------------------------------------- contact functions
# docs/LIMB_PROFILE.md sec. 3.3.  In the fundamental plane at the observer
# [ES92] eq. 8.353-8.354 (L1', L2' reduced cone radii, both from the LOLA
# sphere K_REF): the Sun's disk has radius R_s = (L1' + L2') / 2, the Moon's
# R_m = (L1' - L2') / 2, and P = (xi - x, eta - y) is the Sun's centre
# relative to the Moon's.  The Moon's silhouette is r_M(psi) = R_m (1 +
# delta_rho(psi) / R_REF_KM).  Totality: the whole solar disk inside the
# silhouette; annularity: the whole silhouette inside the solar disk -- the
# geometric definition of the central contacts [NASA-limb].  For delta_rho = 0
# both reduce to m - |L2'|.


@lru_cache(maxsize=4)
def _unit_circle(n: int) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin of the n bin centres ``-pi + (k + 0.5) 2 pi / n`` through libm
    (``math``), as the native core computes them."""
    step = 2.0 * math.pi / n
    phi = [-math.pi + (k + 0.5) * step for k in range(n)]
    c = np.array([math.cos(v) for v in phi])
    s = np.array([math.sin(v) for v in phi])
    c.setflags(write=False)
    s.setflags(write=False)
    return c, s


def g_total(px, py, r_s, r_m, profiles) -> np.ndarray:
    """``G_T = max_phi |Q| - r_M(arg Q)``, ``Q = P + R_s e(phi)``, per instant
    (arrays of n; ``profiles`` (n, n_bins)); totality <=> ``G_T < 0``.  The
    solar limb is sampled at the profile's bin centres."""
    px, py, r_s, r_m = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (px, py, r_s, r_m))
    profiles = np.atleast_2d(profiles)
    c, s = _unit_circle(profiles.shape[1])
    out = np.empty(len(px))
    for i in range(len(px)):
        qx = px[i] + r_s[i] * c
        qy = py[i] + r_s[i] * s
        rho = delta_rho_at(profiles[i], _atan2(qy, qx))
        out[i] = np.max(np.sqrt(qx * qx + qy * qy) - r_m[i] * (1.0 + rho / R_REF_KM))
    return out


def g_annular(px, py, r_s, r_m, profiles) -> np.ndarray:
    """``G_A = max_psi |M(psi) - P| - R_s``, ``M(psi) = r_M(psi) e(psi)`` at the
    profile's bin centres, per instant; annularity <=> ``G_A < 0``."""
    px, py, r_s, r_m = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (px, py, r_s, r_m))
    profiles = np.atleast_2d(profiles)
    c, s = _unit_circle(profiles.shape[1])
    out = np.empty(len(px))
    for i in range(len(px)):
        rad = r_m[i] * (1.0 + profiles[i] / R_REF_KM)
        dx = rad * c - px[i]
        dy = rad * s - py[i]
        out[i] = np.max(np.sqrt(dx * dx + dy * dy)) - r_s[i]
    return out

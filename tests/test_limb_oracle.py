"""Independent 3D check of the limb-profile contacts (docs/LIMB_PROFILE.md sec. 5.2).

The profile mode reduces everything to the fundamental plane: one perspective
silhouette per instant (``app.limb.silhouette``), the Sun and Moon disks from
the cone radii L1'/L2', the observer's height folded into the fundamental
plane (``geo_to_fund``), and the contact functions G_T / G_A.  This oracle
shares none of that.  From the observer at its height above the ellipsoid
(plain ECEF, ``geodesy_oracle.observer_ecef_km`` [WGS84]; ITRS via ERFA
``c2t06a`` [SOFA]) it casts rays at points on the solar limb (the Sun a
sphere of ``constants.SUN_RADIUS_KM``, apparent positions ``LT+S`` [SPICE]),
marches each ray past the Moon in ``MOON_ME`` and compares its distance from
the Moon's centre with the DEM surface there (LOLA heights, bilinear between
pixel centres [LOLA]).  ``H(t) = max over limb rays of min along the ray of
(|p| - r_surface)``: totality <=> H < 0 (every limb ray blocked); annularity
<=> ``-min over rays`` < 0 (none blocked).

The two methods interpolate the DEM differently (grid edges vs bilinear), so
they differ by a few tens of metres of limb height.  A timing difference is
therefore judged as a limb height, ``|dt| x |dG/dt|`` (the rate at which the
limbs close, km/s): at a grazing contact the limbs close ~10x slower than
mid-path, and the same height difference is ~10x more seconds.  Measured
(this module's sites at their heights, docs sec. 9.10): <= 57 m; gate 75 m,
except Vale's C2 at its height, 94 m, a strict xfail (see ``SITES``).

Slow (each H evaluation marches ~3600 rays x ~2200 steps); the whole module
takes about a minute.  Needs the NAIF kernels and ``kernels.bootstrap --limb``.
"""

from __future__ import annotations

import numpy as np
import pytest
from geodesy_oracle import observer_ecef_km

from app import limb
from app.ephemeris import default_metakernel

GATE_KM = 0.075  # limb-height equivalent of a contact-time difference

requires_limb = pytest.mark.skipif(
    not (default_metakernel().exists() and limb.default_band_path().exists()),
    reason="limb band / lunar kernels not installed (python -m kernels.bootstrap --limb)",
)


class Dem:
    """The limb band as a dense (lines, samples) radius grid [km], NaN outside
    the band; bilinear radius at any MOON_ME point (pixel centres per the PDS
    map projection of the LDEM label, :mod:`kernels.limb_band`)."""

    def __init__(self, band: limb.LimbBand):
        h = band.header
        self.ppd = float(h["map_resolution"])
        self.loff = float(h["line_projection_offset"])
        self.soff = float(h["sample_projection_offset"])
        self.clon = float(h["center_longitude"])
        self.samples = int(h["line_samples"])
        g = np.full((int(h["lines"]), self.samples), np.nan)
        pos = 0
        for line, j0, cnt in band.runs:
            g[line, j0:j0 + cnt] = band.offset_km + band.dn[pos:pos + cnt] * band.scale_km
            pos += cnt
        self.g = g

    def clearance(self, p: np.ndarray) -> np.ndarray:
        """``|p| - r_surface(p)`` [km] for MOON_ME points ``p`` (..., 3)."""
        r = np.linalg.norm(p, axis=-1)
        lat = np.degrees(np.arcsin(p[..., 2] / r))
        lon = np.degrees(np.arctan2(p[..., 1], p[..., 0])) % 360.0
        fi = self.loff - lat * self.ppd
        fj = (lon - self.clon) * self.ppd + self.soff
        i0 = np.clip(np.floor(fi).astype(int), 0, self.g.shape[0] - 2)
        j0 = np.floor(fj).astype(int)
        wi, wj = fi - i0, fj - j0
        ja, jb = j0 % self.samples, (j0 + 1) % self.samples
        g = self.g
        surf = ((1 - wi) * ((1 - wj) * g[i0, ja] + wj * g[i0, jb])
                + wi * ((1 - wj) * g[i0 + 1, ja] + wj * g[i0 + 1, jb]))
        return r - surf


def oracle_h(dem: Dem, et: float, lat_deg: float, lon_deg: float, annular: bool = False,
             n_rays: int = 3600, half_span_km: float = 280.0, step_km: float = 0.25,
             height_m: float = 0.0) -> float:
    """H(et) [km]: negative inside the central phase (see module docstring),
    for the observer ``height_m`` [m] above the WGS-84 ellipsoid.

    ``half_span_km`` covers 9.2 deg of lunar arc either side of each ray's
    closest approach (a 10-km peak shows over the limb from < 6.2 deg);
    ``step_km`` / ``n_rays`` are converged to < 0.07 s at a graze (docs 9.6).
    """
    import erfa
    import spiceypy as spice

    from app import ephemeris as ep
    from app.constants import SUN_RADIUS_KM

    tt2, ut1, xp, yp = ep.earth_rotation_times(np.array([et]))
    rc2t = erfa.c2t06a(ep._J2000_JD, tt2[0], ep._J2000_JD, ut1[0], xp[0], yp[0])
    obs = rc2t.T @ observer_ecef_km(lat_deg, lon_deg, height_m)
    moon, lt = spice.spkpos("MOON", et, "J2000", "LT+S", "EARTH")
    sun, _ = spice.spkpos("SUN", et, "J2000", "LT+S", "EARTH")
    moon_t, sun_t = np.asarray(moon) - obs, np.asarray(sun) - obs
    to_me = spice.pxform("J2000", "MOON_ME", et - lt)
    u = sun_t / np.linalg.norm(sun_t)
    a_s = np.arcsin(SUN_RADIUS_KM / np.linalg.norm(sun_t))
    e1 = np.cross(u, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    d = (np.cos(a_s) * u[None, :]
         + np.sin(a_s) * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
    o = to_me @ (-moon_t)          # observer, Moon-centred, MOON_ME
    d_me = d @ to_me.T             # ray directions, MOON_ME
    s0 = -(d_me @ o)               # each ray's closest approach to the centre
    sig = np.arange(-half_span_km, half_span_km + step_km / 2, step_km)
    worst = np.empty(n_rays)
    for k in range(0, n_rays, 200):
        sl = slice(k, k + 200)
        p = o[None, None, :] + (s0[sl, None, None] + sig[None, :, None]) * d_me[sl, None, :]
        worst[sl] = np.nanmin(dem.clearance(p), axis=1)
    return float(-worst.min() if annular else worst.max())


# Sites: mid-path and within ~1.5 km of a limit, total and annular, at their
# heights above the WGS-84 ellipsoid: ground height (SRTM90, or [Irwin21]'s
# 711 m at Vale) plus the EGM96 geoid undulation, as in
# test_besselian_integration._EB2024_* (docs/LIMB_VALIDATION_SOURCES.md); the
# annular point is on open sea, so its height is the undulation alone.
#   label, T0 (UTC, near the site's maximum), lat, lon [deg], height [m]
SITES = [
    ("2024 Indianapolis (mid-path)", "2024-04-08T19:08:00", 39.77, -86.15, 219.0 - 34.3),
    ("2024 Effingham (~1 km inside the N limit)", "2024-04-08T19:03:38", 39.12, -88.55,
     182.0 - 32.6),
    ("2017 Vale OR (~1.5 km inside the S limit; Irwin et al. 2021)", "2017-08-21T17:25:50",
     43.953028, -117.219389, 711.0 - 17.2),
    ("2023 annular, greatest eclipse (open sea)", "2023-10-14T17:59:27", 11.4, -83.1, 5.4),
    ("2017 Vale OR at sea level", "2017-08-21T17:25:50", 43.953028, -117.219389, 0.0),
]
# At Vale's height the observer is 0.69 km deeper in the path and C2 grazes a
# different stretch of limb, where the two DEM interpolations (grid edges +
# perspective binning vs bilinear) differ by more than the gate: profile -
# oracle = -1.85 s at dG/dt = 0.051 km/s, 94 m (C3: +0.05 s, 7 m).  Not the
# height handling: at the sea-level point on the same line of sight (0.69 km
# from the site, away from the Sun) the two differ by -1.88 s, 85 m, with no
# height code involved; the oracle is converged there (4x rays: 0 ms; 0.1 km
# steps: 6 ms).  docs/LIMB_PROFILE.md sec. 9.10.
_VALE_DEM_XFAIL = pytest.mark.xfail(strict=True, reason=(
    "Vale C2 at 694 m: profile and oracle DEM interpolations differ by 94 m of "
    "limb height (85 m at the equivalent sea-level point), above the 75 m gate"))


@pytest.fixture(scope="module")
def dem():
    return Dem(limb.load_band())


@requires_limb
@pytest.mark.parametrize("site", [
    pytest.param(s, id=s[0], marks=_VALE_DEM_XFAIL if s[0].startswith("2017 Vale OR (") else ())
    for s in SITES
])
def test_profile_contacts_match_the_3d_oracle(site, dem):
    """Each profile-mode contact is bracketed by the oracle within GATE_KM of
    limb height: at ``c -+ GATE_KM / |dG/dt|`` the oracle is outside / inside
    the central phase (C2) or inside / outside (C3)."""
    from app import ephemeris as ep
    from app.besselian import BesselianModel
    from app.circumstances import _profile_g, local_raw

    label, t0, lat, lon, h = site
    ep.load_kernels()
    model = BesselianModel(t0_utc=t0)
    raw = local_raw(model, lat, lon, "profile", h)
    assert raw.central, label
    annular = raw.L2_x > 0
    for c, entering in ((raw.c2, True), (raw.c3, False)):
        g = _profile_g(model, lat, lon, np.array([c - 0.5 / 3600, c + 0.5 / 3600]), h)
        rate = abs(g[1] - g[0]) * 6378.137  # km/s
        dt = GATE_KM / rate
        before = oracle_h(dem, model.et0 + c * 3600 - dt, lat, lon, annular, height_m=h)
        after = oracle_h(dem, model.et0 + c * 3600 + dt, lat, lon, annular, height_m=h)
        if entering:
            assert before > 0 > after, (label, "C2", before, after, dt)
        else:
            assert before < 0 < after, (label, "C3", before, after, dt)

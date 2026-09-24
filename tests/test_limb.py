"""Lunar limb profile (``core/src/limb.cpp``, ``kernels/limb_band.py``; docs/LIMB_PROFILE.md).

The synthetic-DEM tests always run: a sphere, a plateau and a trench on a
small grid, where the silhouette is known in closed form (design note sec. 5.1).
The real-DEM tests need the limb band and the lunar kernels
(``python -m kernels.bootstrap --limb``) and skip cleanly without them.
"""

from __future__ import annotations

import _eclipse as E
import numpy as np
import pytest

from app.core import SpiceError, default_band_path, default_metakernel, load_kernels
from kernels import limb_band

N_BINS = E.LIMB_N_BINS
R_REF_KM = E.LIMB_R_REF_KM

# Earth-facing view: z^ = -x of MOON_ME (Moon -> Sun, the Sun behind the Earth),
# y^ = +z (lunar north), x^ = y^ x z^ = -y.
AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
PPD = 16.0

_LABEL = """PDS_VERSION_ID = PDS3
PRODUCT_ID                = "LDEM_SYN"
    LINES                 = {lines}
    LINE_SAMPLES          = {samples}
    SAMPLE_TYPE           = LSB_INTEGER
    SAMPLE_BITS           = 16
    SCALING_FACTOR        = 0.5
    OFFSET                = 1737400.
 COORDINATE_SYSTEM_NAME       = "MEAN EARTH/POLAR AXIS OF DE421"
 CENTER_LONGITUDE             = 180 <deg>
 MAP_RESOLUTION               = {ppd:g} <pix/deg>
 LINE_PROJECTION_OFFSET       = {loff} <pix>
 SAMPLE_PROJECTION_OFFSET     = {soff} <pix>
"""


def _grid(ppd=PPD):
    lines, samples = int(180 * ppd), int(360 * ppd)
    lat = (lines / 2 - 0.5 - np.arange(lines)) / ppd
    lon = (np.arange(samples) - (samples / 2 - 0.5)) / ppd + 180.0
    return lines, samples, lat, lon


def _write_dem(tmp_path, dn, ppd=PPD):
    lines, samples = dn.shape
    img, lbl = tmp_path / "syn.img", tmp_path / "syn.lbl"
    dn.astype("<i2").tofile(img)
    lbl.write_text(_LABEL.format(lines=lines, samples=samples, ppd=ppd,
                                 loff=lines / 2 - 0.5, soff=samples / 2 - 0.5))
    return img, lbl


def _band(tmp_path, dn, ppd=PPD):
    """Cut the synthetic DEM to a band file and make the core read it."""
    img, lbl = _write_dem(tmp_path, dn, ppd)
    out = tmp_path / "syn.bin"
    limb_band.cut(img, lbl, out)
    E.load_limb_band(str(out))
    return out


def silhouette(axes, n_bins=N_BINS, distance_km=float("inf")):
    """The installed band's profile for ``axes`` (``_eclipse.limb_silhouette``)."""
    return E.limb_silhouette(np.ascontiguousarray(axes, dtype=float), int(n_bins),
                             float(distance_km))


def _psi_of(lat_deg, lon_deg):
    """Fundamental-plane angle of a MOON_ME direction under ``AXES``."""
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    n = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return float(np.arctan2(AXES[1] @ n, AXES[0] @ n))


def test_label_parse_and_band_file_round_trip(tmp_path):
    lines, samples, _lat, _lon = _grid(4.0)
    dn = (np.arange(lines * samples) % 7 - 3).reshape(lines, samples).astype(np.int16)
    img, lbl = _write_dem(tmp_path, dn, 4.0)
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    sha_a = limb_band.cut(img, lbl, a)
    assert sha_a == limb_band.cut(img, lbl, b)  # deterministic bytes
    assert a.read_bytes() == b.read_bytes()
    bf = limb_band.read(a)
    assert bf.header["product_id"] == "LDEM_SYN" and bf.header["band_deg"] == 20.0
    # Every kept pixel is in the band, and nothing in the band was dropped.
    _, _, lat, lon = _grid(4.0)
    inband = (np.abs(np.cos(np.radians(lat))[:, None] * np.cos(np.radians(lon))[None, :])
              <= np.sin(np.radians(20.0)))
    assert int(bf.runs[:, 2].sum()) == int(inband.sum())
    rebuilt = np.zeros_like(dn)
    mask = np.zeros(dn.shape, dtype=bool)
    pos = 0
    for line, j0, cnt in bf.runs:
        rebuilt[line, j0:j0 + cnt] = bf.dn[pos:pos + cnt]
        mask[line, j0:j0 + cnt] = True
        pos += cnt
    assert np.array_equal(mask, inband)
    assert np.array_equal(rebuilt[mask], dn[mask])
    with pytest.raises(ValueError):
        (tmp_path / "bad.bin").write_bytes(b"NOTLIMB!" + a.read_bytes()[8:])
        limb_band.read(tmp_path / "bad.bin")
    with pytest.raises(ValueError):
        (tmp_path / "short.bin").write_bytes(a.read_bytes()[:-2])
        limb_band.read(tmp_path / "short.bin")


def test_sphere_silhouette_is_flat(tmp_path):
    lines, samples, _, _ = _grid()
    _band(tmp_path, np.zeros((lines, samples), dtype=np.int16))
    prof = silhouette(AXES)
    assert prof.shape == (N_BINS,)
    # Sampling: the pixel nearest the limb is <= half a pixel (1/32 deg) off
    # it, i.e. R (1 - cos) <= 0.26 m below the sphere.
    assert np.all(prof <= 1e-9) and np.all(prof > -3e-4)


def test_plateau_and_trench_on_the_limb(tmp_path):
    lines, samples, lat, lon = _grid()
    dn = np.zeros((lines, samples), dtype=np.int16)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    # A 3-km plateau of 0.5 deg radius centred on the limb at (lat 20, lon 90).
    dn[np.hypot(la - 20.0, (lo - 90.0) * np.cos(np.radians(20.0))) < 0.5] = 6000
    # A 2-km trench along the parallel lat = -30 (0.3 deg wide) spanning the
    # whole band in longitude, so no point behind it fills it in; it crosses
    # both limbs (lon 90 and 270).
    dn[np.abs(la + 30.0) < 0.15] = -4000
    _band(tmp_path, dn)
    prof = silhouette(AXES)
    psi = np.linspace(-np.pi, np.pi, N_BINS, endpoint=False) + np.pi / N_BINS
    d_plateau = np.abs(np.angle(np.exp(1j * (psi - _psi_of(20.0, 90.0)))))
    d_trench = np.minimum(*[np.abs(np.angle(np.exp(1j * (psi - _psi_of(-30.0, lo_)))))
                            for lo_ in (90.0, 270.0)])
    assert np.max(prof[d_plateau < np.radians(0.3)]) == pytest.approx(3.0, abs=1e-3)
    assert np.all(prof[d_trench < np.radians(0.05)] == pytest.approx(-2.0, abs=0.01))
    far = (d_plateau > np.radians(1.0)) & (d_trench > np.radians(1.0))
    assert np.all(np.abs(prof[far]) < 1e-3)  # the sphere, less the 0.26 m sampling sag


def test_view_axis_outside_the_band_is_an_error(tmp_path):
    lines, samples, _, _ = _grid(4.0)
    _band(tmp_path, np.zeros((lines, samples), dtype=np.int16), 4.0)
    c, s = np.cos(np.radians(13.0)), np.sin(np.radians(13.0))
    tilted = np.array([[0.0, -1.0, 0.0], [-s, 0.0, c], [-c, 0.0, -s]])  # 13 deg off +x
    with pytest.raises(ValueError, match="leaves the limb band's coverage"):
        silhouette(tilted)  # the 20-deg band covers a view axis <= 12 deg off +x
    # Grid edges cover every bin even far finer than the 4-ppd pixel (0.25 deg).
    assert np.all(np.abs(silhouette(AXES, n_bins=36000)) < 5e-3)


def test_delta_rho_at_is_periodic_linear_interpolation():
    prof = np.array([0.0, 1.0, 3.0, -1.0])
    w = 2 * np.pi / 4
    centres = -np.pi + (np.arange(4) + 0.5) * w
    def at(psi):
        out = E.limb_delta_rho_at(prof, np.atleast_1d(np.asarray(psi, dtype=float)))
        return out if np.ndim(psi) else float(out[0])

    assert np.allclose(at(centres), prof)
    assert at(centres[0] + 0.25 * w) == pytest.approx(0.25)
    # Across the -pi/pi seam: halfway from the last bin centre to the first.
    assert at(np.pi) == pytest.approx(-0.5)
    assert at(-np.pi) == pytest.approx(-0.5)
    assert at(centres[1] + 2 * np.pi) == pytest.approx(1.0)


# --------------------------------------------------------------- real DEM

requires_limb = pytest.mark.skipif(
    not (default_metakernel().exists() and default_band_path().exists()),
    reason="limb band / lunar kernels not installed (python -m kernels.bootstrap --limb)",
)


def _moon_frames_loaded() -> bool:
    load_kernels()
    try:
        E.limb_axes(np.array([0.0]), "ITRS", "MOON_ME")
    except SpiceError:
        return False
    return True


@requires_limb
def test_axes_are_orthonormal_and_follow_the_elements(spice):
    if not _moon_frames_loaded():
        pytest.skip("MOON_ME not defined: rerun kernels.bootstrap --limb")
    et = E.utc_to_et("2024-04-08T18:17:20") + np.linspace(-7200, 7200, 5)
    ax, dist = E.limb_axes(et, "ITRS", "MOON_ME")
    for a in ax:
        assert np.allclose(a @ a.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(a) == pytest.approx(1.0, abs=1e-14)
    # z^ is ~the Earth-facing direction (-x of MOON_ME) within the libration.
    assert np.all(ax[:, 2, 0] < -np.cos(np.radians(12.0)))
    # The same plane as the elements: the Moon-fixed pole's image in the plane
    # follows from axes that are a pure rotation of the Earth-frame (x^, y^)
    # construction, so x = (Moon - O) . x^ reproduces the element x.  The Moon's
    # position and orientation come from spiceypy's own pool (``spice``).
    e = E.besselian_instants(et, "ITRS")
    a_e = E.WGS84_A_KM  # the fundamental-plane unit [WGS84]
    # The viewing distance is the element z [ES92] eq. 8.322-6, in km.
    assert dist == pytest.approx(e["z"] * a_e, rel=1e-12)
    for i, t in enumerate(et):
        moon, lt = spice.spkpos("MOON", float(t), "J2000", "LT+S", "EARTH")
        rot = spice.pxform("J2000", "MOON_ME", float(t) - lt)
        xj, yj = rot.T @ ax[i, 0], rot.T @ ax[i, 1]
        assert np.dot(moon, xj) / a_e == pytest.approx(e["x"][i], abs=1e-9)
        assert np.dot(moon, yj) / a_e == pytest.approx(e["y"][i], abs=1e-9)


@requires_limb
def test_real_profile_is_plausible():
    from app.core import ensure_limb_band

    if not _moon_frames_loaded():
        pytest.skip("MOON_ME not defined: rerun kernels.bootstrap --limb")
    ensure_limb_band()
    ax, dist = E.limb_axes(np.array([E.utc_to_et("2024-04-08T18:17:20")]), "ITRS", "MOON_ME")
    prof = silhouette(ax[0], distance_km=dist[0])
    # The silhouette of the LOLA Moon: within the DEM's +-10 km, and, being an
    # upper envelope over the limb zone, above the reference sphere on average
    # (0.46 km for this instant; docs/LIMB_PROFILE.md sec. 9).
    assert -10.0 < prof.min() < -1.0 and 1.0 < prof.max() < 10.0
    assert 0.0 < prof.mean() < 1.0


def test_perspective_silhouette(tmp_path):
    """A peak in front of the limb plane looks larger in perspective by the
    factor 1 / (1 + w / D), w = p . z^ (docs/LIMB_PROFILE.md sec. 3.2, 9.6);
    a smooth sphere stays flat for any distance (its own perspective radius
    ``sphere_radius(D)`` is subtracted)."""
    lines, samples, lat, lon = _grid()
    dn = np.zeros((lines, samples), dtype=np.int16)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    # A 10-km plateau 5 deg in front of the limb (toward the observer, +x),
    # 1 deg across, on the equator.
    dn[np.hypot(la, lo - 85.0) < 0.5] = 20000
    _band(tmp_path, dn)
    D = 384400.0
    ortho = silhouette(AXES)
    persp = silhouette(AXES, distance_km=D)
    # Closed form over the plateau's pixels: orthographic |q|, perspective
    # |q| / (1 + w / D), minus the sphere's own apparent radius.
    sel = dn > 0
    r = R_REF_KM + 10.0
    lat_r, lon_r = np.radians(la[sel]), np.radians(lo[sel])
    p = r * np.stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r),
                      np.sin(lat_r)], axis=-1)
    q = np.hypot(p @ AXES[0], p @ AXES[1])
    w = p @ AXES[2]
    assert ortho.max() == pytest.approx(q.max() - R_REF_KM, abs=1e-6)
    assert persp.max() == pytest.approx((q / (1.0 + w / D)).max() - E.limb_sphere_radius(D),
                                        abs=1e-6)
    # The plateau sits 4.5-5.5 deg in front of the limb; seen in perspective
    # its highest point shows 0.61 km taller than orthographically.
    assert persp.max() - ortho.max() == pytest.approx(0.608, abs=0.005)
    (tmp_path / "flat").mkdir()
    _band(tmp_path / "flat", np.zeros((lines, samples), dtype=np.int16))
    assert np.all(np.abs(silhouette(AXES, distance_km=D)) < 1e-3)


def test_contact_functions_reduce_to_the_mean_limb():
    """With a smooth Moon (delta_rho = 0), G_T = m + L2' and G_A = m - L2'
    (docs sec. 3.3), up to the limb sampling's R_s (1 - cos(pi / n)) bias."""
    rng = np.random.default_rng(2)
    n = 50
    px, py = rng.normal(scale=0.004, size=(2, n))
    r_s = rng.uniform(0.26, 0.28, n)
    r_m = r_s + rng.uniform(-0.01, 0.01, n)
    prof = np.zeros((n, N_BINS))
    m = np.hypot(px, py)
    L2p = r_s - r_m
    bias = 0.28 * (1 - np.cos(np.pi / N_BINS))
    assert np.allclose(E.limb_g_total(px, py, r_s, r_m, prof), m + L2p, atol=bias + 1e-15)
    assert np.allclose(E.limb_g_annular(px, py, r_s, r_m, prof), m - L2p, atol=bias + 1e-15)
    # A valley exactly where the Sun's limb touches opens the contact: G_T grows.
    i = 0
    psi = np.arctan2(py[i], px[i])
    k = int(np.floor((psi + np.pi) / (2 * np.pi) * N_BINS))
    prof[i, k - 3:k + 4] = -1.0
    assert E.limb_g_total(px[:1], py[:1], r_s[:1], r_m[:1], prof[:1])[0] > m[0] + L2p[0]


@requires_limb
def test_profile_search_window_widens_until_its_ends_are_outside():
    """A window whose ends are both inside totality is pushed out until they
    are not, and finds the same contacts as the normal search."""
    from app.besselian import BesselianModel
    from app.circumstances import local_raw

    if not _moon_frames_loaded():
        pytest.skip("MOON_ME not defined: rerun kernels.bootstrap --limb")
    model = BesselianModel(t0_utc="2024-04-08T19:08:00")
    ref = local_raw(model, 39.77, -86.15, "profile")
    central, c2, c3 = E.profile_contacts(model.et0, model.earth_frame, model.half_window_hours,
                                         39.77, -86.15, ref.c2 + 20 / 3600, ref.c3 - 20 / 3600)
    assert central
    assert c2 == pytest.approx(ref.c2, abs=1e-8) and c3 == pytest.approx(ref.c3, abs=1e-8)


@requires_limb
@pytest.mark.parametrize(
    ("lat", "mean_type", "profile_type"),
    [
        # Just inside the mean northern limit near Indianapolis: a limb valley
        # keeps a bead of the Sun in view, so the profile has no totality (the
        # 3D oracle agrees: its H stays >= +331 m, the profile's G >= +296 m).
        (40.464, "total", "partial"),
        # Just outside the mean southern limit: limb peaks cover the Sun's
        # last sliver, so the profile has a brief totality (oracle -264 m,
        # profile -281 m at the deepest).
        (38.448, "partial", "total"),
    ],
)
def test_profile_sets_the_type_magnitude_stays_mean_limb(lat, mean_type, profile_type):
    """Where the limb profile reverses the mean limb's verdict at a 2024 limit
    site, the type follows the profile but the magnitude and obscuration are
    the mean limb's, the published definitions [Espenak] (none exists for a
    real limb; [EB2024] corrects contact times only). So in the graze zone a
    profile partial can report a magnitude > 1."""
    from app.besselian import BesselianModel
    from app.circumstances import local_circumstances

    if not _moon_frames_loaded():
        pytest.skip("MOON_ME not defined: rerun kernels.bootstrap --limb")
    model = BesselianModel(t0_utc="2024-04-08T19:08:00")
    mean = local_circumstances(model, lat, -86.15, "mean")
    prof = local_circumstances(model, lat, -86.15, "profile")
    assert (mean["type"], prof["type"]) == (mean_type, profile_type)
    assert prof["magnitude"] == mean["magnitude"]
    assert prof["obscuration"] == mean["obscuration"]
    assert ("C2" in prof) == (profile_type == "total")

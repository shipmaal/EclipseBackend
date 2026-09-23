"""Lunar limb profile (``app/limb.py``, ``kernels/limb_band.py``; docs/LIMB_PROFILE.md).

The synthetic-DEM tests always run: a sphere, a plateau and a trench on a
small grid, where the silhouette is known in closed form (design note sec. 5.1).
The real-DEM tests need the limb band and the lunar kernels
(``python -m kernels.bootstrap --limb``) and skip cleanly without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import limb
from app.ephemeris import default_metakernel
from kernels import limb_band

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
    img, lbl = _write_dem(tmp_path, dn, ppd)
    out = tmp_path / "syn.bin"
    limb_band.cut(img, lbl, out)
    return limb.band_from_file(limb_band.read(out)), out


@pytest.fixture
def python_backend(monkeypatch):
    from app import native

    monkeypatch.setattr(native, "BACKEND", "python")


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


def test_sphere_silhouette_is_flat(tmp_path, python_backend):
    lines, samples, _, _ = _grid()
    band, _ = _band(tmp_path, np.zeros((lines, samples), dtype=np.int16))
    prof = limb.silhouette(band, AXES)
    assert prof.shape == (limb.N_BINS,)
    # Sampling: the pixel nearest the limb is <= half a pixel (1/32 deg) off
    # it, i.e. R (1 - cos) <= 0.26 m below the sphere.
    assert np.all(prof <= 1e-9) and np.all(prof > -3e-4)


def test_plateau_and_trench_on_the_limb(tmp_path, python_backend):
    lines, samples, lat, lon = _grid()
    dn = np.zeros((lines, samples), dtype=np.int16)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    # A 3-km plateau of 0.5 deg radius centred on the limb at (lat 20, lon 90).
    dn[np.hypot(la - 20.0, (lo - 90.0) * np.cos(np.radians(20.0))) < 0.5] = 6000
    # A 2-km trench along the parallel lat = -30 (0.3 deg wide) spanning the
    # whole band in longitude, so no point behind it fills it in; it crosses
    # both limbs (lon 90 and 270).
    dn[np.abs(la + 30.0) < 0.15] = -4000
    band, _ = _band(tmp_path, dn)
    prof = limb.silhouette(band, AXES)
    psi = np.linspace(-np.pi, np.pi, limb.N_BINS, endpoint=False) + np.pi / limb.N_BINS
    d_plateau = np.abs(np.angle(np.exp(1j * (psi - _psi_of(20.0, 90.0)))))
    d_trench = np.minimum(*[np.abs(np.angle(np.exp(1j * (psi - _psi_of(-30.0, lo_)))))
                            for lo_ in (90.0, 270.0)])
    assert np.max(prof[d_plateau < np.radians(0.3)]) == pytest.approx(3.0, abs=1e-3)
    assert np.all(prof[d_trench < np.radians(0.05)] == pytest.approx(-2.0, abs=0.01))
    far = (d_plateau > np.radians(1.0)) & (d_trench > np.radians(1.0))
    assert np.all(np.abs(prof[far]) < 1e-3)  # the sphere, less the 0.26 m sampling sag


def test_view_axis_outside_the_band_is_an_error(tmp_path, python_backend):
    lines, samples, _, _ = _grid(4.0)
    band, _ = _band(tmp_path, np.zeros((lines, samples), dtype=np.int16), 4.0)
    c, s = np.cos(np.radians(13.0)), np.sin(np.radians(13.0))
    tilted = np.array([[0.0, -1.0, 0.0], [-s, 0.0, c], [-c, 0.0, -s]])  # 13 deg off +x
    with pytest.raises(ValueError, match="band covers 12"):
        limb.silhouette(band, tilted)
    # Grid edges cover every bin even far finer than the 4-ppd pixel (0.25 deg).
    assert np.all(np.abs(limb.silhouette(band, AXES, n_bins=36000)) < 5e-3)


def test_delta_rho_at_is_periodic_linear_interpolation():
    prof = np.array([0.0, 1.0, 3.0, -1.0])
    w = 2 * np.pi / 4
    centres = -np.pi + (np.arange(4) + 0.5) * w
    assert np.allclose(limb.delta_rho_at(prof, centres), prof)
    assert limb.delta_rho_at(prof, centres[0] + 0.25 * w) == pytest.approx(0.25)
    # Across the -pi/pi seam: halfway from the last bin centre to the first.
    assert limb.delta_rho_at(prof, np.pi) == pytest.approx(-0.5)
    assert limb.delta_rho_at(prof, -np.pi) == pytest.approx(-0.5)
    assert limb.delta_rho_at(prof, centres[1] + 2 * np.pi) == pytest.approx(1.0)


# --------------------------------------------------------------- real DEM

requires_limb = pytest.mark.skipif(
    not (default_metakernel().exists() and limb.default_band_path().exists()),
    reason="limb band / lunar kernels not installed (python -m kernels.bootstrap --limb)",
)


def _moon_frames_loaded(eph) -> bool:
    import spiceypy

    eph.load_kernels()
    try:
        spiceypy.pxform("J2000", "MOON_ME", 0.0)
    except spiceypy.utils.exceptions.SpiceyError:
        return False
    return True


@requires_limb
def test_axes_are_orthonormal_and_follow_the_elements(python_backend):
    from app import ephemeris as eph

    if not _moon_frames_loaded(eph):
        pytest.skip("MOON_ME not defined: rerun kernels.bootstrap --limb")
    et = eph.utc_to_et("2024-04-08T18:17:20") + np.linspace(-7200, 7200, 5)
    ax, dist = eph.limb_axes(et)
    for a in ax:
        assert np.allclose(a @ a.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(a) == pytest.approx(1.0, abs=1e-14)
    # z^ is ~the Earth-facing direction (-x of MOON_ME) within the libration.
    assert np.all(ax[:, 2, 0] < -np.cos(np.radians(12.0)))
    # The same plane as the elements: the Moon-fixed pole's image in the plane
    # follows from axes that are a pure rotation of the Earth-frame (x^, y^)
    # construction, so x = (Moon - O) . x^ reproduces the element x.
    import spiceypy

    e = eph.besselian_instants(et, "ITRS")
    # The viewing distance is the element z [ES92] eq. 8.322-6, in km.
    assert dist == pytest.approx(e["z"] * eph.earth_equatorial_radius_km(), rel=1e-12)
    for i, t in enumerate(et):
        moon, lt = spiceypy.spkpos("MOON", float(t), "J2000", "LT+S", "EARTH")
        rot = spiceypy.pxform("J2000", "MOON_ME", float(t) - lt)
        xj, yj = rot.T @ ax[i, 0], rot.T @ ax[i, 1]
        a_e = eph.earth_equatorial_radius_km()
        assert np.dot(moon, xj) / a_e == pytest.approx(e["x"][i], abs=1e-9)
        assert np.dot(moon, yj) / a_e == pytest.approx(e["y"][i], abs=1e-9)


@requires_limb
def test_real_profile_is_plausible(python_backend):
    from app import ephemeris as eph

    if not _moon_frames_loaded(eph):
        pytest.skip("MOON_ME not defined: rerun kernels.bootstrap --limb")
    prof = limb.limb_profiles(eph.utc_to_et("2024-04-08T18:17:20"))[0]
    # The silhouette of the LOLA Moon: within the DEM's +-10 km, and, being an
    # upper envelope over the limb zone, above the reference sphere on average
    # (0.46 km for this instant; docs/LIMB_PROFILE.md sec. 9).
    assert -10.0 < prof.min() < -1.0 and 1.0 < prof.max() < 10.0
    assert 0.0 < prof.mean() < 1.0


def test_empty_bins_are_filled_by_periodic_linear_interpolation():
    rho = np.r_[1.0, -np.inf, -np.inf, 4.0, 5.0, -np.inf, np.arange(7.0, 40.0), -np.inf]
    out = limb._fill_empty(rho)
    assert out[1:3] == pytest.approx([2.0, 3.0])
    assert out[5] == pytest.approx(6.0)
    assert out[-1] == pytest.approx(20.0)  # across the seam: halfway 39 -> 1
    assert np.array_equal(out[np.isfinite(rho)], rho[np.isfinite(rho)])
    with pytest.raises(ValueError, match="too many empty bins"):
        limb._fill_empty(np.r_[rho, np.full(3, -np.inf)])


def test_perspective_silhouette(tmp_path, python_backend):
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
    band, _ = _band(tmp_path, dn)
    D = 384400.0
    ortho = limb.silhouette(band, AXES)
    persp = limb.silhouette(band, AXES, distance_km=D)
    # Closed form over the plateau's pixels: orthographic |q|, perspective
    # |q| / (1 + w / D), minus the sphere's own apparent radius.
    sel = dn > 0
    r = limb.R_REF_KM + 10.0
    lat_r, lon_r = np.radians(la[sel]), np.radians(lo[sel])
    p = r * np.stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r),
                      np.sin(lat_r)], axis=-1)
    q = np.hypot(p @ AXES[0], p @ AXES[1])
    w = p @ AXES[2]
    assert ortho.max() == pytest.approx(q.max() - limb.R_REF_KM, abs=1e-6)
    assert persp.max() == pytest.approx((q / (1.0 + w / D)).max() - limb.sphere_radius(D),
                                        abs=1e-6)
    # The plateau sits 4.5-5.5 deg in front of the limb; seen in perspective
    # its highest point shows 0.61 km taller than orthographically.
    assert persp.max() - ortho.max() == pytest.approx(0.608, abs=0.005)
    (tmp_path / "flat").mkdir()
    flat, _ = _band(tmp_path / "flat", np.zeros((lines, samples), dtype=np.int16))
    assert np.all(np.abs(limb.silhouette(flat, AXES, distance_km=D)) < 1e-3)


def test_contact_functions_reduce_to_the_mean_limb():
    """With a smooth Moon (delta_rho = 0), G_T = m + L2' and G_A = m - L2'
    (docs sec. 3.3), up to the limb sampling's R_s (1 - cos(pi / n)) bias."""
    rng = np.random.default_rng(2)
    n = 50
    px, py = rng.normal(scale=0.004, size=(2, n))
    r_s = rng.uniform(0.26, 0.28, n)
    r_m = r_s + rng.uniform(-0.01, 0.01, n)
    prof = np.zeros((n, limb.N_BINS))
    m = np.hypot(px, py)
    L2p = r_s - r_m
    bias = 0.28 * (1 - np.cos(np.pi / limb.N_BINS))
    assert np.allclose(limb.g_total(px, py, r_s, r_m, prof), m + L2p, atol=bias + 1e-15)
    assert np.allclose(limb.g_annular(px, py, r_s, r_m, prof), m - L2p, atol=bias + 1e-15)
    # A valley exactly where the Sun's limb touches opens the contact: G_T grows.
    i = 0
    psi = np.arctan2(py[i], px[i])
    k = int(np.floor((psi + np.pi) / (2 * np.pi) * limb.N_BINS))
    prof[i, k - 3:k + 4] = -1.0
    assert limb.g_total(px[:1], py[:1], r_s[:1], r_m[:1], prof[:1])[0] > m[0] + L2p[0]


@requires_limb
def test_profile_search_window_widens_until_its_ends_are_outside():
    """A window whose ends are both inside totality is pushed out until they
    are not, and finds the same contacts as the normal search."""
    from app import ephemeris as eph
    from app.besselian import BesselianModel
    from app.circumstances import _profile_contacts, local_raw

    if not _moon_frames_loaded(eph):
        pytest.skip("MOON_ME not defined: rerun kernels.bootstrap --limb")
    model = BesselianModel(t0_utc="2024-04-08T19:08:00")
    ref = local_raw(model, 39.77, -86.15, "profile")
    central, c2, c3 = _profile_contacts(model, 39.77, -86.15, ref.c2 + 20 / 3600,
                                        ref.c3 - 20 / 3600)
    assert central
    assert c2 == pytest.approx(ref.c2, abs=1e-8) and c3 == pytest.approx(ref.c3, abs=1e-8)

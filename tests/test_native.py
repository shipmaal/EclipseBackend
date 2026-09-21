"""Phase-0 acceptance tests for the native core ``_eclipse`` (docs/CPP_ROADMAP.md §5).

The module is built by ``uv sync`` (scikit-build-core + CMake). These tests pin
the vendored toolkit versions to the Python oracle's, prove that a SPICE error
is a Python exception rather than a process abort, and assert bit-level parity
of the SPICE calls the elements layer will use.

Note: the extension links its own CSPICE statically, so it has its *own* kernel
pool — furnishing here does not affect spiceypy's pool and vice versa.
"""

from __future__ import annotations

import os

import erfa
import numpy as np
import pytest
import spiceypy

from app.ephemeris import default_metakernel

native = pytest.importorskip("_eclipse", reason="native core not built (run `uv sync`)")

requires_kernels = pytest.mark.skipif(
    not default_metakernel().exists(),
    reason="SPICE kernels not downloaded (run `python -m kernels.bootstrap`)",
)

# Reference-eclipse instants (near greatest eclipse) used throughout the suite.
REFERENCE_UTC = ["2017-08-21T18:26:40", "2023-10-14T18:00:41", "2024-04-08T18:17:20"]


def test_vendored_toolkits_match_the_python_oracle():
    # Same CSPICE build as spiceypy and the same liberfa as pyerfa: the elements
    # layer can then be required to agree to ~1e-15, not "numerically".
    assert native.toolkit_version() == "CSPICE_N0067"
    assert native.toolkit_version() == spiceypy.tkvrsn("TOOLKIT")
    assert native.erfa_version() == erfa.version.erfa_version


def test_spice_error_is_an_exception_not_an_abort():
    native.kclear()
    assert native.kernel_count() == 0
    with pytest.raises(native.SpiceError, match=r"SPICE\(NOLOADEDFILES\)") as exc_info:
        native.body_position("SUN", 0.0, "J2000", "NONE", "EARTH")
    assert isinstance(exc_info.value, RuntimeError)
    # The error state was reset: the toolkit is still usable afterwards.
    assert native.toolkit_version() == "CSPICE_N0067"
    if default_metakernel().exists():  # leave the pool as other tests expect it
        native.furnish(str(default_metakernel()))


def test_bad_time_string_raises():
    with pytest.raises(native.SpiceError, match="SPICE"):
        native.str_to_et("not a date")


@requires_kernels
def test_time_and_position_parity_with_spiceypy():
    mk = str(default_metakernel())
    native.kclear()
    native.furnish(mk)
    spiceypy.furnsh(mk)
    try:
        for utc in REFERENCE_UTC:
            et = native.str_to_et(utc)
            assert et == spiceypy.str2et(utc)  # exact: same LSK, same code
            assert native.et_to_utc_iso(et, 6) == spiceypy.et2utc(et, "ISOC", 6)
            for body in ("SUN", "MOON"):
                ours = native.body_position(body, et, "J2000", "LT+S", "EARTH")
                ref, lt = spiceypy.spkpos(body, et, "J2000", "LT+S", "EARTH")
                assert list(ours.km) == list(ref), (utc, body)  # bit-identical
                assert ours.light_time_s == lt
    finally:
        native.kclear()
        spiceypy.unload(mk)


# ---------------------------------------------------------------- phase 1 parity
# Roadmap §4 gates. Measured residuals on x86-64/AVX-512 (NumPy SIMD arctan2,
# hypot, tan, arcsin differ from libm by 1 ulp; everything else bit-identical):
# x ~5e-14, mu ~1e-13 deg, times / delta-T exactly 0. On Apple arm64 NumPy's
# own build contracts the multiply-add inside ``np.interp`` into an FMA (the
# C++ is compiled with -ffp-contract=off), so the EOP columns and everything
# downstream of UT1 differ by 1 ulp there; those asserts use the 1e-9 s gate
# with a 1e-12 s / 1e-18 rad record of the actual residual.
_XY_TOL = 1e-13       # x, y, z, l1, l2 [Earth radii]; tan_f1/2
_ANG_TOL = 1e-11      # d, mu, sub-solar lon/lat [deg]
_TIME_TOL_S = 1e-9    # roadmap gate for the time layer
_EOP_TOL_S = 1e-12    # UT1-UTC residual actually observed (0 on x86-64, 1 ulp on arm64)
_EOP_TOL_RAD = 1e-18  # polar motion, same

_WINDOWS = [  # ±3 h around greatest eclipse, plus a pre-IERS epoch (model delta-T)
    ("2017-08-21T15:26:40", "2017-08-21T21:26:40"),
    ("2023-10-14T15:00:41", "2023-10-14T21:00:41"),
    ("2024-04-08T15:17:20", "2024-04-08T21:17:20"),
    ("1919-05-29T10:08:00", "1919-05-29T16:08:00"),
]


@pytest.fixture
def oracle(monkeypatch):
    """The pure-Python ``app.ephemeris`` regardless of ``ECLIPSE_BACKEND``."""
    from app import ephemeris
    from app import native as native_mod

    monkeypatch.setattr(native_mod, "BACKEND", "python")
    return ephemeris


@pytest.fixture
def native_pool():
    """``_eclipse`` with the IERS table and (if present) the kernels loaded."""
    from app import native as native_mod

    mod = native_mod.raw_module()
    if default_metakernel().exists():
        mod.kclear()
        mod.furnish(str(default_metakernel()))
    return mod


def test_delta_t_model_parity(native_pool):
    from app.deltat import decimal_year_from_jd, delta_t_seconds

    years = np.linspace(-1500.0, 3200.0, 200_001)
    assert np.array_equal(native_pool.delta_t_seconds(years), delta_t_seconds(years))  # exact
    jd = np.linspace(1_000_000.0, 3_000_000.0, 1001)
    assert np.array_equal(native_pool.decimal_year_from_jd(jd), decimal_year_from_jd(jd))


def test_eop_parity(native_pool):
    from app.eop import eop, iers_mjd_range

    lo, hi = iers_mjd_range()
    assert native_pool.eop_mjd_range() == (lo, hi)
    mjd = np.concatenate([np.linspace(lo - 100.0, hi + 100.0, 50_001), [lo, hi, 60408.0]])
    (xp, yp, dut1), (rxp, ryp, rdut1) = native_pool.eop(mjd), eop(mjd)
    assert np.max(np.abs(xp - rxp)) <= _EOP_TOL_RAD
    assert np.max(np.abs(yp - ryp)) <= _EOP_TOL_RAD
    assert np.max(np.abs(dut1 - rdut1)) <= _EOP_TOL_S
    # Endpoint hold and exact-node semantics are value-for-value identical.
    assert dut1[-3] == rdut1[-3] and dut1[-2] == rdut1[-2] and dut1[0] == rdut1[0]


@requires_kernels
def test_time_scale_parity(oracle, native_pool):
    oracle.load_kernels()
    for s in ["2017-08-21T18:26:40", "2023-10-14T18:00:41.5", "2024-04-08T18:17:20",
              "2016-12-31T23:59:60", "1972-06-30T12:00:00", "1919-05-29T13:08:00",
              "2200-01-01T00:00:00", "0500-01-01T00:00:00", "1600-06-01T00:00:00"]:
        et = native_pool.utc_to_et(s)
        assert et == oracle.utc_to_et(s), s  # exact
        assert native_pool.et_to_utc(et) == oracle.et_to_utc(et), s
    with pytest.raises(native_pool.SpiceError):
        native_pool.utc_to_et("not a date")


@requires_kernels
def test_earth_rotation_parity(oracle, native_pool):
    oracle.load_kernels()
    et0, et1 = oracle.utc_to_et("1900-01-01T00:00:00"), oracle.utc_to_et("2100-01-01T00:00:00")
    et = np.linspace(et0, et1, 20_001)
    (tt2, ut1, xp, yp), (rtt2, rut1, rxp, ryp) = (
        native_pool.earth_rotation_times(et),
        oracle.earth_rotation_times(et),
    )
    assert np.array_equal(tt2, rtt2)  # exact: et / 86400
    assert np.max(np.abs(ut1 - rut1)) * 86400.0 <= _EOP_TOL_S  # gate 1e-9 s; see header
    assert np.max(np.abs(xp - rxp)) <= _EOP_TOL_RAD
    assert np.max(np.abs(yp - ryp)) <= _EOP_TOL_RAD
    assert np.max(np.abs(native_pool.tt_minus_ut1(et) - oracle.tt_minus_ut1(et))) <= _EOP_TOL_S


def _ephemeris_covers(oracle, et: float) -> bool:
    """False when the loaded SPK lacks the epoch (mirror DE432s spans 1949-2050,
    so the 1919 window is skipped there, as test_besselian_integration does)."""
    import spiceypy

    try:
        spiceypy.spkpos("SUN", et, "J2000", "LT+S", "EARTH")
        return True
    except spiceypy.utils.exceptions.SpiceyError:
        return False


def _frames_available(oracle):
    import spiceypy

    frames = ["ITRS", "TOD", "IAU_EARTH"]
    try:
        spiceypy.spkpos("SUN", oracle.utc_to_et("2024-04-08T18:00:00"), "ITRF93", "LT+S", "EARTH")
        frames.append("ITRF93")
    except spiceypy.utils.exceptions.SpiceyError:
        pass
    return frames


@requires_kernels
def test_besselian_elements_parity(oracle, native_pool):
    oracle.load_kernels()
    worst: dict[str, float] = {}
    for utc0, utc1 in _WINDOWS:
        et = np.linspace(oracle.utc_to_et(utc0), oracle.utc_to_et(utc1), 241)
        if not _ephemeris_covers(oracle, float(et[0])):
            continue
        for frame in _frames_available(oracle):
            if frame == "ITRF93" and utc0 < "1950":
                continue  # the binary Earth PCK starts in 1962
            ours = native_pool.besselian_instants(et, frame)
            ref = oracle.besselian_instants(et, frame)
            assert ours.keys() == ref.keys()
            for k in ref:
                tol = _ANG_TOL if k in ("d", "mu") else _XY_TOL
                diff = float(np.max(np.abs(ours[k] - ref[k])))
                worst[k] = max(worst.get(k, 0.0), diff)
                assert diff <= tol, (utc0, frame, k, diff)
    # Stated so a regression that stays inside the gate is still visible.
    assert worst["x"] < 2e-13 and worst["mu"] < 5e-13


@requires_kernels
def test_sub_solar_parity(oracle, native_pool):
    oracle.load_kernels()
    for utc0, utc1 in _WINDOWS:
        et = np.linspace(oracle.utc_to_et(utc0), oracle.utc_to_et(utc1), 241)
        if not _ephemeris_covers(oracle, float(et[0])):
            continue
        for frame in ("ITRS", "TOD", "IAU_EARTH"):
            lon, lat = native_pool.sub_solar_points(et, frame)
            rlon, rlat = oracle.sub_solar_points(et, frame)
            assert np.max(np.abs(lon - rlon)) <= _ANG_TOL, (utc0, frame)
            assert np.max(np.abs(lat - rlat)) <= _ANG_TOL, (utc0, frame)


@requires_kernels
def test_backend_switch_dispatches_to_native(monkeypatch, native_pool):
    from app import ephemeris
    from app import native as native_mod

    monkeypatch.setattr(native_mod, "BACKEND", "native")
    ephemeris.load_kernels()
    et = np.array([native_pool.utc_to_et("2024-04-08T18:17:20")])
    via_app = ephemeris.besselian_instants(et, "ITRS")
    direct = native_pool.besselian_instants(et, "ITRS")
    for k in direct:
        assert np.array_equal(via_app[k], direct[k]), k
    bi = ephemeris.besselian_instant(float(et[0]), "ITRS")
    assert bi.x == direct["x"][0] and bi.mu == direct["mu"][0]
    pre_iers = "1919-05-29T13:08:00"
    assert ephemeris.utc_to_et(pre_iers) == native_pool.utc_to_et(pre_iers)


def test_native_backend_raises_spiceypy_exception_types(monkeypatch, native_pool):
    """app.main / app.besselian catch spiceypy.utils.exceptions.SpiceyError (and the
    frame probe in test_frame_consistency catches it at collection), so the native
    backend must raise the same class the Python backend would."""
    import spiceypy.utils.exceptions as spice_exc

    from app import ephemeris
    from app import native as native_mod

    monkeypatch.setattr(native_mod, "BACKEND", "native")
    with pytest.raises(spice_exc.SpiceUNPARSEDTIME) as exc_info:
        ephemeris.utc_to_et("not a date")
    assert isinstance(exc_info.value, spice_exc.SpiceyError)
    assert isinstance(exc_info.value.__cause__, native_pool.SpiceError)
    # The raw module still raises its own type with the four fields attached.
    with pytest.raises(native_pool.SpiceError) as raw:
        native_pool.str_to_et("not a date")
    assert raw.value.short_message == "SPICE(UNPARSEDTIME)"
    assert raw.value.traceback_text and raw.value.long_message


def test_unknown_frame_is_a_value_error(native_pool):
    with pytest.raises(ValueError, match="earth_frame"):
        native_pool.besselian_instants(np.array([0.0]), "GCRS")


# ---------------------------------------------------------------- phase 2: numerics
def test_numerics_parity(native_pool):
    """eclipse/numerics.hpp against app.numerics and NumPy: exact (roadmap §4 —
    same algorithm, same iteration count), driven by the SAME Python objective so
    the C++ loop's arithmetic is the only variable."""
    from app import numerics

    # np.arange's fill rule (t[i] = start + i*((start+step)-start), not start + i*step).
    # Bit-identical on x86-64. NumPy's own fill kernel is a single C expression, so
    # on Apple arm64 its build contracts `start + i*delta` into an FMA (same
    # mechanism as np.interp in the EOP test) while libeclipse compiles with
    # -ffp-contract=off. The FMA skips the rounding of the product i*delta, whose
    # magnitude is up to 2*hw, so nodes can differ by one ulp of that product plus
    # one of the result (measured on the arm64 runner: 8.9e-16 at |t| < 4). Assert
    # the rule (length, first two nodes exact) and the fill to that bound; the
    # contact times downstream move by ~1e-15 h, far inside the 1e-9 h gate.
    for start, stop, step in [(-5, 5 + 1e-9, 1 / 60), (-2.0, 2.0 + 1e-9, 2.0 / 60.0),
                              (-90.0, 90.0 + 1e-9, 0.5)]:
        ours, ref = native_pool.np_arange(start, stop, step), np.arange(start, stop, step)
        assert ours.shape == ref.shape, (start, stop, step)
        assert np.array_equal(ours[:2], ref[:2]), (start, stop, step)
        bound = np.spacing(np.abs(ref - start)) + np.spacing(np.abs(ref))
        assert np.all(np.abs(ours - ref) <= bound), (start, stop, step)
    assert native_pool.np_arange(3.0, 1.0, 1.0).size == 0

    # np.remainder on a grid with negatives, exact multiples and signed zeros.
    a = np.concatenate([np.linspace(-1080.0, 1080.0, 4321), [-0.0, 0.0, 360.0, -360.0, 1e-300]])
    for b in (360.0, -360.0, 24.0):
        ours = np.array([native_pool.np_remainder(float(v), b) for v in a])
        ref = np.remainder(a, b)
        assert np.array_equal(ours, ref), b
        assert np.array_equal(np.signbit(ours), np.signbit(ref)), b  # +0 / -0 too

    # sign_changes: exact zero (direction from the next sample) and a NaN pair.
    t = np.arange(8.0)
    f = np.array([1.0, 0.0, 2.0, -1.0, np.nan, -3.0, 4.0, 0.0])
    assert native_pool.sign_changes(t, f) == numerics.sign_changes(t, f)
    assert native_pool.sign_changes(t, f) == [(1, True), (2, False), (5, True)]

    # bisect: 200 brackets of cos(t) - t, bit-identical, 31 objective calls each.
    calls = {"py": 0, "cc": 0}

    def make(key):
        def g(x):
            calls[key] += 1
            return np.cos(x) - x
        return g

    lo = np.linspace(-0.5, 0.7, 200)
    hi = lo + np.linspace(1.3, 2.0, 200)  # every bracket holds the root 0.739...
    ours = native_pool.bisect(make("cc"), lo, hi)
    ref = numerics.bisect(make("py"), lo, hi)
    assert np.array_equal(ours, ref)
    assert calls == {"py": 31, "cc": 31}
    assert np.max(np.abs(ours - 0.7390851332151607)) < 1e-8

    # parabolic_minimum: shifted parabolas plus a non-polynomial bowl, 10 calls each.
    calls = {"py": 0, "cc": 0}
    centres = np.linspace(-1.0, 1.0, 50)

    def make2(key):
        def g(x):
            calls[key] += 1
            xx = x.reshape(3, -1)
            return ((xx - centres) ** 2 + 0.1 * np.cos(3.0 * xx)).ravel()
        return g

    t0 = centres + 0.3
    ours = native_pool.parabolic_minimum(make2("cc"), t0, 2.0)
    ref = numerics.parabolic_minimum(make2("py"), t0, 2.0)
    assert np.array_equal(ours, ref)
    assert calls == {"py": 10, "cc": 10}
    # An objective with a flat (denom == 0 -> NaN step) region does not move t0.
    ours = native_pool.parabolic_minimum(lambda x: np.full_like(x, 2.0), t0, 1.0)
    ref = numerics.parabolic_minimum(lambda x: np.full_like(x, 2.0), t0, 1.0)
    assert np.array_equal(ours, ref) and np.array_equal(ours, t0)


# ---------------------------------------------------------------- phase 2: ellipsoid + geometry
# Roadmap §4 gates for app.geography: reduction / fund_to_geo / geo_to_fund 1e-13,
# shadow_edge_limits_v 1e-9 deg / 1e-6 km, global_contacts 1e-9 h; and the phase
# exit, /central-line byte-identical through the two backends.  Measured
# residuals are recorded in each test after its gate.
_ELL_TOL = 1e-13        # ellipsoid reduction [deg / Earth radii]
_LIMIT_TOL_DEG = 1e-9   # shadow-edge limit points [deg]
_LIMIT_TOL_KM = 1e-6    # path width [km]
_CONTACT_TOL_H = 1e-9   # global contacts [h]
_ECLIPSE_MU_DEG = 120.0  # |mu| bound of "eclipse-realistic" axis hour angles (see below)


def test_reduction_aux_parity(native_pool):
    """_reduction_aux over 200 001 declinations spanning ±90 deg: expected and
    measured bit-identical (0.0 residual on every field) on x86-64/glibc; on the
    Apple arm64 CI runner one field differs by 1 ulp (2.2e-16: platform libm
    sin/cos rounding, e.g. a fused sincos on the C++ side). Same operation order
    in both; the roadmap gate is 1e-13 and a 1-ulp record keeps drift visible."""
    from app.geography import _reduction_aux

    d = np.radians(np.linspace(-90.0, 90.0, 200_001))
    ours = np.array([native_pool.reduction_aux(float(v)) for v in d])  # (n, 6)
    ref = _reduction_aux(d)
    worst = 0.0
    for j, name in enumerate(ref._fields):
        diff = float(np.max(np.abs(ours[:, j] - ref[j])))
        worst = max(worst, diff)
        assert diff <= _ELL_TOL, (name, diff)
    assert worst <= 4.0 * np.finfo(float).eps, worst  # 0.0 on x86-64, 1 ulp on arm64


def _wrapped_deg(a, b):
    """|a - b| in degrees reduced modulo 360 (longitudes at ±180 are one point)."""
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)


def test_fund_to_geo_and_inverse_parity(oracle, native_pool):
    """fund_to_geo_v / geo_to_fund on dense grids against the pure-Python oracle.

    Forward: x, y in [-1.3, 1.3] (121 x 121, axis on and off the Earth) x four
    axis declinations x four hour angles.  NaN masks identical; latitude and the
    longitude (reduced modulo 360) within the 1e-13 gate at every eclipse-
    realistic hour angle (|mu| <= 120 deg).  KNOWN and understood: at mu near
    ±180 deg the intermediate ``degrees(theta) - mu + 180`` reaches ~540 deg,
    whose 1 ulp is 1.14e-13; NumPy's SIMD arctan2 differs from libm's by 1 ulp
    on AVX-512 hosts, which the ``% 360`` then carries into the longitude at
    that magnitude.  That regime is outside any eclipse (|mu| < 120 in every
    reference case) and is bounded here at 1 ulp of that intermediate, not by
    widening the gate.  Inverse: lat x lon grids at the same d, mu; xi/eta/zeta
    within 1e-13 Earth radii.  Measured on x86-64 (this grid): forward lat
    2.9e-14 deg, lon 5.7e-14 deg (1 ulp of a ~360 deg intermediate) at every
    mu including ±179.9 -- the 1.14e-13 case is the analytic ceiling seen during
    the ellipsoid port, not reached on this grid; inverse 2.8e-16 Earth radii.
    """
    from app import geography

    xs = np.linspace(-1.3, 1.3, 121)
    d_set = np.array([-8.24, 7.59, 11.87, 23.4])
    mu_set = np.array([-179.9, 0.0, 89.6, 179.9])
    x, y, d, mu = (a.ravel() for a in np.meshgrid(xs, xs, d_set, mu_set, indexing="ij"))

    lon, lat = native_pool.fund_to_geo(x, y, d, mu)
    rlon, rlat = geography.fund_to_geo_v(x, y, d, mu)
    assert np.array_equal(np.isnan(lat), np.isnan(rlat))
    assert np.array_equal(np.isnan(lon), np.isnan(rlon))
    ok = ~np.isnan(rlat)
    assert ok.sum() > 0.4 * ok.size  # the on-Earth disc is exercised
    lat_diff = float(np.max(np.abs(lat[ok] - rlat[ok])))
    lon_diff = _wrapped_deg(lon[ok], rlon[ok])
    realistic = np.abs(mu[ok]) <= _ECLIPSE_MU_DEG
    lon_diff_realistic = float(np.max(lon_diff[realistic]))
    lon_diff_all = float(np.max(lon_diff))
    assert lat_diff <= _ELL_TOL, lat_diff
    assert lon_diff_realistic <= _ELL_TOL, lon_diff_realistic
    assert lon_diff_all <= np.spacing(540.0), lon_diff_all  # 1 ulp of the wrap intermediate
    print(f"\nfund_to_geo: lat {lat_diff:.3g}, lon {lon_diff_realistic:.3g} (|mu|<=120), "
          f"lon {lon_diff_all:.3g} (all mu)")

    lats = np.linspace(-89.5, 89.5, 181)
    lons = np.linspace(-180.0, 180.0, 181)
    la, lo, d2, mu2 = (a.ravel() for a in np.meshgrid(lats, lons, d_set, mu_set, indexing="ij"))
    ours = native_pool.geo_to_fund(la, lo, d2, mu2)
    ref = geography.geo_to_fund(la, lo, d2, mu2)
    inv_diff = 0.0
    for name, a, b in zip(("xi", "eta", "zeta"), ours, ref, strict=True):
        diff = float(np.max(np.abs(a - b)))
        inv_diff = max(inv_diff, diff)
        assert diff <= _ELL_TOL, (name, diff)
    print(f"geo_to_fund: {inv_diff:.3g}")

    # The (P, 1) x (N,) broadcast app.circumstances relies on, through the
    # dispatching app.geography.geo_to_fund itself.
    from app import native as native_mod

    obs_lat = np.array([[-40.0], [11.4], [25.3], [37.0]])
    obs_lon = np.array([[-160.0], [-83.1], [-104.1], [-87.7]])
    dN, muN = np.linspace(7.5, 7.6, 61), np.linspace(60.0, 120.0, 61)
    ref = geography.geo_to_fund(obs_lat, obs_lon, dN, muN)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(native_mod, "BACKEND", "native")
        via_app = geography.geo_to_fund(obs_lat, obs_lon, dN, muN)
        scalar = geography.geo_to_fund(25.3, -104.1, 7.5862, 89.6)
    assert all(v.shape == (4, 61) for v in via_app)
    for a, b in zip(via_app, ref, strict=True):
        assert np.max(np.abs(a - b)) <= _ELL_TOL
    assert all(isinstance(v, float) for v in scalar)  # scalars in -> Python floats out
    for v, r in zip(scalar, geography.geo_to_fund(25.3, -104.1, 7.5862, 89.6), strict=True):
        assert abs(v - r) <= _ELL_TOL


def _central_line_model(oracle, utc0: str):
    """BesselianModel exactly as /central-line builds it (default window, frame)."""
    from datetime import datetime, timedelta

    from app.besselian import BesselianModel

    t0 = (datetime.fromisoformat(utc0) + timedelta(hours=3)).isoformat()
    return BesselianModel(t0_utc=t0, earth_frame=oracle.DEFAULT_EARTH_FRAME,
                          half_window_hours=2.0)


@requires_kernels
def test_shadow_edge_limits_parity(oracle, native_pool):
    """shadow_edge_limits_v against the oracle on the three modern reference
    tracks (241 instants over ±3 h, bearings from central_track as /central-line
    derives them), umbral (l2, tan_f2, 600 km) and penumbral (l1, tan_f1,
    10 000 km, terminator-clipped).  Gate 1e-9 deg / 1e-6 km.  Measured on
    x86-64: NaN masks identical; limit points 1.42e-13 deg, widths 1.09e-11 km
    -- the same 1-ulp libm-vs-NumPy trig difference that tests/cpp/
    test_geometry.cpp records (it flips one of the 50 bisection decisions at an
    exact tie).  Both sides bisect the *same* oracle elements here, so the
    phase-1 elements residual does not enter; the geometry itself is at the
    ulp level, four orders below the gate."""
    from app.geography import central_track, shadow_edge_limits_v

    oracle.load_kernels()
    worst_deg, worst_km, n_checked = 0.0, 0.0, 0
    for utc0, _utc1 in _WINDOWS[:3]:
        model = _central_line_model(oracle, utc0)
        if not _ephemeris_covers(oracle, model.et0):
            continue
        elems, track = central_track(model, np.linspace(-3.0, 3.0, 241))
        idx = np.array([tp.i for tp in track], dtype=int)
        brg = np.array([tp.bearing for tp in track])
        args = tuple(elems[k][idx] for k in ("x", "y", "d", "mu"))
        for l_key, tf_key, max_km in (("l2", "tan_f2", 600.0), ("l1", "tan_f1", 10_000.0)):
            shadow = (elems[l_key][idx], elems[tf_key][idx], brg)
            ours = native_pool.shadow_edge_limits(*args, *shadow, max_km=max_km, sunlit_only=True)
            ref = shadow_edge_limits_v(*args, *shadow, max_km=max_km)
            for j, (a, b) in enumerate(zip(ours, ref, strict=True)):
                assert np.array_equal(np.isnan(a), np.isnan(b)), (utc0, l_key, j)
                m = ~np.isnan(b)
                diff = float(np.max(np.abs(a[m] - b[m])))
                if j == 4:
                    worst_km = max(worst_km, diff)
                    assert diff <= _LIMIT_TOL_KM, (utc0, l_key, "width", diff)
                else:
                    worst_deg = max(worst_deg, diff)
                    assert diff <= _LIMIT_TOL_DEG, (utc0, l_key, j, diff)
            assert np.sum(~np.isnan(ref[0])) > 100  # a real track, not all-NaN
            n_checked += 1
    assert n_checked == 6
    print(f"\nshadow_edge_limits: points {worst_deg:.3g} deg, width {worst_km:.3g} km")


@requires_kernels
def test_global_contacts_parity(oracle, native_pool):
    """global_contacts at the catalog's greatest-eclipse epochs (+1919 when the
    SPK covers it): identical keys in the oracle's insertion order (P1, P4, U1,
    U4, U2, U3) and |dt| <= 1e-9 h.  Measured on x86-64: 0.0 (the elements
    residual of phase 1 lies below the bisection's 1e-6 h resolution).  Both
    sides evaluate the elements from (et0, frame); the model itself is not
    crossed."""
    from app.besselian import BesselianModel
    from app.geography import global_contacts

    oracle.load_kernels()
    epochs = ["2017-08-21T18:25:30", "2023-10-14T17:59:27", "2024-04-08T18:17:15",
              "1919-05-29T13:08:00"]
    worst, n_checked = 0.0, 0
    for t0 in epochs:
        if not _ephemeris_covers(oracle, oracle.utc_to_et(t0)):
            continue
        model = BesselianModel(t0_utc=t0, half_window_hours=2.5)
        ours = native_pool.global_contacts(model.et0, model.earth_frame, 5.0)
        ref = global_contacts(model)
        assert [name for name, _ in ours] == list(ref)  # keys *and* order
        assert len(ref) >= 2 and list(ref)[:2] == ["P1", "P4"]
        for name, t in ours:
            diff = abs(t - ref[name])
            worst = max(worst, diff)
            assert diff <= _CONTACT_TOL_H, (t0, name, diff)
        n_checked += 1
    assert n_checked >= 3
    print(f"\nglobal_contacts: {worst:.3g} h")


def _diff_paths(a, b, path="$"):
    """Leaf paths where two parsed-JSON trees differ (for the failure message)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k} (missing on one side)")
            else:
                out.extend(_diff_paths(a[k], b[k], f"{path}.{k}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} length {len(a)} != {len(b)}"]
        return [p for i, (x, y) in enumerate(zip(a, b, strict=True))
                for p in _diff_paths(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@requires_kernels
@pytest.mark.parametrize("params", [
    # test_api.py's request, then the endpoint's default 4-hour track.
    {"epoch": "2024-04-08T18:17:15", "start_hours": -0.1, "end_hours": 0.1, "step_minutes": 6},
    {"epoch": "2024-04-08T18:17:15"},
    {"epoch": "2017-08-21T18:25:30"},
    {"epoch": "2023-10-14T17:59:27"},
])
def test_central_line_identical_through_native(monkeypatch, native_pool, params):
    """Phase-2 exit: /central-line is unchanged to the last digit.  The same
    request through the pure-Python backend and through the native one (elements,
    reduction, limits, contacts all in C++) parses to equal JSON at the endpoint's
    own rounding (1e-5 deg, 0.01 km, whole seconds).  Measured: identical for the
    four requests (up to 121 track points each)."""
    from fastapi.testclient import TestClient

    from app import ephemeris
    from app import native as native_mod
    from app.main import app

    client = TestClient(app)
    monkeypatch.setattr(native_mod, "BACKEND", "python")
    ephemeris.load_kernels()
    body_python = client.get("/central-line", params=params).json()
    monkeypatch.setattr(native_mod, "BACKEND", "native")
    body_native = client.get("/central-line", params=params).json()
    assert body_native["count"] == body_python["count"] > 0
    assert set(body_native["contacts"]) == {"P1", "U1", "U2", "U3", "U4", "P4"}
    assert body_native == body_python, "\n".join(_diff_paths(body_native, body_python))


def test_geography_round_trip_through_native(monkeypatch, native_pool):
    """tests/test_geography.py::test_geo_to_fund_round_trip with the dispatching
    geo_to_fund routed to the native core (fund_to_geo stays scalar Python)."""
    from app import native as native_mod
    from app.geography import fund_to_geo, geo_to_fund

    monkeypatch.setattr(native_mod, "BACKEND", "native")
    d_deg, mu_deg = 7.5862, 89.6
    checked = 0
    for lat in (-40.0, -10.0, 0.0, 20.0, 45.0):
        for lon in (-160.0, -104.0, -40.0, 0.0, 80.0):
            xi, eta, zeta = geo_to_fund(lat, lon, d_deg, mu_deg)
            assert isinstance(zeta, float)
            if zeta <= 0.0:
                continue
            lon2, lat2 = fund_to_geo(xi, eta, d_deg, mu_deg)
            assert lat2 == pytest.approx(lat, abs=1e-6), (lat, lon)
            assert lon2 == pytest.approx(lon, abs=1e-6), (lat, lon)
            checked += 1
    assert checked >= 3


# ---------------------------------------------------------------- phase 3: circumstances
# Roadmap §4 gates for app.circumstances: contacts / t_max 1e-9 h, magnitude
# and obscuration 1e-12, grid 1e-12 with the bool arrays bit-identical, horizon
# flags identical; and the phase exit, /circumstances and /map byte-identical
# through the two backends plus the 0.5-degree global-grid benchmark.
# Measured residuals are recorded in each test after its gate.
_LOCAL_TIME_TOL_H = 1e-9   # c1, c4, c2, c3, t_max [h]
_LOCAL_MAG_TOL = 1e-12     # magnitude, obscuration, L2_x
_LOCAL_ALT_TOL_DEG = 1e-11  # Sun altitude / azimuth at the events [deg]
_GRID_TOL = 1e-12          # magnitude, obscuration, t_max_hours, sun_alt

# (epoch near greatest eclipse) as /circumstances and /map receive it.
_CIRC_EPOCHS = ["2017-08-21T18:25:30", "2023-10-14T17:59:27", "2024-04-08T18:17:15"]
# Off-path observers shared by every eclipse: NYC, a sunset-in-progress site
# (W Ireland on 2024-04-08), the night side (central India), outside the
# penumbra (S America on 2024-04-08), both poles and the antimeridian.
_CIRC_FIXED_SITES = [(40.71, -74.01), (53.3, -9.0), (20.0, 77.0), (-30.0, -60.0),
                     (89.9, 0.0), (-89.9, 0.0), (0.0, 179.99), (0.0, -179.99)]


def _circumstances_model(epoch: str, half_window_hours: float = 2.5):
    """BesselianModel exactly as /circumstances builds it (default window, frame)."""
    from app.besselian import BesselianModel
    from app.ephemeris import DEFAULT_EARTH_FRAME

    return BesselianModel(t0_utc=epoch, earth_frame=DEFAULT_EARTH_FRAME,
                          half_window_hours=half_window_hours)


def _map_model(epoch: str):
    """BesselianModel exactly as /map builds it (window_hours 3.0, default frame)."""
    return _circumstances_model(epoch, half_window_hours=3.0)


def _path_sites(model):
    """The greatest-eclipse point (axis on the fundamental plane at T0, reduced
    to the ellipsoid) and three central-line points along the track."""
    from app.geography import central_track, fund_to_geo

    e = model.evaluate(np.array([0.0]))
    lon, lat = fund_to_geo(e["x"][0], e["y"][0], e["d"][0], e["mu"][0])
    _elems, track = central_track(model, np.array([-0.9, -0.3, 0.6]))
    assert len(track) == 3, "central line expected at all three offsets"
    return [(float(lat), float(lon))] + [(tp.lat, tp.lon) for tp in track]


def _native_local_raw(native_pool, model, lat, lon):
    from app.circumstances import _LocalRaw

    raw = native_pool.local_circumstances(model.et0, model.earth_frame,
                                          model.half_window_hours, float(lat), float(lon))
    return _LocalRaw(*raw)


@requires_kernels
def test_local_circumstances_parity(oracle, native_pool):
    """_local_raw against the native core, field by field, at 13 observers per
    modern reference eclipse (39 sites): the greatest-eclipse point and three
    central-line points (total / annular phases with C2, C3), NYC (partial), the
    sunset site (max and C4 below the horizon), the night side (geometric
    penumbra, eclipse False), a site outside the penumbra (geometric False),
    the poles, the antimeridian, and the greatest point with a 1-hour fit
    window (the bracketing widens 1.0 -> 1.5 -> 2.25 h).  Flags and NaN masks
    identical; contacts / t_max within 1e-9 h, magnitude / obscuration / L2_x
    within 1e-12, altitudes / azimuths within 1e-11 deg.  Measured on x86-64
    (39 sites): contacts C1-C4 0.0 (bit-identical: the bisection's 30 decisions
    land on the same doubles), t_max 2.55e-12 h (the parabolic step's
    ``(y0 - y2) / denom`` amplifies 1 ulp of the magnitude samples; the same
    figure tests/cpp/test_circumstances.cpp records), magnitude / obscuration /
    L2_x 1.5e-14, altitude 2.1e-14 deg, azimuth 5.7e-14 deg -- the 1-ulp
    libm-vs-NumPy SIMD trig noise of phases 1-2 and nothing above it.  The
    oracle's ValueError (partial roots all one-signed) was not reached at any
    site of a 5-degree global scan of the three eclipses at both windows, so
    it has no parity case here; the native core maps it to ValueError."""
    from app.circumstances import _local_raw

    oracle.load_kernels()
    worst = {"contact": 0.0, "t_max": 0.0, "mag": 0.0, "alt": 0.0, "az": 0.0}
    n_sites = n_central = n_hidden = n_none = 0
    for epoch in _CIRC_EPOCHS:
        model = _circumstances_model(epoch)
        if not _ephemeris_covers(oracle, model.et0):
            continue
        cases = [(model, s) for s in _path_sites(model) + _CIRC_FIXED_SITES]
        cases.append((_circumstances_model(epoch, half_window_hours=1.0), cases[0][1]))
        for m, (lat, lon) in cases:
            ref = _local_raw(m, lat, lon)
            ours = _native_local_raw(native_pool, m, lat, lon)
            site = (epoch, m.half_window_hours, lat, lon)
            for name in ("geometric", "central", "eclipse", "below"):
                assert getattr(ours, name) == getattr(ref, name), (site, name)
            for name in ("c1", "c4", "c2", "c3", "t_max"):
                a, b = getattr(ours, name), getattr(ref, name)
                assert np.isnan(a) == np.isnan(b), (site, name)
                if not np.isnan(b):
                    key = "t_max" if name == "t_max" else "contact"
                    worst[key] = max(worst[key], abs(a - b))
                    assert abs(a - b) <= _LOCAL_TIME_TOL_H, (site, name, a, b)
            for name in ("magnitude", "obscuration", "L2_x"):
                a, b = getattr(ours, name), getattr(ref, name)
                assert np.isnan(a) == np.isnan(b), (site, name)
                if not np.isnan(b):
                    worst["mag"] = max(worst["mag"], abs(a - b))
                    assert abs(a - b) <= _LOCAL_MAG_TOL, (site, name, a, b)
            for name, key in (("alt_deg", "alt"), ("az_deg", "az")):
                a, b = np.array(getattr(ours, name)), np.array(getattr(ref, name))
                assert np.array_equal(np.isnan(a), np.isnan(b)), (site, name)
                ok = ~np.isnan(b)
                if ok.any():
                    diff = float(np.max(np.abs(a[ok] - b[ok])))
                    worst[key] = max(worst[key], diff)
                    assert diff <= _LOCAL_ALT_TOL_DEG, (site, name, diff)
            n_sites += 1
            n_central += ref.central
            n_hidden += ref.geometric and not ref.eclipse
            n_none += not ref.geometric
    assert n_sites >= 26 and n_central >= 8 and n_hidden >= 2 and n_none >= 2
    print(f"\nlocal_circumstances ({n_sites} sites): contacts {worst['contact']:.3g} h, "
          f"t_max {worst['t_max']:.3g} h, magnitude/obscuration/L2_x {worst['mag']:.3g}, "
          f"alt {worst['alt']:.3g} deg, az {worst['az']:.3g} deg")


@requires_kernels
def test_circumstances_grid_parity(oracle, native_pool):
    """circumstances_grid on the 10-degree global grid (19 x 36 = 684 cells) of
    the three modern eclipses, with the /map model (3-hour window, 2-minute
    step): the four float arrays within 1e-12 and ``visible`` / ``central``
    bit-identical; and the native result independent of the thread count
    (``threads=1`` equal to the OpenMP default on all six arrays).  Measured on
    x86-64 (3 x 684 cells): ``visible`` / ``central`` identical, t_max_hours
    0.0 (the same grid instant is the argmax), magnitude 7.3e-14, obscuration
    6.3e-14, sun_alt 5.7e-14 deg -- 1 ulp of values of order 1 / 60 deg, the
    libm-vs-NumPy trig noise of phases 1-2."""
    from app.circumstances import circumstances_grid

    oracle.load_kernels()
    lats = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    lons = np.arange(-180.0, 180.0, 10.0)
    la, lo = (a.ravel() for a in np.meshgrid(lats, lons, indexing="ij"))
    keys = ("magnitude", "obscuration", "t_max_hours", "sun_alt", "visible", "central")
    worst: dict[str, float] = {}
    n_checked = n_central = 0
    for epoch in _CIRC_EPOCHS:
        model = _map_model(epoch)
        if not _ephemeris_covers(oracle, model.et0):
            continue
        ref = circumstances_grid(model, la, lo, step_minutes=2.0)
        ours = native_pool.circumstances_grid(model.et0, model.earth_frame,
                                              model.half_window_hours, la, lo, 2.0)
        serial = native_pool.circumstances_grid(model.et0, model.earth_frame,
                                                model.half_window_hours, la, lo, 2.0, threads=1)
        assert tuple(ours) == keys and tuple(ref) == keys
        for k in keys:
            assert ours[k].dtype == ref[k].dtype and ours[k].shape == ref[k].shape, (epoch, k)
            assert np.array_equal(ours[k], serial[k]), (epoch, k, "thread count")
            if ref[k].dtype == bool:
                assert np.array_equal(ours[k], ref[k]), (epoch, k)
            else:
                diff = float(np.max(np.abs(ours[k] - ref[k])))
                worst[k] = max(worst.get(k, 0.0), diff)
                assert diff <= _GRID_TOL, (epoch, k, diff)
        assert ref["visible"].sum() > 50  # a real map
        n_central += int(ref["central"].sum())
        n_checked += 1
    assert n_checked == 3 and n_central >= 1  # the 10-degree grid meets the 2024 path
    print("\ncircumstances_grid: " + ", ".join(f"{k} {v:.3g}" for k, v in worst.items()))


@requires_kernels
@pytest.mark.parametrize("path, params", [
    # test_api.py's /circumstances request (a 400: the same error body on both
    # backends), then the reference sites of test_besselian_integration.py.
    ("/circumstances", {"epoch": "2024 APR 08 18:17:15", "lat": 25.3, "lon": -104.1}),
    ("/circumstances", {"epoch": "2024-04-08T18:17:15", "lat": 25.3, "lon": -104.1}),
    ("/circumstances", {"epoch": "2017-08-21T18:25:30", "lat": 37.0, "lon": -87.7}),
    ("/circumstances", {"epoch": "2023-10-14T17:59:27", "lat": 11.4, "lon": -83.1}),
    ("/circumstances", {"epoch": "2024-04-08T18:17:15", "lat": 40.71, "lon": -74.01}),
    ("/circumstances", {"epoch": "2024-04-08T18:17:15", "lat": 53.3, "lon": -9.0}),
    ("/circumstances", {"epoch": "2024-04-08T18:17:15", "lat": 20.0, "lon": 77.0}),
    ("/circumstances", {"epoch": "2024-04-08T18:17:15", "lat": -30.0, "lon": -60.0}),
    # test_api.py's /map request, then the default 2-degree maps of the three.
    ("/map", {"epoch": "2024-04-08T18:17:15", "lat_step": 10, "lon_step": 10}),
    ("/map", {"epoch": "2024-04-08T18:17:15"}),
    ("/map", {"epoch": "2017-08-21T18:25:30"}),
    ("/map", {"epoch": "2023-10-14T17:59:27"}),
])
def test_circumstances_endpoints_identical_through_native(monkeypatch, native_pool, path, params):
    """Phase-3 exit: /circumstances and /map are unchanged to the last digit.
    The same request through the pure-Python backend and through the native one
    (elements, reduction, contacts, magnitudes, horizon all in C++) parses to
    equal JSON at the endpoints' own rounding (1e-4 magnitude, 0.1 deg, 0.1 s,
    whole clock seconds; 1e-3 on the map).  Measured: identical for the twelve
    requests (status codes included; the default maps are 91 x 180 cells)."""
    from fastapi.testclient import TestClient

    from app import ephemeris
    from app import native as native_mod
    from app.main import app

    client = TestClient(app)
    monkeypatch.setattr(native_mod, "BACKEND", "python")
    ephemeris.load_kernels()
    r_python = client.get(path, params=params)
    monkeypatch.setattr(native_mod, "BACKEND", "native")
    r_native = client.get(path, params=params)
    body_python, body_native = r_python.json(), r_native.json()
    assert r_native.status_code == r_python.status_code
    if path == "/map" and r_python.status_code == 200:
        assert sum(map(sum, body_python["central"])) >= 1
    elif r_python.status_code == 200:
        assert body_python["lat"] == params["lat"]
    assert body_native == body_python, "\n".join(_diff_paths(body_native, body_python))


@requires_kernels
@pytest.mark.skipif(os.environ.get("ECLIPSE_BENCH") != "1",
                    reason="benchmark: run with ECLIPSE_BENCH=1 (ECLIPSE_BENCH_MAX_S = gate)")
def test_circumstances_grid_benchmark(native_pool):
    """Roadmap §5 phase 3: the 0.5-degree global grid (361 x 720 = 259 920
    observers, /map's 3-hour window at a 2-minute step = 181 instants) through
    the native circumstances_grid in < 0.5 s.  Called directly because the grid
    exceeds /map's MAX_MAP_CELLS (150 000).  Measured on the development host
    (4 cores, x86-64, OpenMP default threads): 0.38 s (1.39 s serial) against
    4.0 s for the Python circumstances_grid on the same grid, whose six arrays
    the native ones match as in test_circumstances_grid_parity (bools
    identical, floats <= 9.4e-14); the 0.5 s figure is the developer-host
    criterion, and CI runs it with ECLIPSE_BENCH_MAX_S=2.0 as a regression
    guard on shared 2-4 vCPU runners."""
    import time

    from app.ephemeris import utc_to_et

    model_et0 = utc_to_et("2024-04-08T18:17:15")
    lats = np.arange(-90.0, 90.0 + 1e-9, 0.5)
    lons = np.arange(-180.0, 180.0, 0.5)
    la, lo = (a.ravel() for a in np.meshgrid(lats, lons, indexing="ij"))
    assert la.size == 361 * 720
    native_pool.circumstances_grid(model_et0, "ITRS", 3.0, la[:720], lo[:720], 2.0)  # warm-up
    t0 = time.perf_counter()
    g = native_pool.circumstances_grid(model_et0, "ITRS", 3.0, la, lo, 2.0)
    elapsed = time.perf_counter() - t0
    assert g["central"].sum() > 0 and g["visible"].sum() > 10_000
    limit = float(os.environ.get("ECLIPSE_BENCH_MAX_S", "0.5"))
    print(f"\ncircumstances_grid 0.5-degree global grid: {elapsed:.3f} s "
          f"({os.cpu_count()} CPUs; gate {limit} s)")
    assert elapsed < limit, f"{elapsed:.3f} s >= {limit} s"

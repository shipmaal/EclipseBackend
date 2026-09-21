"""Phase-0 acceptance tests for the native core ``_eclipse`` (docs/CPP_ROADMAP.md §5).

The module is built by ``uv sync`` (scikit-build-core + CMake). These tests pin
the vendored toolkit versions to the Python oracle's, prove that a SPICE error
is a Python exception rather than a process abort, and assert bit-level parity
of the SPICE calls the elements layer will use.

Note: the extension links its own CSPICE statically, so it has its *own* kernel
pool — furnishing here does not affect spiceypy's pool and vice versa.
"""

from __future__ import annotations

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
    assert np.array_equal(native_pool.np_arange(-5, 5 + 1e-9, 1 / 60),
                          np.arange(-5, 5 + 1e-9, 1 / 60))
    assert np.array_equal(native_pool.np_arange(-2.0, 2.0 + 1e-9, 2.0 / 60.0),
                          np.arange(-2.0, 2.0 + 1e-9, 2.0 / 60.0))
    assert np.array_equal(native_pool.np_arange(-90.0, 90.0 + 1e-9, 0.5),
                          np.arange(-90.0, 90.0 + 1e-9, 0.5))
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

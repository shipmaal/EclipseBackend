"""Behaviour of the native core ``_eclipse`` as a Python extension: errors are
exceptions (never aborts), the kernel pool and EOP table are what the API
relies on, parallel results do not depend on the thread count, and the two
benchmarks CI gates on.

The extension links its own CSPICE statically, so it has its *own* kernel
pool; spiceypy (a separate CSPICE build, test-only) is used here as an
independent reference for the raw SPICE calls.
"""

from __future__ import annotations

import math
import os

import _eclipse as E
import numpy as np
import pytest

from app.core import default_metakernel, load_kernels, metakernel

requires_kernels = pytest.mark.skipif(
    not default_metakernel().exists(),
    reason="SPICE kernels not downloaded (run `python -m kernels.bootstrap`)",
)

REFERENCE_UTC = ["2017-08-21T18:26:40", "2023-10-14T18:00:41", "2024-04-08T18:17:20"]


@pytest.fixture
def pool():
    """``_eclipse`` with the EOP table and (if present) the kernels loaded, fresh."""
    load_kernels()
    if default_metakernel().exists():
        E.kclear()
        E.furnish(str(metakernel()))
    return E


def test_vendored_toolkit_versions():
    assert E.toolkit_version() == "CSPICE_N0067"
    assert E.erfa_version().startswith("2.")


def test_spice_error_is_an_exception_not_an_abort():
    E.kclear()
    try:
        assert E.kernel_count() == 0
        with pytest.raises(E.SpiceError, match=r"SPICE\(NOLOADEDFILES\)") as exc_info:
            E.body_position("SUN", 0.0, "J2000", "NONE", "EARTH")
        assert isinstance(exc_info.value, RuntimeError)
        assert exc_info.value.short_message == "SPICE(NOLOADEDFILES)"
        # The error state was reset: the toolkit is still usable afterwards.
        assert E.toolkit_version() == "CSPICE_N0067"
    finally:
        if default_metakernel().exists():  # leave the pool as other tests expect it
            E.furnish(str(metakernel()))


def test_bad_time_string_raises():
    with pytest.raises(E.SpiceError, match="SPICE"):
        E.str_to_et("not a date")


def test_utc_to_et_spice_error_is_raised_and_reset(pool):
    # tparse_c's own SPICE error goes through spice_call: raised, and the global
    # error state is reset, so the next call works (review item C5).
    with pytest.raises(E.SpiceError) as exc_info:
        pool.utc_to_et("")
    assert exc_info.value.short_message == "SPICE(EMPTYSTRING)"
    assert pool.utc_to_et("1919-05-29T13:08:00") < 0.0


def test_unknown_frame_is_a_value_error(pool):
    with pytest.raises(ValueError, match="earth_frame"):
        pool.besselian_instants(np.array([0.0]), "GCRS")


def test_eop_table_is_the_iers_file():
    """The core parses finals2000A.all itself (``load_eop_file``); at the table's
    nodes the interpolation returns the file's Bulletin A values exactly
    (columns per IERS ``readme.finals2000A``), polar motion in radians."""
    from astropy_iers_data import IERS_A_FILE

    load_kernels()
    assert E.eop_source() == str(IERS_A_FILE)
    rows = []
    with open(IERS_A_FILE) as fh:
        for line in fh:
            if not line[18:27].strip():
                break
            rows.append((float(line[7:15]), float(line[18:27]), float(line[37:46]),
                         float(line[58:68])))
    lo, hi = E.eop_mjd_range()
    assert (lo, hi) == (rows[0][0], rows[-1][0])
    pick = [rows[i] for i in (0, len(rows) // 3, len(rows) // 2, -1)]
    xp, yp, dut1 = E.eop(np.array([r[0] for r in pick]))
    arcsec = math.pi / (180.0 * 3600.0)
    for k, r in enumerate(pick):
        assert xp[k] == r[1] * arcsec and yp[k] == r[2] * arcsec and dut1[k] == r[3]


@requires_kernels
def test_time_and_position_agree_with_spiceypy(pool, spice):
    """The core's CSPICE and spiceypy's (a separate build of the same N0067
    toolkit) read the same kernels identically."""
    for utc in REFERENCE_UTC:
        et = pool.str_to_et(utc)
        assert et == spice.str2et(utc)
        assert pool.et_to_utc_iso(et, 6) == spice.et2utc(et, "ISOC", 6)
        for body in ("SUN", "MOON"):
            ours = pool.body_position(body, et, "J2000", "LT+S", "EARTH")
            ref, lt = spice.spkpos(body, et, "J2000", "LT+S", "EARTH")
            assert list(ours.km) == list(ref), (utc, body)
            assert ours.light_time_s == lt


@requires_kernels
def test_kernel_pool_change_mid_vector_is_an_error_never_mixed(pool):
    """A furnish/kclear between two spkpos batches (2048 instants each) raises
    instead of returning a vector from two kernel sets (review item C6).  Timing
    decides whether a reload lands mid-vector, so each run must be either the
    untouched result or the RuntimeError, never anything else."""
    import threading

    et0 = pool.utc_to_et("2024-04-08T18:17:20")
    et = et0 + np.linspace(-3e6, 3e6, 200_000)
    ref = pool.axis_separation(et)
    mk = str(metakernel())
    stop = threading.Event()

    def reload_loop():
        while not stop.is_set():
            pool.furnish(mk)  # same kernels: only the generation changes

    t = threading.Thread(target=reload_loop)
    t.start()
    try:
        for _ in range(5):
            try:
                got = pool.axis_separation(et)
            except RuntimeError as exc:
                assert not isinstance(exc, pool.SpiceError)
                assert "kernel pool changed" in str(exc)
            else:
                for g, r in zip(got, ref, strict=True):
                    assert np.array_equal(g, r)
    finally:
        stop.set()
        t.join()
        pool.kclear()
        pool.furnish(mk)
    assert pool.axis_separation(et[:10])[0].shape == (10,)


def _nan_safe(obj):
    """Nested tuples/lists with NaN replaced by a sentinel, for == comparison."""
    if isinstance(obj, (list, tuple)):
        return type(obj)(_nan_safe(v) for v in obj)
    if isinstance(obj, float) and math.isnan(obj):
        return "NaN"
    return obj


@requires_kernels
def test_catalog_thread_count_independence(pool):
    """The per-event hybrid / detail loop is OpenMP-parallel; serial (threads=1)
    and the default give value-identical tuples (no cross-event reductions)."""
    et_a = pool.utc_to_et("2023-01-01T00:00:00")
    et_b = pool.utc_to_et("2024-12-31T00:00:00")
    default = pool.find_eclipses(et_a, et_b, "ITRS", True)
    serial = pool.find_eclipses(et_a, et_b, "ITRS", True, threads=1)
    assert [t[1] for t in default] == ["hybrid", "annular", "total", "annular"]
    assert _nan_safe(serial) == _nan_safe(default)
    assert _nan_safe(pool.find_eclipses(et_a, et_b, "ITRS", True, threads=3)) == \
        _nan_safe(default)


@requires_kernels
@pytest.mark.skipif(os.environ.get("ECLIPSE_BENCH") != "1",
                    reason="benchmark: run with ECLIPSE_BENCH=1 (ECLIPSE_BENCH_MAX_S = gate)")
def test_circumstances_grid_benchmark(pool):
    """The 0.5-degree global grid (361 x 720 = 259 920 observers, /map's 3-hour
    window at a 2-minute step = 181 instants) in < 0.5 s.  Called directly
    because the grid exceeds /map's MAX_MAP_CELLS (150 000).  Measured on the
    development host (4 cores, x86-64, OpenMP default threads): 0.38 s (1.39 s
    serial); CI runs it with ECLIPSE_BENCH_MAX_S=2.0 as a regression guard on
    shared 2-4 vCPU runners."""
    import time

    et0 = pool.utc_to_et("2024-04-08T18:17:15")
    lats = np.arange(-90.0, 90.0 + 1e-9, 0.5)
    lons = np.arange(-180.0, 180.0, 0.5)
    la, lo = (a.ravel() for a in np.meshgrid(lats, lons, indexing="ij"))
    assert la.size == 361 * 720
    pool.circumstances_grid(et0, "ITRS", 3.0, la[:720], lo[:720], 2.0)  # warm-up
    t0 = time.perf_counter()
    g = pool.circumstances_grid(et0, "ITRS", 3.0, la, lo, 2.0)
    elapsed = time.perf_counter() - t0
    assert g["central"].sum() > 0 and g["visible"].sum() > 10_000
    limit = float(os.environ.get("ECLIPSE_BENCH_MAX_S", "0.5"))
    print(f"\ncircumstances_grid 0.5-degree global grid: {elapsed:.3f} s "
          f"({os.cpu_count()} CPUs; gate {limit} s)")
    assert elapsed < limit, f"{elapsed:.3f} s >= {limit} s"


@requires_kernels
@pytest.mark.skipif(os.environ.get("ECLIPSE_BENCH") != "1",
                    reason="benchmark: run with ECLIPSE_BENCH=1 "
                           "(ECLIPSE_CATALOG_BENCH_MAX_S = gate)")
def test_catalog_century_benchmark(pool):
    """A century scan with detail=True in < 10 s through app.catalog.find_eclipses
    (/eclipses caps detail scans at 10 years).  2000-2100 when the loaded SPK
    covers it (DE440s), else 1950-2050 (the DE432s mirror, 1949-2050).
    Measured on the development host (4 cores, x86-64, OpenMP default
    threads): 2000-2100, 226 eclipses, detail on 8.5 s (12.1 s serial),
    detail off 4.07 s; CI runs it with ECLIPSE_CATALOG_BENCH_MAX_S=30 as a
    regression guard on shared 2-4 vCPU runners."""
    import time

    from app.catalog import find_eclipses

    try:
        pool.axis_separation(np.array([pool.utc_to_et("2100-01-08T00:00:00")]))
        start, end = "2000-01-01", "2100-01-01"
    except pool.SpiceError:
        start, end = "1950-01-01", "2050-01-01"
    find_eclipses("2024-01-01", "2024-12-31", detail=True)  # warm-up
    t0 = time.perf_counter()
    rows = find_eclipses(start, end, detail=True)
    elapsed = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows_off = find_eclipses(start, end, detail=False)
    elapsed_off = time.perf_counter() - t0
    assert 200 <= len(rows) == len(rows_off) <= 260
    assert all("contacts" in r for r in rows)
    limit = float(os.environ.get("ECLIPSE_CATALOG_BENCH_MAX_S", "10.0"))
    print(f"\nfind_eclipses {start}..{end} ({len(rows)} eclipses): detail on {elapsed:.2f} s, "
          f"detail off {elapsed_off:.2f} s ({os.cpu_count()} CPUs; gate {limit} s)")
    assert elapsed < limit, f"{elapsed:.2f} s >= {limit} s"

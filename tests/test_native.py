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

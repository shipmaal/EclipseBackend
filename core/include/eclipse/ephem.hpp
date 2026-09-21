// ephem — the ONLY part of libeclipse that touches CSPICE and ERFA.
//
// Every function here takes the process-global SPICE lock (the kernel pool is
// not thread-safe; ``app/ephemeris.SPICE_LOCK`` is the same rule in Python),
// runs in RETURN error mode and throws ``eclipse::spice_error`` on failure.
// Everything else in the library is pure math and must not include this
// header's implementation details (roadmap §2, §7).
//
// Units: times are ``et`` = TDB seconds past J2000 as CSPICE defines them
// [SPICE]; positions in km; angles in degrees at the API boundary.
#pragma once

#include <array>
#include <string>
#include <string_view>

#include "eclipse/spice_error.hpp"

namespace eclipse::ephem {

/// ``tkvrsn_c("TOOLKIT")`` — e.g. ``"CSPICE_N0067"``. Pinned by a test so the
/// vendored toolkit cannot silently diverge from the spiceypy oracle.
std::string toolkit_version();

/// ``eraVersion()`` of the vendored liberfa, e.g. ``"2.0.1"`` (must equal the
/// version pyerfa bundles for bit-identical Earth orientation).
std::string erfa_version();

/// ``eraSofaVersion()`` — the SOFA release liberfa was derived from.
std::string sofa_version();

/// Load a kernel or metakernel (``furnsh_c``). Paths are resolved by CSPICE
/// relative to the current working directory, exactly as in spiceypy.
void furnish(std::string_view path);

/// Unload every kernel and clear the pool (``kclear_c``).
void kclear();

/// Number of loaded kernels of ``kind`` (``ktotal_c``; default "ALL").
int kernel_count(std::string_view kind = "ALL");

/// UTC (or any ``str2et_c`` string, e.g. ``"2024-04-08T18:00:00"`` or
/// ``"... TDT"``) -> ``et`` seconds [SPICE str2et_c].
double str_to_et(std::string_view time_string);

/// ``et`` -> ISO calendar UTC string with ``precision`` fractional-second
/// digits (``et2utc_c`` with format "ISOC").
std::string et_to_utc_iso(double et, int precision = 6);

/// Result of ``spkpos_c``: position of ``target`` relative to ``observer`` in
/// km, expressed in ``frame``, plus one-way light time in seconds.
struct Position {
    std::array<double, 3> km;
    double light_time_s;
};

/// Apparent position via ``spkpos_c`` [SPICE]. ``abcorr`` "LT+S" gives the
/// light-time + stellar-aberration corrected direction used for the Besselian
/// elements (``app/ephemeris._geocentric_vectors``).
Position body_position(std::string_view target, double et, std::string_view frame,
                       std::string_view abcorr, std::string_view observer);

}  // namespace eclipse::ephem

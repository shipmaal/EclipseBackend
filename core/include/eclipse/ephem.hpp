// ephem — the ONLY part of libeclipse that touches CSPICE and ERFA.
//
// Every function here takes the process-global SPICE lock (the kernel pool is
// not thread-safe; ``app/ephemeris.SPICE_LOCK`` is the same rule in Python),
// runs in RETURN error mode and throws ``eclipse::spice_error`` on failure.
// Everything else in the library is pure math and must not include this
// header's implementation details (roadmap §2, §7).
//
// This is the port of app/ephemeris.py; that module's docstring is the
// scientific reference for what follows (frames, ``mu`` convention, time
// scales and delta-T). Units: ``et`` = TDB seconds past J2000 [SPICE];
// positions in km at the SPICE boundary and in Earth equatorial radii in the
// fundamental plane; angles in **degrees** at this API's boundary.
#pragma once

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "eclipse/elements.hpp"
#include "eclipse/spice_error.hpp"

namespace eclipse::ephem {

// ------------------------------------------------------------ toolkit / pool

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

/// Solar radius [km]: ``constants::SUN_RADIUS_KM``, 696 000 km (IAU 1976) — the
/// radius the [Espenak] k1/k2 are paired with (``app/ephemeris.sun_radius_km``);
/// not read from the PCK, whose pck00011 value is 695 700 km (review item W1).
double sun_radius_km();

// --------------------------------------------------------------- raw SPICE

/// Any ``str2et_c`` string (e.g. ``"2024-04-08T18:00:00"`` or ``"... TDT"``)
/// -> ``et`` seconds [SPICE str2et_c]. No era logic: see ``utc_to_et``.
double str_to_et(std::string_view time_string);

/// ``et`` -> ISO calendar UTC string with ``precision`` fractional-second
/// digits (``et2utc_c`` with format "ISOC"). No era logic: see ``et_to_utc``.
std::string et_to_utc_iso(double et, int precision = 6);

/// Result of ``spkpos_c``: position of ``target`` relative to ``observer`` in
/// km, expressed in ``frame``, plus one-way light time in seconds.
struct Position {
    std::array<double, 3> km;
    double light_time_s;
};

/// Apparent position via ``spkpos_c`` [SPICE]. ``abcorr`` "LT+S" gives the
/// light-time + stellar-aberration corrected direction used for the Besselian
/// elements.
Position body_position(std::string_view target, double et, std::string_view frame,
                       std::string_view abcorr, std::string_view observer);

// ------------------------------------------------------------- time scales
// The IERS-era rule of app/ephemeris.py: inside the Bulletin A table
// (``eclipse::eop``, which must be set) the epoch string is UTC and the
// leap-second kernel applies; outside it the string is read as UT1 and
// TT = UT1 + delta-T from the [Espenak] model (``eclipse::deltat``).

/// ``app.ephemeris.utc_to_et``: epoch string -> TDB seconds past J2000.
double utc_to_et(std::string_view utc);

/// ``app.ephemeris.et_to_utc``: inverse, ISO ``YYYY-MM-DDTHH:MM:SS`` (whole s).
std::string et_to_utc(double et);

/// ``app.ephemeris.earth_rotation_times``: per instant, TT and UT1 as
/// fractional days past J2000 (second halves of two-part JDs whose first half
/// is ``constants::J2000_JD``) and polar motion in radians (0 outside the
/// IERS era). TDB is used for TT (<= 1.7 ms, sub-mas; review item A4).
struct EarthRotation {
    std::vector<double> tt2, ut1, xp, yp;
};
EarthRotation earth_rotation_times(std::span<const double> et);

/// Delta-T = TT - UT1 [s] as actually used at ``et`` (measured or modelled).
std::vector<double> tt_minus_ut1(std::span<const double> et);

// ---------------------------------------------------------------- geometry

/// ``app.ephemeris.besselian_instants``: the eight elements plus ``z`` at each
/// ``et`` [ES92] eq. 8.322-6, 8.323-1/6/7 with k1/k2 [Espenak].
Elements besselian_instants(std::span<const double> et, Frame frame = Frame::ITRS);

/// ``app.ephemeris.axis_separation``: the Moon's cylindrical coordinates about
/// the shadow axis at each ``et``, frame-free — ``rho = hypot(x, y)`` (the axis'
/// distance from the Earth's centre in the fundamental plane) and ``z`` (the
/// Moon's distance along the axis toward the Sun), both in Earth equatorial
/// radii, from the SAME [ES92] eq. 8.322-6 expressions as ``besselian_instants``
/// (one shared ``fundamental_xyz``) applied to the un-rotated apparent (LT+S)
/// J2000 vectors. Every Earth-orientation frame rotates those two vectors by
/// one common rotation and ``x, y, z`` are built from the Sun-Moon and Moon
/// directions alone, so they are rotation-invariant: no ERFA, no EOP. This is
/// the catalog's scan objective (``eclipse::catalog``), which reads nothing
/// else. The SPICE lock is taken per batch of 2048 instants.
struct AxisSeparation {
    std::vector<double> rho, z;
};
AxisSeparation axis_separation(std::span<const double> et);

/// ``app.ephemeris.sub_solar_points``: geographic (lon, lat) [deg] of the
/// sub-solar point at each ``et``; lon east-positive in (-180, 180].
struct SubSolar {
    std::vector<double> lon_deg, lat_deg;
};
SubSolar sub_solar_points(std::span<const double> et, Frame frame = Frame::ITRS);

/// ``app.ephemeris.limb_axes``: the fundamental-plane unit vectors x^ (east),
/// y^ (north), z^ (shadow axis, toward the Sun) [ES92] 8.322 expressed in
/// ``moon_frame`` (``"MOON_ME"``, the LOLA DEM's frame, or ``"IAU_MOON"``),
/// one row-major 3x3 per ``et`` (rows x^, y^, z^). z^ is the apparent (LT+S)
/// Moon->Sun direction, y^ the ``frame``'s pole projected onto the plane,
/// x^ = y^ x z^; the Moon's orientation is at ``et - lt`` (docs/LIMB_PROFILE.md
/// sec. 3.1-3.2).
std::vector<std::array<double, 9>> limb_axes(std::span<const double> et, Frame frame,
                                             std::string_view moon_frame);

}  // namespace eclipse::ephem

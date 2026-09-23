// Physical and geodetic constants — the C++ twin of app/constants.py and the
// lunar-radius / time constants of app/ephemeris.py. Every value is DERIVED
// the same way, in the same floating-point order, as the Python oracle.
#pragma once

#include <numbers>

namespace eclipse::constants {

// --- WGS-84 reference ellipsoid [WGS84] (NIMA TR8350.2, 3rd ed., 2000) ------
inline constexpr double WGS84_A_KM = 6378.137;                    // semi-major axis a [km]
inline constexpr double WGS84_INV_FLATTENING = 298.257223563;     // 1/f
inline constexpr double WGS84_F = 1.0 / WGS84_INV_FLATTENING;     // f = (a - b) / a
inline constexpr double WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F);  // b = a(1 - f) [km]
inline constexpr double WGS84_E2 = WGS84_F * (2.0 - WGS84_F);       // e^2 = f(2 - f)

// Fundamental-plane unit radius: Besselian x, y, l1, l2 are in Earth equatorial
// radii [ES92] ch. 8, and the SAME radius scales the geocentric Sun/Moon
// vectors and defines the reduction ellipsoid (review item A1).
inline constexpr double EARTH_EQUATORIAL_RADIUS_KM = WGS84_A_KM;

// IUGG arithmetic mean radius R_1 = (2a + b) / 3 for short great-circle offsets.
inline constexpr double EARTH_MEAN_RADIUS_KM = (2.0 * WGS84_A_KM + WGS84_B_KM) / 3.0;

// --- Lunar radius / Earth equatorial radius [Espenak] ------------------------
// Two values: the penumbral one is the IAU mean lunar radius, the umbral one is
// reduced for the mean lunar-limb profile (app/ephemeris.py K_PENUMBRA/K_UMBRA).
inline constexpr double K_PENUMBRA = 0.2725076;
inline constexpr double K_UMBRA = 0.2722810;

// --- Solar radius (app/constants.py SUN_RADIUS_KM) ----------------------------
// IAU (1976) 696 000 km, paired with k1/k2 in the cone angles
// sin f = (d_s +/- k) / G [ES92] eq. 8.323-1 [Espenak]; a constant, not the
// PCK's (pck00011 carries 695 700 km; review item W1).
inline constexpr double SUN_RADIUS_KM = 696000.0;

// --- Time -------------------------------------------------------------------
inline constexpr double J2000_JD = 2451545.0;   // Julian date of J2000.0
inline constexpr double MJD_OFFSET = 2400000.5; // JD - MJD
inline constexpr double SECONDS_PER_DAY = 86400.0;

// --- Angle conversions, written exactly as NumPy evaluates them ---------------
// np.degrees(x) == x * (180.0 / pi); np.radians(x) == x * (pi / 180.0).
inline constexpr double RAD_TO_DEG = 180.0 / std::numbers::pi;
inline constexpr double DEG_TO_RAD = std::numbers::pi / 180.0;
inline constexpr double ARCSEC_TO_RAD = std::numbers::pi / (180.0 * 3600.0);  // app/eop.py

}  // namespace eclipse::constants

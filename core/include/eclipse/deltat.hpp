// Delta-T = TT - UT1 polynomial model [Espenak] sec. 2.5 — port of app/deltat.py.
//
// Used only OUTSIDE the IERS era (before 1973 / beyond the Bulletin A table),
// where UTC is undefined or leap seconds are unknown; inside it delta-T is
// measured (leap-second kernel + IERS UT1-UTC). See app/ephemeris.py "Time
// scales and delta-T".
#pragma once

#include <span>
#include <vector>

namespace eclipse::deltat {

/// Delta-T [s] at decimal ``year`` (e.g. 1919.41): the piecewise polynomials
/// of [Espenak] sec. 2.5 (NASA ``deltatpoly2004``), evaluated by Horner's rule
/// in the same order as ``np.polyval`` so the result is bit-identical to the
/// Python oracle. Continuous at the segment boundaries to ~1 s.
double delta_t_seconds(double year);

/// Vector form of ``delta_t_seconds``.
std::vector<double> delta_t_seconds(std::span<const double> years);

/// Decimal year of a Julian date (any uniform scale):
/// ``2000.0 + (jd - 2451545.0) / 365.25``.
double decimal_year_from_jd(double jd);

}  // namespace eclipse::deltat

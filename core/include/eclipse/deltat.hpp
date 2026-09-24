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

/// Delta-T [s] at decimal ``year`` after the end of the measured record
/// (item C2). The record ends at ``year_end`` with the measured value
/// ``dt_end_s``; the [Espenak] model differs from it there (by +6.8 s at the
/// end of the 2027 Bulletin A table: the 2005-2050 polynomial was fitted
/// before Delta-T levelled off near 69 s), and switching to the model would
/// move every later eclipse by that step. So the offset is carried and
/// tapered linearly to zero at the end of the model's 2005-2050 segment
/// [Espenak] sec. 2.5 (or ten years after ``year_end``, if later):
///   dT(y) = model(y) + (dt_end - model(y_end)) * (y_h - y) / (y_h - y_end),
/// continuous with the measurement at ``y_end`` and with the model at
/// ``y_h``. The taper is this project's rule, not a published one: any
/// Delta-T beyond the record is a prediction (a few seconds a decade).
/// For ``year <= year_end`` this is the model alone.
double delta_t_after_record(double year, double year_end, double dt_end_s);

/// Vector form of ``delta_t_seconds``.
std::vector<double> delta_t_seconds(std::span<const double> years);

/// Decimal year of a Julian date (any uniform scale):
/// ``2000.0 + (jd - 2451545.0) / 365.25``.
double decimal_year_from_jd(double jd);

}  // namespace eclipse::deltat

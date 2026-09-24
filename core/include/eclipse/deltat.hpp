// Delta-T = TT - UT1 outside the IERS era, where UTC is undefined (before
// 1973) or UT1-UTC is not yet tabulated (past the Bulletin A table); inside
// it delta-T is measured (leap-second kernel + IERS UT1-UTC, ``ephem``).
//
// Two published sources, chosen by ``ephem`` per epoch:
//   * past the Bulletin A table, the USNO prediction table [USNO]
//     (``load_predictions``, ``predicted``) while it lasts;
//   * everywhere else, the Espenak & Meeus polynomial model [Espenak]
//     sec. 2.5 (``delta_t_seconds``).
// Neither is joined to its neighbour: the steps at the two ends of the
// prediction table are recorded as known open discrepancies (CLAUDE.md
// convention 7), not smoothed.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace eclipse::deltat {

/// Delta-T [s] at decimal ``year`` (e.g. 1919.41): the piecewise polynomials
/// of [Espenak] sec. 2.5 (NASA ``deltatpoly2004``), evaluated by Horner's rule
/// in the same order as ``np.polyval`` so the result is bit-identical to the
/// Python oracle. Continuous at the segment boundaries to ~1 s.
double delta_t_seconds(double year);

/// A published Delta-T prediction table [USNO]: ``mjd`` (ascending),
/// TT - UT1 [s] and its error estimate [s]. Replaces any earlier table.
/// All three spans must have the same length, at least 2.
void set_predictions(std::span<const double> mjd, std::span<const double> dt_s,
                     std::span<const double> err_s);

/// Parse the USNO ``deltat.preds`` file (tab/space separated: MJD, year,
/// TT-UT [s], an optional UT1-UTC [s], error [s]; one header line) and
/// install it (``set_predictions``). Returns the number of rows. Throws
/// ``std::runtime_error`` if the file cannot be read or is malformed.
std::size_t load_predictions(const std::string& path);

/// The file ``load_predictions`` read ("" for none or ``set_predictions``).
std::string predictions_source();

/// First and last MJD of the prediction table; (NaN, NaN) when none is set.
std::pair<double, double> predictions_mjd_range();

/// A counter bumped by every ``set_predictions`` / ``load_predictions``.
std::uint64_t predictions_generation();

/// Delta-T from the prediction table at ``mjd``: linear interpolation
/// between its rows (the table's own quarterly spacing), and the error
/// estimate interpolated the same way. ``std::nullopt`` outside the table or
/// when none is set.
struct Predicted {
    double dt_s;
    double err_s;
};
std::optional<Predicted> predicted(double mjd);

/// Vector form of ``delta_t_seconds``.
std::vector<double> delta_t_seconds(std::span<const double> years);

/// Decimal year of a Julian date (any uniform scale):
/// ``2000.0 + (jd - 2451545.0) / 365.25``.
double decimal_year_from_jd(double jd);

}  // namespace eclipse::deltat

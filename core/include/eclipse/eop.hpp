// IERS Earth orientation parameters — port of app/eop.py.
//
// The table is INJECTED, not parsed (roadmap §3): Python reads finals2000A.all
// from astropy-iers-data and hands the Bulletin A columns to ``set_table``;
// there is no second source of EOP truth. Interpolation reproduces
// ``np.interp`` (linear, endpoints held) exactly [IERS2010].
#pragma once

#include <span>
#include <utility>

namespace eclipse::eop {

/// Install the Bulletin A table: ``mjd`` (UTC, ascending), polar motion
/// ``xp``/``yp`` in **arcseconds**, ``dut1`` = UT1-UTC in **seconds**. All
/// four spans must have the same non-zero length. Replaces any earlier table.
void set_table(std::span<const double> mjd, std::span<const double> xp_arcsec,
               std::span<const double> yp_arcsec, std::span<const double> dut1_s);

/// True once ``set_table`` has been called.
bool has_table();

/// First and last MJD of the table. Throws ``std::logic_error`` if unset.
std::pair<double, double> mjd_range();

/// True where ``mjd`` (any scale; the boundary tolerance is a day) has EOP.
bool in_iers_era(double mjd);

/// Interpolated EOP at a UTC MJD: polar motion in **radians**, UT1-UTC in
/// **seconds**; outside the table the endpoint values are held (np.interp).
struct Eop {
    double xp_rad;
    double yp_rad;
    double dut1_s;
};
Eop interpolate(double mjd_utc);

}  // namespace eclipse::eop

// IERS Earth orientation parameters: polar motion and UT1-UTC [IERS2010].
//
// The core reads the IERS ``finals2000A.all`` file itself (``load_file``; the
// API passes the copy bundled by the ``astropy-iers-data`` package) and
// interpolates the Bulletin A values linearly between the daily rows,
// holding the end values outside the table (``np.interp`` semantics).
// UT1-UTC is interpolated as UT1-TAI across a leap second (``interpolate``).
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <utility>

namespace eclipse::eop {

/// Install the Bulletin A table: ``mjd`` (UTC, ascending), polar motion
/// ``xp``/``yp`` in **arcseconds**, ``dut1`` = UT1-UTC in **seconds**. All
/// four spans must have the same non-zero length. Replaces any earlier table.
void set_table(std::span<const double> mjd, std::span<const double> xp_arcsec,
               std::span<const double> yp_arcsec, std::span<const double> dut1_s);

/// Parse an IERS ``finals2000A.all`` file and install its Bulletin A table
/// (``set_table``). Fixed-width columns (IERS ``readme.finals2000A``): MJD in
/// characters 8-15, PM-x 19-27, PM-y 38-46 [arcsec], UT1-UTC 59-68 [s]; the
/// table ends at the first row without a Bulletin A polar motion (the file's
/// far-future padding). Returns the number of rows. Throws
/// ``std::runtime_error`` if the file cannot be read or has no rows.
std::size_t load_file(const std::string& path);

/// The file ``load_file`` read ("" for none, or a table from ``set_table``).
std::string source();

/// A counter bumped by every ``set_table`` / ``load_file``: identifies the
/// installed table (caches of EOP-dependent results key on it).
std::uint64_t generation();

/// True once ``set_table`` has been called.
bool has_table();

/// First and last MJD of the table. Throws ``std::logic_error`` if unset.
std::pair<double, double> mjd_range();

/// True where ``mjd`` lies inside the table's [first, last] MJD, inclusive.
/// The scale is the caller's (the table is UTC; a TT MJD differs by about a
/// minute, far below the daily row spacing).
bool in_iers_era(double mjd);

/// Interpolated EOP at a UTC MJD: polar motion in **radians**, UT1-UTC in
/// **seconds**; outside the table the endpoint values are held (np.interp).
/// Between two rows that straddle a leap second (a UT1-UTC change of more
/// than 0.5 s), UT1-UTC is interpolated as UT1-TAI, so it is continuous up
/// to the step at the row after the leap second [IERS2010] ch. 5.
struct Eop {
    double xp_rad;
    double yp_rad;
    double dut1_s;
};
Eop interpolate(double mjd_utc);

}  // namespace eclipse::eop

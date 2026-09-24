// IERS Earth orientation parameters: polar motion and UT1-UTC [IERS2010].
//
// The core reads the IERS ``finals2000A.all`` file itself (``load_file``; the
// API passes the copy bundled by the ``astropy-iers-data`` package) and
// interpolates the Bulletin A values linearly between the daily rows,
// holding the end values outside the table (``np.interp`` semantics).
#pragma once

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

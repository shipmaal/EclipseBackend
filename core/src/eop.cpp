#include "eclipse/eop.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "eclipse/constants.hpp"

namespace eclipse::eop {

namespace {

struct Table {
    std::vector<double> mjd, xp, yp, dut1;
};

std::mutex& table_mutex() {
    static std::mutex m;
    return m;
}

std::shared_ptr<const Table>& table_slot() {
    static std::shared_ptr<const Table> t;
    return t;
}

std::shared_ptr<const Table> table() {
    std::scoped_lock lock(table_mutex());
    auto t = table_slot();
    if (!t) throw std::logic_error("eclipse::eop: EOP table not set (call set_table first)");
    return t;
}

// ``np.interp(x, xp, fp)`` for one point: NumPy's arr_interp (compiled_base.c):
// j = index with xp[j] <= x < xp[j+1]; x < xp[0] -> fp[0]; x >= xp[-1] -> fp[-1];
// exact hit -> fp[j]; else slope*(x - xp[j]) + fp[j], recomputed from the right
// end if that produced a NaN.
double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    const auto n = static_cast<std::ptrdiff_t>(xp.size());
    const auto it = std::upper_bound(xp.begin(), xp.end(), x);
    const std::ptrdiff_t j = (it - xp.begin()) - 1;  // largest j with xp[j] <= x, or -1
    if (j < 0) return fp[0];
    if (j == n - 1) return fp[static_cast<size_t>(n - 1)];
    const auto k = static_cast<size_t>(j);
    if (xp[k] == x) return fp[k];
    const double slope = (fp[k + 1] - fp[k]) / (xp[k + 1] - xp[k]);
    double r = slope * (x - xp[k]) + fp[k];
    if (std::isnan(r)) {
        r = slope * (x - xp[k + 1]) + fp[k + 1];
        if (std::isnan(r) && fp[k] == fp[k + 1]) r = fp[k];
    }
    return r;
}

}  // namespace

void set_table(std::span<const double> mjd, std::span<const double> xp_arcsec,
               std::span<const double> yp_arcsec, std::span<const double> dut1_s) {
    if (mjd.empty() || mjd.size() != xp_arcsec.size() || mjd.size() != yp_arcsec.size() ||
        mjd.size() != dut1_s.size())
        throw std::invalid_argument("eclipse::eop::set_table: four equal-length, non-empty columns");
    if (!std::is_sorted(mjd.begin(), mjd.end()))
        throw std::invalid_argument("eclipse::eop::set_table: mjd must be ascending");
    auto t = std::make_shared<Table>();
    t->mjd.assign(mjd.begin(), mjd.end());
    t->xp.assign(xp_arcsec.begin(), xp_arcsec.end());
    t->yp.assign(yp_arcsec.begin(), yp_arcsec.end());
    t->dut1.assign(dut1_s.begin(), dut1_s.end());
    std::scoped_lock lock(table_mutex());
    table_slot() = std::move(t);
}

bool has_table() {
    std::scoped_lock lock(table_mutex());
    return static_cast<bool>(table_slot());
}

std::pair<double, double> mjd_range() {
    const auto t = table();
    return {t->mjd.front(), t->mjd.back()};
}

bool in_iers_era(double mjd) {
    const auto [lo, hi] = mjd_range();
    return mjd >= lo && mjd <= hi;
}

Eop interpolate(double mjd_utc) {
    const auto t = table();
    // app/eop.py: ``np.interp(...) * _ARCSEC_TO_RAD`` — interpolate, then scale.
    return {interp(mjd_utc, t->mjd, t->xp) * constants::ARCSEC_TO_RAD,
            interp(mjd_utc, t->mjd, t->yp) * constants::ARCSEC_TO_RAD,
            interp(mjd_utc, t->mjd, t->dut1)};
}

}  // namespace eclipse::eop

#include "eclipse/eop.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "eclipse/constants.hpp"

namespace eclipse::eop {

namespace {

struct Table {
    std::vector<double> mjd, xp, yp, dut1;
    std::string source;
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

// UT1-UTC at a UTC MJD. UT1-UTC steps by an integer second at each leap
// second, and a straight line across the step is wrong by up to half a
// second on the day before it. UT1-TAI has no step, so interpolate that
// instead, as the IERS practice is [IERS2010] ch. 5 (the daily Bulletin A
// values are tabulated at 0h UTC; a leap second is inserted at the end of
// the day before the row that carries the step). The step is recognised in
// the table itself: UT1-UTC changes by a few milliseconds a day, so any
// change between rows larger than half a second is a leap second, of the
// integer size s = round(delta). On [mjd_k, mjd_k+1) the value is
//   dut1_k + w (dut1_k+1 - s - dut1_k),
// i.e. UT1-TAI interpolated plus the TAI-UTC that holds on day k. Exact
// hits and the ends behave as ``interp``.
double interp_dut1(double x, const std::vector<double>& mjd, const std::vector<double>& dut1) {
    const auto n = static_cast<std::ptrdiff_t>(mjd.size());
    const auto it = std::upper_bound(mjd.begin(), mjd.end(), x);
    const std::ptrdiff_t j = (it - mjd.begin()) - 1;
    if (j < 0 || j == n - 1) return interp(x, mjd, dut1);
    const auto k = static_cast<size_t>(j);
    const double step = std::round(dut1[k + 1] - dut1[k]);
    if (step == 0.0 || mjd[k] == x) return interp(x, mjd, dut1);
    const double w = (x - mjd[k]) / (mjd[k + 1] - mjd[k]);
    return dut1[k] + w * (dut1[k + 1] - step - dut1[k]);
}

// ``line[a:b].strip()`` (Python slice semantics: clipped to the line).
std::string field(const std::string& line, std::size_t a, std::size_t b) {
    if (a >= line.size()) return {};
    std::string f = line.substr(a, std::min(b, line.size()) - a);
    const auto first = f.find_first_not_of(" \t\r");
    if (first == std::string::npos) return {};
    const auto last = f.find_last_not_of(" \t\r");
    return f.substr(first, last - first + 1);
}

double number(const std::string& f, const std::string& path) {
    char* end = nullptr;
    const double v = std::strtod(f.c_str(), &end);  // correctly rounded, as Python float()
    if (f.empty() || end != f.c_str() + f.size())
        throw std::runtime_error("eclipse::eop::load_file: bad number '" + f + "' in " + path);
    return v;
}

void install(std::shared_ptr<Table> t) {
    std::scoped_lock lock(table_mutex());
    table_slot() = std::move(t);
}

}  // namespace

std::size_t load_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("eclipse::eop::load_file: cannot read " + path);
    auto t = std::make_shared<Table>();
    std::string line;
    while (std::getline(in, line)) {
        const std::string pm_x = field(line, 18, 27);
        if (pm_x.empty()) break;  // no Bulletin A polar motion beyond this row
        t->mjd.push_back(number(field(line, 7, 15), path));
        t->xp.push_back(number(pm_x, path));
        t->yp.push_back(number(field(line, 37, 46), path));
        t->dut1.push_back(number(field(line, 58, 68), path));
    }
    if (t->mjd.empty()) throw std::runtime_error("eclipse::eop::load_file: no rows in " + path);
    if (!std::is_sorted(t->mjd.begin(), t->mjd.end()))
        throw std::runtime_error("eclipse::eop::load_file: MJD not ascending in " + path);
    t->source = path;
    const std::size_t n = t->mjd.size();
    install(std::move(t));
    return n;
}

std::string source() {
    std::scoped_lock lock(table_mutex());
    return table_slot() ? table_slot()->source : std::string();
}

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
    install(std::move(t));
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
    // Interpolate in arcseconds, then scale to radians.
    return {interp(mjd_utc, t->mjd, t->xp) * constants::ARCSEC_TO_RAD,
            interp(mjd_utc, t->mjd, t->yp) * constants::ARCSEC_TO_RAD,
            interp_dut1(mjd_utc, t->mjd, t->dut1)};
}

}  // namespace eclipse::eop

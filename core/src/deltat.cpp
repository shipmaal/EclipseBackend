#include "eclipse/deltat.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <limits>

namespace eclipse::deltat {

namespace {

// np.polyval: ``y = 0; for c in p: y = y * x + c`` (highest power first).
double polyval(std::initializer_list<double> p, double x) {
    double y = 0.0;
    for (const double c : p) y = y * x + c;
    return y;
}

// Long-term parabola argument: u = (y - 1820) / 100 [Espenak] sec. 2.5.
double u(double y) { return (y - 1820.0) / 100.0; }

// -20 + 32 u^2 (np: ``u ** 2`` is ``square``, i.e. u * u).
double parabola(double y) { return -20.0 + 32.0 * (u(y) * u(y)); }

}  // namespace

double delta_t_seconds(double y) {
    // Segment boundaries and coefficients: [Espenak] sec. 2.5, identical to
    // app/deltat.py (which is the citation trail for each line).
    if (y < -500.0) return parabola(y);
    if (y < 500.0)
        return polyval({0.0090316521, 0.022174192, -0.1798452, -5.952053, 33.78311, -1014.41, 10583.6},
                       y / 100.0);
    if (y < 1600.0)
        return polyval({0.0083572073, -0.005050998, -0.8503463, 0.319781, 71.23472, -556.01, 1574.2},
                       (y - 1000.0) / 100.0);
    if (y < 1700.0) return polyval({1.0 / 7129.0, -0.01532, -0.9808, 120.0}, y - 1600.0);
    if (y < 1800.0)
        return polyval({-1.0 / 1174000.0, 0.00013336, -0.0059285, 0.1603, 8.83}, y - 1700.0);
    if (y < 1860.0)
        return polyval({0.000000000875, -0.0000001699, 0.0000121272, -0.00037436, 0.0041116,
                        0.0068612, -0.332447, 13.72},
                       y - 1800.0);
    if (y < 1900.0)
        return polyval({1.0 / 233174.0, -0.0004473624, 0.01680668, -0.251754, 0.5737, 7.62},
                       y - 1860.0);
    if (y < 1920.0)
        return polyval({-0.000197, 0.0061966, -0.0598939, 1.494119, -2.79}, y - 1900.0);
    if (y < 1941.0) return polyval({0.0020936, -0.076100, 0.84493, 21.20}, y - 1920.0);
    if (y < 1961.0) return polyval({1.0 / 2547.0, -1.0 / 233.0, 0.407, 29.07}, y - 1950.0);
    if (y < 1986.0) return polyval({-1.0 / 718.0, -1.0 / 260.0, 1.067, 45.45}, y - 1975.0);
    if (y < 2005.0)
        return polyval({0.00002373599, 0.000651814, 0.0017275, -0.060374, 0.3345, 63.86},
                       y - 2000.0);
    if (y < 2050.0) return polyval({0.005589, 0.32217, 62.92}, y - 2000.0);
    if (y < 2150.0) return parabola(y) - 0.5628 * (2150.0 - y);
    return parabola(y);
}

std::vector<double> delta_t_seconds(std::span<const double> years) {
    std::vector<double> out;
    out.reserve(years.size());
    for (const double y : years) out.push_back(delta_t_seconds(y));
    return out;
}

namespace {

struct Predictions {
    std::vector<double> mjd, dt, err;
    std::string source;
};

std::mutex& pred_mutex() {
    static std::mutex m;
    return m;
}

std::shared_ptr<const Predictions>& pred_slot() {
    static std::shared_ptr<const Predictions> p;
    return p;
}

std::uint64_t& pred_generation() {
    static std::uint64_t g = 0;
    return g;
}

void install(std::shared_ptr<Predictions> p) {
    if (p->mjd.size() < 2 || p->dt.size() != p->mjd.size() || p->err.size() != p->mjd.size())
        throw std::invalid_argument("eclipse::deltat: predictions need >= 2 equal-length rows");
    for (std::size_t i = 1; i < p->mjd.size(); ++i)
        if (!(p->mjd[i] > p->mjd[i - 1]))
            throw std::invalid_argument("eclipse::deltat: prediction MJD must increase");
    std::scoped_lock lock(pred_mutex());
    pred_slot() = std::move(p);
    ++pred_generation();
}

std::shared_ptr<const Predictions> snapshot() {
    std::scoped_lock lock(pred_mutex());
    return pred_slot();
}

}  // namespace

void set_predictions(std::span<const double> mjd, std::span<const double> dt_s,
                     std::span<const double> err_s) {
    auto p = std::make_shared<Predictions>();
    p->mjd.assign(mjd.begin(), mjd.end());
    p->dt.assign(dt_s.begin(), dt_s.end());
    p->err.assign(err_s.begin(), err_s.end());
    install(std::move(p));
}

std::size_t load_predictions(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("eclipse::deltat::load_predictions: cannot read " + path);
    auto p = std::make_shared<Predictions>();
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (header) {  // "MJD  Year  TT-UT  UT1-UTC  Error"
            header = false;
            continue;
        }
        std::istringstream ss(line);
        std::vector<double> v;
        for (std::string tok; ss >> tok;) {
            char* end = nullptr;
            const double x = std::strtod(tok.c_str(), &end);
            if (end != tok.c_str() + tok.size())
                throw std::runtime_error("eclipse::deltat::load_predictions: bad number '" + tok +
                                         "' in " + path);
            v.push_back(x);
        }
        if (v.empty()) continue;
        // MJD, year, TT-UT, [UT1-UTC,] error: the optional column is present
        // only near the issue date.
        if (v.size() != 4 && v.size() != 5)
            throw std::runtime_error("eclipse::deltat::load_predictions: bad row in " + path);
        p->mjd.push_back(v[0]);
        p->dt.push_back(v[2]);
        p->err.push_back(v.back());
    }
    p->source = path;
    const std::size_t n = p->mjd.size();
    try {
        install(std::move(p));
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(std::string("eclipse::deltat::load_predictions: ") + e.what() +
                                 " in " + path);
    }
    return n;
}

std::string predictions_source() {
    const auto p = snapshot();
    return p ? p->source : std::string();
}

std::pair<double, double> predictions_mjd_range() {
    const auto p = snapshot();
    if (!p) return {std::nan(""), std::nan("")};
    return {p->mjd.front(), p->mjd.back()};
}

std::uint64_t predictions_generation() {
    std::scoped_lock lock(pred_mutex());
    return pred_generation();
}

std::optional<Predicted> predicted(double mjd) {
    const auto p = snapshot();
    if (!p || !(mjd >= p->mjd.front() && mjd <= p->mjd.back())) return std::nullopt;
    auto it = std::upper_bound(p->mjd.begin(), p->mjd.end(), mjd);
    std::size_t k = static_cast<std::size_t>(it - p->mjd.begin());
    if (k == p->mjd.size()) --k;  // mjd == last row
    const std::size_t j = k - 1;
    const double w = (mjd - p->mjd[j]) / (p->mjd[k] - p->mjd[j]);
    return Predicted{p->dt[j] + w * (p->dt[k] - p->dt[j]), p->err[j] + w * (p->err[k] - p->err[j])};
}

double decimal_year_from_jd(double jd) { return 2000.0 + (jd - 2451545.0) / 365.25; }

}  // namespace eclipse::deltat

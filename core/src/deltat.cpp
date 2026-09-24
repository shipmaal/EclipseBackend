#include "eclipse/deltat.hpp"

#include <algorithm>
#include <initializer_list>
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

double delta_t_after_record(double year, double year_end, double dt_end_s) {
    const double model = delta_t_seconds(year);
    if (!(year > year_end)) return model;
    // The horizon: the end of [Espenak]'s 2005-2050 segment, or ten years on.
    const double y_h = std::max(2050.0, year_end + 10.0);
    if (year >= y_h) return model;
    const double offset = dt_end_s - delta_t_seconds(year_end);
    return model + offset * ((y_h - year) / (y_h - year_end));
}

std::vector<double> delta_t_seconds(std::span<const double> years) {
    std::vector<double> out;
    out.reserve(years.size());
    for (const double y : years) out.push_back(delta_t_seconds(y));
    return out;
}

double decimal_year_from_jd(double jd) { return 2000.0 + (jd - 2451545.0) / 365.25; }

}  // namespace eclipse::deltat

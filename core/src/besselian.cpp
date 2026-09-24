#include "eclipse/besselian.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "eclipse/ellipsoid.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/geometry.hpp"
#include "eclipse/numerics.hpp"

namespace eclipse::besselian {

namespace {

std::vector<double> et_of(double et0, std::span<const double> t_hours) {
    std::vector<double> et(t_hours.size());
    for (std::size_t i = 0; i < et.size(); ++i) et[i] = et0 + t_hours[i] * 3600.0;
    return et;
}

template <std::size_t N>
std::array<double, N> fixed(const std::vector<double>& c) {
    std::array<double, N> a{};
    std::copy(c.begin(), c.end(), a.begin());
    return a;
}

double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (const double x : v) s += x;
    return s / static_cast<double>(v.size());
}

}  // namespace

Elements elements_direct(double et0, Frame frame, std::span<const double> t_hours, double k1,
                         double k2) {
    Elements e = ephem::besselian_instants(et_of(et0, t_hours), frame, k1, k2);
    unwrap_mu_deg(e);
    return e;
}

Rates element_rates(double et0, Frame frame, std::span<const double> t_hours) {
    const std::size_t n = t_hours.size();
    const double dt = geometry::RATE_DT_H;
    std::vector<double> t(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        t[i] = t_hours[i] - dt;
        t[n + i] = t_hours[i] + dt;
    }
    const Elements e = elements_direct(et0, frame, t);
    Rates r;
    const auto diff = [&](const std::vector<double>& v, std::vector<double>& out) {
        out.resize(n);
        for (std::size_t i = 0; i < n; ++i) out[i] = (v[n + i] - v[i]) / (2.0 * dt);
    };
    diff(e.x, r.x);
    diff(e.y, r.y);
    diff(e.d, r.d);
    diff(e.l1, r.l1);
    diff(e.l2, r.l2);
    r.mu.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        r.mu[i] = (numerics::np_remainder(e.mu[n + i] - e.mu[i] + 180.0, 360.0) - 180.0) / (2.0 * dt);
    return r;
}

std::vector<double> polyfit(std::span<const double> t, std::span<const double> y, int deg) {
    const std::size_t m = t.size();
    const auto n = static_cast<std::size_t>(deg + 1);
    if (deg < 0 || m < n || y.size() != m)
        throw std::invalid_argument("besselian::polyfit: need t.size() == y.size() > deg >= 0");
    // Column-scaled Vandermonde matrix A[i][j] = t_i^j / scl_j (column-major).
    std::vector<double> a(m * n), scl(n);
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double p = std::pow(t[i], static_cast<double>(j));
            a[j * m + i] = p;
            s += p * p;
        }
        scl[j] = s > 0.0 ? std::sqrt(s) : 1.0;
        for (std::size_t i = 0; i < m; ++i) a[j * m + i] /= scl[j];
    }
    std::vector<double> b(y.begin(), y.end());
    // Householder QR: reduce A to R in place, applying each reflector to b.
    for (std::size_t k = 0; k < n; ++k) {
        double norm = 0.0;
        for (std::size_t i = k; i < m; ++i) norm += a[k * m + i] * a[k * m + i];
        norm = std::sqrt(norm);
        if (norm == 0.0) throw std::invalid_argument("besselian::polyfit: rank-deficient system");
        const double alpha = a[k * m + k] > 0.0 ? -norm : norm;
        std::vector<double> v(a.begin() + static_cast<std::ptrdiff_t>(k * m + k),
                              a.begin() + static_cast<std::ptrdiff_t>(k * m + m));
        v[0] -= alpha;
        double vv = 0.0;
        for (const double x : v) vv += x * x;
        const auto reflect = [&](double* col) {
            double s = 0.0;
            for (std::size_t i = 0; i < v.size(); ++i) s += v[i] * col[k + i];
            s = 2.0 * s / vv;
            for (std::size_t i = 0; i < v.size(); ++i) col[k + i] -= s * v[i];
        };
        for (std::size_t j = k; j < n; ++j) reflect(&a[j * m]);
        reflect(b.data());
    }
    // Back substitution R c' = (Q^T b)[0:n], then undo the column scaling.
    std::vector<double> c(n);
    for (std::size_t kk = n; kk-- > 0;) {
        double s = b[kk];
        for (std::size_t j = kk + 1; j < n; ++j) s -= a[j * m + kk] * c[j];
        c[kk] = s / a[kk * m + kk];
    }
    for (std::size_t j = 0; j < n; ++j) c[j] /= scl[j];
    return c;
}

Polynomials fit_polynomials(double et0, Frame frame, double half_window_hours, double step_hours) {
    const std::vector<double> t =
        numerics::arange(-half_window_hours, half_window_hours + step_hours / 2.0, step_hours);
    if (t.size() < 4)
        throw std::invalid_argument("besselian::fit_polynomials: fewer than 4 samples in the window");
    const Elements e = elements_direct(et0, frame, t);  // mu unwrapped over the samples
    Polynomials p{};
    p.x = fixed<4>(polyfit(t, e.x, 3));
    p.y = fixed<4>(polyfit(t, e.y, 3));
    p.d = fixed<3>(polyfit(t, e.d, 2));
    p.l1 = fixed<3>(polyfit(t, e.l1, 2));
    p.l2 = fixed<3>(polyfit(t, e.l2, 2));
    p.mu = fixed<2>(polyfit(t, e.mu, 1));
    p.tan_f1 = mean(e.tan_f1);
    p.tan_f2 = mean(e.tan_f2);
    return p;
}

CentralLine central_line(double et0, Frame frame, std::span<const double> t_hours) {
    const Elements e = elements_direct(et0, frame, t_hours);
    const ellipsoid::Geographic g = ellipsoid::fund_to_geo(e.x, e.y, e.d, e.mu);

    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < t_hours.size(); ++i)
        if (!std::isnan(g.lat_deg[i])) idx.push_back(i);
    const std::size_t n = idx.size();

    CentralLine c;
    c.t_hours.resize(n);
    c.lat_deg.resize(n);
    c.lon_deg.resize(n);
    c.bearing_deg.resize(n);
    c.penumbra_km.resize(n);
    c.umbra_km.resize(n);
    c.is_total.resize(n);
    std::vector<double> x(n), y(n), d(n), mu(n), l1(n), l2(n), tf1(n), tf2(n);
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t i = idx[k];
        c.t_hours[k] = t_hours[i];
        c.lat_deg[k] = g.lat_deg[i];
        c.lon_deg[k] = g.lon_deg[i];
        x[k] = e.x[i], y[k] = e.y[i], d[k] = e.d[i], mu[k] = e.mu[i];
        l1[k] = e.l1[i], l2[k] = e.l2[i], tf1[k] = e.tan_f1[i], tf2[k] = e.tan_f2[i];
        const geometry::ShadowRadii r =
            geometry::shadow_radii(x[k], y[k], d[k], l1[k], l2[k], tf1[k], tf2[k]);
        c.penumbra_km[k] = r.penumbra_km;
        c.umbra_km[k] = r.umbra_km;
        c.is_total[k] = r.is_total;
    }
    // Along-track bearing: previous on-Earth point to the next (clamped).
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t p = k == 0 ? 0 : k - 1, q = std::min(n - 1, k + 1);
        c.bearing_deg[k] = n > 1 ? geometry::bearing_deg(c.lat_deg[p], c.lon_deg[p],
                                                         c.lat_deg[q], c.lon_deg[q])
                                 : 0.0;
    }
    if (n == 0) return c;

    // Umbral limits: the path envelope, from the element rates at each point.
    const Rates r = element_rates(et0, frame, c.t_hours);
    const geometry::EdgeRates er{r.x, r.y, r.d, r.mu, r.l2};
    geometry::EdgeLimits u = geometry::shadow_edge_limits(x, y, d, mu, l2, tf2, c.bearing_deg,
                                                          UMBRA_LIMIT_MAX_KM, true, &er);
    c.north_lat = std::move(u.north_lat);
    c.north_lon = std::move(u.north_lon);
    c.south_lat = std::move(u.south_lat);
    c.south_lon = std::move(u.south_lon);
    c.width_km = std::move(u.width_km);
    // Penumbral limits: the shadow outline at the instant, clipped to the terminator.
    geometry::EdgeLimits p = geometry::shadow_edge_limits(x, y, d, mu, l1, tf1, c.bearing_deg,
                                                          PENUMBRA_LIMIT_MAX_KM, true, nullptr);
    c.pen_north_lat = std::move(p.north_lat);
    c.pen_north_lon = std::move(p.north_lon);
    c.pen_south_lat = std::move(p.south_lat);
    c.pen_south_lon = std::move(p.south_lon);
    return c;
}

}  // namespace eclipse::besselian

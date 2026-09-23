// See eclipse/limb.hpp. Arithmetic follows app/limb.py's NumPy evaluation
// order exactly (roadmap §7); the bin maximum is order-independent.
#include "eclipse/limb.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "eclipse/constants.hpp"

namespace eclipse::limb {

namespace {

struct Band {
    std::vector<std::int32_t> line, first, count;
    std::vector<std::size_t> offset;  // first point of each run
    std::vector<std::size_t> line_runs;  // runs of line i: [line_runs[i], line_runs[i + 1])
    std::vector<std::int16_t> dn;
    std::vector<std::int32_t> right, down;  // neighbour point index, -1: none
    std::vector<double> cos_lat, sin_lat, cos_lon, sin_lon;
    double offset_km = 0.0, scale_km = 0.0, band_deg = 0.0;
};

std::mutex& band_mutex() {
    static std::mutex m;
    return m;
}

std::shared_ptr<const Band>& band_slot() {
    static std::shared_ptr<const Band> b;
    return b;
}

std::shared_ptr<const Band> band() {
    std::scoped_lock lock(band_mutex());
    auto b = band_slot();
    if (!b) throw std::logic_error("eclipse::limb: limb band not set (call set_band first)");
    return b;
}

// app.limb._fill_empty: periodic linear interpolation across empty bins,
// ``a + (b - a) * (m / L)``; too many empty bins is an error.
void fill_empty(std::vector<double>& rho) {
    const std::size_t n = rho.size();
    std::vector<std::size_t> filled;
    for (std::size_t k = 0; k < n; ++k)
        if (std::isfinite(rho[k])) filled.push_back(k);
    if (filled.size() == n) return;
    if (static_cast<double>(n - filled.size()) > MAX_EMPTY_FRACTION * static_cast<double>(n) ||
        filled.empty())
        throw std::invalid_argument(
            "eclipse::limb::silhouette: limb profile has too many empty bins: n_bins too fine for "
            "this DEM");
    const std::vector<double> in = rho;
    for (std::size_t f = 0; f < filled.size(); ++f) {
        const std::size_t a_i = filled[f], b_i = filled[(f + 1) % filled.size()];
        const std::size_t gap = (b_i + n - a_i) % n;
        if (gap <= 1) continue;
        const double a = in[a_i], b = in[b_i];
        for (std::size_t m = 1; m < gap; ++m)
            rho[(a_i + m) % n] = a + (b - a) * (static_cast<double>(m) / static_cast<double>(gap));
    }
}

}  // namespace

void set_band(std::span<const std::int32_t> line, std::span<const std::int32_t> first,
              std::span<const std::int32_t> count, std::span<const std::int16_t> dn,
              std::span<const std::int32_t> right, std::span<const std::int32_t> down,
              std::span<const double> cos_lat, std::span<const double> sin_lat,
              std::span<const double> cos_lon, std::span<const double> sin_lon,
              double offset_km, double scale_km, double band_deg) {
    if (line.size() != first.size() || line.size() != count.size() ||
        cos_lat.size() != sin_lat.size() || cos_lon.size() != sin_lon.size())
        throw std::invalid_argument("eclipse::limb::set_band: mismatched array lengths");
    auto b = std::make_shared<Band>();
    b->offset.reserve(line.size());
    std::size_t total = 0;
    for (std::size_t r = 0; r < line.size(); ++r) {
        if (line[r] < 0 || static_cast<std::size_t>(line[r]) >= cos_lat.size() || first[r] < 0 ||
            count[r] < 0 ||
            static_cast<std::size_t>(first[r]) + static_cast<std::size_t>(count[r]) > cos_lon.size())
            throw std::invalid_argument("eclipse::limb::set_band: run outside the DEM grid");
        b->offset.push_back(total);
        total += static_cast<std::size_t>(count[r]);
    }
    if (total != dn.size() || right.size() != total || down.size() != total)
        throw std::invalid_argument("eclipse::limb::set_band: runs do not cover dn/right/down");
    for (std::size_t p = 0; p < total; ++p)
        if (right[p] >= static_cast<std::int32_t>(total) || down[p] >= static_cast<std::int32_t>(total))
            throw std::invalid_argument("eclipse::limb::set_band: neighbour index out of range");
    for (std::size_t r = 1; r < line.size(); ++r)
        if (line[r] < line[r - 1])
            throw std::invalid_argument("eclipse::limb::set_band: runs must be in line order");
    b->line_runs.assign(cos_lat.size() + 1, 0);
    for (const std::int32_t li : line) ++b->line_runs[static_cast<std::size_t>(li) + 1];
    for (std::size_t i = 1; i < b->line_runs.size(); ++i) b->line_runs[i] += b->line_runs[i - 1];
    b->line.assign(line.begin(), line.end());
    b->first.assign(first.begin(), first.end());
    b->count.assign(count.begin(), count.end());
    b->dn.assign(dn.begin(), dn.end());
    b->right.assign(right.begin(), right.end());
    b->down.assign(down.begin(), down.end());
    b->cos_lat.assign(cos_lat.begin(), cos_lat.end());
    b->sin_lat.assign(sin_lat.begin(), sin_lat.end());
    b->cos_lon.assign(cos_lon.begin(), cos_lon.end());
    b->sin_lon.assign(sin_lon.begin(), sin_lon.end());
    b->offset_km = offset_km;
    b->scale_km = scale_km;
    b->band_deg = band_deg;
    std::scoped_lock lock(band_mutex());
    band_slot() = std::move(b);
}

bool has_band() {
    std::scoped_lock lock(band_mutex());
    return static_cast<bool>(band_slot());
}

std::vector<double> silhouette(const std::array<double, 9>& axes, int n_bins) {
    if (n_bins < 1) throw std::invalid_argument("eclipse::limb::silhouette: n_bins < 1");
    const auto b = band();
    const double cover = std::cos((b->band_deg - VISIBLE_DEG) * constants::DEG_TO_RAD);
    if (std::abs(axes[6]) < cover)
        throw std::invalid_argument(
            "eclipse::limb::silhouette: the view axis leaves the limb band's coverage");

    const double x0 = axes[0], x1 = axes[1], x2 = axes[2];
    const double y0 = axes[3], y1 = axes[4], y2 = axes[5];
    const double floor_km = R_REF_KM - FLOOR_KM;
    const double pi = std::numbers::pi;
    const double per_rad = static_cast<double>(n_bins) / (2.0 * pi);
    const double half = static_cast<double>(n_bins) / 2.0;
    const double nbd = static_cast<double>(n_bins);
    const auto nb = static_cast<std::size_t>(n_bins);
    const std::size_t samples = b->cos_lon.size();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    std::vector<double> rho(nb, neg_inf);

    // One DEM point -> (s, u): radius and continuous bin coordinate (u only if
    // s > floor), in app/limb.py silhouette's operation order.
    struct Proj {
        double s, u;
        bool near;
    };
    auto project = [&](std::size_t li, std::size_t j, std::size_t p) {
        const double cl = b->cos_lat[li];
        const double rad = b->offset_km + static_cast<double>(b->dn[p]) * b->scale_km;
        const double px = rad * (cl * b->cos_lon[j]);
        const double py = rad * (cl * b->sin_lon[j]);
        const double pz = rad * b->sin_lat[li];
        const double qx = (px * x0 + py * x1) + pz * x2;
        const double qy = (px * y0 + py * y1) + pz * y2;
        Proj r{std::sqrt(qx * qx + qy * qy), 0.0, false};
        if (r.s > floor_km) {
            r.near = true;
            r.u = (std::atan2(qy, qx) + pi) * per_rad;
        }
        return r;
    };
    auto wrap_bin = [&](std::ptrdiff_t i) {  // Python % n_bins
        auto k = i % static_cast<std::ptrdiff_t>(n_bins);
        if (k < 0) k += n_bins;
        return static_cast<std::size_t>(k);
    };

    const auto n_lines = static_cast<std::ptrdiff_t>(b->cos_lat.size());

    // Lines are processed in order with two row buffers (this line, the next),
    // so each point is projected once per block of lines: its right neighbour
    // is in the same row buffer and its down neighbour in the next one. The
    // max over points and edges is the same whatever the order.
#ifdef _OPENMP
#pragma omp parallel if (!omp_in_parallel())
#endif
    {
        std::vector<double> local(nb, neg_inf);
        std::vector<Proj> cur(samples), nxt(samples);
        auto put = [&](std::size_t k, double v) {
            if (v > local[k]) local[k] = v;
        };
        // app.limb._edge_crossings for one edge.
        auto edge = [&](const Proj& a, const Proj& e) {
            double du = e.u - a.u;
            du = du > half ? du - nbd : (du <= -half ? du + nbd : du);
            const double ub = a.u + du;
            const double first = std::floor(std::min(a.u, ub)) + 1.0;
            const double last = std::floor(std::max(a.u, ub));
            const auto span = static_cast<std::ptrdiff_t>(last - first);
            for (std::ptrdiff_t m = 0; m <= span; ++m) {
                const double bnd = first + static_cast<double>(m);
                const double frac = (bnd - a.u) / du;
                const double rad = a.s + (e.s - a.s) * frac;
                const auto ib = static_cast<std::ptrdiff_t>(bnd);
                put(wrap_bin(ib - 1), rad);
                put(wrap_bin(ib), rad);
            }
        };
        auto fill_row = [&](std::size_t li, std::vector<Proj>& row) {
            for (std::size_t r = b->line_runs[li]; r < b->line_runs[li + 1]; ++r) {
                const auto j0 = static_cast<std::size_t>(b->first[r]);
                const auto cnt = static_cast<std::size_t>(b->count[r]);
                for (std::size_t c = 0; c < cnt; ++c)
                    row[j0 + c] = project(li, j0 + c, b->offset[r] + c);
            }
        };
        std::ptrdiff_t filled_next = -1;  // line held in ``nxt``, if any
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t ll = 0; ll < n_lines; ++ll) {
            const auto li = static_cast<std::size_t>(ll);
            if (b->line_runs[li] == b->line_runs[li + 1]) continue;
            if (filled_next == ll)
                std::swap(cur, nxt);
            else
                fill_row(li, cur);
            const bool has_below = ll + 1 < n_lines;
            if (has_below) {
                fill_row(li + 1, nxt);
                filled_next = ll + 1;
            }
            for (std::size_t r = b->line_runs[li]; r < b->line_runs[li + 1]; ++r) {
                const auto j0 = static_cast<std::size_t>(b->first[r]);
                const auto cnt = static_cast<std::size_t>(b->count[r]);
                const std::size_t p0 = b->offset[r];
                for (std::size_t c = 0; c < cnt; ++c) {
                    const std::size_t j = j0 + c, p = p0 + c;
                    const Proj& a = cur[j];
                    if (!a.near) continue;
                    auto k = static_cast<std::ptrdiff_t>(std::floor(a.u));
                    k = std::min<std::ptrdiff_t>(k, n_bins - 1);
                    put(static_cast<std::size_t>(k), a.s);
                    if (b->right[p] >= 0) {
                        const Proj& e = cur[(j + 1) % samples];
                        if (e.near) edge(a, e);
                    }
                    if (b->down[p] >= 0 && has_below) {
                        const Proj& e = nxt[j];
                        if (e.near) edge(a, e);
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp critical(eclipse_limb_merge)
#endif
        for (std::size_t k = 0; k < nb; ++k)
            if (local[k] > rho[k]) rho[k] = local[k];
    }
    fill_empty(rho);
    for (double& v : rho) v = v - R_REF_KM;
    return rho;
}

std::vector<double> delta_rho_at(std::span<const double> profile, std::span<const double> psi) {
    const auto n = static_cast<std::ptrdiff_t>(profile.size());
    if (n < 1) throw std::invalid_argument("eclipse::limb::delta_rho_at: empty profile");
    const double pi = std::numbers::pi;
    const double per_rad = static_cast<double>(n) / (2.0 * pi);
    std::vector<double> out(psi.size());
    for (std::size_t i = 0; i < psi.size(); ++i) {
        const double u = (psi[i] + pi) * per_rad - 0.5;
        const double k0 = std::floor(u);
        const double w = u - k0;
        // Python's % on int64: non-negative for a positive modulus.
        auto i0 = static_cast<std::ptrdiff_t>(k0) % n;
        if (i0 < 0) i0 += n;
        const std::ptrdiff_t i1 = (i0 + 1) % n;
        const double a = profile[static_cast<std::size_t>(i0)];
        out[i] = a + w * (profile[static_cast<std::size_t>(i1)] - a);
    }
    return out;
}

}  // namespace eclipse::limb

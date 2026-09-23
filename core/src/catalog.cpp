// See eclipse/catalog.hpp: the port of app/catalog.py's raw layer
// (``_rho_at``, ``_coarse_grid``, ``_local_minima``, ``_scan_candidates``, the
// parabolic refinement, ``_classify``, ``_hybrid``, ``_catalog_raw`` and
// ``_detail_raw``) function for function, every expression in the Python's
// exact floating-point operation order (roadmap §7: port, don't improve).
//
// Physics: the shadow axis' distance from the Earth's centre in the
// fundamental plane ``rho = hypot(x, y)`` [ES92] eq. 8.322-6 (through
// ``ephem::axis_separation``), an eclipse when ``rho < 1 + l1`` [ES92]
// sec. 8.34, the reduced cone radii ``L' = l - zeta tan f`` [ES92] eq. 8.353,
// the Canon type codes, ``gamma`` and the greatest-eclipse magnitude
// [Espenak]. The header comment lists the operation-order rules this file
// keeps; nothing here touches SPICE or ERFA except through ``ephem``.
//
// The Python evaluates the hybrid grid once per event and the detail inside
// the classification loop; here all hybrid grids share one ``elements_at``
// call and the detail of every kept event runs in an OpenMP loop after the
// classification. Both reorderings are value-preserving: every element is a
// per-instant quantity and every detail reads only its own event, so the
// numbers are identical to the oracle's and to a serial run.
#include "eclipse/catalog.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP  // CMake defines ECLIPSE_HAVE_OPENMP and adds -fopenmp together
#include <omp.h>
#endif

#include "eclipse/ellipsoid.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/numerics.hpp"

namespace eclipse::catalog {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

/// The ``ElementsAtEt`` contract: every element read here has one value per
/// requested instant (a shorter result would index out of range).
void check_elements(const Elements& e, std::size_t n, const char* who) {
    for (const auto* v : {&e.x, &e.y, &e.d, &e.mu, &e.l1, &e.l2, &e.tan_f1, &e.tan_f2}) {
        if (v->size() != n)
            throw std::invalid_argument(std::string(who) +
                                        ": elements_at returned a wrong number of instants");
    }
}

/// ``_hybrid``'s sampling offsets [h]: ``arange(-span, span + 1e-9, step)``.
std::vector<double> hybrid_offsets_h() {
    return numerics::arange(-HYBRID_HALF_SPAN_H, HYBRID_HALF_SPAN_H + 1e-9, HYBRID_STEP_H);
}

/// The decision of ``_hybrid`` on the elements of ONE event's grid, stored at
/// ``[begin, begin + n)`` of ``e`` — several events' grids can share one
/// ``elements_at`` call because every element is a per-instant quantity.
/// ``fund_to_geo`` on all, ``on`` = the on-Earth indices; false when none;
/// else ``max(l2[on.front()], l2[on.back()]) > 0.0``: at the ends of the
/// central line ``zeta = 0`` so the umbral ground radius is ``l2`` itself
/// [ES92] eq. 8.353, and a positive one there with ``L2' < 0`` at greatest
/// eclipse is the umbra's tip lifting off before the path ends [Espenak].
bool hybrid_decide(const Elements& e, std::size_t begin, std::size_t n) {
    std::size_t first = n, last = n;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t k = begin + i;
        const ellipsoid::LonLat p = ellipsoid::fund_to_geo_one(e.x[k], e.y[k], e.d[k], e.mu[k]);
        if (std::isnan(p.lat_deg)) continue;
        if (first == n) first = i;
        last = i;
    }
    if (first == n) return false;
    // Python's ``max(a, b) > 0.0`` on two finite floats.
    return std::max(e.l2[begin + first], e.l2[begin + last]) > 0.0;
}

}  // namespace

// ------------------------------------------------------------------ sources

std::vector<double> rho_at(std::span<const double> et) {
    // np.where(z > 0.0, rho, np.inf) on axis_separation(et): the far-side
    // (full-moon) minima never qualify.
    const ephem::AxisSeparation a = ephem::axis_separation(et);
    std::vector<double> out(et.size());
    for (std::size_t i = 0; i < et.size(); ++i) out[i] = a.z[i] > 0.0 ? a.rho[i] : kInf;
    return out;
}

Sources sources_from_ephem(Frame frame) {
    Sources s;
    s.rho_at = [](std::span<const double> et) { return rho_at(et); };
    s.elements_at = [frame](std::span<const double> et) {
        return ephem::besselian_instants(et, frame);  // mu wrapped, never unwrapped here
    };
    s.frame = frame;
    return s;
}

// --------------------------------------------------------------------- scan

std::vector<double> coarse_grid(double et_a, double et_b) {
    // np.arange(et_a - step, et_b + 2 * step, step)
    return numerics::arange(et_a - COARSE_STEP_S, et_b + 2 * COARSE_STEP_S, COARSE_STEP_S);
}

std::vector<std::size_t> local_minima(std::span<const double> rho) {
    // (rho[1:-1] < rho[:-2]) & (rho[1:-1] <= rho[2:]) & (rho[1:-1] < _CANDIDATE_RHO),
    // flatnonzero + 1: grid order. Every comparison is false for NaN / +inf
    // at [i], as in NumPy.
    std::vector<std::size_t> out;
    if (rho.size() < 3) return out;
    for (std::size_t i = 1; i + 1 < rho.size(); ++i) {
        if (rho[i] < rho[i - 1] && rho[i] <= rho[i + 1] && rho[i] < CANDIDATE_RHO) out.push_back(i);
    }
    return out;
}

std::vector<double> scan_candidates(double et_a, double et_b, const RhoAt& rho_at) {
    const std::vector<double> et = coarse_grid(et_a, et_b);
    // Chunked exactly as the oracle (a memory bound; per-instant values).
    std::vector<double> rho;
    rho.reserve(et.size());
    const std::span<const double> all(et);
    for (std::size_t b = 0; b < et.size(); b += COARSE_CHUNK) {
        const std::size_t end = std::min(et.size(), b + COARSE_CHUNK);
        const std::vector<double> part = rho_at(all.subspan(b, end - b));
        if (part.size() != end - b)
            throw std::invalid_argument("scan_candidates: rho_at returned a wrong length");
        rho.insert(rho.end(), part.begin(), part.end());
    }
    const std::vector<std::size_t> idx = local_minima(rho);
    std::vector<double> out(idx.size());
    for (std::size_t j = 0; j < idx.size(); ++j) out[j] = et[idx[j]];
    return out;
}

std::vector<double> refine_greatest(std::span<const double> cand_et, const RhoAt& rho_at) {
    // parabolic_minimum(_rho_at, cand, _COARSE_STEP_S): 10 iterations, one
    // objective call on the stacked 3n vector per iteration.
    return numerics::parabolic_minimum(rho_at, cand_et, COARSE_STEP_S);
}

// ----------------------------------------------------------- classification

Classification classify(double rho_g, double l1, double l2, double tan_f1, double tan_f2,
                        bool central, double zeta) {
    if (central) {
        // Reduced cone radii at the greatest-eclipse ground point [ES92] eq.
        // 8.353; total when the umbra reaches the ground (L2' < 0); the
        // magnitude is the Moon/Sun diameter ratio [Espenak].
        const double L1p = l1 - zeta * tan_f1;
        const double L2p = l2 - zeta * tan_f2;
        const Kind kind = L2p < 0.0 ? Kind::Total : Kind::Annular;
        return Classification{kind, (L1p - L2p) / (L1p + L2p), L2p};
    }
    if (rho_g > 1.0 + std::abs(l2)) {
        // Partial: the umbra never touches; the magnitude at the closest limb
        // point (zeta = 0) [Espenak], in the oracle's bracket order.
        return Classification{Kind::Partial, (l1 - (rho_g - 1.0)) / (l1 + l2), kNaN};
    }
    // Non-central: the umbra grazes the limb; the kind follows the sign of l2.
    const Kind kind = l2 < 0.0 ? Kind::Total : Kind::Annular;
    return Classification{kind, (l1 - l2) / (l1 + l2), kNaN};
}

bool hybrid(double et_g, double L2p_greatest, const ElementsAtEt& elements_at) {
    if (L2p_greatest >= 0.0) return false;  // annular at closest approach: annular throughout
    const std::vector<double> offs = hybrid_offsets_h();
    std::vector<double> et(offs.size());
    for (std::size_t i = 0; i < offs.size(); ++i) et[i] = et_g + offs[i] * 3600.0;
    const Elements e = elements_at(et);
    check_elements(e, et.size(), "hybrid");
    return hybrid_decide(e, 0, et.size());
}

// ------------------------------------------------------------------- detail

Detail add_detail(double et_g, Frame frame, bool central, double lat_g_deg, double lon_g_deg) {
    // BesselianModel(t0_utc=et_to_utc(et_g), ...): the model's epoch is the
    // WHOLE-SECOND greatest-eclipse string re-parsed, not et_g itself.
    Detail d;
    d.et0 = ephem::utc_to_et(ephem::et_to_utc(et_g));
    d.contacts = geometry::global_contacts(d.et0, frame, CONTACT_HALF_WINDOW_H);
    if (!central) return d;

    // local_raw(model, round(lat, 2), round(lon, 2)): the point the row reports.
    d.local = circumstances::local_circumstances(d.et0, frame, DETAIL_HALF_WINDOW_H,
                                                 numerics::py_round(lat_g_deg, 2),
                                                 numerics::py_round(lon_g_deg, 2));

    // central_track(model, [-dt, 0, +dt]): evaluate_direct (mu unwrapped),
    // fund_to_geo on all, the on-Earth points in order, each point's bearing
    // from its previous on-Earth neighbour to its next (clamped; 0.0 when
    // alone).
    const double t_h[3] = {-TRACK_DT_H, 0.0, TRACK_DT_H};
    std::vector<double> et(3);
    for (std::size_t i = 0; i < 3; ++i) et[i] = d.et0 + t_h[i] * 3600.0;
    Elements e = ephem::besselian_instants(et, frame);
    unwrap_mu_deg(e);
    const ellipsoid::Geographic g = ellipsoid::fund_to_geo(e.x, e.y, e.d, e.mu);

    struct Point {
        std::size_t i;
        double t, lat, lon;
    };
    std::vector<Point> valid;
    for (std::size_t i = 0; i < 3; ++i)
        if (!std::isnan(g.lat_deg[i])) valid.push_back({i, t_h[i], g.lat_deg[i], g.lon_deg[i]});

    const std::size_t n = valid.size();
    for (std::size_t k = 0; k < n; ++k) {
        const Point& tp = valid[k];
        if (!(std::abs(tp.t) < 1e-9)) continue;  // the t = 0 point (first match)
        const Point& prev = valid[k == 0 ? 0 : k - 1];
        const Point& nxt = valid[std::min(n - 1, k + 1)];
        const double brg = n > 1 ? geometry::bearing_deg(prev.lat, prev.lon, nxt.lat, nxt.lon) : 0.0;
        // shadow_edge_limits(x, y, d, mu, l2, tan_f2, bearing): the scalar
        // oracle's (None, None, 0.0) when the north point is NaN.
        const std::size_t i = tp.i;
        const double x[] = {e.x[i]}, y[] = {e.y[i]}, dd[] = {e.d[i]}, mu[] = {e.mu[i]};
        const double l[] = {e.l2[i]}, tf[] = {e.tan_f2[i]}, b[] = {brg};
        // element_rates(model, [0.0]): evaluate_direct at [0 - dt, 0 + dt]
        // (mu unwrapped), central differences, the mu difference wrapped.
        const double dt = geometry::RATE_DT_H;
        const std::vector<double> et_r = {d.et0 + (0.0 - dt) * 3600.0, d.et0 + (0.0 + dt) * 3600.0};
        Elements er = ephem::besselian_instants(et_r, frame);
        unwrap_mu_deg(er);
        const double rx[] = {(er.x[1] - er.x[0]) / (2.0 * dt)};
        const double ry[] = {(er.y[1] - er.y[0]) / (2.0 * dt)};
        const double rd[] = {(er.d[1] - er.d[0]) / (2.0 * dt)};
        const double rmu[] = {(numerics::np_remainder(er.mu[1] - er.mu[0] + 180.0, 360.0) - 180.0) /
                              (2.0 * dt)};
        const double rl[] = {(er.l2[1] - er.l2[0]) / (2.0 * dt)};
        const geometry::EdgeRates rates{rx, ry, rd, rmu, rl};
        const geometry::EdgeLimits lim = geometry::shadow_edge_limits(
            x, y, dd, mu, l, tf, b, UMBRA_LIMIT_MAX_KM, true, &rates);
        d.width_km = std::isnan(lim.north_lat[0]) ? 0.0 : lim.width_km[0];
        break;
    }
    return d;
}

// -------------------------------------------------------------------- event

std::vector<Event> find_eclipses(double et_a, double et_b, const Sources& sources, bool detail,
                                 int threads) {
    if (!sources.rho_at || !sources.elements_at)
        throw std::invalid_argument("find_eclipses: Sources.rho_at and elements_at must be set");

    const std::vector<double> cand = scan_candidates(et_a, et_b, sources.rho_at);
    if (cand.empty()) return {};

    // Refine every candidate at once to the instant of greatest eclipse, then
    // the elements there in the catalog's frame (one call, mu raw / wrapped).
    const std::vector<double> et_g = refine_greatest(cand, sources.rho_at);
    const std::size_t n = et_g.size();
    const Elements e = sources.elements_at(et_g);
    check_elements(e, n, "find_eclipses");
    const ellipsoid::Geographic geo = ellipsoid::fund_to_geo(e.x, e.y, e.d, e.mu);

    std::vector<Event> events;
    std::vector<double> L2p;  // per kept event, for the hybrid test
    for (std::size_t k = 0; k < n; ++k) {
        const double rho_g = std::hypot(e.x[k], e.y[k]);
        // not (et_a <= et_g <= et_b) or rho_g > 1.0 + l1 -> dropped without trace
        // (a NaN et_g fails the range test exactly as in Python).
        if (!(et_a <= et_g[k] && et_g[k] <= et_b) || rho_g > 1.0 + e.l1[k]) continue;
        const bool central = !std::isnan(geo.lat_deg[k]);
        double zeta = kNaN;
        if (central)  // the UNROUNDED point, the raw wrapped mu
            zeta = ellipsoid::geo_to_fund_one(geo.lat_deg[k], geo.lon_deg[k], e.d[k], e.mu[k]).zeta;
        const Classification c =
            classify(rho_g, e.l1[k], e.l2[k], e.tan_f1[k], e.tan_f2[k], central, zeta);
        Event ev;
        ev.et_g = et_g[k];
        ev.kind = c.kind;
        ev.central = central;
        ev.gamma = std::copysign(rho_g, e.y[k]);  // signed rho, positive north [Espenak]
        ev.magnitude = c.magnitude;
        ev.lat_deg = geo.lat_deg[k];
        ev.lon_deg = geo.lon_deg[k];
        events.push_back(ev);
        L2p.push_back(c.L2p);
    }

    // _hybrid for every central event that is total at greatest eclipse: the
    // per-event 5-minute grids concatenated into ONE elements_at call
    // (elementwise identical to the oracle's call per event), then split.
    std::vector<std::size_t> need;
    for (std::size_t j = 0; j < events.size(); ++j)
        if (events[j].central && !(L2p[j] >= 0.0)) need.push_back(j);
    if (!need.empty()) {
        const std::vector<double> offs = hybrid_offsets_h();
        const std::size_t m = offs.size();
        std::vector<double> et(need.size() * m);
        for (std::size_t q = 0; q < need.size(); ++q)
            for (std::size_t i = 0; i < m; ++i) et[q * m + i] = events[need[q]].et_g + offs[i] * 3600.0;
        const Elements eh = sources.elements_at(et);
        check_elements(eh, et.size(), "find_eclipses (hybrid)");
        for (std::size_t q = 0; q < need.size(); ++q)
            if (hybrid_decide(eh, q * m, m)) events[need[q]].kind = Kind::Hybrid;
    }

    if (detail) {
        // Each slot is written by one thread; the ephemeris calls inside
        // serialize on the SPICE lock; the pure math is lock-free. No
        // reductions, so the result is identical for any thread count. An
        // exception is captured per slot and the lowest-index one rethrown
        // after the region (the Python loop's first-failure semantics).
        const std::size_t count = events.size();
        std::vector<std::exception_ptr> errors(count);
#ifdef _OPENMP
        const int nt = threads > 0 ? threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic) num_threads(nt)
#else
        (void)threads;
#endif
        for (std::ptrdiff_t jj = 0; jj < static_cast<std::ptrdiff_t>(count); ++jj) {
            const auto j = static_cast<std::size_t>(jj);
            Event& ev = events[j];
            try {
                ev.detail = add_detail(ev.et_g, sources.frame, ev.central, ev.lat_deg, ev.lon_deg);
            } catch (...) {
                errors[j] = std::current_exception();
            }
        }
        for (const std::exception_ptr& p : errors)
            if (p) std::rethrow_exception(p);
    }
    return events;
}

std::vector<Event> find_eclipses(double et_a, double et_b, Frame frame, bool detail, int threads) {
    return find_eclipses(et_a, et_b, sources_from_ephem(frame), detail, threads);
}

}  // namespace eclipse::catalog

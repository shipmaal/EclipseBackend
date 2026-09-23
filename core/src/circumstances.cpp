// See eclipse/circumstances.hpp. The port of app/circumstances.py (roadmap §3,
// §4 row "circumstances", §5 phase 3), written in the Python oracle's exact
// floating-point operation order (left to right as NumPy evaluates it;
// ``np.hypot`` is ``std::hypot``, ``np.radians`` / ``np.degrees`` the
// ``constants::DEG_TO_RAD`` / ``RAD_TO_DEG`` products) so parity is bit-level
// up to the 1-ulp libm-vs-NumPy SIMD noise — do not reassociate (roadmap §7).
//
// No SPICE here: the ephemeris enters only through ``Model``'s callables, and
// ``model_from_ephem`` is the one place binding them to ``ephem``. Every
// ``np.where`` of the oracle evaluates both branches and selects; the C++
// evaluates only the selected branch, which is bit-identical because the
// expressions have no side effects, and never traps (IEEE division by zero and
// invalid operations produce inf / NaN, as under the oracle's
// ``np.errstate(divide="ignore", invalid="ignore")``).
#include "eclipse/circumstances.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP  // CMake defines ECLIPSE_HAVE_OPENMP and adds -fopenmp together
#include <omp.h>
#endif

#include "eclipse/constants.hpp"
#include "eclipse/ellipsoid.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/limb.hpp"
#include "eclipse/numerics.hpp"

namespace eclipse::circumstances {

namespace {

using constants::DEG_TO_RAD;
using constants::RAD_TO_DEG;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// ``evaluate_direct``'s ``et0 + t * 3600.0`` [TDB s] for offsets ``t`` [h].
std::vector<double> et_of(double et0, std::span<const double> t_hours) {
    std::vector<double> et(t_hours.size());
    for (std::size_t i = 0; i < t_hours.size(); ++i) et[i] = et0 + t_hours[i] * 3600.0;
    return et;
}

/// The ``ElementsAt`` contract: every element read here has one value per
/// requested instant (a shorter result would index out of range).
void check_elements(const Elements& e, std::size_t n, const char* who) {
    if (e.x.size() != n || e.y.size() != n || e.d.size() != n || e.mu.size() != n ||
        e.l1.size() != n || e.l2.size() != n || e.tan_f1.size() != n || e.tan_f2.size() != n)
        throw std::invalid_argument(std::string(who) + ": elements_at returned a wrong-length result");
}

/// The ``SubSolarAt`` contract: one (lon, lat) per requested instant.
void check_sub_solar(const ephem::SubSolar& ss, std::size_t n, const char* who) {
    if (ss.lon_deg.size() != n || ss.lat_deg.size() != n)
        throw std::invalid_argument(std::string(who) + ": sub_solar_at returned a wrong-length result");
}

/// The per-instant half of ``geo_to_fund`` ([ES92] eq. 8.331, ``_reduction_aux``
/// of ``np.radians(d)``), hoisted so a P x N grid evaluates it N times, not
/// P x N times; each value is exactly what ``ellipsoid::geo_to_fund_one``'s
/// un-hoisted overload would compute for that instant.
std::vector<ellipsoid::ReductionAux> instant_aux(const Elements& e) {
    std::vector<ellipsoid::ReductionAux> aux;
    aux.reserve(e.size());
    for (std::size_t i = 0; i < e.size(); ++i)
        aux.push_back(ellipsoid::reduction_aux(e.d[i] * DEG_TO_RAD));
    return aux;
}

/// ``_series`` for ONE observer over the N instants of ``e``: the body shared
/// by both ``series`` overloads and the grid loop. ``cos_beta`` / ``sin_beta``
/// are the observer's parametric-latitude factors (``parametric_latitude``,
/// [Meeus98] ch. 11, eq. 11.1) and ``aux`` the per-instant reduction. Writes
/// N values to each output.
///   m    = hypot(xi - x, eta - y)          axis separation
///   L1'  = l1 - zeta tan_f1                penumbral radius at the observer [ES92] eq. 8.353
///   L2'  = l2 - zeta tan_f2                umbral / antumbral radius        [ES92] eq. 8.353
///   mag  = (L1' - m) / (L1' + L2')         magnitude [Espenak], [ES92] eq. 8.354
void series_row(const Elements& e, std::span<const ellipsoid::ReductionAux> aux, double cos_beta,
                double sin_beta, double lon_deg, double* m, double* L1p, double* L2p,
                double* mag) {
    const std::size_t n = e.size();
    for (std::size_t i = 0; i < n; ++i) {
        const ellipsoid::Fund f =
            ellipsoid::geo_to_fund_one(cos_beta, sin_beta, lon_deg, aux[i], e.mu[i]);
        m[i] = std::hypot(f.xi - e.x[i], f.eta - e.y[i]);
        L1p[i] = e.l1[i] - f.zeta * e.tan_f1[i];
        L2p[i] = e.l2[i] - f.zeta * e.tan_f2[i];
        mag[i] = (L1p[i] - m[i]) / (L1p[i] + L2p[i]);
    }
}

Series make_series(std::size_t n) {
    Series s;
    s.m.resize(n);
    s.L1p.resize(n);
    s.L2p.resize(n);
    s.mag.resize(n);
    return s;
}

/// ``_series`` through a ``Model`` at offsets ``t`` [h]: ``elements_at`` (the
/// ``evaluate_direct`` contract, ``mu`` already unwrapped) then the geometry.
Series model_series(const Model& model, double lat_deg, double lon_deg, std::span<const double> t,
                    const char* who) {
    const Elements e = model.elements_at(t);
    check_elements(e, t.size(), who);
    return series(e, lat_deg, lon_deg);
}

/// The Python's ``_NO_ECLIPSE_RAW``: every field a placeholder.
LocalRaw no_eclipse_raw() {
    LocalRaw r{};
    r.geometric = false;
    r.central = false;
    r.c1 = r.c4 = r.c2 = r.c3 = r.t_max = r.magnitude = r.obscuration = r.L2_x = kNaN;
    r.alt_deg.fill(kNaN);
    r.az_deg.fill(kNaN);
    r.below.fill(false);
    r.eclipse = false;
    return r;
}

/// Python's ``min(roots, key=t)`` / ``max(roots, key=t)`` over the roots
/// ``pred`` selects: the FIRST extreme on ties (CPython replaces the running
/// best only on a strict ``<`` / ``>``, so a NaN key never replaces it);
/// ``nullptr`` for an empty selection (Python raises ``ValueError``).
template <class Pred>
const Root* extreme_root(const std::vector<Root>& rs, bool want_max, Pred pred) {
    const Root* best = nullptr;
    for (const Root& r : rs) {
        if (!pred(r)) continue;
        if (best == nullptr || (want_max ? r.t > best->t : r.t < best->t)) best = &r;
    }
    return best;
}

}  // namespace

// -------------------------------------------------------------------- model

Model model_from_ephem(double et0, Frame frame, double half_window_hours) {
    // BesselianModel.evaluate_direct: besselian_instants(et0 + t * 3600.0, frame)
    // then e["mu"] = np.degrees(np.unwrap(np.radians(e["mu"]))) — applied to
    // whatever vector is passed (the grid, the bracket vector, the 1-instant
    // [t_max]); the round trip is not bit-identity, so it is never skipped.
    const ElementsAt elements_at = [et0, frame](std::span<const double> t_hours) {
        Elements e = ephem::besselian_instants(et_of(et0, t_hours), frame);
        unwrap_mu_deg(e);
        return e;
    };
    // _sun_altaz / circumstances_grid: sub_solar_points(et0 + t * 3600.0, frame).
    const SubSolarAt sub_solar_at = [et0, frame](std::span<const double> t_hours) {
        return ephem::sub_solar_points(et_of(et0, t_hours), frame);
    };
    // Profile mode: evaluate_direct(t, K_REF, K_REF) and profiles_at(et0 + t * 3600).
    const ElementsAt elements_ref_at = [et0, frame](std::span<const double> t_hours) {
        Elements e = ephem::besselian_instants(et_of(et0, t_hours), frame, limb::K_REF,
                                               limb::K_REF);
        unwrap_mu_deg(e);
        return e;
    };
    const ProfilesAt profiles_at = [et0, frame](std::span<const double> t_hours) {
        return limb::profiles_at(et_of(et0, t_hours), frame);
    };
    return Model{elements_at, sub_solar_at, half_window_hours, elements_ref_at, profiles_at};
}

// ------------------------------------------------------------------- series

Series series(const Elements& e, double lat_deg, double lon_deg) {
    const std::size_t n = e.size();
    check_elements(e, n, "series");
    const std::vector<ellipsoid::ReductionAux> aux = instant_aux(e);
    const double beta = ellipsoid::parametric_latitude(lat_deg);
    Series s = make_series(n);
    series_row(e, aux, std::cos(beta), std::sin(beta), lon_deg, s.m.data(), s.L1p.data(),
               s.L2p.data(), s.mag.data());
    return s;
}

Series series(const Elements& e, std::span<const double> lat_deg, std::span<const double> lon_deg) {
    const std::size_t P = lat_deg.size();
    if (lon_deg.size() != P) throw std::invalid_argument("series: lat_deg, lon_deg differ in length");
    const std::size_t n = e.size();
    check_elements(e, n, "series");
    const std::vector<ellipsoid::ReductionAux> aux = instant_aux(e);
    Series s = make_series(P * n);
    for (std::size_t p = 0; p < P; ++p) {
        const double beta = ellipsoid::parametric_latitude(lat_deg[p]);
        const std::size_t off = p * n;
        series_row(e, aux, std::cos(beta), std::sin(beta), lon_deg[p], s.m.data() + off,
                   s.L1p.data() + off, s.L2p.data() + off, s.mag.data() + off);
    }
    return s;
}

// -------------------------------------------------------------- obscuration

double overlap_area(double r, double R, double d) {
    // _overlap_area's np.where order: disjoint, then contained, else the lens.
    // NaN inputs fail both comparisons and fall through to the (NaN) lens, as
    // in NumPy.
    if (d >= r + R) return 0.0;
    if (d <= R - r) return std::numbers::pi * r * r;
    // Circular-segment (lens) area of two intersecting circles: the two
    // sector-minus-triangle pieces, with the kite half-area ``tri`` from the
    // four-factor (Heron-like) product. (numerical)
    const double a = numerics::np_clip((d * d + r * r - R * R) / (2 * d * r), -1.0, 1.0);
    const double b = numerics::np_clip((d * d + R * R - r * r) / (2 * d * R), -1.0, 1.0);
    const double tri =
        0.5 * std::sqrt(numerics::np_maximum(
                  0.0, (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)));
    return r * r * std::acos(a) + R * R * std::acos(b) - tri;
}

double obscuration(double L1p, double L2p, double m) {
    // Covered fraction of the Sun's AREA [Espenak]: the Moon/Sun disks as two
    // circles of radius ratio q = (L1' - L2') / (L1' + L2') and centre
    // separation 2 m / (L1' + L2') in Sun radii; overlap / (pi 1^2).
    const double denom = L1p + L2p;
    const double safe = denom > 0.0 ? denom : 1.0;
    const double q = (L1p - L2p) / safe;
    const double sep = 2.0 * m / safe;
    const double r = numerics::np_minimum(q, 1.0);
    const double R = numerics::np_maximum(q, 1.0);
    return denom > 0.0 ? overlap_area(r, R, sep) / std::numbers::pi : 0.0;
}

// ------------------------------------------------------------------ horizon

AltAz sun_altaz(double lat_deg, double lon_deg, double ss_lon_deg, double ss_lat_deg) {
    // Spherical cos c = sin phi sin delta + cos phi cos delta cos H with the
    // sub-solar latitude as the Sun's declination and H = radians(ss_lon - lon)
    // (ONE radians of the difference) — geodetic phi is exact here (the
    // ellipsoid normal is the zenith; solar parallax < 9" neglected).
    const double p1 = lat_deg * DEG_TO_RAD, p2 = ss_lat_deg * DEG_TO_RAD;
    const double dl = (ss_lon_deg - lon_deg) * DEG_TO_RAD;
    const double cos_c = std::sin(p1) * std::sin(p2) + std::cos(p1) * std::cos(p2) * std::cos(dl);
    const double alt = 90.0 - std::acos(numerics::np_clip(cos_c, -1.0, 1.0)) * RAD_TO_DEG;
    const double az = geometry::bearing_deg(lat_deg, lon_deg, ss_lat_deg, ss_lon_deg);
    return AltAz{alt, az};
}

double sun_alt_grid(double lat_deg, double lon_deg, double ss_lon_deg, double ss_lat_deg) {
    // circumstances_grid's own expression: dl = radians(ss_lon) - radians(lon)
    // (two radians, then the difference) — a different rounding from
    // sun_altaz, kept separate for parity.
    const double p1 = lat_deg * DEG_TO_RAD, p2 = ss_lat_deg * DEG_TO_RAD;
    const double dl = ss_lon_deg * DEG_TO_RAD - lon_deg * DEG_TO_RAD;
    const double cos_c = std::sin(p1) * std::sin(p2) + std::cos(p1) * std::cos(p2) * std::cos(dl);
    return 90.0 - std::acos(numerics::np_clip(cos_c, -1.0, 1.0)) * RAD_TO_DEG;
}

// -------------------------------------------------------------------- roots

std::vector<Root> roots(std::span<const double> t, std::span<const double> f) {
    if (f.size() != t.size()) throw std::invalid_argument("roots: t, f differ in length");
    std::vector<Root> out;
    for (const numerics::SignChange& sc : numerics::sign_changes(t, f)) {
        const std::size_t i = sc.i;
        if (f[i] == 0.0) {
            out.push_back(Root{t[i], sc.rising, i});
        } else {
            // Linear interpolation: t[i] - f[i] * (t[i + 1] - t[i]) / (f[i + 1] - f[i]),
            // left to right. (numerical)
            const double tc = t[i] - ((f[i] * (t[i + 1] - t[i])) / (f[i + 1] - f[i]));
            out.push_back(Root{tc, sc.rising, i});
        }
    }
    return out;
}

// --------------------------------------------------------------- bracketing

Bracketed bracketed_series(const Model& model, double lat_deg, double lon_deg) {
    double hw = model.half_window_hours;
    for (;;) {
        std::vector<double> t = numerics::arange(-hw, hw + 1e-9, GRID_STEP_HOURS);
        if (t.empty()) throw std::invalid_argument("bracketed_series: empty sampling grid");
        Series s = model_series(model, lat_deg, lon_deg, t, "bracketed_series");
        // outside = m - L1' > 0 when the observer is outside the penumbra.
        const std::size_t last = t.size() - 1;
        const bool edges_clear = (s.m[0] - s.L1p[0]) > 0.0 && (s.m[last] - s.L1p[last]) > 0.0;
        if (edges_clear || hw >= MAX_HALF_WINDOW_HOURS) return Bracketed{std::move(t), std::move(s)};
        hw = std::min(hw * 1.5, MAX_HALF_WINDOW_HOURS);
    }
}

std::vector<double> refine_contacts(const Model& model, double lat_deg, double lon_deg,
                                    std::span<const double> t_lo, std::span<const double> t_hi,
                                    std::span<const std::uint8_t> central) {
    const std::size_t n = t_lo.size();
    if (t_hi.size() != n || central.size() != n)
        throw std::invalid_argument("refine_contacts: t_lo, t_hi, central differ in length");
    // np.where(central, m - |L2'|, m - L1'): the C2/C3 condition where central,
    // the C1/C4 one elsewhere; one series per objective call.
    const auto f = [&](std::span<const double> tt) -> std::vector<double> {
        const Series s = model_series(model, lat_deg, lon_deg, tt, "refine_contacts");
        std::vector<double> out(tt.size());
        for (std::size_t k = 0; k < tt.size(); ++k)
            out[k] = central[k] != 0 ? s.m[k] - std::abs(s.L2p[k]) : s.m[k] - s.L1p[k];
        return out;
    };
    return numerics::bisect(f, t_lo, t_hi);
}

double refine_maximum(std::span<const double> t, std::span<const double> mag, std::size_t imax) {
    if (mag.size() != t.size()) throw std::invalid_argument("refine_maximum: t, mag differ in length");
    if (imax >= t.size()) throw std::invalid_argument("refine_maximum: imax out of range");
    // Three-point parabola through the grid neighbours; the vertex only when
    // the fit is a maximum (denom < 0). The step is the constant grid step,
    // not the arange delta, exactly as the Python. (numerical)
    if (0 < imax && imax + 1 < t.size()) {
        const double y0 = mag[imax - 1], y1 = mag[imax], y2 = mag[imax + 1];
        const double denom = y0 - 2.0 * y1 + y2;
        if (denom < 0.0) return t[imax] + (((0.5 * GRID_STEP_HOURS) * (y0 - y2)) / denom);
    }
    return t[imax];
}

// -------------------------------------------------------- local circumstances

std::vector<double> profile_g(const Model& model, double lat_deg, double lon_deg,
                              std::span<const double> t_hours) {
    if (!model.elements_ref_at || !model.profiles_at)
        throw std::invalid_argument("profile_g: the model has no limb-profile hooks");
    const std::size_t n = t_hours.size();
    const Elements e = model.elements_ref_at(t_hours);
    check_elements(e, n, "profile_g");
    const std::vector<double> prof = model.profiles_at(t_hours);
    const auto nb = static_cast<std::size_t>(limb::N_BINS);
    if (prof.size() != n * nb) throw std::invalid_argument("profile_g: profiles_at returned a wrong size");
    const double beta = ellipsoid::parametric_latitude(lat_deg);
    const double cb = std::cos(beta), sb = std::sin(beta);
    const std::vector<ellipsoid::ReductionAux> aux = instant_aux(e);
    // Per instant: the Python's masked g_total / g_annular calls, one at a time
    // (each instant is independent, so the grouping does not change a bit).
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const ellipsoid::Fund f = ellipsoid::geo_to_fund_one(cb, sb, lon_deg, aux[i], e.mu[i]);
        const double px = f.xi - e.x[i];
        const double py = f.eta - e.y[i];
        const double L1p = e.l1[i] - f.zeta * e.tan_f1[i];
        const double L2p = e.l2[i] - f.zeta * e.tan_f2[i];
        const double r_s[1] = {(L1p + L2p) / 2.0};
        const double r_m[1] = {(L1p - L2p) / 2.0};
        const double pxs[1] = {px}, pys[1] = {py};
        const std::span<const double> p(prof.data() + i * nb, nb);
        out[i] = L2p < 0.0 ? limb::g_total(pxs, pys, r_s, r_m, p)[0]
                           : limb::g_annular(pxs, pys, r_s, r_m, p)[0];
    }
    return out;
}

ProfileContacts profile_contacts(const Model& model, double lat_deg, double lon_deg, double t_lo,
                                 double t_hi) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> t, g;
    for (int k = 0; k <= PROFILE_MAX_WIDEN; ++k) {
        t = numerics::arange(t_lo, t_hi + 1e-12, PROFILE_STEP_H);
        g = profile_g(model, lat_deg, lon_deg, t);
        if (!(g.front() < 0.0 || g.back() < 0.0)) break;
        if (g.front() < 0.0) t_lo = t_lo - PROFILE_MARGIN_H;
        if (g.back() < 0.0) t_hi = t_hi + PROFILE_MARGIN_H;
    }
    bool any_neg = false;
    for (const double v : g) any_neg = any_neg || v < 0.0;
    if (!any_neg || g.front() < 0.0 || g.back() < 0.0) return {false, nan, nan};
    const std::vector<numerics::SignChange> ch = numerics::sign_changes(t, g);
    std::size_t i2 = t.size(), i3 = 0;
    bool have2 = false, have3 = false;
    for (const auto& c : ch) {
        if (!c.rising && (!have2 || c.i < i2)) i2 = c.i, have2 = true;
        if (c.rising && (!have3 || c.i > i3)) i3 = c.i, have3 = true;
    }
    if (!have2 || !have3) return {false, nan, nan};
    const double lo[2] = {t[i2], t[i3]}, hi[2] = {t[i2 + 1], t[i3 + 1]};
    const std::vector<double> c = numerics::bisect(
        [&](std::span<const double> tt) { return profile_g(model, lat_deg, lon_deg, tt); }, lo, hi);
    return {true, c[0], c[1]};
}

LocalRaw local_circumstances(const Model& model, double lat_deg, double lon_deg, bool profile) {
    const Bracketed b = bracketed_series(model, lat_deg, lon_deg);
    const std::vector<double>& t = b.t;
    const Series& s = b.s;
    const std::size_t n = t.size();

    // Partial contacts: roots of m - L1'.
    std::vector<double> f(n);
    for (std::size_t i = 0; i < n; ++i) f[i] = s.m[i] - s.L1p[i];
    const std::vector<Root> partial = roots(t, f);
    // mag.max() <= 0: np.max propagates NaN, and NaN <= 0 is false (continues).
    if (partial.size() < 2 || numerics::np_max(s.mag) <= 0.0) return no_eclipse_raw();

    // C1 = the falling root with the smallest crossing, C4 = the rising root
    // with the largest; an empty selection is the Python's ValueError.
    const Root* r1 = extreme_root(partial, false, [](const Root& r) { return !r.rising; });
    const Root* r4 = extreme_root(partial, true, [](const Root& r) { return r.rising; });
    if (r1 == nullptr || r4 == nullptr)
        throw std::invalid_argument(
            "local_circumstances: partial roots are all one-signed (no C1/C4 pair)");
    const std::size_t i1 = r1->i, i4 = r4->i;

    const std::size_t imax = numerics::argmax_first(s.mag);
    // Central phase: observer inside the umbra / antumbra at maximum.
    bool central_phase = s.m[imax] < std::abs(s.L2p[imax]);
    // Brackets in the Python's list order: C1, C4, then C2, C3 when central.
    std::vector<double> t_lo{t[i1], t[i4]}, t_hi{t[i1 + 1], t[i4 + 1]};
    std::vector<std::uint8_t> is_central{0, 0};
    if (central_phase) {
        std::vector<double> g(n);
        for (std::size_t i = 0; i < n; ++i) g[i] = s.m[i] - std::abs(s.L2p[i]);
        const std::vector<Root> central = roots(t, g);
        const double t_imax = t[imax];
        // C2 = the last entering (falling) root at or before t[imax], C3 = the
        // first exiting (rising) root at or after it.
        const Root* r2 = extreme_root(
            central, true, [t_imax](const Root& r) { return !r.rising && r.t <= t_imax; });
        const Root* r3 = extreme_root(
            central, false, [t_imax](const Root& r) { return r.rising && r.t >= t_imax; });
        if (r2 != nullptr && r3 != nullptr) {
            t_lo.push_back(t[r2->i]);
            t_hi.push_back(t[r2->i + 1]);
            is_central.push_back(1);
            t_lo.push_back(t[r3->i]);
            t_hi.push_back(t[r3->i + 1]);
            is_central.push_back(1);
        } else {
            central_phase = false;
        }
    }
    const std::vector<double> times = refine_contacts(model, lat_deg, lon_deg, t_lo, t_hi, is_central);
    const double c1 = times[0], c4 = times[1];
    const double tmax = refine_maximum(t, s.mag, imax);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double c2 = central_phase ? times[2] : nan;
    double c3 = central_phase ? times[3] : nan;

    if (profile) {
        bool search;
        double lo, hi;
        if (central_phase) {
            lo = c2 - PROFILE_MARGIN_H;
            hi = c3 + PROFILE_MARGIN_H;
            search = true;
        } else {
            search = s.m[imax] - std::abs(s.L2p[imax]) <
                     PROFILE_NEAR_KM / constants::EARTH_EQUATORIAL_RADIUS_KM;
            lo = t[imax] - PROFILE_GRAZE_H;
            hi = t[imax] + PROFILE_GRAZE_H;
        }
        if (search) {
            const ProfileContacts pc = profile_contacts(model, lat_deg, lon_deg, lo, hi);
            central_phase = pc.central;
            c2 = pc.c2;
            c3 = pc.c3;
        } else {
            central_phase = false;
        }
    }

    // Quantities at the refined maximum: a fresh 1-instant series.
    const double t1[1] = {tmax};
    const Series sx = model_series(model, lat_deg, lon_deg, t1, "local_circumstances");
    const double m_x = sx.m[0], L1_x = sx.L1p[0], L2_x = sx.L2p[0];
    // Eclipse magnitude [Espenak]: the Moon/Sun apparent diameter ratio when
    // central, the covered fraction of the Sun's diameter when partial.
    const double magnitude =
        central_phase ? (L1_x - L2_x) / (L1_x + L2_x) : (L1_x - m_x) / (L1_x + L2_x);

    // Events C1, max, C4 (+ C2, C3): one sub-solar evaluation for all of them.
    const std::size_t n_events = central_phase ? 5 : 3;
    std::vector<double> ev_times{c1, tmax, c4};
    if (central_phase) {
        ev_times.push_back(c2);
        ev_times.push_back(c3);
    }
    const ephem::SubSolar ss = model.sub_solar_at(ev_times);
    check_sub_solar(ss, n_events, "local_circumstances");

    LocalRaw r = no_eclipse_raw();
    std::size_t n_below = 0;
    for (std::size_t k = 0; k < n_events; ++k) {
        const AltAz a = sun_altaz(lat_deg, lon_deg, ss.lon_deg[k], ss.lat_deg[k]);
        r.alt_deg[k] = a.alt_deg;
        r.az_deg[k] = a.az_deg;
        // Sun below the rise/set altitude h0 [Meeus98] ch. 15.
        r.below[k] = a.alt_deg <= HORIZON_ALT_DEG;
        if (r.below[k]) ++n_below;
    }
    r.geometric = true;
    r.central = central_phase;
    r.c1 = c1;
    r.c4 = c4;
    if (central_phase) {
        r.c2 = c2;
        r.c3 = c3;
    }
    r.t_max = tmax;
    r.magnitude = magnitude;
    r.obscuration = obscuration(L1_x, L2_x, m_x);
    r.L2_x = L2_x;
    // The cone geometry continues through the Earth; a night-side observer is
    // "inside" it but sees nothing.
    r.eclipse = n_below != n_events;
    return r;
}

LocalRaw local_circumstances(double et0, Frame frame, double half_window_hours, double lat_deg,
                             double lon_deg, bool profile) {
    return local_circumstances(model_from_ephem(et0, frame, half_window_hours), lat_deg, lon_deg,
                               profile);
}

// --------------------------------------------------------------------- grid

GridResult circumstances_grid(const Model& model, std::span<const double> lat_deg,
                              std::span<const double> lon_deg, double step_minutes, int threads) {
    const std::size_t P = lat_deg.size();
    if (lon_deg.size() != P)
        throw std::invalid_argument("circumstances_grid: lat_deg, lon_deg differ in length");
    const double hw = model.half_window_hours;
    const std::vector<double> t = numerics::arange(-hw, hw + 1e-9, step_minutes / 60.0);
    if (t.empty()) throw std::invalid_argument("circumstances_grid: empty sampling grid");
    const std::size_t N = t.size();

    // Everything that touches the ephemeris, once, serially (under ephem's
    // SPICE lock when the Model is model_from_ephem's): the sub-solar points
    // and the elements. The Python evaluates the elements once per observer
    // chunk, on the same t — identical values.
    const ephem::SubSolar ss = model.sub_solar_at(t);
    check_sub_solar(ss, N, "circumstances_grid");
    const Elements e = model.elements_at(t);
    check_elements(e, N, "circumstances_grid");
    const std::vector<ellipsoid::ReductionAux> aux = instant_aux(e);

    GridResult out;
    out.magnitude.assign(P, 0.0);
    out.obscuration.assign(P, 0.0);
    out.t_max_hours.assign(P, 0.0);
    out.sun_alt.assign(P, 0.0);
    out.visible.assign(P, 0);
    out.central.assign(P, 0);

    // Pure math from here: each observer writes only its own slot of each
    // output and reads only shared constants, so the result is identical for
    // any thread count (no reductions, no shared accumulators). The series
    // row buffers are per thread.
#ifdef _OPENMP
    const int nt = threads > 0 ? threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt)
#else
    (void)threads;
#endif
    {
        std::vector<double> m(N), L1p(N), L2p(N), mag(N);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::size_t p = 0; p < P; ++p) {
            const double beta = ellipsoid::parametric_latitude(lat_deg[p]);
            series_row(e, aux, std::cos(beta), std::sin(beta), lon_deg[p], m.data(), L1p.data(),
                       L2p.data(), mag.data());
            const std::size_t imax = numerics::argmax_first(mag);
            const double mag_x = mag[imax], m_x = m[imax], L1_x = L1p[imax], L2_x = L2p[imax];
            const bool has = mag_x > 0.0;
            const bool central = has && (m_x < std::abs(L2_x));
            // Central magnitude is the diameter ratio [Espenak], as in
            // local_circumstances.
            const double magnitude = central ? (L1_x - L2_x) / (L1_x + L2_x) : mag_x;
            const double alt = sun_alt_grid(lat_deg[p], lon_deg[p], ss.lon_deg[imax], ss.lat_deg[imax]);
            out.magnitude[p] = has ? magnitude : 0.0;
            out.obscuration[p] = has ? obscuration(L1_x, L2_x, m_x) : 0.0;
            out.t_max_hours[p] = t[imax];
            out.sun_alt[p] = alt;
            out.visible[p] = (has && alt > HORIZON_ALT_DEG) ? 1 : 0;
            out.central[p] = central ? 1 : 0;
        }
    }
    return out;
}

GridResult circumstances_grid(double et0, Frame frame, double half_window_hours,
                              std::span<const double> lat_deg, std::span<const double> lon_deg,
                              double step_minutes, int threads) {
    return circumstances_grid(model_from_ephem(et0, frame, half_window_hours), lat_deg, lon_deg,
                              step_minutes, threads);
}

}  // namespace eclipse::circumstances

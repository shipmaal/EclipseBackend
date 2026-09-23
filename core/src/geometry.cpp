// See eclipse/geometry.hpp. Arithmetic is written in the Python oracle's exact
// operation order (left-to-right as NumPy evaluates it; ``**2`` as ``v * v``)
// so that parity is bit-level up to libm-vs-NumPy 1-ulp noise — do not
// reassociate (roadmap §7). No SPICE here: the only ephemeris access is the
// ``ephem::besselian_instants`` call bound in the second ``global_contacts``.
#include "eclipse/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "eclipse/constants.hpp"
#include "eclipse/ellipsoid.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/numerics.hpp"

namespace eclipse::geometry {

namespace {

using constants::DEG_TO_RAD;
using constants::EARTH_MEAN_RADIUS_KM;
using constants::RAD_TO_DEG;
using constants::WGS84_A_KM;
using constants::WGS84_F;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

// ------------------------------------------------------------ shadow radii

ShadowRadii shadow_radii(double x, double y, double d_deg, double l1, double l2, double tan_f1,
                         double tan_f2) {
    // eta1 = y / rho1 [ES92] eq. 8.331; zeta on the auxiliary sphere [ES92] eq. 8.332.
    const double eta1 = y / ellipsoid::reduction_aux(d_deg * DEG_TO_RAD).rho1;
    const double disc = 1.0 - x * x - eta1 * eta1;
    if (disc < 0.0) return ShadowRadii{0.0, 0.0, false};
    const double zeta = std::sqrt(disc);

    // Cone radii reduced to the observer's distance below the fundamental
    // plane, L = l - zeta tan f [ES92] eq. 8.353.
    const double L1 = l1 - zeta * tan_f1;  // penumbra radius at the observer (Earth radii)
    const double L2 = l2 - zeta * tan_f2;  // umbra radius; negative => total, positive => annular
    return ShadowRadii{std::abs(L1) * WGS84_A_KM, std::abs(L2) * WGS84_A_KM, L2 < 0.0};
}

// ------------------------------------------------------ great-circle helpers
// Standard spherical trigonometry on a sphere of radius EARTH_MEAN_RADIUS_KM
// (app.geography._destination / _haversine_km / bearing).

LatLon destination(double lat_deg, double lon_deg, double bearing_deg, double dist_km) {
    const double ad = dist_km / EARTH_MEAN_RADIUS_KM;
    const double lat1 = lat_deg * DEG_TO_RAD, brg = bearing_deg * DEG_TO_RAD;
    // No clip on the asin argument: the Python has none.
    const double lat2 =
        std::asin(std::sin(lat1) * std::cos(ad) + std::cos(lat1) * std::sin(ad) * std::cos(brg));
    const double lon2 = lon_deg * DEG_TO_RAD +
                        std::atan2(std::sin(brg) * std::sin(ad) * std::cos(lat1),
                                   std::cos(ad) - std::sin(lat1) * std::sin(lat2));
    return LatLon{lat2 * RAD_TO_DEG, numerics::wrap_180(lon2 * RAD_TO_DEG)};
}

double haversine_km(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
    const double p1 = lat1_deg * DEG_TO_RAD, p2 = lat2_deg * DEG_TO_RAD;
    const double dp = (lat2_deg - lat1_deg) * DEG_TO_RAD, dl = (lon2_deg - lon1_deg) * DEG_TO_RAD;
    const double sp = std::sin(dp / 2), sl = std::sin(dl / 2);
    const double a = sp * sp + std::cos(p1) * std::cos(p2) * (sl * sl);
    return 2 * EARTH_MEAN_RADIUS_KM * std::asin(std::sqrt(a));
}

double geodesic_km(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
    // Andoyer [Meeus98] ch. 11, in app.geography.geodesic_km's operation order.
    const double fh = ((lat1_deg + lat2_deg) / 2.0) * DEG_TO_RAD;
    const double gh = ((lat1_deg - lat2_deg) / 2.0) * DEG_TO_RAD;
    const double lh = (numerics::np_remainder(lon1_deg - lon2_deg + 180.0, 360.0) - 180.0) / 2.0 *
                      DEG_TO_RAD;
    const double sin_g = std::sin(gh), cos_g = std::cos(gh);
    const double sin_f = std::sin(fh), cos_f = std::cos(fh);
    const double sin_l = std::sin(lh), cos_l = std::cos(lh);
    const double s = sin_g * sin_g * (cos_l * cos_l) + cos_f * cos_f * (sin_l * sin_l);
    const double c = cos_g * cos_g * (cos_l * cos_l) + sin_f * sin_f * (sin_l * sin_l);
    const double w = std::atan(std::sqrt(s / c));
    const double r = std::sqrt(s * c) / w;
    const double dist = 2.0 * w * WGS84_A_KM;
    const double h1 = (3.0 * r - 1.0) / (2.0 * c);
    const double h2 = (3.0 * r + 1.0) / (2.0 * s);
    return dist * (1.0 + WGS84_F * h1 * (sin_f * sin_f) * (cos_g * cos_g) -
                   WGS84_F * h2 * (cos_f * cos_f) * (sin_g * sin_g));
}

double bearing_deg(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
    const double p1 = lat1_deg * DEG_TO_RAD, p2 = lat2_deg * DEG_TO_RAD;
    const double dl = (lon2_deg - lon1_deg) * DEG_TO_RAD;
    const double y = std::sin(dl) * std::cos(p2);
    const double x = std::cos(p1) * std::sin(p2) - std::sin(p1) * std::cos(p2) * std::cos(dl);
    return numerics::np_remainder(std::atan2(y, x) * RAD_TO_DEG + 360.0, 360.0);
}

std::vector<double> bearing_deg(std::span<const double> lat1_deg, std::span<const double> lon1_deg,
                                std::span<const double> lat2_deg, std::span<const double> lon2_deg) {
    const std::size_t n = lat1_deg.size();
    if (lon1_deg.size() != n || lat2_deg.size() != n || lon2_deg.size() != n)
        throw std::invalid_argument("bearing_deg: lat1, lon1, lat2, lon2 differ in length");
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = bearing_deg(lat1_deg[i], lon1_deg[i], lat2_deg[i], lon2_deg[i]);
    return out;
}

// ------------------------------------------------------- shadow-edge limits

EdgeLimits shadow_edge_limits(std::span<const double> x, std::span<const double> y,
                              std::span<const double> d_deg, std::span<const double> mu_deg,
                              std::span<const double> l, std::span<const double> tan_f,
                              std::span<const double> path_bearing_deg, double max_km,
                              bool sunlit_only, const EdgeRates* rates) {
    const std::size_t n = x.size();
    if (y.size() != n || d_deg.size() != n || mu_deg.size() != n || l.size() != n ||
        tan_f.size() != n || path_bearing_deg.size() != n)
        throw std::invalid_argument("shadow_edge_limits: argument arrays differ in length");
    if (rates && (rates->dx.size() != n || rates->dy.size() != n || rates->dd_deg.size() != n ||
                  rates->dmu_deg.size() != n || rates->dl.size() != n))
        throw std::invalid_argument("shadow_edge_limits: rates differ in length from x");

    // Residual of the edge condition at a point (lat, lon) for instant i:
    // |axis separation| - |l - zeta tan f| = 0 [ES92] / [MeeusSE], with the
    // point reduced to the fundamental plane on the WGS-84 ellipsoid
    // (geo_to_fund [ES92] eq. 8.331). With sunlit_only, a point beyond the
    // terminator (zeta <= 0) counts as outside the shadow (residual 1.0);
    // a NaN zeta keeps the NaN residual, as ``np.where(zeta <= 0.0, 1.0, r)``.
    const auto instant_residual = [&](double lat, double lon, std::size_t i) {
        const ellipsoid::Fund f = ellipsoid::geo_to_fund_one(lat, lon, d_deg[i], mu_deg[i]);
        const double r = std::hypot(f.xi - x[i], f.eta - y[i]) - std::abs(l[i] - f.zeta * tan_f[i]);
        return (sunlit_only && f.zeta <= 0.0) ? 1.0 : r;
    };

    // Path envelope (the oracle's ``envelope_residual``): squared axis
    // separation and zeta at tau [h] from instant i, elements linear in tau,
    // [ES92] eq. 8.331 via geo_to_fund_one; the tau of closest approach by
    // Newton steps with central differences; then the edge condition there.
    struct Sep {
        double s2, zeta;
    };
    const auto separation2 = [&](double lat, double lon, std::size_t i, double tau) {
        const ellipsoid::Fund f = ellipsoid::geo_to_fund_one(
            lat, lon, d_deg[i] + rates->dd_deg[i] * tau, mu_deg[i] + rates->dmu_deg[i] * tau);
        const double u = (x[i] + rates->dx[i] * tau) - f.xi;
        const double v = (y[i] + rates->dy[i] * tau) - f.eta;
        return Sep{u * u + v * v, f.zeta};
    };
    const auto envelope_residual = [&](double lat, double lon, std::size_t i) {
        const double h = ENVELOPE_H;
        double tau = 0.0;
        for (int it = 0; it < ENVELOPE_ITERATIONS; ++it) {
            const double s_m = separation2(lat, lon, i, tau - h).s2;
            const double s_0 = separation2(lat, lon, i, tau).s2;
            const double s_p = separation2(lat, lon, i, tau + h).s2;
            const double g = (s_p - s_m) / (2.0 * h);
            const double c = (s_p - 2.0 * s_0 + s_m) / (h * h);
            if (c > 0.0) {
                // np.clip: maximum then minimum, NaN propagating.
                const double t = tau - g / c;
                tau = std::isnan(t) ? t
                                    : std::min(std::max(t, -ENVELOPE_MAX_TAU_H), ENVELOPE_MAX_TAU_H);
            }
        }
        const Sep sep = separation2(lat, lon, i, tau);
        const double r = std::sqrt(sep.s2) - std::abs((l[i] + rates->dl[i] * tau) - sep.zeta * tan_f[i]);
        return (sunlit_only && sep.zeta <= 0.0) ? 1.0 : r;
    };
    const auto residual = [&](double lat, double lon, std::size_t i) {
        return rates ? envelope_residual(lat, lon, i) : instant_residual(lat, lon, i);
    };

    EdgeLimits out;
    out.north_lat.resize(n);
    out.north_lon.resize(n);
    out.south_lat.resize(n);
    out.south_lon.resize(n);
    out.width_km.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
        // Central point (NaN where the axis misses the Earth; NaN then flows
        // through every comparison as false, exactly as in NumPy).
        const ellipsoid::LonLat c = ellipsoid::fund_to_geo_one(x[i], y[i], d_deg[i], mu_deg[i]);
        const double lat0 = c.lat_deg, lon0 = c.lon_deg;
        // The two perpendicular searches: bearing - 90 first, bearing + 90 second.
        const double bearings[2] = {numerics::np_remainder(path_bearing_deg[i] - 90.0, 360.0),
                                    numerics::np_remainder(path_bearing_deg[i] + 90.0, 360.0)};

        const bool inside = residual(lat0, lon0, i) < 0.0;  // central point inside this shadow
        bool found[2];
        double lat_e[2], lon_e[2];
        for (int j = 0; j < 2; ++j) {
            double s_lo = 0.0, s_hi = max_km;
            {
                const LatLon p = destination(lat0, lon0, bearings[j], s_hi);
                found[j] = (residual(p.lat_deg, p.lon_deg, i) >= 0.0) && inside;
            }
            // 50 fixed halvings on the offset distance [km]; decision ``r < 0``
            // (a NaN residual moves hi), final point at 0.5 (lo + hi). This is
            // the oracle's own loop, not app.numerics.bisect. (numerical)
            for (int it = 0; it < 50; ++it) {
                const double s_mid = 0.5 * (s_lo + s_hi);
                const LatLon p = destination(lat0, lon0, bearings[j], s_mid);
                const double r = residual(p.lat_deg, p.lon_deg, i);
                if (r < 0.0)
                    s_lo = s_mid;
                else
                    s_hi = s_mid;
            }
            const LatLon p = destination(lat0, lon0, bearings[j], 0.5 * (s_lo + s_hi));
            lat_e[j] = p.lat_deg;
            lon_e[j] = p.lon_deg;
        }

        const bool ok = found[0] && found[1];
        if (!ok) {
            lat_e[0] = lat_e[1] = kNaN;
            lon_e[0] = lon_e[1] = kNaN;
        }
        out.width_km[i] = ok ? geodesic_km(lat_e[0], lon_e[0], lat_e[1], lon_e[1]) : 0.0;
        // Northern point = greater latitude (tie -> the first; NaN -> false -> second).
        const bool first_is_north = lat_e[0] >= lat_e[1];
        out.north_lat[i] = first_is_north ? lat_e[0] : lat_e[1];
        out.north_lon[i] = first_is_north ? lon_e[0] : lon_e[1];
        out.south_lat[i] = first_is_north ? lat_e[1] : lat_e[0];
        out.south_lon[i] = first_is_north ? lon_e[1] : lon_e[0];
    }
    return out;
}

// ---------------------------------------------------------- global contacts

namespace {

// ``rho_and_radii`` of app.geography.global_contacts: the axis distance on the
// auxiliary circle, rho = hypot(x, y / rho1) [ES92] eq. 8.331, with l1 and |l2|.
struct RhoRadii {
    std::vector<double> rho, l1, l2_abs;
};

RhoRadii rho_and_radii(const ElementsAt& elements_at, std::span<const double> t) {
    const Elements e = elements_at(t);
    if (e.x.size() != t.size() || e.y.size() != t.size() || e.d.size() != t.size() ||
        e.l1.size() != t.size() || e.l2.size() != t.size())
        throw std::invalid_argument("global_contacts: elements_at returned a wrong-length result");
    RhoRadii r;
    r.rho.resize(t.size());
    r.l1 = e.l1;
    r.l2_abs.resize(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        const ellipsoid::ReductionAux aux = ellipsoid::reduction_aux(e.d[i] * DEG_TO_RAD);
        r.rho[i] = std::hypot(e.x[i], e.y[i] / aux.rho1);
        r.l2_abs[i] = std::abs(e.l2[i]);
    }
    return r;
}

// The three contact conditions [ES92] sec. 8.34, in the oracle's dict order.
enum class Kind { PE, UE, UI };

double condition(Kind k, double rho, double l1, double l2_abs) {
    switch (k) {
    case Kind::PE: return rho - (1.0 + l1);      // penumbra external: P1 (falling), P4 (rising)
    case Kind::UE: return rho - (1.0 + l2_abs);  // umbra external: U1, U4
    case Kind::UI: return rho - (1.0 - l2_abs);  // umbra internal: U2, U3
    }
    return kNaN;
}

}  // namespace

std::vector<Contact> global_contacts(const ElementsAt& elements_at, double half_window_hours) {
    // 1-minute grid over +/- half_window_hours (a solar eclipse lasts at most
    // ~6 h on Earth), NumPy's arange fill rule.
    const std::vector<double> t = numerics::arange(-half_window_hours, half_window_hours + 1e-9,
                                                   1.0 / 60.0);
    const RhoRadii g = rho_and_radii(elements_at, t);

    struct Bracket {
        const char* name;
        Kind kind;
        std::size_t i;  // sign change between t[i] and t[i + 1]
    };
    std::vector<Bracket> brackets;
    const struct {
        Kind kind;
        const char* first;
        const char* last;
    } conditions[] = {{Kind::PE, "P1", "P4"}, {Kind::UE, "U1", "U4"}, {Kind::UI, "U2", "U3"}};
    std::vector<double> f(t.size());
    for (const auto& c : conditions) {
        for (std::size_t i = 0; i < t.size(); ++i)
            f[i] = condition(c.kind, g.rho[i], g.l1[i], g.l2_abs[i]);
        // First falling crossing -> first contact; last rising -> last contact.
        bool has_falling = false, has_rising = false;
        std::size_t min_falling = 0, max_rising = 0;
        for (const numerics::SignChange& sc : numerics::sign_changes(t, f)) {
            if (sc.rising) {
                if (!has_rising || sc.i > max_rising) max_rising = sc.i;
                has_rising = true;
            } else {
                if (!has_falling || sc.i < min_falling) min_falling = sc.i;
                has_falling = true;
            }
        }
        if (has_falling) brackets.push_back({c.first, c.kind, min_falling});
        if (has_rising) brackets.push_back({c.last, c.kind, max_rising});
    }
    if (brackets.empty()) return {};

    // One vectorized objective over all brackets: each evaluates its own
    // condition (the oracle's ``np.select`` on ``kinds``).
    const auto f_all = [&](std::span<const double> tt) -> std::vector<double> {
        const RhoRadii r = rho_and_radii(elements_at, tt);
        std::vector<double> out(tt.size());
        for (std::size_t k = 0; k < tt.size(); ++k)
            out[k] = condition(brackets[k].kind, r.rho[k], r.l1[k], r.l2_abs[k]);
        return out;
    };
    std::vector<double> t_lo(brackets.size()), t_hi(brackets.size());
    for (std::size_t k = 0; k < brackets.size(); ++k) {
        t_lo[k] = t[brackets[k].i];
        t_hi[k] = t[brackets[k].i + 1];
    }
    const std::vector<double> roots = numerics::bisect(f_all, t_lo, t_hi);

    std::vector<Contact> out;
    out.reserve(brackets.size());
    for (std::size_t k = 0; k < brackets.size(); ++k) out.push_back({brackets[k].name, roots[k]});
    return out;
}

std::vector<Contact> global_contacts(double et0, Frame frame, double half_window_hours) {
    // BesselianModel.evaluate_direct: besselian_instants(et0 + t * 3600.0, frame).
    // (Its mu unwrap is irrelevant here: only x, y, d, l1, l2 are read.)
    const ElementsAt at = [et0, frame](std::span<const double> t_hours) {
        std::vector<double> et(t_hours.size());
        for (std::size_t i = 0; i < t_hours.size(); ++i) et[i] = et0 + t_hours[i] * 3600.0;
        return ephem::besselian_instants(et, frame);
    };
    return global_contacts(at, half_window_hours);
}

}  // namespace eclipse::geometry

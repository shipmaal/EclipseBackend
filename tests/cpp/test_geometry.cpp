// Shadow geometry (eclipse/geometry.hpp) — parity against the ``gc`` / ``rad``
// / ``lim`` records of fixtures/geometry_cases.txt (the Python oracle at
// literal inputs: offline, no kernels), the ``contacts`` records of the three
// modern per-eclipse fixtures (kernels required; skipped otherwise), and a
// kernel-free unit test of ``global_contacts`` through the injectable
// ``ElementsAt`` with hand-computable crossings.
//
// Tolerances are the roadmap §4 gates: limits 1e-9 deg / 1e-6 km, contacts
// 1e-9 h; the great-circle helpers and shadow_radii use the same gates.
// Measured against the fixtures (x86-64, GCC 13): ``gc`` lat 7.1e-15 deg,
// lon / distance / bearing bit-identical; ``rad`` bit-identical; ``lim`` lat
// 4.3e-14 deg, lon 3.4e-13 deg, width 1.1e-11 km (a 1-ulp libm-vs-NumPy
// difference in the trig flips one of the 50 bisection decisions at an exact
// tie, moving a 10 000 km penumbral search by ~1e-11 km and a 600 km umbral
// one by ~2e-12 km); ``contacts`` bit-identical (0 h) at all three eclipses.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "eclipse/constants.hpp"
#include "eclipse/elements.hpp"
#include "eclipse/eop.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/geometry.hpp"
#include "fixtures.hpp"

using Catch::Matchers::WithinAbs;
namespace geo = eclipse::geometry;

namespace {
// Roadmap §4 gates.
constexpr double kDegTol = 1e-9;   // latitudes, longitudes, bearings [deg]
constexpr double kKmTol = 1e-6;    // distances, widths, radii [km]
constexpr double kHourTol = 1e-9;  // contact times [h]
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::vector<fixtures::Record> records(const char* kind) {
    std::vector<fixtures::Record> out;
    for (auto& r : fixtures::read("geometry_cases.txt"))
        if (r.kind == kind) out.push_back(std::move(r));
    return out;
}

// Same skip logic as test_elements.cpp: kernels bootstrapped and the pinned
// DE440s on disk, EOP subset installed, metakernel furnished.
bool setup(const std::vector<fixtures::Record>& recs) {
    if (!std::filesystem::exists(fixtures::kMetakernel)) {
        SKIP("kernels not bootstrapped: " + fixtures::kMetakernel.string());
        return false;
    }
    if (!fixtures::pinned_spk_present(recs)) {
        SKIP("fixtures pin de440s.bsp (NAIF set); kernel set on disk differs");
        return false;
    }
    std::vector<double> mjd, xp, yp, dut1;
    for (const auto& r : fixtures::read("eop_subset.txt")) {
        if (r.kind != "eop") continue;
        mjd.push_back(r.num(0));
        xp.push_back(r.num(1));
        yp.push_back(r.num(2));
        dut1.push_back(r.num(3));
    }
    eclipse::eop::set_table(mjd, xp, yp, dut1);
    eclipse::ephem::kclear();
    eclipse::ephem::furnish(fixtures::kMetakernel.string());
    return true;
}

void check_contacts_fixture(const std::string& name) {
    const auto recs = fixtures::read(name);
    if (!setup(recs)) return;
    int checked = 0;
    for (const auto& r : recs) {
        if (r.kind != "contacts") continue;
        INFO(name << " " << r.tokens.at(0) << " " << r.tokens.at(1));
        const eclipse::Frame frame = eclipse::frame_from_string(r.tokens.at(1));
        const std::vector<geo::Contact> c = geo::global_contacts(r.num(2), frame, r.num(3));
        // The record lists (name, t_hours) pairs in the oracle dict's insertion
        // order; the same names in the same order, each to the 1e-9 h gate.
        const size_t n_pairs = (r.tokens.size() - 4) / 2;
        REQUIRE(c.size() == n_pairs);
        for (size_t k = 0; k < n_pairs; ++k) {
            INFO("contact " << k << " " << r.tokens.at(4 + 2 * k));
            CHECK(c[k].name == r.tokens.at(4 + 2 * k));
            CHECK_THAT(c[k].t_hours, WithinAbs(r.num(5 + 2 * k), kHourTol));
        }
        ++checked;
    }
    CHECK(checked == 1);
}

// Synthetic elements for the kernel-free global_contacts test: the axis moves
// along x at 0.5 Earth radii per hour through a fixed y at declination 0, with
// constant cone radii, so every contact is a closed-form crossing of
// hypot(x, y / rho1) with 1 + l1, 1 + |l2| and 1 - |l2|.
struct Synthetic {
    double y = 0.0, l1 = 0.55, l2 = -0.012;
    mutable int calls = 0;
    eclipse::Elements operator()(std::span<const double> t) const {
        ++calls;
        eclipse::Elements e;
        e.reserve(t.size());
        for (const double ti : t) {
            e.x.push_back(0.5 * ti);
            e.y.push_back(y);
            e.z.push_back(0.0);
            e.d.push_back(0.0);
            e.mu.push_back(0.0);
            e.l1.push_back(l1);
            e.l2.push_back(l2);
            e.tan_f1.push_back(0.0046);
            e.tan_f2.push_back(0.0046);
        }
        return e;
    }
    // Hour at which hypot(0.5 t, y / rho1) == r for t > 0 (NaN if never).
    double crossing(double r) const {
        const double rho1 = std::sqrt(1.0 - eclipse::constants::WGS84_E2);  // d = 0
        const double y1 = y / rho1;
        return std::sqrt(r * r - y1 * y1) / 0.5;
    }
};
}  // namespace

TEST_CASE("great-circle helpers parity: _destination, _haversine_km, bearing", "[parity]") {
    const auto recs = records("gc");
    REQUIRE(recs.size() >= 21);
    std::vector<double> lat1, lon1, lat2, lon2, brg12;
    for (const auto& r : recs) {
        INFO("lat=" << r.tokens.at(0) << " lon=" << r.tokens.at(1) << " brg=" << r.tokens.at(2)
                    << " dist=" << r.tokens.at(3));
        const geo::LatLon p = geo::destination(r.num(0), r.num(1), r.num(2), r.num(3));
        CHECK_THAT(p.lat_deg, WithinAbs(r.num(4), kDegTol));
        CHECK_THAT(p.lon_deg, WithinAbs(r.num(5), kDegTol));
        CHECK(p.lon_deg >= -180.0);
        CHECK(p.lon_deg < 180.0);
        // The oracle computed distance and bearing from ITS destination; feed the
        // fixture's values so the helpers are checked on identical inputs.
        CHECK_THAT(geo::haversine_km(r.num(0), r.num(1), r.num(4), r.num(5)),
                   WithinAbs(r.num(6), kKmTol));
        CHECK_THAT(geo::bearing_deg(r.num(0), r.num(1), r.num(4), r.num(5)),
                   WithinAbs(r.num(7), kDegTol));
        lat1.push_back(r.num(0));
        lon1.push_back(r.num(1));
        lat2.push_back(r.num(4));
        lon2.push_back(r.num(5));
        brg12.push_back(r.num(7));
    }
    // Array form: identical to the scalar one, and the equal-length contract.
    const std::vector<double> b = geo::bearing_deg(lat1, lon1, lat2, lon2);
    REQUIRE(b.size() == recs.size());
    for (size_t i = 0; i < recs.size(); ++i)
        CHECK(b[i] == geo::bearing_deg(lat1[i], lon1[i], lat2[i], lon2[i]));
    const double one[] = {0.0};
    CHECK_THROWS_AS(geo::bearing_deg(lat1, lon1, lat2, one), std::invalid_argument);

    // Bearings lie in [0, 360) (NumPy's %); a due-west step keeps 270 exactly.
    for (const double v : b) {
        CHECK(v >= 0.0);
        CHECK(v < 360.0);
    }
    CHECK(geo::bearing_deg(0.0, 10.0, 0.0, 9.0) == 270.0);
    // Zero distance returns the start point unchanged.
    const geo::LatLon same = geo::destination(10.0, 20.0, 0.0, 0.0);
    CHECK(same.lat_deg == 10.0);
    CHECK(same.lon_deg == 20.0);
    CHECK(geo::haversine_km(10.0, 20.0, 10.0, 20.0) == 0.0);
}

TEST_CASE("shadow_radii parity: [ES92] 8.353", "[parity]") {
    const auto recs = records("rad");
    REQUIRE(recs.size() >= 39);
    size_t n_off = 0, n_total = 0, n_annular = 0;
    for (const auto& r : recs) {
        INFO("x=" << r.tokens.at(0) << " y=" << r.tokens.at(1) << " d=" << r.tokens.at(2));
        const geo::ShadowRadii s =
            geo::shadow_radii(r.num(0), r.num(1), r.num(2), r.num(3), r.num(4), r.num(5), r.num(6));
        CHECK_THAT(s.penumbra_km, WithinAbs(r.num(7), kKmTol));
        CHECK_THAT(s.umbra_km, WithinAbs(r.num(8), kKmTol));
        CHECK(s.is_total == (r.num(9) != 0.0));
        if (r.num(7) == 0.0 && r.num(8) == 0.0) {
            CHECK_FALSE(s.is_total);
            ++n_off;
        } else if (s.is_total) {
            ++n_total;
        } else {
            ++n_annular;
        }
    }
    CHECK(n_off >= 3);      // the +/-3 h track ends fall off the Earth
    CHECK(n_total >= 10);   // 2017 and 2024 are total
    CHECK(n_annular >= 5);  // 2023 is annular
}

TEST_CASE("shadow_edge_limits parity: shadow_edge_limits_v, umbral and penumbral", "[parity]") {
    const auto recs = records("lim");
    REQUIRE(recs.size() >= 64);
    // Group the records by (max_km, sunlit) so each group is one vectorized
    // call, exactly as the oracle dumped them.
    struct Group {
        double max_km;
        bool sunlit;
        std::vector<double> x, y, d, mu, l, tf, brg, n_lat, n_lon, s_lat, s_lon, width;
    };
    std::vector<Group> groups;
    for (const auto& r : recs) {
        const double max_km = r.num(7);
        const bool sunlit = r.num(8) != 0.0;
        Group* g = nullptr;
        for (auto& cand : groups)
            if (cand.max_km == max_km && cand.sunlit == sunlit) g = &cand;
        if (g == nullptr) {
            groups.push_back(Group{max_km, sunlit, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}});
            g = &groups.back();
        }
        g->x.push_back(r.num(0));
        g->y.push_back(r.num(1));
        g->d.push_back(r.num(2));
        g->mu.push_back(r.num(3));
        g->l.push_back(r.num(4));
        g->tf.push_back(r.num(5));
        g->brg.push_back(r.num(6));
        g->n_lat.push_back(r.num(9));
        g->n_lon.push_back(r.num(10));
        g->s_lat.push_back(r.num(11));
        g->s_lon.push_back(r.num(12));
        g->width.push_back(r.num(13));
    }
    REQUIRE(groups.size() == 3);  // umbra 600 km sunlit; penumbra 10000 km sunlit / not

    size_t n_nan = 0, n_found = 0, n_terminator_clipped = 0;
    for (const auto& g : groups) {
        const geo::EdgeLimits e =
            geo::shadow_edge_limits(g.x, g.y, g.d, g.mu, g.l, g.tf, g.brg, g.max_km, g.sunlit);
        REQUIRE(e.north_lat.size() == g.x.size());
        for (size_t i = 0; i < g.x.size(); ++i) {
            INFO("max_km=" << g.max_km << " sunlit=" << g.sunlit << " x=" << g.x[i]
                           << " y=" << g.y[i] << " l=" << g.l[i]);
            // NaN-mask identity on all four coordinates, and width 0 where NaN.
            CHECK(std::isnan(e.north_lat[i]) == std::isnan(g.n_lat[i]));
            CHECK(std::isnan(e.north_lon[i]) == std::isnan(g.n_lon[i]));
            CHECK(std::isnan(e.south_lat[i]) == std::isnan(g.s_lat[i]));
            CHECK(std::isnan(e.south_lon[i]) == std::isnan(g.s_lon[i]));
            if (std::isnan(g.n_lat[i])) {
                CHECK(e.width_km[i] == 0.0);
                CHECK(g.width[i] == 0.0);
                ++n_nan;
                continue;
            }
            CHECK_THAT(e.north_lat[i], WithinAbs(g.n_lat[i], kDegTol));
            CHECK_THAT(e.north_lon[i], WithinAbs(g.n_lon[i], kDegTol));
            CHECK_THAT(e.south_lat[i], WithinAbs(g.s_lat[i], kDegTol));
            CHECK_THAT(e.south_lon[i], WithinAbs(g.s_lon[i], kDegTol));
            CHECK_THAT(e.width_km[i], WithinAbs(g.width[i], kKmTol));
            CHECK(e.north_lat[i] >= e.south_lat[i]);
            ++n_found;
        }
        if (g.max_km == 600.0 && g.sunlit) {
            // The scalar contract of the fixture's off-Earth sample (x = 2):
            // NaN limits, zero width.
            const double x[] = {2.0}, y[] = {0.0}, d[] = {7.5}, mu[] = {90.0}, l[] = {-0.0106},
                         tf[] = {0.00464}, brg[] = {45.0};
            const geo::EdgeLimits off = geo::shadow_edge_limits(x, y, d, mu, l, tf, brg);
            CHECK(std::isnan(off.north_lat[0]));
            CHECK(std::isnan(off.north_lon[0]));
            CHECK(std::isnan(off.south_lat[0]));
            CHECK(std::isnan(off.south_lon[0]));
            CHECK(off.width_km[0] == 0.0);
        }
    }
    // Where sunlit_only clips a night-side penumbral limit to the terminator,
    // the unclipped search of the same instant differs (or is not found).
    const Group* sun = nullptr;
    const Group* night = nullptr;
    for (const auto& g : groups) {
        if (g.max_km == 10000.0) (g.sunlit ? sun : night) = &g;
    }
    REQUIRE(sun != nullptr);
    REQUIRE(night != nullptr);
    REQUIRE(sun->x.size() == night->x.size());
    for (size_t i = 0; i < sun->x.size(); ++i) {
        const bool s_nan = std::isnan(sun->n_lat[i]), n_nan_ = std::isnan(night->n_lat[i]);
        if (s_nan != n_nan_ || (!s_nan && sun->n_lat[i] != night->n_lat[i])) ++n_terminator_clipped;
    }
    CHECK(n_nan >= 3);  // two night-side penumbral rows (sunlit=0) + the off-Earth sample
    CHECK(n_found >= 50);
    CHECK(n_terminator_clipped >= 1);

    // Equal-length contract.
    const double one[] = {0.0};
    CHECK_THROWS_AS(geo::shadow_edge_limits(groups[0].x, groups[0].y, groups[0].d, groups[0].mu,
                                            groups[0].l, groups[0].tf, one),
                    std::invalid_argument);
    // NaN inputs give NaN limits and zero width rather than throwing.
    const double xn[] = {kNaN}, y0[] = {0.0}, d0[] = {0.0}, mu0[] = {0.0}, l0[] = {-0.01},
                 tf0[] = {0.0046}, b0[] = {0.0};
    const geo::EdgeLimits en = geo::shadow_edge_limits(xn, y0, d0, mu0, l0, tf0, b0);
    CHECK(std::isnan(en.north_lat[0]));
    CHECK(en.width_km[0] == 0.0);
}

TEST_CASE("global_contacts: synthetic elements, closed-form crossings, oracle order") {
    // x = 0.5 t, y = 0, d = 0, l1 = 0.55, l2 = -0.012: rho = 0.5 |t|, so the
    // contacts are at |t| = 1.55 / 0.5 = 3.1 (P), 1.012 / 0.5 = 2.024 (U ext)
    // and 0.988 / 0.5 = 1.976 (U int) — none on a 1-minute grid node.
    Synthetic syn;
    const std::vector<geo::Contact> c = geo::global_contacts(std::cref(syn), 5.0);
    REQUIRE(c.size() == 6);
    const char* names[] = {"P1", "P4", "U1", "U4", "U2", "U3"};
    const double times[] = {-3.1, 3.1, -2.024, 2.024, -1.976, 1.976};
    for (size_t k = 0; k < 6; ++k) {
        INFO("contact " << k);
        CHECK(c[k].name == names[k]);
        // 30 halvings of a 1-minute bracket: |error| <= 1/60/2^31 h ~ 7.8e-12.
        CHECK_THAT(c[k].t_hours, WithinAbs(times[k], 1e-10));
    }
    // One vectorized evaluation for the grid, one for f(t_lo), one per bisection
    // iteration: 32 calls regardless of how many contacts exist.
    CHECK(syn.calls == 32);

    // A narrower window omits the contacts outside it (order kept).
    syn.calls = 0;
    const std::vector<geo::Contact> inner = geo::global_contacts(std::cref(syn), 2.0);
    REQUIRE(inner.size() == 2);
    CHECK(inner[0].name == "U2");
    CHECK(inner[1].name == "U3");
    CHECK_THAT(inner[0].t_hours, WithinAbs(-1.976, 1e-10));
    CHECK_THAT(inner[1].t_hours, WithinAbs(1.976, 1e-10));

    // Penumbra only (axis passes 1.2 Earth radii from the centre): P1 and P4
    // at the closed-form crossing of hypot(0.5 t, y / rho1) = 1 + l1.
    Synthetic partial;
    partial.y = 1.2;
    partial.l2 = 0.01;
    const std::vector<geo::Contact> p = geo::global_contacts(std::cref(partial), 5.0);
    REQUIRE(p.size() == 2);
    CHECK(p[0].name == "P1");
    CHECK(p[1].name == "P4");
    const double tp = partial.crossing(1.55);
    CHECK_THAT(p[0].t_hours, WithinAbs(-tp, 1e-10));
    CHECK_THAT(p[1].t_hours, WithinAbs(tp, 1e-10));

    // No contact at all (the axis never comes within 1 + l1): empty, one call.
    Synthetic miss;
    miss.y = 3.0;
    CHECK(geo::global_contacts(std::cref(miss), 5.0).empty());
    CHECK(miss.calls == 1);

    // A wrong-length elements function is an error, not a silent read.
    const geo::ElementsAt bad = [](std::span<const double>) { return eclipse::Elements{}; };
    CHECK_THROWS_AS(geo::global_contacts(bad, 1.0), std::invalid_argument);
}

TEST_CASE("global_contacts parity: 2017-08-21 total", "[kernels][parity]") {
    check_contacts_fixture("se2017aug21.txt");
}
TEST_CASE("global_contacts parity: 2023-10-14 annular", "[kernels][parity]") {
    check_contacts_fixture("se2023oct14.txt");
}
TEST_CASE("global_contacts parity: 2024-04-08 total", "[kernels][parity]") {
    check_contacts_fixture("se2024apr08.txt");
}

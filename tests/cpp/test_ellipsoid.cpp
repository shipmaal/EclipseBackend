// Ellipsoid reduction (eclipse/ellipsoid.hpp) — parity against the ``red`` /
// ``f2g`` / ``g2f`` records of fixtures/geometry_cases.txt (the Python oracle
// at literal inputs, so this runs offline, no kernels) plus the round-trip
// test of tests/test_geography.py::test_geo_to_fund_round_trip.
//
// Tolerance is the roadmap §4 gate (1e-13). Measured residuals against the
// fixture: ``red`` 0.0 (bit-identical), ``g2f`` 0.0 (bit-identical), ``f2g``
// 1.42e-14 deg (a 1-ulp libm-vs-NumPy difference in the inverse trig), and the
// round trip recovers lat/lon to 3.98e-13 deg.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include "eclipse/constants.hpp"
#include "eclipse/ellipsoid.hpp"
#include "fixtures.hpp"

using Catch::Matchers::WithinAbs;
namespace ell = eclipse::ellipsoid;

namespace {
constexpr double kTol = 1e-13;  // roadmap §4 parity gate for ellipsoid.hpp
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::vector<fixtures::Record> records(const char* kind) {
    std::vector<fixtures::Record> out;
    for (auto& r : fixtures::read("geometry_cases.txt"))
        if (r.kind == kind) out.push_back(std::move(r));
    return out;
}
}  // namespace

TEST_CASE("reduction_aux parity: _reduction_aux [ES92] 8.331", "[parity]") {
    const auto recs = records("red");
    REQUIRE(recs.size() >= 25);
    for (const auto& r : recs) {
        INFO("d = " << r.tokens.at(0) << " deg");
        const ell::ReductionAux a = ell::reduction_aux(r.num(0) * eclipse::constants::DEG_TO_RAD);
        CHECK_THAT(a.rho1, WithinAbs(r.num(1), kTol));
        CHECK_THAT(a.rho2, WithinAbs(r.num(2), kTol));
        CHECK_THAT(a.sin_d1, WithinAbs(r.num(3), kTol));
        CHECK_THAT(a.cos_d1, WithinAbs(r.num(4), kTol));
        CHECK_THAT(a.sin_d1_d2, WithinAbs(r.num(5), kTol));
        CHECK_THAT(a.cos_d1_d2, WithinAbs(r.num(6), kTol));
    }
    // Identities of the auxiliary declinations: sin^2 + cos^2 = 1 for d1 and d1 - d2.
    const ell::ReductionAux a = ell::reduction_aux(0.13);
    CHECK_THAT(a.sin_d1 * a.sin_d1 + a.cos_d1 * a.cos_d1, WithinAbs(1.0, 1e-15));
    CHECK_THAT(a.sin_d1_d2 * a.sin_d1_d2 + a.cos_d1_d2 * a.cos_d1_d2, WithinAbs(1.0, 1e-15));
}

TEST_CASE("fund_to_geo parity: fund_to_geo_v [ES92] 8.331-8.334, NaN off the Earth", "[parity]") {
    const auto recs = records("f2g");
    REQUIRE(recs.size() >= 40);
    size_t n_nan = 0, n_on = 0;
    for (const auto& r : recs) {
        INFO("x=" << r.tokens.at(0) << " y=" << r.tokens.at(1) << " d=" << r.tokens.at(2)
                  << " mu=" << r.tokens.at(3));
        const ell::LonLat p = ell::fund_to_geo_one(r.num(0), r.num(1), r.num(2), r.num(3));
        const double lon = r.num(4), lat = r.num(5);
        // NaN-mask identity: NaN expected <=> NaN produced, for both fields.
        CHECK(std::isnan(p.lon_deg) == std::isnan(lon));
        CHECK(std::isnan(p.lat_deg) == std::isnan(lat));
        if (std::isnan(lon)) {
            CHECK(std::isnan(lat));
            ++n_nan;
            continue;
        }
        CHECK_THAT(p.lon_deg, WithinAbs(lon, kTol));
        CHECK_THAT(p.lat_deg, WithinAbs(lat, kTol));
        ++n_on;
    }
    CHECK(n_nan >= 3);  // the OFF_EARTH cases of tools/dump_oracle.py are in the fixture
    CHECK(n_on >= 20);

    // Array form: same values, and the equal-length contract.
    std::vector<double> x, y, d, mu;
    for (const auto& r : recs) {
        x.push_back(r.num(0));
        y.push_back(r.num(1));
        d.push_back(r.num(2));
        mu.push_back(r.num(3));
    }
    const ell::Geographic g = ell::fund_to_geo(x, y, d, mu);
    REQUIRE(g.lon_deg.size() == recs.size());
    for (size_t i = 0; i < recs.size(); ++i) {
        const ell::LonLat p = ell::fund_to_geo_one(x[i], y[i], d[i], mu[i]);
        CHECK((std::isnan(p.lon_deg) ? std::isnan(g.lon_deg[i]) : g.lon_deg[i] == p.lon_deg));
        CHECK((std::isnan(p.lat_deg) ? std::isnan(g.lat_deg[i]) : g.lat_deg[i] == p.lat_deg));
    }
    const double one[] = {0.0};
    CHECK_THROWS_AS(ell::fund_to_geo(x, y, d, one), std::invalid_argument);

    // NaN inputs propagate (NumPy semantics) rather than throw.
    const ell::LonLat pn = ell::fund_to_geo_one(kNaN, 0.0, 7.5, 90.0);
    CHECK(std::isnan(pn.lon_deg));
    CHECK(std::isnan(pn.lat_deg));
    // Longitude is wrapped into [-180, 180) with NumPy's %.
    for (const auto& r : recs) {
        if (std::isnan(r.num(4))) continue;
        const ell::LonLat p = ell::fund_to_geo_one(r.num(0), r.num(1), r.num(2), r.num(3));
        CHECK(p.lon_deg >= -180.0);
        CHECK(p.lon_deg < 180.0);
    }
}

TEST_CASE("geo_to_fund parity: geo_to_fund [ES92] 8.331, [Meeus98] 11.1", "[parity]") {
    const auto recs = records("g2f");
    REQUIRE(recs.size() >= 49);
    std::vector<double> lat, lon, d, mu;
    for (const auto& r : recs) {
        INFO("lat=" << r.tokens.at(0) << " lon=" << r.tokens.at(1) << " d=" << r.tokens.at(2)
                    << " mu=" << r.tokens.at(3));
        const ell::Fund f = ell::geo_to_fund_one(r.num(0), r.num(1), r.num(2), r.num(3));
        CHECK_THAT(f.xi, WithinAbs(r.num(4), kTol));
        CHECK_THAT(f.eta, WithinAbs(r.num(5), kTol));
        CHECK_THAT(f.zeta, WithinAbs(r.num(6), kTol));
        lat.push_back(r.num(0));
        lon.push_back(r.num(1));
        d.push_back(r.num(2));
        mu.push_back(r.num(3));
    }
    const ell::Fundamental f = ell::geo_to_fund(lat, lon, d, mu);
    REQUIRE(f.xi.size() == recs.size());
    for (size_t i = 0; i < recs.size(); ++i) {
        const ell::Fund p = ell::geo_to_fund_one(lat[i], lon[i], d[i], mu[i]);
        CHECK(f.xi[i] == p.xi);
        CHECK(f.eta[i] == p.eta);
        CHECK(f.zeta[i] == p.zeta);
    }
    const double one[] = {0.0};
    CHECK_THROWS_AS(ell::geo_to_fund(lat, one, d, mu), std::invalid_argument);
}

TEST_CASE("geo_to_fund / fund_to_geo round trip on the near side") {
    // tests/test_geography.py::test_geo_to_fund_round_trip: the two are exact
    // inverses on the shadow-facing side (zeta > 0); the far side maps to the
    // near-side solution and is skipped. Measured max error 3.98e-13 deg.
    const double d_deg = 7.5862, mu_deg = 89.6;
    int checked = 0;
    for (const double lat : {-40.0, -10.0, 0.0, 20.0, 45.0}) {
        for (const double lon : {-160.0, -104.0, -40.0, 0.0, 80.0}) {
            const ell::Fund f = ell::geo_to_fund_one(lat, lon, d_deg, mu_deg);
            if (f.zeta <= 0.0) continue;
            INFO("lat=" << lat << " lon=" << lon);
            const ell::LonLat p = ell::fund_to_geo_one(f.xi, f.eta, d_deg, mu_deg);
            CHECK_THAT(p.lat_deg, WithinAbs(lat, 1e-9));
            CHECK_THAT(p.lon_deg, WithinAbs(lon, 1e-9));
            ++checked;
        }
    }
    CHECK(checked >= 3);
    // The point on the ellipsoid lies on the unit auxiliary sphere: the
    // reduced coordinates satisfy xi^2 + eta1^2 + zeta1^2 = 1 [ES92] 8.331.
    const ell::ReductionAux a = ell::reduction_aux(d_deg * eclipse::constants::DEG_TO_RAD);
    const ell::Fund f = ell::geo_to_fund_one(20.0, -40.0, d_deg, mu_deg);
    const double eta1 = f.eta / a.rho1;
    const double zeta1 = (f.zeta / a.rho2 + eta1 * a.sin_d1_d2) / a.cos_d1_d2;
    CHECK_THAT(f.xi * f.xi + eta1 * eta1 + zeta1 * zeta1, WithinAbs(1.0, 1e-14));
}

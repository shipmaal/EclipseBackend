// Eclipse catalog (eclipse/catalog.hpp) — phase 4 parity against the Python
// oracle app/catalog.py: the ``ismin`` / ``classify`` / ``pyround`` records of
// fixtures/catalog_cases.txt (literal inputs: offline, no kernels), a
// kernel-free synthetic scan through the injectable ``Sources``, and the
// ``scan`` / ``event`` records of fixtures/catalog_2019_2024.txt (kernels
// required; skipped otherwise) at the gates below, plus thread-count
// independence of the catalog's detail loop and of ``ephem``'s OpenMP element
// loop, and the hidden century-scan benchmark (``[!benchmark]``, roadmap §5).
//
// Gates (roadmap §4, revised by the header's "CONDITIONING OF et_g" note):
//  * candidate set and coarse grid: identical (bit-level);
//  * ``et_g``: |delta| <= 1e-2 s — the greatest-eclipse instant is determined
//    only to ~1 ms by double precision (flat parabola; the refinement's own
//    last steps are 1e-4..1e-3 s), so the roadmap's 1e-6 s is not attainable
//    for et_g itself and is NOT what this file asserts;
//  * ``greatest_utc``: the whole-second string equal to the oracle's, except
//    when the oracle's et_g lies within |delta et_g| of a .5-second rounding
//    boundary — then a 1-s difference is admissible, asserted as such and
//    WARNed with the case (never silent, never looser);
//  * kind / central: identical;
//  * gamma, magnitude: 1e-12 at the oracle's own et_g (the classification
//    replayed there through the public units) and, for the catalog's own
//    row, 1e-12 plus what the et_g band implies — the quantity's excursion
//    over et_g_oracle +/- 1e-2 s, evaluated per event (both are quadratic at
//    the minimum, so the band moves them by ~1e-13 at most);
//  * lat / lon: 1e-11 deg at the oracle's et_g, and for the catalog's row
//    1e-11 deg plus the band excursion (the central point moves ~0.5-1 km/s,
//    i.e. up to ~1e-4 deg over the 1e-2 s band: the et_g conditioning, not the
//    geometry, dominates this residual);
//  * et0 bit-identical when greatest_utc matches; contacts 1e-9 h (names and
//    order identical); LocalRaw at the phase-3 gates (contacts 1e-9 h,
//    magnitude / obscuration / L2' 1e-12, altitudes / azimuths 1e-11 deg,
//    flags identical); width 1e-6 km.
//
// Measured against the fixtures (x86-64 4-core, GCC 13, glibc, NumPy 2.x
// oracle; DE440s; 30 event records = 13 + 2 eclipses x detail 0 / 1):
// ``ismin``, ``classify`` and ``pyround`` bit-identical; the scan grids
// (26 295 and 4 383 samples) and candidate sets (26 and 4) identical; et_g
// 3.3e-3 s (inside the 1e-2 band; median well below), 0 greatest_utc boundary
// cases; at the oracle's et_g: gamma 1.1e-14, magnitude 2.8e-15, lat 5.5e-14
// deg, lon 1.5e-12 deg (the 2021-06-10 point at lat 80.8, where the longitude
// amplifies the 1-ulp x / mu noise by 1 / cos lat); the catalog's own row:
// gamma 1.4e-12, magnitude 1.3e-10, lat 2.6e-5 deg, lon 1.4e-5 deg against
// band excursions of 1.0e-11, 1.2e-9 (the Moon's radial motion changes l1,
// l2 by ~1e-7 R/s), 8.4e-5 deg and 2.1e-4 deg — every row residual is the
// et_g shift times the quantity's rate, not a geometry difference; et0 and
// the global contacts bit-identical (0), LocalRaw times 1.0e-14 h, magnitude
// / obscuration / L2' 0, altitudes / azimuths 5.7e-14 deg, width 9.9e-13 km.
// The ``catalog residuals:`` line each run prints is the live record.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "eclipse/catalog.hpp"
#include "eclipse/circumstances.hpp"
#include "eclipse/ellipsoid.hpp"
#include "eclipse/eop.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/numerics.hpp"
#include "fixtures.hpp"

using Catch::Matchers::WithinAbs;
namespace cat = eclipse::catalog;
namespace circ = eclipse::circumstances;
namespace ephem = eclipse::ephem;

namespace {

// Gates (see the file comment).
constexpr double kEtGTol = 1e-2;      // et_g [s]: the double-precision conditioning band
constexpr double kMagTol = 1e-12;     // gamma, magnitude, LocalRaw magnitude / obscuration / L2'
constexpr double kDegTol = 1e-11;     // lat / lon, altitudes / azimuths [deg]
constexpr double kHourTol = 1e-9;     // contacts, t_max [h]
constexpr double kWidthTol = 1e-6;    // width [km]
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<fixtures::Record> records(const char* file, const char* kind) {
    std::vector<fixtures::Record> out;
    for (auto& r : fixtures::read(file))
        if (r.kind == kind) out.push_back(std::move(r));
    return out;
}

// Same skip logic as test_elements.cpp / test_geometry.cpp: kernels
// bootstrapped and the pinned DE440s on disk, EOP subset installed,
// metakernel furnished.
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
    ephem::kclear();
    ephem::furnish(fixtures::kMetakernel.string());
    return true;
}

// A NaN in the oracle must be a NaN here; otherwise the gate applies.
void check_nan_or_within(double actual, double expected, double tol) {
    if (std::isnan(expected))
        CHECK(std::isnan(actual));
    else
        CHECK_THAT(actual, WithinAbs(expected, tol));
}

// Bit-identity including NaN pairs and the sign of zero.
bool same_bits(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return a == b && std::signbit(a) == std::signbit(b);
}

// Largest |actual - expected| seen, ignoring NaN pairs.
struct MaxAbs {
    double v = 0.0;
    void add(double actual, double expected) {
        if (std::isnan(expected) || std::isnan(actual)) return;
        v = std::max(v, std::abs(actual - expected));
    }
};

cat::Kind kind_from_string(const std::string& s) {
    if (s == "partial") return cat::Kind::Partial;
    if (s == "annular") return cat::Kind::Annular;
    if (s == "total") return cat::Kind::Total;
    if (s == "hybrid") return cat::Kind::Hybrid;
    throw std::invalid_argument("unknown kind " + s);
}

// ------------------------------------------------------------ synthetic scan
// A kernel-free objective with the shape of the real one: near each syzygy
// ``rho = sqrt(g^2 + v^2 (t - t_k)^2)`` (the axis passing the Earth's centre
// at distance g with the Moon's relative speed v ~ 0.5 Earth radii / hour,
// so rho'' = v^2 / g ~ 6e-8 R/s^2 as in the header's conditioning note), and
// ``+inf`` over a 24-hour far-side (full moon) segment between them.
struct Bowl {
    double t1 = 5.0 * 86400.0 + 1234.5;   // first new moon [s]
    double t2 = 25.0 * 86400.0 + 777.25;  // second new moon [s]
    double g1 = 0.3, g2 = 0.8;
    double v = 0.5 / 3600.0;              // Earth radii per second
    double far_lo = 15.0 * 86400.0 - 12.0 * 3600.0, far_hi = 15.0 * 86400.0 + 12.0 * 3600.0;
    mutable int calls = 0;
    mutable std::vector<std::size_t> sizes;

    double rho(double t) const {
        if (t >= far_lo && t <= far_hi) return kInf;
        const double u1 = t - t1, u2 = t - t2;
        return std::abs(u1) <= std::abs(u2) ? std::sqrt(g1 * g1 + v * v * u1 * u1)
                                            : std::sqrt(g2 * g2 + v * v * u2 * u2);
    }
    cat::RhoAt objective() const {
        return [this](std::span<const double> et) {
            ++calls;
            sizes.push_back(et.size());
            std::vector<double> out(et.size());
            for (std::size_t i = 0; i < et.size(); ++i) out[i] = rho(et[i]);
            return out;
        };
    }
};

}  // namespace

// ---------------------------------------------------------------- constants

TEST_CASE("catalog constants are app/catalog.py's, written as the Python writes them") {
    CHECK(cat::COARSE_STEP_S == 7200.0);          // 2.0 * 3600.0
    CHECK(cat::COARSE_CHUNK == 50000);
    CHECK(cat::CANDIDATE_RHO == 2.6);
    CHECK(cat::HYBRID_HALF_SPAN_H == 3.5);
    CHECK(cat::HYBRID_STEP_H == 5.0 / 60.0);
    CHECK(cat::DETAIL_HALF_WINDOW_H == 2.5);
    CHECK(cat::CONTACT_HALF_WINDOW_H == 5.0);
    CHECK(cat::TRACK_DT_H == 0.02);
    CHECK(cat::UMBRA_LIMIT_MAX_KM == 600.0);
}

TEST_CASE("catalog Kind strings are the oracle's row values") {
    CHECK(std::string(cat::to_string(cat::Kind::Partial)) == "partial");
    CHECK(std::string(cat::to_string(cat::Kind::Annular)) == "annular");
    CHECK(std::string(cat::to_string(cat::Kind::Total)) == "total");
    CHECK(std::string(cat::to_string(cat::Kind::Hybrid)) == "hybrid");
}

// ----------------------------------------------------------- offline parity

TEST_CASE("_local_minima parity: ismin records", "[parity]") {
    // Measured: identical index lists at all 14 records (ties, plateaus,
    // inf / NaN samples, the threshold, n < 3, and a real 20-day slice of the
    // 2024 coarse grid with its far-side inf run).
    const auto recs = records("catalog_cases.txt", "ismin");
    REQUIRE(recs.size() >= 8);
    for (const auto& r : recs) {
        const auto n = static_cast<std::size_t>(r.num(0));
        INFO("n=" << n << (n ? " rho0=" + r.tokens.at(1) : std::string()));
        std::vector<double> rho(n);
        for (std::size_t i = 0; i < n; ++i) rho[i] = r.num(1 + i);
        const auto k = static_cast<std::size_t>(r.num(1 + n));
        std::vector<std::size_t> expected(k);
        for (std::size_t j = 0; j < k; ++j) expected[j] = static_cast<std::size_t>(r.num(2 + n + j));
        CHECK(cat::local_minima(rho) == expected);
    }
}

TEST_CASE("_local_minima: ties, plateaus, inf and NaN, grid order") {
    using V = std::vector<std::size_t>;
    CHECK(cat::local_minima(std::vector<double>{}) == V{});
    CHECK(cat::local_minima(std::vector<double>{1.0}) == V{});
    CHECK(cat::local_minima(std::vector<double>{1.0, 0.5}) == V{});
    CHECK(cat::local_minima(std::vector<double>{3.0, 2.0, 1.0, 2.0, 3.0}) == V{2});
    // A plateau's FIRST sample wins (strict below the previous, at or below the next).
    CHECK(cat::local_minima(std::vector<double>{3.0, 1.0, 1.0, 1.0, 3.0}) == V{1});
    // An exact tie with the previous sample is not a minimum.
    CHECK(cat::local_minima(std::vector<double>{1.0, 1.0, 2.0}) == V{});
    // The ends never qualify.
    CHECK(cat::local_minima(std::vector<double>{1.0, 2.0, 3.0}) == V{});
    CHECK(cat::local_minima(std::vector<double>{3.0, 2.0, 1.0}) == V{});
    // +inf (far side) never qualifies; a finite dip between infs does.
    CHECK(cat::local_minima(std::vector<double>{kInf, 2.0, kInf, 1.0, kInf}) == V{1, 3});
    CHECK(cat::local_minima(std::vector<double>{2.0, kInf, 2.0}) == V{});
    // NaN fails every comparison: neither a minimum nor a valid neighbour.
    CHECK(cat::local_minima(std::vector<double>{2.0, kNaN, 1.0, 2.0}) == V{});
    CHECK(cat::local_minima(std::vector<double>{2.0, 1.0, kNaN}) == V{});
    // The threshold is strict.
    CHECK(cat::local_minima(std::vector<double>{3.0, 2.6, 3.0}) == V{});
    CHECK(cat::local_minima(std::vector<double>{3.0, std::nextafter(2.6, 0.0), 3.0}) == V{1});
    // Grid order, never sorted by depth.
    CHECK(cat::local_minima(std::vector<double>{2.0, 1.0, 2.0, 0.5, 0.4, 0.4, 0.9, 2.0, kInf, 2.5,
                                                2.4, 2.5}) == V{1, 4, 10});
}

TEST_CASE("_classify parity: classify records", "[parity]") {
    // Measured: kind identical and magnitude / L2' bit-identical at all 13
    // records (pure +, -, *, / in the oracle's order).
    const auto recs = records("catalog_cases.txt", "classify");
    REQUIRE(recs.size() >= 10);
    for (const auto& r : recs) {
        INFO("rho_g=" << r.tokens.at(0) << " l2=" << r.tokens.at(2) << " central=" << r.tokens.at(5)
                      << " zeta=" << r.tokens.at(6));
        const cat::Classification c = cat::classify(r.num(0), r.num(1), r.num(2), r.num(3), r.num(4),
                                                    r.num(5) != 0.0, r.num(6));
        CHECK(c.kind == kind_from_string(r.tokens.at(7)));
        CHECK(same_bits(c.magnitude, r.num(8)));
        CHECK(same_bits(c.L2p, r.num(9)));
    }
}

TEST_CASE("Python round parity: pyround records", "[parity]") {
    // Measured: bit-identical (value and sign of zero) at all records: the
    // half-even hard cases, the catalog's 1/2/4-place roundings, NaN / inf.
    const auto recs = records("catalog_cases.txt", "pyround");
    REQUIRE(recs.size() >= 27);
    for (const auto& r : recs) {
        INFO("x=" << r.tokens.at(0) << " n=" << r.tokens.at(1));
        const double got = eclipse::numerics::py_round(r.num(0), static_cast<int>(r.num(1)));
        CHECK(same_bits(got, r.num(2)));
    }
}

// ------------------------------------------------------------ synthetic scan

TEST_CASE("coarse_grid is NumPy's arange(et_a - step, et_b + 2 step, step)") {
    const std::vector<double> g = cat::coarse_grid(0.0, 30.0 * 86400.0);
    // ceil((2592000 + 14400 + 7200) / 7200) = 363 samples from -7200.
    REQUIRE(g.size() == 363);
    CHECK(g.front() == -7200.0);
    CHECK(g[1] == 0.0);
    CHECK(g.back() == -7200.0 + 362.0 * 7200.0);
    CHECK(g == eclipse::numerics::arange(-7200.0, 30.0 * 86400.0 + 14400.0, 7200.0));
    CHECK(cat::coarse_grid(100.0, 100.0).size() == 3);  // et_b == et_a still brackets
}

TEST_CASE("synthetic scan: candidates, call counts and the refined minimum") {
    const Bowl bowl;
    const cat::RhoAt f = bowl.objective();
    const double et_a = 0.0, et_b = 30.0 * 86400.0;

    // The coarse grid has 363 samples; only the two syzygies dip below the
    // 2.6 threshold (rho reaches 2.6 within 5.2 h of each), the far-side inf
    // run yields nothing, and the candidates come in grid order.
    const std::vector<double> cand = cat::scan_candidates(et_a, et_b, f);
    CHECK(bowl.calls == 1);  // one chunk: the grid is shorter than COARSE_CHUNK
    REQUIRE(bowl.sizes.size() == 1);
    CHECK(bowl.sizes[0] == 363);
    REQUIRE(cand.size() == 2);
    // Nearest grid samples: t1 = 433234.5 -> grid index 61 (432000); t2 =
    // 2160777.25 -> index 301 (2160000).
    CHECK(cand[0] == -7200.0 + 61.0 * 7200.0);
    CHECK(cand[1] == -7200.0 + 301.0 * 7200.0);

    // Refinement: 10 iterations, each ONE objective call on the 3N vector.
    bowl.calls = 0;
    bowl.sizes.clear();
    const std::vector<double> et_g = cat::refine_greatest(cand, f);
    CHECK(bowl.calls == 10);
    for (const std::size_t s : bowl.sizes) CHECK(s == 3 * cand.size());
    REQUIRE(et_g.size() == 2);
    CHECK_THAT(et_g[0], WithinAbs(bowl.t1, 1e-6));
    CHECK_THAT(et_g[1], WithinAbs(bowl.t2, 1e-6));

    // A candidate whose probes reach the inf segment does not move (NaN step
    // -> the Python's t0 += nan): the far side is never refined into.
    const double near_far[] = {bowl.far_lo - 3600.0};
    const std::vector<double> stuck = cat::refine_greatest(near_far, f);
    CHECK(std::isnan(stuck[0]));
}

TEST_CASE("find_eclipses through injected Sources: range keep, hybrid grid, synthetic elements") {
    // Elements that make every candidate central with a total path whose
    // umbra lifts off at the ends (l2 > 0 there), so the hybrid grid is
    // exercised: x = 0.5 t_h (t_h = hours from the bowl's minimum), y = 0.2,
    // d = 10, mu = 15 t_h, l1 = 0.54, l2 = -0.005 + 0.004 |t_h| (negative at
    // greatest eclipse, positive beyond 1.25 h), tan_f1 = tan_f2 = 0.0046.
    const Bowl bowl;
    cat::Sources src;
    src.rho_at = bowl.objective();
    int element_calls = 0;
    std::vector<std::size_t> element_sizes;
    src.elements_at = [&](std::span<const double> et) {
        ++element_calls;
        element_sizes.push_back(et.size());
        eclipse::Elements e;
        e.reserve(et.size());
        for (const double t : et) {
            const double u = std::abs(t - bowl.t1) <= std::abs(t - bowl.t2) ? t - bowl.t1 : t - bowl.t2;
            const double th = u / 3600.0;
            e.x.push_back(0.5 * th);
            e.y.push_back(0.2);
            e.z.push_back(60.0);
            e.d.push_back(10.0);
            e.mu.push_back(eclipse::numerics::wrap_180(15.0 * th));
            e.l1.push_back(0.54);
            e.l2.push_back(-0.005 + 0.004 * std::abs(th));
            e.tan_f1.push_back(0.0046);
            e.tan_f2.push_back(0.0046);
        }
        return e;
    };
    src.frame = eclipse::Frame::ITRS;

    SECTION("both syzygies kept: central, hybrid; one elements call for the greatest instants and one for all hybrid grids") {
        const std::vector<cat::Event> ev = cat::find_eclipses(0.0, 30.0 * 86400.0, src, false);
        REQUIRE(ev.size() == 2);
        CHECK(element_calls == 2);
        REQUIRE(element_sizes.size() == 2);
        CHECK(element_sizes[0] == 2);        // the two et_g
        CHECK(element_sizes[1] == 2 * 85);   // arange(-3.5, 3.5 + 1e-9, 1/12) has 85 samples, per event
        for (const cat::Event& x : ev) {
            CHECK(x.central);
            CHECK(x.kind == cat::Kind::Hybrid);
            CHECK(!x.detail.has_value());
            // gamma = copysign(hypot(x, y), y) with x ~ 0 at the refined minimum.
            CHECK_THAT(x.gamma, WithinAbs(0.2, 1e-9));
            CHECK(x.gamma > 0.0);
            // Central magnitude (L1' - L2') / (L1' + L2') with zeta from the point.
            const eclipse::ellipsoid::LonLat p = eclipse::ellipsoid::fund_to_geo_one(0.0, 0.2, 10.0, 0.0);
            const double zeta = eclipse::ellipsoid::geo_to_fund_one(p.lat_deg, p.lon_deg, 10.0, 0.0).zeta;
            const cat::Classification c = cat::classify(0.2, 0.54, -0.005, 0.0046, 0.0046, true, zeta);
            CHECK_THAT(x.magnitude, WithinAbs(c.magnitude, 1e-9));
            CHECK_THAT(x.lat_deg, WithinAbs(p.lat_deg, 1e-7));
            CHECK_THAT(x.lon_deg, WithinAbs(p.lon_deg, 1e-7));
        }
        CHECK_THAT(ev[0].et_g, WithinAbs(bowl.t1, 1e-6));
        CHECK_THAT(ev[1].et_g, WithinAbs(bowl.t2, 1e-6));
    }
    SECTION("the range keep is inclusive on et_g, not on the coarse candidate") {
        // A window ending just before the second minimum drops it, although
        // its coarse candidate (the grid sample 777.25 s earlier) is inside.
        const std::vector<cat::Event> ev = cat::find_eclipses(0.0, bowl.t2 - 1e-3, src, false);
        REQUIRE(ev.size() == 1);
        CHECK_THAT(ev[0].et_g, WithinAbs(bowl.t1, 1e-6));
        // ... and a window starting exactly at the first minimum keeps it
        // only if the refined instant lands at or after et_a (it lands within
        // 1e-6 s of t1 on either side, so use a start 1 ms earlier).
        const std::vector<cat::Event> ev2 = cat::find_eclipses(bowl.t1 - 1e-3, 30.0 * 86400.0, src, false);
        REQUIRE(ev2.size() == 2);
    }
    SECTION("the hybrid decision alone through the public hybrid()") {
        CHECK(cat::hybrid(bowl.t1, -0.01, src.elements_at));
        CHECK_FALSE(cat::hybrid(bowl.t1, 0.0, src.elements_at));   // annular at closest approach
        CHECK_FALSE(cat::hybrid(bowl.t1, 0.01, src.elements_at));
        // With the umbra on the ground at both ends there is no hybrid.
        cat::ElementsAtEt always_total = [&](std::span<const double> et) {
            eclipse::Elements e = src.elements_at(et);
            for (double& v : e.l2) v = -0.01;
            return e;
        };
        CHECK_FALSE(cat::hybrid(bowl.t1, -0.01, always_total));
        // With the axis off the Earth on the whole grid: no on-Earth sample -> false.
        cat::ElementsAtEt off_earth = [&](std::span<const double> et) {
            eclipse::Elements e = src.elements_at(et);
            for (double& v : e.x) v = 3.0;
            return e;
        };
        CHECK_FALSE(cat::hybrid(bowl.t1, -0.01, off_earth));
    }
    SECTION("a partial candidate: rho_g > 1 + |l2| -> partial, not central, NaN point") {
        cat::Sources part = src;
        part.elements_at = [&](std::span<const double> et) {
            eclipse::Elements e = src.elements_at(et);
            for (double& v : e.y) v = 1.2;
            return e;
        };
        const std::vector<cat::Event> ev = cat::find_eclipses(0.0, 30.0 * 86400.0, part, false);
        REQUIRE(ev.size() == 2);
        for (const cat::Event& x : ev) {
            CHECK(x.kind == cat::Kind::Partial);
            CHECK_FALSE(x.central);
            CHECK(std::isnan(x.lat_deg));
            CHECK(std::isnan(x.lon_deg));
            CHECK_THAT(x.gamma, WithinAbs(1.2, 1e-9));
            CHECK_THAT(x.magnitude, WithinAbs((0.54 - (1.2 - 1.0)) / (0.54 + -0.005), 1e-9));
        }
        // rho_g > 1 + l1 drops the event entirely.
        part.elements_at = [&](std::span<const double> et) {
            eclipse::Elements e = src.elements_at(et);
            for (double& v : e.y) v = 1.6;
            return e;
        };
        CHECK(cat::find_eclipses(0.0, 30.0 * 86400.0, part, false).empty());
    }
    SECTION("an empty window yields no events and no elements call") {
        // Between the syzygies: no coarse minimum below the threshold.
        const std::vector<cat::Event> ev = cat::find_eclipses(10.0 * 86400.0, 12.0 * 86400.0, src, false);
        CHECK(ev.empty());
        CHECK(element_calls == 0);
    }
}

// ------------------------------------------------------- kernel-backed parity

namespace {

// One ``event`` record of catalog_2019_2024.txt.
struct EventRec {
    std::string label;
    eclipse::Frame frame;
    bool detail;
    double et_g;
    cat::Kind kind;
    bool central;
    double gamma, magnitude, lat, lon;
    std::string greatest_utc;
    // detail = 1
    double et0 = kNaN;
    std::vector<std::pair<std::string, double>> contacts;
    bool has_local = false;
    std::vector<double> local;  // the 26 _LocalRaw fields
    double width = kNaN;        // nan when absent
};

EventRec parse_event(const fixtures::Record& r) {
    EventRec e;
    e.label = r.tokens.at(0);
    e.frame = eclipse::frame_from_string(r.tokens.at(1));
    e.detail = r.num(2) != 0.0;
    e.et_g = r.num(3);
    e.kind = kind_from_string(r.tokens.at(4));
    e.central = r.num(5) != 0.0;
    e.gamma = r.num(6);
    e.magnitude = r.num(7);
    e.lat = r.num(8);
    e.lon = r.num(9);
    e.greatest_utc = r.tokens.at(10);
    if (!e.detail) {
        REQUIRE(r.tokens.size() == 11);
        return e;
    }
    e.et0 = r.num(11);
    const auto nc = static_cast<std::size_t>(r.num(12));
    std::size_t pos = 13;
    for (std::size_t k = 0; k < nc; ++k, pos += 2) e.contacts.emplace_back(r.tokens.at(pos), r.num(pos + 1));
    if (r.tokens.at(pos) == "-") {
        ++pos;
    } else {
        e.has_local = true;
        for (std::size_t k = 0; k < 26; ++k) e.local.push_back(r.num(pos + k));
        pos += 26;
    }
    e.width = r.num(pos);
    REQUIRE(pos + 1 == r.tokens.size());
    return e;
}

// The scan window of a label.
struct Window {
    double et_a, et_b;
};
std::map<std::string, Window> windows() {
    std::map<std::string, Window> w;
    for (const auto& r : records("catalog_2019_2024.txt", "scan")) w[r.tokens.at(0)] = {r.num(1), r.num(2)};
    return w;
}

// The classification replayed at ONE given instant through the public units,
// exactly as find_eclipses does after the refinement: isolates the geometry
// / classification parity from the et_g conditioning.
struct AtInstant {
    bool central;
    cat::Kind kind;
    double gamma, magnitude, lat, lon;
};
AtInstant classify_at(double et, const cat::Sources& src) {
    const double one[] = {et};
    const eclipse::Elements e = src.elements_at(one);
    const double rho_g = std::hypot(e.x[0], e.y[0]);
    const eclipse::ellipsoid::LonLat p = eclipse::ellipsoid::fund_to_geo_one(e.x[0], e.y[0], e.d[0], e.mu[0]);
    AtInstant a;
    a.central = !std::isnan(p.lat_deg);
    double zeta = kNaN;
    if (a.central) zeta = eclipse::ellipsoid::geo_to_fund_one(p.lat_deg, p.lon_deg, e.d[0], e.mu[0]).zeta;
    const cat::Classification c = cat::classify(rho_g, e.l1[0], e.l2[0], e.tan_f1[0], e.tan_f2[0], a.central, zeta);
    a.kind = c.kind;
    if (a.central && cat::hybrid(et, c.L2p, src.elements_at)) a.kind = cat::Kind::Hybrid;
    a.gamma = std::copysign(rho_g, e.y[0]);
    a.magnitude = c.magnitude;
    a.lat = p.lat_deg;
    a.lon = p.lon_deg;
    return a;
}

// The excursion of each row quantity over the et_g band [et - kEtGTol, et +
// kEtGTol]: what a shift of the refined instant inside the band can move it.
struct Band {
    double gamma = 0.0, magnitude = 0.0, lat = 0.0, lon = 0.0;
};
Band band_excursion(double et, const cat::Sources& src) {
    const AtInstant c = classify_at(et, src);
    Band b;
    for (const double s : {-kEtGTol, kEtGTol}) {
        const AtInstant a = classify_at(et + s, src);
        b.gamma = std::max(b.gamma, std::abs(a.gamma - c.gamma));
        b.magnitude = std::max(b.magnitude, std::abs(a.magnitude - c.magnitude));
        if (c.central && a.central) {
            b.lat = std::max(b.lat, std::abs(a.lat - c.lat));
            b.lon = std::max(b.lon, std::abs(eclipse::numerics::wrap_180(a.lon - c.lon)));
        }
    }
    return b;
}

// Fractional UTC second of ``et`` (IERS era: et2utc's rounding boundary is at .5).
double utc_fraction(double et) {
    const std::string s = ephem::et_to_utc_iso(et, 6);  // YYYY-MM-DDTHH:MM:SS.ffffff
    return std::strtod(s.substr(s.rfind('.')).c_str(), nullptr);
}

struct Residuals {
    MaxAbs et_g, gamma, magnitude, lat, lon, gamma_at, magnitude_at, lat_at, lon_at, et0, contacts,
        local_times, local_mags, local_angles, width;
    double band_gamma = 0.0, band_mag = 0.0, band_lat = 0.0, band_lon = 0.0;
    int boundary_cases = 0, events = 0;
};

void check_local(const circ::LocalRaw& a, const std::vector<double>& f, Residuals& res) {
    REQUIRE(f.size() == 26);
    CHECK(a.geometric == (f[0] != 0.0));
    CHECK(a.central == (f[1] != 0.0));
    const double* times[] = {&a.c1, &a.c4, &a.c2, &a.c3, &a.t_max};
    for (std::size_t k = 0; k < 5; ++k) {
        INFO("local time field " << k);
        check_nan_or_within(*times[k], f[2 + k], kHourTol);
        res.local_times.add(*times[k], f[2 + k]);
    }
    check_nan_or_within(a.magnitude, f[7], kMagTol);
    check_nan_or_within(a.obscuration, f[8], kMagTol);
    check_nan_or_within(a.L2_x, f[9], kMagTol);
    res.local_mags.add(a.magnitude, f[7]);
    res.local_mags.add(a.obscuration, f[8]);
    res.local_mags.add(a.L2_x, f[9]);
    for (std::size_t k = 0; k < 5; ++k) {
        INFO("local event " << k);
        check_nan_or_within(a.alt_deg[k], f[10 + k], kDegTol);
        check_nan_or_within(a.az_deg[k], f[15 + k], kDegTol);
        res.local_angles.add(a.alt_deg[k], f[10 + k]);
        res.local_angles.add(a.az_deg[k], f[15 + k]);
        CHECK(a.below[k] == (f[20 + k] != 0.0));
    }
    CHECK(a.eclipse == (f[25] != 0.0));
}

void check_detail(const cat::Detail& d, const EventRec& rec, Residuals& res) {
    CHECK(same_bits(d.et0, rec.et0));
    res.et0.add(d.et0, rec.et0);
    REQUIRE(d.contacts.size() == rec.contacts.size());
    for (std::size_t k = 0; k < d.contacts.size(); ++k) {
        INFO("contact " << rec.contacts[k].first);
        CHECK(d.contacts[k].name == rec.contacts[k].first);
        CHECK_THAT(d.contacts[k].t_hours, WithinAbs(rec.contacts[k].second, kHourTol));
        res.contacts.add(d.contacts[k].t_hours, rec.contacts[k].second);
    }
    CHECK(d.local.has_value() == rec.has_local);
    if (d.local && rec.has_local) check_local(*d.local, rec.local, res);
    CHECK(d.width_km.has_value() == !std::isnan(rec.width));
    if (d.width_km && !std::isnan(rec.width)) {
        CHECK_THAT(*d.width_km, WithinAbs(rec.width, kWidthTol));
        res.width.add(*d.width_km, rec.width);
    }
}

void check_event(const cat::Event& ev, const EventRec& rec, const cat::Sources& src, Residuals& res) {
    ++res.events;
    // et_g within the conditioning band.
    const double d_et = std::abs(ev.et_g - rec.et_g);
    CHECK(d_et <= kEtGTol);
    res.et_g.add(ev.et_g, rec.et_g);

    // The row strings / flags.
    CHECK(ev.kind == rec.kind);
    CHECK(ev.central == rec.central);
    const std::string utc = ephem::et_to_utc(ev.et_g);
    bool utc_matches = utc == rec.greatest_utc;
    if (!utc_matches) {
        // Admissible only as a rounding-boundary flip inside the band.
        const double frac = utc_fraction(rec.et_g);
        INFO("greatest_utc " << utc << " != " << rec.greatest_utc << " (oracle fraction " << frac
                             << ", |delta et_g| " << d_et << ")");
        REQUIRE(std::abs(frac - 0.5) <= d_et);
        WARN("greatest_utc boundary case " << rec.label << " " << rec.greatest_utc << ": native " << utc
                                           << ", oracle et_g fraction " << frac << ", |delta et_g| " << d_et);
        ++res.boundary_cases;
    }

    // gamma / magnitude / point at the oracle's et_g: pure geometry parity.
    const AtInstant at = classify_at(rec.et_g, src);
    CHECK(at.kind == rec.kind);
    CHECK(at.central == rec.central);
    CHECK_THAT(at.gamma, WithinAbs(rec.gamma, kMagTol));
    CHECK_THAT(at.magnitude, WithinAbs(rec.magnitude, kMagTol));
    check_nan_or_within(at.lat, rec.lat, kDegTol);
    check_nan_or_within(at.lon, rec.lon, kDegTol);
    res.gamma_at.add(at.gamma, rec.gamma);
    res.magnitude_at.add(at.magnitude, rec.magnitude);
    res.lat_at.add(at.lat, rec.lat);
    res.lon_at.add(at.lon, rec.lon);

    // The catalog's own row: the same gates plus the band excursion.
    const Band band = band_excursion(rec.et_g, src);
    res.band_gamma = std::max(res.band_gamma, band.gamma);
    res.band_mag = std::max(res.band_mag, band.magnitude);
    res.band_lat = std::max(res.band_lat, band.lat);
    res.band_lon = std::max(res.band_lon, band.lon);
    CHECK_THAT(ev.gamma, WithinAbs(rec.gamma, kMagTol + band.gamma));
    CHECK_THAT(ev.magnitude, WithinAbs(rec.magnitude, kMagTol + band.magnitude));
    check_nan_or_within(ev.lat_deg, rec.lat, kDegTol + band.lat);
    check_nan_or_within(ev.lon_deg, rec.lon, kDegTol + band.lon);
    res.gamma.add(ev.gamma, rec.gamma);
    res.magnitude.add(ev.magnitude, rec.magnitude);
    res.lat.add(ev.lat_deg, rec.lat);
    res.lon.add(ev.lon_deg, rec.lon);

    CHECK(ev.detail.has_value() == rec.detail);
    if (!rec.detail || !ev.detail) return;
    if (utc_matches) {
        check_detail(*ev.detail, rec, res);
    } else {
        // The oracle's model epoch is its own whole second: replay the detail
        // there so the phase-2/3 gates still apply to every number.
        const cat::Detail d = cat::add_detail(rec.et_g, rec.frame, rec.central, rec.lat, rec.lon);
        check_detail(d, rec, res);
    }
}

void print_residuals(const char* what, const Residuals& r) {
    std::cout << "catalog residuals: " << what << ": " << r.events << " events, " << r.boundary_cases
              << " greatest_utc boundary case(s); et_g " << r.et_g.v << " s; at oracle et_g: gamma "
              << r.gamma_at.v << " magnitude " << r.magnitude_at.v << " lat " << r.lat_at.v << " lon "
              << r.lon_at.v << "; catalog row: gamma " << r.gamma.v << " magnitude " << r.magnitude.v
              << " lat " << r.lat.v << " lon " << r.lon.v << " (band excursions: gamma " << r.band_gamma
              << " magnitude " << r.band_mag << " lat " << r.band_lat << " lon " << r.band_lon
              << "); et0 " << r.et0.v << " contacts " << r.contacts.v << " local times " << r.local_times.v
              << " local mags " << r.local_mags.v << " local angles " << r.local_angles.v << " width "
              << r.width.v << "\n";
}

}  // namespace

TEST_CASE("_coarse_grid / _scan_candidates parity: scan records", "[kernels][parity]") {
    // Measured: grid counts and the first / second / last samples bit-identical,
    // candidate sets identical (26 in 2019-2024, 4 in 2024).
    const auto recs = fixtures::read("catalog_2019_2024.txt");
    if (!setup(recs)) return;
    int checked = 0;
    for (const auto& r : recs) {
        if (r.kind != "scan") continue;
        INFO("scan " << r.tokens.at(0));
        const double et_a = r.num(1), et_b = r.num(2);
        const std::vector<double> grid = cat::coarse_grid(et_a, et_b);
        REQUIRE(grid.size() == static_cast<std::size_t>(r.num(3)));
        CHECK(grid[0] == r.num(4));
        CHECK(grid[1] == r.num(5));
        CHECK(grid.back() == r.num(6));
        const auto n_cand = static_cast<std::size_t>(r.num(7));
        std::vector<double> expected(n_cand);
        for (std::size_t j = 0; j < n_cand; ++j) expected[j] = r.num(8 + j);
        const cat::RhoAt f = [](std::span<const double> et) { return cat::rho_at(et); };
        const std::vector<double> cand = cat::scan_candidates(et_a, et_b, f);
        CHECK(cand == expected);
        ++checked;
    }
    CHECK(checked == 2);
}

TEST_CASE("_catalog_raw parity: event records, detail 0 and 1, both windows", "[kernels][parity]") {
    const auto recs = fixtures::read("catalog_2019_2024.txt");
    if (!setup(recs)) return;
    const std::map<std::string, Window> win = windows();
    const cat::Sources src = cat::sources_from_ephem(eclipse::Frame::ITRS);

    // Group the records by (label, detail), in file order.
    std::map<std::pair<std::string, bool>, std::vector<EventRec>> groups;
    std::vector<std::pair<std::string, bool>> order;
    for (const auto& r : recs) {
        if (r.kind != "event") continue;
        EventRec e = parse_event(r);
        const auto key = std::make_pair(e.label, e.detail);
        if (!groups.count(key)) order.push_back(key);
        groups[key].push_back(std::move(e));
    }
    REQUIRE(order.size() == 4);

    Residuals res;
    for (const auto& key : order) {
        const std::vector<EventRec>& expected = groups[key];
        INFO("window " << key.first << " detail " << key.second);
        REQUIRE(win.count(key.first) == 1);
        const Window w = win.at(key.first);
        const std::vector<cat::Event> got = cat::find_eclipses(w.et_a, w.et_b, src, key.second);
        REQUIRE(got.size() == expected.size());
        for (std::size_t k = 0; k < got.size(); ++k) {
            INFO("event " << expected[k].greatest_utc << " " << cat::to_string(expected[k].kind));
            check_event(got[k], expected[k], src, res);
        }
    }
    CHECK(res.events == 2 * (13 + 2));
    print_residuals("event records (ITRS, 2019-2024 and 2024, detail 0 and 1)", res);
}

TEST_CASE("find_eclipses: threads=1 equals the OpenMP default on every field", "[kernels]") {
    const auto recs = fixtures::read("catalog_2019_2024.txt");
    if (!setup(recs)) return;
    const Window w = windows().at("y2024");
    const std::vector<cat::Event> serial = cat::find_eclipses(w.et_a, w.et_b, eclipse::Frame::ITRS, true, 1);
    const std::vector<cat::Event> par = cat::find_eclipses(w.et_a, w.et_b, eclipse::Frame::ITRS, true, 0);
    REQUIRE(serial.size() == 2);
    REQUIRE(par.size() == serial.size());
    for (std::size_t k = 0; k < serial.size(); ++k) {
        const cat::Event &a = serial[k], &b = par[k];
        CHECK(same_bits(a.et_g, b.et_g));
        CHECK(a.kind == b.kind);
        CHECK(a.central == b.central);
        CHECK(same_bits(a.gamma, b.gamma));
        CHECK(same_bits(a.magnitude, b.magnitude));
        CHECK(same_bits(a.lat_deg, b.lat_deg));
        CHECK(same_bits(a.lon_deg, b.lon_deg));
        REQUIRE(a.detail.has_value());
        REQUIRE(b.detail.has_value());
        CHECK(same_bits(a.detail->et0, b.detail->et0));
        REQUIRE(a.detail->contacts.size() == b.detail->contacts.size());
        for (std::size_t c = 0; c < a.detail->contacts.size(); ++c) {
            CHECK(a.detail->contacts[c].name == b.detail->contacts[c].name);
            CHECK(same_bits(a.detail->contacts[c].t_hours, b.detail->contacts[c].t_hours));
        }
        REQUIRE(a.detail->local.has_value());
        REQUIRE(b.detail->local.has_value());
        const circ::LocalRaw &la = *a.detail->local, &lb = *b.detail->local;
        CHECK(la.geometric == lb.geometric);
        CHECK(la.central == lb.central);
        CHECK(la.eclipse == lb.eclipse);
        const double *pa[] = {&la.c1, &la.c4, &la.c2, &la.c3, &la.t_max, &la.magnitude, &la.obscuration, &la.L2_x};
        const double *pb[] = {&lb.c1, &lb.c4, &lb.c2, &lb.c3, &lb.t_max, &lb.magnitude, &lb.obscuration, &lb.L2_x};
        for (std::size_t i = 0; i < 8; ++i) CHECK(same_bits(*pa[i], *pb[i]));
        for (std::size_t i = 0; i < 5; ++i) {
            CHECK(same_bits(la.alt_deg[i], lb.alt_deg[i]));
            CHECK(same_bits(la.az_deg[i], lb.az_deg[i]));
            CHECK(la.below[i] == lb.below[i]);
        }
        REQUIRE(a.detail->width_km.has_value() == b.detail->width_km.has_value());
        if (a.detail->width_km) CHECK(same_bits(*a.detail->width_km, *b.detail->width_km));
    }
}

TEST_CASE("ephem element loop: OpenMP threads give the bits of the serial loop", "[kernels]") {
    // geocentric_vectors' ERFA loop is parallel for n >= 64: the per-instant
    // results must not depend on the thread count (no reductions, ERFA is
    // reentrant). 1000 instants across the 2024-04-08 window in the two
    // ERFA frames, serial (1 thread) vs the default team, bit for bit.
    const auto recs = fixtures::read("catalog_2019_2024.txt");
    if (!setup(recs)) return;
    const double et0 = ephem::utc_to_et("2024-04-08T18:17:20");
    std::vector<double> et(1000);
    for (std::size_t i = 0; i < et.size(); ++i) et[i] = et0 + (static_cast<double>(i) - 500.0) * 36.0;
    for (const eclipse::Frame frame : {eclipse::Frame::ITRS, eclipse::Frame::TOD}) {
        INFO(eclipse::to_string(frame));
#ifdef _OPENMP
        const int before = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        const eclipse::Elements serial = ephem::besselian_instants(et, frame);
        const ephem::SubSolar ss_serial = ephem::sub_solar_points(et, frame);
#ifdef _OPENMP
        omp_set_num_threads(before);
        CHECK(omp_get_max_threads() == before);
#endif
        const eclipse::Elements par = ephem::besselian_instants(et, frame);
        const ephem::SubSolar ss_par = ephem::sub_solar_points(et, frame);
        REQUIRE(par.size() == serial.size());
        for (std::size_t i = 0; i < et.size(); ++i) {
            CHECK(same_bits(serial.x[i], par.x[i]));
            CHECK(same_bits(serial.y[i], par.y[i]));
            CHECK(same_bits(serial.z[i], par.z[i]));
            CHECK(same_bits(serial.d[i], par.d[i]));
            CHECK(same_bits(serial.mu[i], par.mu[i]));
            CHECK(same_bits(serial.l1[i], par.l1[i]));
            CHECK(same_bits(serial.l2[i], par.l2[i]));
            CHECK(same_bits(serial.tan_f1[i], par.tan_f1[i]));
            CHECK(same_bits(serial.tan_f2[i], par.tan_f2[i]));
            CHECK(same_bits(ss_serial.lon_deg[i], ss_par.lon_deg[i]));
            CHECK(same_bits(ss_serial.lat_deg[i], ss_par.lat_deg[i]));
        }
    }
}

TEST_CASE("century scan benchmark 2000-2100, detail off and on", "[!benchmark][kernels]") {
    // Roadmap §5 phase 4: a century scan with detail in < 10 s. The scan is
    // 438k SPICE instants (serial under the lock by design); the detail loop
    // is OpenMP-parallel over the ~225 events. Prints the timings and the
    // core count; the numbers are reported, not asserted.
    const auto recs = fixtures::read("catalog_2019_2024.txt");
    if (!setup(recs)) return;
    const double et_a = ephem::utc_to_et("2000-01-01T00:00:00");
    const double et_b = ephem::utc_to_et("2100-01-01T00:00:00");
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::cout << "catalog benchmark: hardware threads " << std::thread::hardware_concurrency()
              << ", OpenMP max threads " << threads << ", grid " << cat::coarse_grid(et_a, et_b).size()
              << " instants\n";
    for (const bool detail : {false, true}) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<cat::Event> ev = cat::find_eclipses(et_a, et_b, eclipse::Frame::ITRS, detail);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        int central = 0;
        for (const cat::Event& e : ev) central += e.central ? 1 : 0;
        std::cout << "catalog benchmark: detail=" << detail << ": " << ev.size() << " eclipses (" << central
                  << " central) in " << s << " s\n";
        CHECK(ev.size() > 200);
    }
    // The serial detail loop, for the speed-up figure.
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<cat::Event> ev = cat::find_eclipses(et_a, et_b, eclipse::Frame::ITRS, true, 1);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "catalog benchmark: detail=1 threads=1: " << ev.size() << " eclipses in " << s << " s\n";
}

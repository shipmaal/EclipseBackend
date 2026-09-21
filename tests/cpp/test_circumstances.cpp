// Local circumstances (eclipse/circumstances.hpp) — parity against the
// ``ovl`` / ``obsc`` / ``unw`` / ``altaz`` / ``roots`` records of
// fixtures/circumstances_cases.txt (the Python oracle at literal inputs:
// offline, no kernels), the ``local`` / ``grid`` records of the three modern
// per-eclipse fixtures (kernels required; skipped otherwise), kernel-free unit
// tests through the injectable ``Model`` callables (a synthetic axis crossing
// with closed-form contacts, call counts, the horizon rule, the widening
// loop, the ValueError case), the grid-vs-scalar test and the hidden
// 0.5-degree global-grid benchmark (``[!benchmark]``, roadmap §5).
//
// Gates (roadmap §4): contacts 1e-9 h, magnitude 1e-12, grid 1e-12; the
// altitudes use the sub-solar gate 1e-11 deg. Measured against the fixtures
// (x86-64 AVX-512, GCC 13, glibc 2.39, NumPy 2.0.2 oracle): ``ovl``,
// ``obsc``, ``unw`` and ``roots`` bit-identical; ``altaz`` alt and az
// bit-identical, alt_grid 1.4e-14 deg (one record, NumPy's SIMD arccos vs
// libm); ``local`` (30 records): C1/C4/C2/C3 bit-identical (0 h), t_max
// 2.5e-12 h (the parabolic vertex divides a 1-ulp magnitude difference by
// the ~1e-7 curvature denominator), magnitude 1.5e-14, obscuration 1.4e-14,
// L2_x 0, altitudes 2.1e-14 deg, azimuths 5.7e-14 deg, every flag identical;
// ``grid`` (27 records): magnitude 2.4e-14, obscuration 2.9e-14, t_max_hours
// 0 (bit-identical), sun_alt 1.4e-14 deg, flags identical. The nonzero
// residuals are the 1-ulp libm-vs-NumPy-SIMD trig noise of phases 1-2 carried
// through the geometry, not a tolerance to widen.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "eclipse/circumstances.hpp"
#include "eclipse/constants.hpp"
#include "eclipse/elements.hpp"
#include "eclipse/eop.hpp"
#include "eclipse/ephem.hpp"
#include "fixtures.hpp"

using Catch::Matchers::WithinAbs;
namespace circ = eclipse::circumstances;
using eclipse::constants::DEG_TO_RAD;

namespace {
// Roadmap §4 gates.
constexpr double kHourTol = 1e-9;   // contact times, t_max [h]
constexpr double kMagTol = 1e-12;   // magnitude, obscuration, L2', grid values
constexpr double kDegTol = 1e-11;   // altitudes / azimuths [deg] (the sub-solar gate)

std::vector<fixtures::Record> records(const char* kind) {
    std::vector<fixtures::Record> out;
    for (auto& r : fixtures::read("circumstances_cases.txt"))
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
    eclipse::ephem::kclear();
    eclipse::ephem::furnish(fixtures::kMetakernel.string());
    return true;
}

// A NaN in the oracle must be a NaN here; otherwise the gate applies.
void check_nan_or_within(double actual, double expected, double tol) {
    if (std::isnan(expected))
        CHECK(std::isnan(actual));
    else
        CHECK_THAT(actual, WithinAbs(expected, tol));
}

// Largest |actual - expected| seen, ignoring NaN pairs: the ovl / obsc tests
// assert it as the recorded (bit-identical) residual.
struct MaxAbs {
    double v = 0.0;
    void add(double actual, double expected) {
        if (std::isnan(expected) || std::isnan(actual)) return;
        v = std::max(v, std::abs(actual - expected));
    }
};

// Synthetic eclipse for the kernel-free tests: the shadow axis crosses the
// observer at (0, 0) at t = 0. With d = 0 and mu = 15 t deg the observer's
// fundamental-plane position is xi = sin(15 t deg), eta = 0, zeta = cos(15 t
// deg); the axis is at x = sin(15 t deg) + 0.5 t, y = 0, so the separation is
// m = 0.5 |t| and the reduced radii L1' = l1 - cos(15 t deg) tan f, L2' = l2 -
// cos(15 t deg) tan f are even in t: every contact is symmetric about t = 0
// and the closed-form crossing of 0.5 t with L1' (C4) or |L2'| (C3) is a
// contraction fixed point. ``elements_at`` applies ``unwrap_mu_deg`` exactly
// as ``model_from_ephem`` does. The sub-solar point is fixed: at the observer
// ("day", altitude +90) or its antipode ("night", -90).
struct Synthetic {
    double l1 = 0.55, l2 = -0.012, tan_f = 0.0046, y = 0.0;
    double ss_lon = 0.0, ss_lat = 0.0;
    mutable int element_calls = 0, sub_solar_calls = 0;
    mutable std::vector<std::size_t> element_sizes;

    circ::Model model(double half_window_hours) const {
        circ::Model m;
        m.elements_at = [this](std::span<const double> t) {
            ++element_calls;
            element_sizes.push_back(t.size());
            eclipse::Elements e;
            e.reserve(t.size());
            for (const double ti : t) {
                e.x.push_back(std::sin(15.0 * ti * DEG_TO_RAD) + 0.5 * ti);
                e.y.push_back(y);
                e.z.push_back(0.0);
                e.d.push_back(0.0);
                e.mu.push_back(15.0 * ti);
                e.l1.push_back(l1);
                e.l2.push_back(l2);
                e.tan_f1.push_back(tan_f);
                e.tan_f2.push_back(tan_f);
            }
            eclipse::unwrap_mu_deg(e);
            return e;
        };
        m.sub_solar_at = [this](std::span<const double> t) {
            ++sub_solar_calls;
            eclipse::ephem::SubSolar ss;
            ss.lon_deg.assign(t.size(), ss_lon);
            ss.lat_deg.assign(t.size(), ss_lat);
            return ss;
        };
        m.half_window_hours = half_window_hours;
        return m;
    }
    // Fixed point of t = 2 (l - cos(15 t deg) tan f) (C4 for l = l1) or of
    // t = 2 |l2 - cos(15 t deg) tan f| (C3): the positive crossing.
    double crossing(double l, bool absolute) const {
        double t = 1.0;
        for (int i = 0; i < 100; ++i) {
            const double L = l - std::cos(15.0 * t * DEG_TO_RAD) * tan_f;
            t = 2.0 * (absolute ? std::abs(L) : L);
        }
        return t;
    }
};
}  // namespace

// --------------------------------------------------------------- offline parity

TEST_CASE("_overlap_area parity: ovl records", "[parity]") {
    // Measured: bit-identical at all 13 records (disjoint, contained, lens,
    // r = 0, the near-tangent 1.9999 case).
    const auto recs = records("ovl");
    REQUIRE(recs.size() >= 13);
    MaxAbs res;
    for (const auto& r : recs) {
        INFO("r=" << r.tokens.at(0) << " R=" << r.tokens.at(1) << " d=" << r.tokens.at(2));
        const double a = circ::overlap_area(r.num(0), r.num(1), r.num(2));
        CHECK_THAT(a, WithinAbs(r.num(3), kMagTol));
        res.add(a, r.num(3));
    }
    CHECK(res.v <= kMagTol);
}

TEST_CASE("_obscuration parity: obsc records", "[parity]") {
    // Measured: bit-identical at all 14 records (total, annular on-axis,
    // no overlap, denom <= 0, the real 2017 / 2023 radii).
    const auto recs = records("obsc");
    REQUIRE(recs.size() >= 14);
    MaxAbs res;
    for (const auto& r : recs) {
        INFO("L1p=" << r.tokens.at(0) << " L2p=" << r.tokens.at(1) << " m=" << r.tokens.at(2));
        const double o = circ::obscuration(r.num(0), r.num(1), r.num(2));
        CHECK_THAT(o, WithinAbs(r.num(3), kMagTol));
        res.add(o, r.num(3));
    }
    CHECK(res.v <= kMagTol);
}

TEST_CASE("evaluate_direct's mu unwrap parity: unw records", "[parity]") {
    // np.degrees(np.unwrap(np.radians(mu))) is exact (numerics::unwrap is
    // NumPy's float branch expression by expression): bit-identical at all
    // 9 records, including the no-wrap ones whose round trip is not identity.
    const auto recs = records("unw");
    REQUIRE(recs.size() >= 9);
    for (const auto& r : recs) {
        const auto n = static_cast<std::size_t>(r.num(0));
        INFO("n=" << n << " v0=" << r.tokens.at(1));
        eclipse::Elements e;
        for (std::size_t i = 0; i < n; ++i) e.mu.push_back(r.num(1 + i));
        eclipse::unwrap_mu_deg(e);
        REQUIRE(e.mu.size() == n);
        for (std::size_t i = 0; i < n; ++i) {
            INFO("i=" << i);
            CHECK(e.mu[i] == r.num(1 + n + i));
        }
    }
}

TEST_CASE("_sun_altaz and the grid altitude parity: altaz records", "[parity]") {
    // Both altitude expressions (one radians of the difference vs two) and
    // the azimuth, to the 1e-11 deg sub-solar gate. Measured over the 16
    // records (poles, the antimeridian pair, zenith / nadir): alt and az
    // bit-identical; alt_grid bit-identical except 1.4e-14 deg at (0, 179.99)
    // vs (-95.6, 7.6), where NumPy's AVX-512 arccos returns the neighbouring
    // double of glibc's acos (verified directly: np.arccos 1.474092161460371
    // vs math.acos 1.4740921614603708 at the same cos_c) — the 1-ulp
    // libm-vs-NumPy noise of roadmap §5, not a tolerance to widen.
    const auto recs = records("altaz");
    REQUIRE(recs.size() >= 16);
    for (const auto& r : recs) {
        INFO("lat=" << r.tokens.at(0) << " lon=" << r.tokens.at(1) << " ss_lon=" << r.tokens.at(2)
                    << " ss_lat=" << r.tokens.at(3));
        const circ::AltAz a = circ::sun_altaz(r.num(0), r.num(1), r.num(2), r.num(3));
        CHECK_THAT(a.alt_deg, WithinAbs(r.num(4), kDegTol));
        CHECK_THAT(a.az_deg, WithinAbs(r.num(5), kDegTol));
        const double alt_grid = circ::sun_alt_grid(r.num(0), r.num(1), r.num(2), r.num(3));
        CHECK_THAT(alt_grid, WithinAbs(r.num(6), kDegTol));
    }
}

TEST_CASE("_roots parity: roots records", "[parity]") {
    // Linear-interpolated crossings, exact zeros, NaN skipped, consecutive
    // zeros: bit-identical crossings, same direction and bracket index at all
    // 7 records.
    const auto recs = records("roots");
    REQUIRE(recs.size() >= 7);
    for (const auto& r : recs) {
        const auto n = static_cast<std::size_t>(r.num(0));
        std::vector<double> t(n), f(n);
        for (std::size_t i = 0; i < n; ++i) {
            t[i] = r.num(1 + i);
            f[i] = r.num(1 + n + i);
        }
        INFO("t0=" << r.tokens.at(1) << " f0=" << r.tokens.at(1 + n));
        const auto k = static_cast<std::size_t>(r.num(1 + 2 * n));
        const std::vector<circ::Root> found = circ::roots(t, f);
        REQUIRE(found.size() == k);
        for (std::size_t j = 0; j < k; ++j) {
            INFO("root " << j);
            const std::size_t base = 2 + 2 * n + 3 * j;
            CHECK(found[j].t == r.num(base));
            CHECK(found[j].rising == (r.num(base + 1) != 0.0));
            CHECK(found[j].i == static_cast<std::size_t>(r.num(base + 2)));
        }
    }
    // Equal-length spans only.
    const double t3[] = {0.0, 1.0, 2.0}, f2[] = {1.0, -1.0};
    CHECK_THROWS_AS(circ::roots(t3, f2), std::invalid_argument);
}

// ---------------------------------------------------------------- unit tests

TEST_CASE("series: the two overloads agree bit for bit and validate lengths") {
    const Synthetic syn;
    const circ::Model model = syn.model(2.5);
    const std::vector<double> t = eclipse::numerics::arange(-1.0, 1.0 + 1e-9, 0.25);
    const eclipse::Elements e = model.elements_at(t);
    const double lats[] = {0.0, 30.0, -45.0}, lons[] = {0.0, 10.0, -100.0};
    const circ::Series many = circ::series(e, lats, lons);
    REQUIRE(many.m.size() == 3 * t.size());
    for (std::size_t p = 0; p < 3; ++p) {
        const circ::Series one = circ::series(e, lats[p], lons[p]);
        REQUIRE(one.m.size() == t.size());
        for (std::size_t i = 0; i < t.size(); ++i) {
            const std::size_t k = p * t.size() + i;
            CHECK(many.m[k] == one.m[i]);
            CHECK(many.L1p[k] == one.L1p[i]);
            CHECK(many.L2p[k] == one.L2p[i]);
            CHECK(many.mag[k] == one.mag[i]);
        }
    }
    // On the axis at t = 0: m = 0, L1' = l1 - tan f, L2' = l2 - tan f.
    const circ::Series axis = circ::series(e, 0.0, 0.0);
    const std::size_t i0 = 4;  // t = 0
    CHECK(t[i0] == 0.0);
    CHECK_THAT(axis.m[i0], WithinAbs(0.0, 1e-15));
    CHECK(axis.L1p[i0] == syn.l1 - syn.tan_f);
    CHECK(axis.L2p[i0] == syn.l2 - syn.tan_f);
    const double lat2[] = {0.0, 1.0}, lon1[] = {0.0};
    CHECK_THROWS_AS(circ::series(e, lat2, lon1), std::invalid_argument);
    eclipse::Elements bad = e;
    bad.l2.pop_back();
    CHECK_THROWS_AS(circ::series(bad, 0.0, 0.0), std::invalid_argument);
}

TEST_CASE("refine_maximum: parabola vertex, ends and non-maximum fits") {
    const double t[] = {0.0, 1.0, 2.0};
    const double sym[] = {0.0, 1.0, 0.0};
    CHECK(circ::refine_maximum(t, sym, 1) == 1.0);  // y0 == y2: no shift
    const double skew[] = {0.5, 1.0, 0.0};
    // denom = 0.5 - 2 + 0 = -1.5; t + ((0.5 * step) * 0.5) / -1.5, left to right.
    const double expected = 1.0 + (((0.5 * circ::GRID_STEP_HOURS) * 0.5) / -1.5);
    CHECK(circ::refine_maximum(t, skew, 1) == expected);
    CHECK(circ::refine_maximum(t, skew, 0) == 0.0);  // an end: the grid instant
    CHECK(circ::refine_maximum(t, skew, 2) == 2.0);
    const double convex[] = {1.0, 0.0, 1.0};
    CHECK(circ::refine_maximum(t, convex, 1) == 1.0);  // denom > 0: not a maximum
    const double flat[] = {1.0, 1.0, 1.0};
    CHECK(circ::refine_maximum(t, flat, 1) == 1.0);  // denom == 0
    const double two[] = {0.0, 1.0};
    CHECK_THROWS_AS(circ::refine_maximum(t, two, 1), std::invalid_argument);
    CHECK_THROWS_AS(circ::refine_maximum(t, sym, 3), std::invalid_argument);
}

TEST_CASE("local_circumstances on the synthetic crossing: total, day side") {
    const Synthetic syn;  // l2 < 0: total
    const circ::Model model = syn.model(2.5);
    const circ::LocalRaw r = circ::local_circumstances(model, 0.0, 0.0);
    REQUIRE(r.geometric);
    CHECK(r.central);
    CHECK(r.eclipse);
    CHECK(r.L2_x < 0.0);
    // Ordering C1 < C2 < t_max < C3 < C4.
    CHECK(r.c1 < r.c2);
    CHECK(r.c2 < r.t_max);
    CHECK(r.t_max < r.c3);
    CHECK(r.c3 < r.c4);
    // Symmetry about the crossing and the closed-form contacts (30 bisection
    // steps on a 30-s bracket resolve ~1e-11 h).
    CHECK_THAT(r.c1 + r.c4, WithinAbs(0.0, kHourTol));
    CHECK_THAT(r.c2 + r.c3, WithinAbs(0.0, kHourTol));
    CHECK_THAT(r.t_max, WithinAbs(0.0, kHourTol));
    CHECK_THAT(r.c4, WithinAbs(syn.crossing(syn.l1, false), kHourTol));
    CHECK_THAT(r.c3, WithinAbs(syn.crossing(syn.l2, true), kHourTol));
    // Magnitude at maximum is the diameter ratio [Espenak]; obscuration 1.
    const double L1 = syn.l1 - syn.tan_f, L2 = syn.l2 - syn.tan_f;
    CHECK_THAT(r.magnitude, WithinAbs((L1 - L2) / (L1 + L2), kMagTol));
    CHECK(r.magnitude > 1.0);
    CHECK(r.obscuration == 1.0);
    CHECK_THAT(r.L2_x, WithinAbs(L2, kMagTol));
    // Sun at the zenith for every event; nothing below the horizon.
    for (int k = 0; k < 5; ++k) {
        INFO("event " << k);
        CHECK(r.alt_deg[k] == 90.0);
        CHECK_FALSE(r.below[k]);
    }
    // Call counts: 1 grid (no widening) + 1 f(t_lo) + 30 bisection midpoints
    // + 1 at t_max; sub-solar once (the five event times in one call).
    CHECK(syn.element_calls == 33);
    CHECK(syn.sub_solar_calls == 1);
    REQUIRE(syn.element_sizes.size() == 33);
    CHECK(syn.element_sizes[0] == 601);  // arange(-2.5, 2.5 + 1e-9, 1/120)
    for (std::size_t k = 1; k <= 31; ++k) CHECK(syn.element_sizes[k] == 4);  // [C1, C4, C2, C3]
    CHECK(syn.element_sizes[32] == 1);
}

TEST_CASE("local_circumstances on the synthetic crossing: annular") {
    Synthetic syn;
    syn.l2 = 0.012;  // l2 > 0: annular
    const circ::Model model = syn.model(2.5);
    const circ::LocalRaw r = circ::local_circumstances(model, 0.0, 0.0);
    REQUIRE(r.geometric);
    CHECK(r.central);
    CHECK(r.eclipse);
    CHECK(r.L2_x > 0.0);
    CHECK(r.c1 < r.c2);
    CHECK(r.c2 < r.t_max);
    CHECK(r.t_max < r.c3);
    CHECK(r.c3 < r.c4);
    CHECK_THAT(r.c2 + r.c3, WithinAbs(0.0, kHourTol));
    CHECK_THAT(r.c3, WithinAbs(syn.crossing(syn.l2, true), kHourTol));
    const double L1 = syn.l1 - syn.tan_f, L2 = syn.l2 - syn.tan_f;
    CHECK_THAT(r.magnitude, WithinAbs((L1 - L2) / (L1 + L2), kMagTol));
    CHECK(r.magnitude < 1.0);
    // Annular on the axis: the covered fraction of the Sun's area is q^2.
    const double q = (L1 - L2) / (L1 + L2);
    CHECK_THAT(r.obscuration, WithinAbs(q * q, kMagTol));
}

TEST_CASE("local_circumstances: partial observer, night side, axis miss") {
    SECTION("partial: off the axis but inside the penumbra") {
        const Synthetic syn;
        const circ::LocalRaw r = circ::local_circumstances(syn.model(2.5), 20.0, 0.0);
        REQUIRE(r.geometric);
        CHECK_FALSE(r.central);
        CHECK(r.eclipse);
        CHECK(std::isnan(r.c2));
        CHECK(std::isnan(r.c3));
        CHECK(r.c1 < r.t_max);
        CHECK(r.t_max < r.c4);
        CHECK(r.magnitude > 0.0);
        CHECK(r.magnitude < 1.0);
        CHECK(r.obscuration > 0.0);
        CHECK(r.obscuration < 1.0);
        for (int k = 3; k < 5; ++k) {
            CHECK(std::isnan(r.alt_deg[k]));
            CHECK(std::isnan(r.az_deg[k]));
            CHECK_FALSE(r.below[k]);
        }
        // Two brackets only: [C1, C4].
        for (std::size_t k = 1; k <= 31; ++k) CHECK(syn.element_sizes[k] == 2);
    }
    SECTION("night: the geometry holds but the Sun is down at every event") {
        Synthetic syn;
        syn.ss_lon = 180.0;  // antipodal sub-solar point
        const circ::LocalRaw r = circ::local_circumstances(syn.model(2.5), 0.0, 0.0);
        REQUIRE(r.geometric);
        CHECK(r.central);
        CHECK_FALSE(r.eclipse);
        for (int k = 0; k < 5; ++k) {
            INFO("event " << k);
            CHECK(r.alt_deg[k] == -90.0);
            CHECK(r.below[k]);
        }
        // The Python compares the altitude to h0 = -0.8333 deg [Meeus98] ch. 15.
        CHECK(circ::HORIZON_ALT_DEG == -0.8333);
    }
    SECTION("axis miss: the observer never enters the penumbra") {
        Synthetic syn;
        syn.y = 3.0;
        const circ::LocalRaw r = circ::local_circumstances(syn.model(2.5), 0.0, 0.0);
        CHECK_FALSE(r.geometric);
        CHECK_FALSE(r.central);
        CHECK_FALSE(r.eclipse);
        CHECK(std::isnan(r.c1));
        CHECK(std::isnan(r.t_max));
        CHECK(std::isnan(r.magnitude));
        for (int k = 0; k < 5; ++k) {
            CHECK(std::isnan(r.alt_deg[k]));
            CHECK_FALSE(r.below[k]);
        }
        CHECK(syn.sub_solar_calls == 0);
    }
}

TEST_CASE("bracketed_series widens from a small half-window until the edges are clear") {
    // C1/C4 are at +/-1.09 h: 0.5 h and 0.75 h are inside the penumbra at both
    // ends, 1.125 h is clear — three grid evaluations, two widening steps,
    // each on arange(-hw, hw + 1e-9, 1/120).
    const Synthetic syn;
    const circ::Model model = syn.model(0.5);
    const circ::Bracketed b = circ::bracketed_series(model, 0.0, 0.0);
    CHECK(syn.element_calls == 3);
    REQUIRE(syn.element_sizes.size() == 3);
    CHECK(syn.element_sizes[0] == 121);
    CHECK(syn.element_sizes[1] == 181);
    CHECK(syn.element_sizes[2] == 271);
    CHECK(b.t.size() == 271);
    CHECK(b.t[0] == -1.125);
    CHECK(b.s.m.size() == 271);
    CHECK(b.s.m[0] - b.s.L1p[0] > 0.0);
    CHECK(b.s.m[270] - b.s.L1p[270] > 0.0);
    // The full local circumstances from the same seed: 3 grid + 1 + 30 + 1,
    // and the same contacts as from the 2.5 h seed to the bisection precision.
    const circ::LocalRaw narrow = circ::local_circumstances(model, 0.0, 0.0);
    CHECK(syn.element_calls == 3 + 35);
    const Synthetic wide;
    const circ::LocalRaw ref = circ::local_circumstances(wide.model(2.5), 0.0, 0.0);
    CHECK_THAT(narrow.c1, WithinAbs(ref.c1, kHourTol));
    CHECK_THAT(narrow.c4, WithinAbs(ref.c4, kHourTol));
    CHECK_THAT(narrow.c2, WithinAbs(ref.c2, kHourTol));
    CHECK_THAT(narrow.c3, WithinAbs(ref.c3, kHourTol));
    CHECK_THAT(narrow.magnitude, WithinAbs(ref.magnitude, kMagTol));
}

TEST_CASE("bracketed_series stops at the 6 h cap; one-signed partial roots throw") {
    // An elements_at whose m - L1' is -0.5 at the first grid point, 1 at odd
    // indices and exactly 0 at even ones (tan f = 0, so L1' = l1 = 0.5 and m
    // = |x|): every sign change is rising (product < 0 at i = 0, exact zeros
    // followed by a positive sample after), the peak magnitude is positive,
    // and the first sample is never clear, so the window widens 2.5 -> 3.75
    // -> 5.625 -> 6 h (four evaluations) and the C1 selection is empty — the
    // Python's ValueError from min() of an empty sequence.
    int calls = 0;
    std::vector<std::size_t> sizes;
    circ::Model model;
    model.elements_at = [&](std::span<const double> t) {
        ++calls;
        sizes.push_back(t.size());
        eclipse::Elements e;
        for (std::size_t i = 0; i < t.size(); ++i) {
            e.x.push_back(i == 0 ? 0.0 : (i % 2 == 1 ? 1.5 : 0.5));
            e.y.push_back(0.0);
            e.z.push_back(0.0);
            e.d.push_back(0.0);
            e.mu.push_back(0.0);
            e.l1.push_back(0.5);
            e.l2.push_back(-0.01);
            e.tan_f1.push_back(0.0);
            e.tan_f2.push_back(0.0);
        }
        return e;
    };
    model.sub_solar_at = [](std::span<const double> t) {
        eclipse::ephem::SubSolar ss;
        ss.lon_deg.assign(t.size(), 0.0);
        ss.lat_deg.assign(t.size(), 0.0);
        return ss;
    };
    model.half_window_hours = 2.5;
    CHECK_THROWS_AS(circ::local_circumstances(model, 0.0, 0.0), std::invalid_argument);
    CHECK(calls == 4);
    REQUIRE(sizes.size() == 4);
    CHECK(sizes[0] == 601);   // 2.5 h
    CHECK(sizes[1] == 901);   // 3.75 h
    CHECK(sizes[2] == 1351);  // 5.625 h
    CHECK(sizes[3] == 1441);  // 6 h (the cap; min(5.625 * 1.5, 6))
}

TEST_CASE("circumstances_grid on the synthetic model: thread counts agree bit for bit") {
    const Synthetic syn;
    const circ::Model model = syn.model(2.5);
    // On the axis, off it, the pole (outside the penumbra: eta = rho1 > L1'
    // throughout, while the axis sweeps x across the whole equator), the
    // night-side antipode (inside the continued cone, Sun at -90), and more.
    const double lats[] = {0.0, 10.0, -20.0, 89.9, 0.0, 0.0, -89.9, 45.0};
    const double lons[] = {0.0, 0.0, 5.0, 0.0, 180.0, 90.0, 0.0, -30.0};
    const circ::GridResult serial = circ::circumstances_grid(model, lats, lons, 2.0, 1);
    const circ::GridResult par = circ::circumstances_grid(model, lats, lons, 2.0, 0);
    const circ::GridResult three = circ::circumstances_grid(model, lats, lons, 2.0, 3);
    CHECK(serial.magnitude == par.magnitude);
    CHECK(serial.obscuration == par.obscuration);
    CHECK(serial.t_max_hours == par.t_max_hours);
    CHECK(serial.sun_alt == par.sun_alt);
    CHECK(serial.visible == par.visible);
    CHECK(serial.central == par.central);
    CHECK(serial.magnitude == three.magnitude);
    CHECK(serial.obscuration == three.obscuration);
    CHECK(serial.t_max_hours == three.t_max_hours);
    CHECK(serial.sun_alt == three.sun_alt);
    CHECK(serial.visible == three.visible);
    CHECK(serial.central == three.central);
    // Elements and sub-solar points evaluated once per grid, whatever the
    // thread count.
    CHECK(syn.element_calls == 3);
    CHECK(syn.sub_solar_calls == 3);
    REQUIRE(serial.magnitude.size() == 8);
    // On the axis: central, visible, obscuration 1, at the grid instant nearest t = 0.
    CHECK(serial.central[0] == 1);
    CHECK(serial.visible[0] == 1);
    CHECK(serial.obscuration[0] == 1.0);
    CHECK_THAT(serial.t_max_hours[0], WithinAbs(0.0, 1e-13));
    CHECK(serial.sun_alt[0] == 90.0);
    // Outside the penumbra: zeros, but t_max and sun_alt still reported.
    CHECK(serial.magnitude[3] == 0.0);
    CHECK(serial.obscuration[3] == 0.0);
    CHECK(serial.visible[3] == 0);
    CHECK(serial.central[3] == 0);
    // The antipode: inside the continued cone (magnitude > 0) but the Sun is
    // at -90 deg, so not visible.
    CHECK(serial.magnitude[4] > 0.0);
    CHECK(serial.sun_alt[4] == -90.0);
    CHECK(serial.visible[4] == 0);
    const double lat1[] = {0.0};
    CHECK_THROWS_AS(circ::circumstances_grid(model, lat1, lons), std::invalid_argument);
}

TEST_CASE("circumstances_grid agrees with local_circumstances at the same points") {
    // The grid reports the maximum at its 2-minute instant; the scalar path
    // refines it. On the axis the maximum is the cusp at t = 0, which both
    // grids contain to ~1e-14 h, so the magnitudes agree to the 1e-12 gate.
    // Off the axis the sampled maximum is up to a minute from the refined one
    // and the magnitude curve is flat there (a few 1e-5): same flags, t_max
    // within a step, magnitude within 1e-3 — the same relation the oracle's
    // ``local`` and ``grid`` fixture records show at the partial sites.
    const Synthetic syn;
    const circ::Model model = syn.model(2.5);
    const double lats[] = {0.0, 10.0, -20.0, 0.0, 60.0};
    const double lons[] = {0.0, 0.0, 5.0, 180.0, 0.0};
    const circ::GridResult g = circ::circumstances_grid(model, lats, lons, 2.0, 1);
    for (std::size_t p = 0; p < 5; ++p) {
        INFO("observer " << p);
        const circ::LocalRaw r = circ::local_circumstances(model, lats[p], lons[p]);
        const bool has = g.magnitude[p] > 0.0;
        CHECK(has == r.geometric);
        if (!has) continue;
        CHECK((g.central[p] != 0) == r.central);
        CHECK((g.visible[p] != 0) == r.eclipse);
        CHECK_THAT(g.t_max_hours[p], WithinAbs(r.t_max, 2.0 / 60.0));
        if (lats[p] == 0.0) {
            CHECK_THAT(g.magnitude[p], WithinAbs(r.magnitude, kMagTol));
            CHECK_THAT(g.obscuration[p], WithinAbs(r.obscuration, kMagTol));
            CHECK_THAT(g.t_max_hours[p], WithinAbs(r.t_max, 1e-13));
        } else {
            CHECK_THAT(g.magnitude[p], WithinAbs(r.magnitude, 1e-3));
            CHECK(g.magnitude[p] <= r.magnitude + kMagTol);  // the refined maximum is the maximum
        }
        // The grid's altitude at its own instant vs the scalar's at maximum.
        CHECK_THAT(g.sun_alt[p], WithinAbs(r.alt_deg[1], kDegTol));
    }
}

// ------------------------------------------------------- kernel-backed parity

namespace {

// ``local`` records: label frame et0 hw lat lon then the _LocalRaw fields.
void check_local_fixture(const std::string& name) {
    const auto recs = fixtures::read(name);
    if (!setup(recs)) return;
    int checked = 0;
    for (const auto& r : recs) {
        if (r.kind != "local") continue;
        INFO(name << " hw=" << r.tokens.at(3) << " lat=" << r.tokens.at(4) << " lon=" << r.tokens.at(5));
        REQUIRE(r.tokens.size() == 32);
        const eclipse::Frame frame = eclipse::frame_from_string(r.tokens.at(1));
        const circ::LocalRaw a = circ::local_circumstances(r.num(2), frame, r.num(3), r.num(4), r.num(5));
        CHECK(a.geometric == (r.num(6) != 0.0));
        CHECK(a.central == (r.num(7) != 0.0));
        const double* times[] = {&a.c1, &a.c4, &a.c2, &a.c3, &a.t_max};
        for (std::size_t k = 0; k < 5; ++k) {
            INFO("time field " << k);
            check_nan_or_within(*times[k], r.num(8 + k), kHourTol);
        }
        check_nan_or_within(a.magnitude, r.num(13), kMagTol);
        check_nan_or_within(a.obscuration, r.num(14), kMagTol);
        check_nan_or_within(a.L2_x, r.num(15), kMagTol);
        for (std::size_t k = 0; k < 5; ++k) {
            INFO("event " << k);
            check_nan_or_within(a.alt_deg[k], r.num(16 + k), kDegTol);
            check_nan_or_within(a.az_deg[k], r.num(21 + k), kDegTol);
            CHECK(a.below[k] == (r.num(26 + k) != 0.0));
        }
        CHECK(a.eclipse == (r.num(31) != 0.0));
        ++checked;
    }
    CHECK(checked == 10);
}

// ``grid`` records: label frame et0 hw step lat lon magnitude obscuration
// t_max_hours sun_alt visible central.
void check_grid_fixture(const std::string& name) {
    const auto recs = fixtures::read(name);
    if (!setup(recs)) return;
    int checked = 0;
    for (const auto& r : recs) {
        if (r.kind != "grid") continue;
        INFO(name << " lat=" << r.tokens.at(5) << " lon=" << r.tokens.at(6));
        REQUIRE(r.tokens.size() == 13);
        const eclipse::Frame frame = eclipse::frame_from_string(r.tokens.at(1));
        const double lat[] = {r.num(5)}, lon[] = {r.num(6)};
        const circ::GridResult g =
            circ::circumstances_grid(r.num(2), frame, r.num(3), lat, lon, r.num(4));
        REQUIRE(g.magnitude.size() == 1);
        CHECK_THAT(g.magnitude[0], WithinAbs(r.num(7), kMagTol));
        CHECK_THAT(g.obscuration[0], WithinAbs(r.num(8), kMagTol));
        CHECK_THAT(g.t_max_hours[0], WithinAbs(r.num(9), kMagTol));
        CHECK_THAT(g.sun_alt[0], WithinAbs(r.num(10), kMagTol));
        CHECK(g.visible[0] == (r.num(11) != 0.0 ? 1 : 0));
        CHECK(g.central[0] == (r.num(12) != 0.0 ? 1 : 0));
        ++checked;
    }
    CHECK(checked == 9);
}

}  // namespace

TEST_CASE("local_circumstances parity: 2017-08-21 local records", "[parity][kernels]") {
    check_local_fixture("se2017aug21.txt");
}

TEST_CASE("local_circumstances parity: 2023-10-14 local records", "[parity][kernels]") {
    check_local_fixture("se2023oct14.txt");
}

TEST_CASE("local_circumstances parity: 2024-04-08 local records", "[parity][kernels]") {
    check_local_fixture("se2024apr08.txt");
}

TEST_CASE("circumstances_grid parity: 2017-08-21 grid records", "[parity][kernels]") {
    check_grid_fixture("se2017aug21.txt");
}

TEST_CASE("circumstances_grid parity: 2023-10-14 grid records", "[parity][kernels]") {
    check_grid_fixture("se2023oct14.txt");
}

TEST_CASE("circumstances_grid parity: 2024-04-08 grid records", "[parity][kernels]") {
    check_grid_fixture("se2024apr08.txt");
}

// ----------------------------------------------------------------- benchmark

TEST_CASE("0.5-degree global circumstances_grid benchmark", "[!benchmark]") {
    // Hidden (run with ``eclipse_tests "[!benchmark]"``): the /map default
    // grid — 361 x 720 observers, half_window_hours 3, 2-minute step (181
    // instants) — serial and at the OpenMP default thread count, on the
    // synthetic model (the pure-math observer loop alone) and, with kernels,
    // on the 2024-04-08 ephemeris model. Roadmap §5 target: < 0.5 s on 4 cores.
    std::vector<double> lats, lons;
    for (int i = 0; i <= 360; ++i)
        for (int j = 0; j < 720; ++j) {
            lats.push_back(-90.0 + 0.5 * i);
            lons.push_back(-180.0 + 0.5 * j);
        }
    const auto run = [&](const circ::Model& model, int threads) {
        const auto t0 = std::chrono::steady_clock::now();
        const circ::GridResult g = circ::circumstances_grid(model, lats, lons, 2.0, threads);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "  threads=" << threads << ": " << s << " s (" << g.magnitude.size()
                  << " observers)\n";
        return g;
    };
    const Synthetic syn;
    std::cout << "synthetic model:\n";
    const circ::GridResult a = run(syn.model(3.0), 1);
    const circ::GridResult b = run(syn.model(3.0), 0);
    CHECK(a.magnitude == b.magnitude);
    const auto recs = fixtures::read("se2024apr08.txt");
    if (!setup(recs)) return;
    for (const auto& r : recs) {
        if (r.kind != "grid") continue;
        const circ::Model model =
            circ::model_from_ephem(r.num(2), eclipse::frame_from_string(r.tokens.at(1)), r.num(3));
        std::cout << "2024-04-08 ephemeris model:\n";
        const circ::GridResult c = run(model, 1);
        const circ::GridResult d = run(model, 0);
        CHECK(c.magnitude == d.magnitude);
        CHECK(c.sun_alt == d.sun_alt);
        break;
    }
}

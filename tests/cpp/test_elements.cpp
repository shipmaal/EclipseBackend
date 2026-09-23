// Phase 1 parity against the Python oracle (roadmap §4) from the fixtures
// tools/dump_oracle.py wrote: time scales, Earth rotation, Besselian elements
// and sub-solar points at the reference eclipses. Tolerances are the roadmap
// gates; the measured residuals (NumPy SIMD atan2/hypot/tan/asin vs libm,
// ~1 ulp) are ~5e-14 in x and ~1e-13 deg in mu.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "eclipse/constants.hpp"
#include "eclipse/eop.hpp"
#include "eclipse/ephem.hpp"
#include "fixtures.hpp"

using Catch::Matchers::WithinAbs;
namespace ephem = eclipse::ephem;

namespace {

// Roadmap §4 parity gates.
constexpr double kXyTol = 1e-13;       // x, y, z, l1, l2 [Earth radii]; tan_f
constexpr double kAngTol = 1e-11;      // d, mu, sub-solar lon/lat [deg]
constexpr double kTimeTol = 1e-9;      // seconds
constexpr double kDay = eclipse::constants::SECONDS_PER_DAY;

// Load the EOP subset and kernels once; SKIP (not fail) when the environment
// lacks the pinned kernel set.
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

void check_fixture(const std::string& name) {
    const auto recs = fixtures::read(name);
    if (!setup(recs)) return;
    // ITRF93 reads the binary Earth PCK, which NAIF regenerates daily under one
    // name: its rows replay only against the byte-identical file (``pck``
    // header record); tests/test_native.py checks ITRF93 parity live.
    const bool pck_pinned = fixtures::pinned_pck_matches(recs);
    bool warned_pck = false;

    for (const auto& r : recs) {
        if (r.kind != "case") continue;
        const std::string label = r.tokens.at(0), frame = r.tokens.at(1);
        if (frame == "ITRF93" && !pck_pinned) {
            if (!warned_pck)
                WARN(name << ": ITRF93 rows skipped; the binary Earth PCK on disk is not the "
                             "one the fixtures were dumped with (regenerate them to replay)");
            warned_pck = true;
            continue;
        }
        INFO(name << " " << label << " " << frame);
        const double et = r.num(2);
        const double e_[] = {et};
        eclipse::Elements el;
        ephem::SubSolar ss;
        try {
            el = ephem::besselian_instants(e_, eclipse::frame_from_string(frame));
            ss = ephem::sub_solar_points(e_, eclipse::frame_from_string(frame));
        } catch (const eclipse::spice_error& e) {
            // The mirror kernel set has no binary Earth PCK (ITRF93) and DE432s
            // does not cover 1919: the oracle dump has the same gap.
            WARN(label << " " << frame << ": " << e.short_message());
            continue;
        }
        CHECK_THAT(el.x[0], WithinAbs(r.num(3), kXyTol));
        CHECK_THAT(el.y[0], WithinAbs(r.num(4), kXyTol));
        CHECK_THAT(el.z[0], WithinAbs(r.num(5), kXyTol));
        CHECK_THAT(el.d[0], WithinAbs(r.num(6), kAngTol));
        CHECK_THAT(el.mu[0], WithinAbs(r.num(7), kAngTol));
        CHECK_THAT(el.l1[0], WithinAbs(r.num(8), kXyTol));
        CHECK_THAT(el.l2[0], WithinAbs(r.num(9), kXyTol));
        CHECK_THAT(el.tan_f1[0], WithinAbs(r.num(10), kXyTol));
        CHECK_THAT(el.tan_f2[0], WithinAbs(r.num(11), kXyTol));

        const ephem::EarthRotation ert = ephem::earth_rotation_times(e_);
        CHECK_THAT(ert.tt2[0] * kDay, WithinAbs(r.num(12) * kDay, kTimeTol));
        CHECK_THAT(ert.ut1[0] * kDay, WithinAbs(r.num(13) * kDay, kTimeTol));
        CHECK_THAT(ert.xp[0], WithinAbs(r.num(14), 1e-18));
        CHECK_THAT(ert.yp[0], WithinAbs(r.num(15), 1e-18));

        CHECK_THAT(ss.lon_deg[0], WithinAbs(r.num(16), kAngTol));
        CHECK_THAT(ss.lat_deg[0], WithinAbs(r.num(17), kAngTol));
    }

    // ``axis <et> <rho> <z>`` (phase 4): app.ephemeris.axis_separation at the
    // same instants — the frame-free scan objective; gate 1e-13 (the x/y gate;
    // measured 0 to ~5e-14, the phase-1 hypot / atan2 ulp noise).
    int n_axis = 0;
    for (const auto& r : recs) {
        if (r.kind != "axis") continue;
        const double e_[] = {r.num(0)};
        INFO(name << " axis et=" << r.tokens.at(0));
        const ephem::AxisSeparation s = ephem::axis_separation(e_);
        REQUIRE(s.rho.size() == 1);
        CHECK_THAT(s.rho[0], WithinAbs(r.num(1), kXyTol));
        CHECK_THAT(s.z[0], WithinAbs(r.num(2), kXyTol));
        ++n_axis;
    }
    CHECK(n_axis == 13);
}

}  // namespace

TEST_CASE("Besselian elements parity: 2017-08-21 total", "[kernels][parity]") {
    check_fixture("se2017aug21.txt");
}
TEST_CASE("Besselian elements parity: 2023-10-14 annular", "[kernels][parity]") {
    check_fixture("se2023oct14.txt");
}
TEST_CASE("Besselian elements parity: 2024-04-08 total", "[kernels][parity]") {
    check_fixture("se2024apr08.txt");
}
TEST_CASE("Besselian elements parity: 1919-05-29 (pre-IERS, model delta-T)", "[kernels][parity]") {
    check_fixture("se1919may29.txt");
}

TEST_CASE("utc_to_et / et_to_utc follow the IERS-era rule", "[kernels][parity]") {
    const auto recs = fixtures::read("utc_cases.txt");
    if (!setup(recs)) return;
    for (const auto& r : recs) {
        if (r.kind != "utc") continue;
        INFO(r.tokens.at(0));
        const double et = ephem::utc_to_et(r.tokens.at(0));
        CHECK_THAT(et, WithinAbs(r.num(1), kTimeTol));
        CHECK(ephem::et_to_utc(et) == r.tokens.at(2));
    }
}

TEST_CASE("EOP interpolation reproduces np.interp semantics") {
    const double mjd[] = {100.0, 101.0, 103.0};
    const double xp[] = {0.1, 0.3, 0.7}, yp[] = {1.0, 1.0, 1.0}, dut1[] = {-0.5, 0.5, 0.5};
    eclipse::eop::set_table(mjd, xp, yp, dut1);
    using eclipse::constants::ARCSEC_TO_RAD;
    CHECK(eclipse::eop::interpolate(99.0).xp_rad == 0.1 * ARCSEC_TO_RAD);     // held left
    CHECK(eclipse::eop::interpolate(104.0).xp_rad == 0.7 * ARCSEC_TO_RAD);    // held right
    CHECK(eclipse::eop::interpolate(101.0).xp_rad == 0.3 * ARCSEC_TO_RAD);    // exact node
    CHECK(eclipse::eop::interpolate(102.0).xp_rad == 0.5 * ARCSEC_TO_RAD);    // midpoint
    CHECK(eclipse::eop::interpolate(100.5).dut1_s == 0.0);
    CHECK(eclipse::eop::in_iers_era(100.0));
    CHECK(eclipse::eop::in_iers_era(103.0));
    CHECK_FALSE(eclipse::eop::in_iers_era(103.5));
}

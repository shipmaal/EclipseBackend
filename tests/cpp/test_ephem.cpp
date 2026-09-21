// Phase 0 acceptance tests for the SPICE/ERFA layer (docs/CPP_ROADMAP.md §2, §5).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <string>

#include "eclipse/ephem.hpp"

namespace ephem = eclipse::ephem;

namespace {
const std::filesystem::path kMetakernel = std::filesystem::path(ECLIPSE_KERNEL_DIR) / "eclipse.tm";
bool kernels_present() { return std::filesystem::exists(kMetakernel); }
}  // namespace

TEST_CASE("vendored toolkit versions are the ones the Python oracle uses") {
    // spiceypy 6/7 wrap CSPICE N0067; pyerfa 2.0.1.x bundles liberfa 2.0.1.
    CHECK(ephem::toolkit_version() == "CSPICE_N0067");
    CHECK(ephem::erfa_version() == "2.0.1");
    CHECK(ephem::sofa_version() == "20231011");
}

TEST_CASE("a SPICE error is an exception, not a process abort") {
    // The acceptance test for "link SPICE to C++": with no SPK loaded spkpos_c
    // signals SPICE(NOLOADEDFILES); in CSPICE's default ABORT mode that would
    // exit the process. We must get a spice_error and a usable toolkit after.
    ephem::kclear();
    REQUIRE(ephem::kernel_count() == 0);

    REQUIRE_THROWS_AS(ephem::body_position("SUN", 0.0, "J2000", "NONE", "EARTH"),
                      eclipse::spice_error);
    try {
        ephem::body_position("SUN", 0.0, "J2000", "NONE", "EARTH");
        FAIL("expected spice_error");
    } catch (const eclipse::spice_error& e) {
        CHECK(e.short_message() == "SPICE(NOLOADEDFILES)");
        CHECK(std::string(e.what()).find("SPICE(NOLOADEDFILES)") != std::string::npos);
        CHECK(!e.traceback().empty());
    }

    // reset_c() ran: the error state is clear and the toolkit still works.
    CHECK(ephem::toolkit_version() == "CSPICE_N0067");
    CHECK(ephem::kernel_count() == 0);
}

TEST_CASE("furnish + time scales round-trip", "[kernels]") {
    if (!kernels_present()) SKIP("kernels not bootstrapped: " + kMetakernel.string());
    ephem::kclear();
    ephem::furnish(kMetakernel.string());
    CHECK(ephem::kernel_count() >= 3);

    // Leap-second kernel loaded: str2et/et2utc must invert exactly at 6 digits.
    const double et = ephem::str_to_et("2024-04-08T18:17:00.000000");
    CHECK(ephem::et_to_utc_iso(et, 6) == "2024-04-08T18:17:00.000000");

    // Sun from Earth, apparent, ~1 AU: sanity bounds, exact parity is asserted
    // against spiceypy in tests/test_native.py.
    const auto sun = ephem::body_position("SUN", et, "J2000", "LT+S", "EARTH");
    const double r = std::sqrt(sun.km[0] * sun.km[0] + sun.km[1] * sun.km[1] + sun.km[2] * sun.km[2]);
    CHECK_THAT(r, Catch::Matchers::WithinAbs(1.495978707e8, 3.0e6));  // ±2 % of 1 AU
    CHECK_THAT(sun.light_time_s, Catch::Matchers::WithinAbs(499.0, 10.0));
    ephem::kclear();
}

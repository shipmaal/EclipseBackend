// eclipse::besselian (the model layer: polynomial fit, direct elements, rates,
// central line) and eop::load_file. The fit and the parser need no kernels;
// the model functions are exercised end to end, against published values, by
// tests/test_besselian_integration.py.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "eclipse/besselian.hpp"
#include "eclipse/constants.hpp"
#include "eclipse/eop.hpp"

using Catch::Matchers::WithinAbs;
namespace bs = eclipse::besselian;

TEST_CASE("polyfit recovers an exact polynomial of its degree") {
    const std::vector<double> c = {0.3, -0.27, 6.4e-5, -4.7e-6};  // a 2024-like y(t)
    std::vector<double> t, y;
    for (double s = -2.0; s <= 2.0 + 1e-12; s += 1.0) {
        t.push_back(s);
        y.push_back(c[0] + s * (c[1] + s * (c[2] + s * c[3])));
    }
    const std::vector<double> f = bs::polyfit(t, y, 3);
    REQUIRE(f.size() == 4);
    for (std::size_t j = 0; j < 4; ++j) CHECK_THAT(f[j], WithinAbs(c[j], 1e-15));
}

TEST_CASE("polyfit is the least-squares line through noisy points") {
    // y = 1 + 2 t plus +-0.1 alternating: the symmetric residuals cancel.
    const std::vector<double> t = {-2, -1, 0, 1, 2};
    const std::vector<double> y = {-3.1, -0.9, 0.9, 3.1, 5.0};
    const std::vector<double> f = bs::polyfit(t, y, 1);
    // Closed form: slope = sum(t y) / sum(t^2), intercept = mean(y) (mean t = 0).
    double sty = 0, stt = 0, sy = 0;
    for (std::size_t i = 0; i < t.size(); ++i) sty += t[i] * y[i], stt += t[i] * t[i], sy += y[i];
    CHECK_THAT(f[0], WithinAbs(sy / 5.0, 1e-14));
    CHECK_THAT(f[1], WithinAbs(sty / stt, 1e-14));
}

TEST_CASE("polyfit rejects an underdetermined system") {
    const std::vector<double> t = {0.0, 1.0}, y = {1.0, 2.0};
    CHECK_THROWS_AS(bs::polyfit(t, y, 2), std::invalid_argument);
}

TEST_CASE("eop::load_file reads the Bulletin A columns of finals2000A.all") {
    // Three rows in the IERS fixed-width format (readme.finals2000A), the last
    // with no Bulletin A polar motion: the table ends before it.
    const auto path = std::filesystem::temp_directory_path() / "eclipse_finals_test.all";
    {
        std::ofstream f(path);
        f << "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  "
             "0.0000 0.1916  P    -0.766    0.199    -0.720    0.300   .143000   .137000   "
             ".8075000   -18.637    -3.667  \n"
          << "73 1 3 41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  "
             "3.5563 0.1916  P    -0.751    0.199    -0.701    0.300   .141000   .134000   "
             ".8044000   -18.636    -3.571  \n"
          << "73 1 4 41686.00                                                              \n";
    }
    REQUIRE(eclipse::eop::load_file(path.string()) == 2);
    CHECK(eclipse::eop::source() == path.string());
    const auto [lo, hi] = eclipse::eop::mjd_range();
    CHECK(lo == 41684.0);
    CHECK(hi == 41685.0);
    const double as = eclipse::constants::ARCSEC_TO_RAD;
    const auto a = eclipse::eop::interpolate(41684.0);
    CHECK(a.xp_rad == 0.120733 * as);
    CHECK(a.yp_rad == 0.136966 * as);
    CHECK(a.dut1_s == 0.8084178);
    const auto m = eclipse::eop::interpolate(41684.5);  // linear between the rows
    CHECK_THAT(m.dut1_s, WithinAbs(0.5 * (0.8084178 + 0.8056163), 1e-15));
    std::filesystem::remove(path);
    CHECK_THROWS_AS(eclipse::eop::load_file(path.string()), std::runtime_error);
}

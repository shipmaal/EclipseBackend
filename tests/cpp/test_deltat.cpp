// Delta-T model [Espenak] — same reference values as tests/test_deltat.py.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

#include "eclipse/deltat.hpp"
#include "fixtures.hpp"

using Catch::Matchers::WithinAbs;
namespace dt = eclipse::deltat;

TEST_CASE("delta-T matches Meeus Table 10.A observed values") {
    // [Meeus98] Table 10.A (rounded to 0.1 s there); the model is a fit.
    const std::pair<double, double> table[] = {{1900, -2.8}, {1920, 21.2}, {1950, 29.1},
                                               {1960, 33.1}, {1980, 50.5}, {2000, 63.8}};
    for (const auto& [year, expected] : table) {
        INFO("year " << year);
        CHECK_THAT(dt::delta_t_seconds(year), WithinAbs(expected, 0.3));
    }
}

TEST_CASE("delta-T segments are continuous to about a second") {
    for (const double edge : {-500.0, 500.0, 1600.0, 1700.0, 1800.0, 1860.0, 1900.0, 1920.0,
                              1941.0, 1961.0, 1986.0, 2005.0, 2050.0, 2150.0}) {
        INFO("edge " << edge);
        CHECK(std::abs(dt::delta_t_seconds(edge + 1e-6) - dt::delta_t_seconds(edge - 1e-6)) < 1.5);
    }
}

TEST_CASE("decimal year of a Julian date") {
    CHECK_THAT(dt::decimal_year_from_jd(2451545.0), WithinAbs(2000.0, 1e-12));
    CHECK_THAT(dt::decimal_year_from_jd(2451545.0 + 365.25), WithinAbs(2001.0, 1e-12));
}

TEST_CASE("vector form equals scalar form") {
    const double years[] = {-1000.0, 0.0, 1000.0, 1919.41, 2024.3, 2500.0};
    const auto v = dt::delta_t_seconds(years);
    for (size_t i = 0; i < std::size(years); ++i) CHECK(v[i] == dt::delta_t_seconds(years[i]));
}

TEST_CASE("delta-T predictions: linear between rows, none outside the table") {
    namespace dt = eclipse::deltat;
    const std::vector<double> mjd = {100.0, 200.0, 300.0}, v = {69.0, 70.0, 72.0},
                              err = {0.1, 0.2, 0.4};
    dt::set_predictions(mjd, v, err);
    CHECK(dt::predictions_mjd_range() == std::pair<double, double>{100.0, 300.0});
    const auto mid = dt::predicted(250.0);
    REQUIRE(mid.has_value());
    CHECK_THAT(mid->dt_s, WithinAbs(71.0, 1e-12));
    CHECK_THAT(mid->err_s, WithinAbs(0.3, 1e-12));
    CHECK(dt::predicted(100.0)->dt_s == 69.0);
    CHECK(dt::predicted(300.0)->dt_s == 72.0);
    CHECK_FALSE(dt::predicted(99.9).has_value());
    CHECK_FALSE(dt::predicted(300.1).has_value());
    const std::vector<double> bad = {100.0, 100.0};
    CHECK_THROWS_AS(dt::set_predictions(bad, std::vector<double>{1, 2}, std::vector<double>{1, 2}),
                    std::invalid_argument);
}

TEST_CASE("load_predictions reads the vendored USNO deltat.preds") {
    namespace dt = eclipse::deltat;
    const auto path = fixtures::kDir.parent_path().parent_path().parent_path() / "third_party" /
                      "usno" / "deltat.preds";
    REQUIRE(dt::load_predictions(path.string()) == 46);  // 2022.50 - 2033.75, quarterly
    CHECK(dt::predictions_source() == path.string());
    CHECK(dt::predictions_mjd_range() == std::pair<double, double>{59762.0, 63871.0});
    CHECK(dt::predicted(59762.0)->dt_s == 69.29);  // a row with the UT1-UTC column
    CHECK(dt::predicted(63871.0)->dt_s == 71.25);  // a row without it
    CHECK(dt::predicted(63871.0)->err_s == 1.0);
    CHECK_THROWS_AS(dt::load_predictions((fixtures::kDir / "no_such_file").string()),
                    std::runtime_error);
}

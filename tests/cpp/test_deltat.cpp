// Delta-T model [Espenak] — same reference values as tests/test_deltat.py.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "eclipse/deltat.hpp"

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

// Eclipse catalog (eclipse/catalog.hpp) — phase 4. Work package A pins the
// contract's constants to app/catalog.py's values and the row strings; work
// package B adds the parity tests against fixtures/catalog_cases.txt (the
// ``ismin`` / ``classify`` records, kernel-free) and
// fixtures/catalog_2019_2024.txt (the ``scan`` / ``event`` records, kernels
// required) at the roadmap §4 gate: identical rows, times to 1e-6 s.
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>

#include "eclipse/catalog.hpp"
#include "fixtures.hpp"

namespace cat = eclipse::catalog;

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

TEST_CASE("catalog fixtures are present for work package B") {
    // Kernel-free records dumped by tools/dump_oracle.py; B replays them.
    int n_ismin = 0, n_classify = 0;
    for (const auto& r : fixtures::read("catalog_cases.txt")) {
        if (r.kind == "ismin") ++n_ismin;
        if (r.kind == "classify") ++n_classify;
    }
    CHECK(n_ismin >= 8);
    CHECK(n_classify >= 10);
    int n_scan = 0, n_event = 0;
    for (const auto& r : fixtures::read("catalog_2019_2024.txt")) {
        if (r.kind == "scan") ++n_scan;
        if (r.kind == "event") ++n_event;
    }
    CHECK(n_scan == 2);
    CHECK(n_event == 2 * (13 + 2));
}

TEST_CASE("catalog math is not ported yet (phase 4 WP-B): the stubs throw") {
    const double rho[] = {2.0, 1.0, 2.0};
    CHECK_THROWS_AS(cat::local_minima(rho), std::logic_error);
    CHECK_THROWS_AS(cat::coarse_grid(0.0, 1.0), std::logic_error);
}

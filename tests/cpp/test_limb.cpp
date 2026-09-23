// Lunar limb profile (docs/LIMB_PROFILE.md PR 1): eclipse::limb and
// ephem::limb_axes against the Python oracle's limb_cases.txt, bit-identical
// (the oracle calls libm's atan2, app.limb._atan2), plus closed-form checks on
// a sphere that need no fixture.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "eclipse/constants.hpp"
#include "eclipse/eop.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/limb.hpp"
#include "fixtures.hpp"

namespace limb = eclipse::limb;
namespace ephem = eclipse::ephem;

namespace {

std::vector<double> nums(const fixtures::Record& r, size_t from = 0) {
    std::vector<double> v;
    for (size_t i = from; i < r.tokens.size(); ++i) v.push_back(r.num(i));
    return v;
}

template <class T>
std::vector<T> ints(const fixtures::Record& r) {
    std::vector<T> v;
    v.reserve(r.tokens.size());
    for (const auto& t : r.tokens) v.push_back(static_cast<T>(std::stol(t)));
    return v;
}

// Install the fixture's synthetic 1-ppd band into the core.
void install_fixture_band(const std::vector<fixtures::Record>& recs) {
    std::vector<std::int32_t> line, first, count, right, down;
    std::vector<std::int16_t> dn;
    std::vector<double> cos_lat, sin_lat, cos_lon, sin_lon, meta;
    for (const auto& r : recs) {
        if (r.kind == "lband") meta = nums(r);
        else if (r.kind == "lcoslat") cos_lat = nums(r);
        else if (r.kind == "lsinlat") sin_lat = nums(r);
        else if (r.kind == "lcoslon") cos_lon = nums(r);
        else if (r.kind == "lsinlon") sin_lon = nums(r);
        else if (r.kind == "ldn") dn = ints<std::int16_t>(r);
        else if (r.kind == "lright") right = ints<std::int32_t>(r);
        else if (r.kind == "ldown") down = ints<std::int32_t>(r);
        else if (r.kind == "lrun") {
            line.push_back(static_cast<std::int32_t>(std::stol(r.tokens.at(0))));
            first.push_back(static_cast<std::int32_t>(std::stol(r.tokens.at(1))));
            count.push_back(static_cast<std::int32_t>(std::stol(r.tokens.at(2))));
        }
    }
    REQUIRE(meta.size() == 3);
    limb::set_band(line, first, count, dn, right, down, cos_lat, sin_lat, cos_lon, sin_lon,
                   meta[0], meta[1], meta[2]);
}

}  // namespace

TEST_CASE("limb silhouette and delta_rho_at parity (synthetic band)") {
    const auto recs = fixtures::read("limb_cases.txt");
    install_fixture_band(recs);
    REQUIRE(limb::has_band());
    int n_sil = 0, n_dra = 0;
    for (const auto& r : recs) {
        if (r.kind == "lsil") {
            const int n_bins = static_cast<int>(r.num(0));
            const auto v = nums(r, 1);
            REQUIRE(v.size() == 9 + static_cast<size_t>(n_bins));
            std::array<double, 9> axes{};
            for (size_t i = 0; i < 9; ++i) axes[i] = v[i];
            const auto prof = limb::silhouette(axes, n_bins);
            for (int k = 0; k < n_bins; ++k) CHECK(prof[static_cast<size_t>(k)] == v[9 + static_cast<size_t>(k)]);
            ++n_sil;
        } else if (r.kind == "ldra") {
            const auto v = nums(r);
            const auto n = static_cast<size_t>(v.at(0));
            const std::vector<double> prof(v.begin() + 1, v.begin() + 1 + static_cast<std::ptrdiff_t>(n));
            const auto m = static_cast<size_t>(v.at(1 + n));
            const std::vector<double> psi(v.begin() + 2 + static_cast<std::ptrdiff_t>(n),
                                          v.begin() + 2 + static_cast<std::ptrdiff_t>(n + m));
            const auto out = limb::delta_rho_at(prof, psi);
            for (size_t i = 0; i < m; ++i) CHECK(out[i] == v.at(2 + n + m + i));
            ++n_dra;
        }
    }
    CHECK(n_sil == 8);
    CHECK(n_dra == 1);
    std::array<double, 9> pole_view{1, 0, 0, 0, 1, 0, 0, 0, 1};  // z^ = +z: off the band
    CHECK_THROWS_AS(limb::silhouette(pole_view, 90), std::invalid_argument);
}

TEST_CASE("limb silhouette of a sphere is flat (closed form)") {
    // A 4-ppd sphere, DN = 0 everywhere in the band, seen straight on: every
    // bin is R_REF within the grid's sampling sag R (1 - cos(half pixel)).
    const double ppd = 4.0;
    const size_t lines = 720, samples = 1440;
    std::vector<double> cos_lat(lines), sin_lat(lines), cos_lon(samples), sin_lon(samples);
    const double d2r = eclipse::constants::DEG_TO_RAD;
    for (size_t i = 0; i < lines; ++i) {
        const double lat = (359.5 - static_cast<double>(i)) / ppd * d2r;
        cos_lat[i] = std::cos(lat);
        sin_lat[i] = std::sin(lat);
    }
    for (size_t j = 0; j < samples; ++j) {
        const double lon = ((static_cast<double>(j) - 719.5) / ppd + 180.0) * d2r;
        cos_lon[j] = std::cos(lon);
        sin_lon[j] = std::sin(lon);
    }
    // Band rows and neighbours built the way app.limb._neighbours does.
    std::vector<std::int32_t> line, first, count;
    std::vector<std::vector<std::int64_t>> idx(lines, std::vector<std::int64_t>(samples, -1));
    std::int64_t n = 0;
    const double limit = std::sin(20.0 * d2r);
    for (size_t i = 0; i < lines; ++i) {
        size_t j = 0;
        while (j < samples) {
            if (std::abs(cos_lat[i] * cos_lon[j]) > limit) { ++j; continue; }
            const size_t j0 = j;
            while (j < samples && std::abs(cos_lat[i] * cos_lon[j]) <= limit) idx[i][j++] = n++;
            line.push_back(static_cast<std::int32_t>(i));
            first.push_back(static_cast<std::int32_t>(j0));
            count.push_back(static_cast<std::int32_t>(j - j0));
        }
    }
    std::vector<std::int32_t> right(static_cast<size_t>(n), -1), down(static_cast<size_t>(n), -1);
    for (size_t i = 0; i < lines; ++i)
        for (size_t j = 0; j < samples; ++j) {
            const auto p = idx[i][j];
            if (p < 0) continue;
            right[static_cast<size_t>(p)] = static_cast<std::int32_t>(idx[i][(j + 1) % samples]);
            if (i + 1 < lines) down[static_cast<size_t>(p)] = static_cast<std::int32_t>(idx[i + 1][j]);
        }
    const std::vector<std::int16_t> dn(static_cast<size_t>(n), 0);
    limb::set_band(line, first, count, dn, right, down, cos_lat, sin_lat, cos_lon, sin_lon,
                   1737.4, 0.0005, 20.0);
    const std::array<double, 9> axes{0, -1, 0, 0, 0, 1, -1, 0, 0};  // z^ = -x (Earth-facing)
    const auto prof = limb::silhouette(axes, 3600);
    const double sag = 1737.4 * (1.0 - std::cos(0.5 / ppd * d2r));
    for (const double v : prof) {
        CHECK(v <= 1e-9);
        CHECK(v >= -sag - 1e-9);
    }
}

TEST_CASE("limb_axes parity", "[kernels]") {
    const auto recs = fixtures::read("limb_cases.txt");
    if (!std::filesystem::exists(fixtures::kMetakernel)) SKIP("kernels not bootstrapped");
    if (!fixtures::pinned_spk_present(recs)) SKIP("fixtures pin de440s.bsp (NAIF set)");
    size_t n_lax = 0;
    for (const auto& r : recs) n_lax += r.kind == "lax";
    if (n_lax == 0) SKIP("fixtures dumped without MOON_ME (kernels.bootstrap --limb)");
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
    try {
        (void)ephem::limb_axes(std::vector<double>{0.0}, eclipse::Frame::TOD, "MOON_ME");
    } catch (const eclipse::spice_error&) {
        SKIP("MOON_ME not defined by the kernels on disk (kernels.bootstrap --limb)");
    }
    for (const auto& r : recs) {
        if (r.kind != "lax") continue;
        const double et = r.num(0);
        const auto frame = eclipse::frame_from_string(r.tokens.at(1));
        const auto ax = ephem::limb_axes(std::vector<double>{et}, frame, r.tokens.at(2));
        for (size_t i = 0; i < 9; ++i) CHECK(ax[0][i] == r.num(3 + i));
    }
    CHECK(n_lax == 54);
}

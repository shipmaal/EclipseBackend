// eclipse/numerics.hpp — hand-value tests of the NumPy-semantics helpers and
// the app.numerics ports (bisect / sign_changes / parabolic_minimum). The
// live Python parity (same objective, bit-identical results, same call
// counts) is tests/test_native.py::test_numerics_parity; here the rules are
// pinned with values checked against NumPy by hand.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include "eclipse/numerics.hpp"

using Catch::Matchers::WithinAbs;
namespace num = eclipse::numerics;

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// A vectorized objective that counts its calls (one call per iteration is
// part of the contract: it is what makes the Python's array evaluation cheap).
template <class G>
struct Counting {
    G g;
    int calls = 0;
    std::vector<double> operator()(std::span<const double> t) {
        ++calls;
        std::vector<double> out(t.size());
        for (size_t i = 0; i < t.size(); ++i) out[i] = g(t[i]);
        return out;
    }
};
}  // namespace

TEST_CASE("np_remainder follows NumPy's float % (signed zero, negative operands)") {
    // Values checked against np.remainder (NumPy 2.0.2).
    CHECK(num::np_remainder(-1.0, 360.0) == 359.0);
    CHECK(num::np_remainder(1.0, -360.0) == -359.0);
    CHECK(num::np_remainder(-1.0, -360.0) == -1.0);
    CHECK(num::np_remainder(725.0, 360.0) == 5.0);
    // Exact multiples give a zero carrying the DIVISOR's sign.
    CHECK(num::np_remainder(360.0, 360.0) == 0.0);
    CHECK_FALSE(std::signbit(num::np_remainder(360.0, 360.0)));
    CHECK_FALSE(std::signbit(num::np_remainder(-360.0, 360.0)));
    CHECK_FALSE(std::signbit(num::np_remainder(-0.0, 360.0)));
    CHECK(std::signbit(num::np_remainder(720.0, -360.0)));
    CHECK(std::isnan(num::np_remainder(kNaN, 360.0)));
    // wrap_180 maps into [-180, 180): both +180 and -180 give -180 (NumPy checked).
    CHECK(num::wrap_180(180.0) == -180.0);
    CHECK(num::wrap_180(-180.0) == -180.0);
    CHECK(num::wrap_180(540.0) == -180.0);
    CHECK(num::wrap_180(-181.0) == 179.0);
    CHECK(num::wrap_180(190.0) == -170.0);
}

TEST_CASE("np_sign and np_clip propagate NaN like NumPy") {
    CHECK(num::np_sign(3.5) == 1.0);
    CHECK(num::np_sign(-2.0) == -1.0);
    CHECK(num::np_sign(0.0) == 0.0);
    CHECK(num::np_sign(-0.0) == 0.0);
    CHECK(std::isnan(num::np_sign(kNaN)));
    CHECK_FALSE(num::np_sign(kNaN) == num::np_sign(kNaN));  // NaN != NaN, as np.sign ==
    CHECK(num::np_clip(2.0, -1.0, 1.0) == 1.0);
    CHECK(num::np_clip(-2.0, -1.0, 1.0) == -1.0);
    CHECK(num::np_clip(0.25, -1.0, 1.0) == 0.25);
    CHECK(std::isnan(num::np_clip(kNaN, -1.0, 1.0)));
}

TEST_CASE("arange uses NumPy's delta fill rule, not start + i*step") {
    // np.arange(-5, 5 + 1e-9, 1/60): 601 elements, and the fill increment is
    // delta = (start + step) - start = 0.016666666666666607, not 1/60.
    const std::vector<double> t = num::arange(-5.0, 5.0 + 1e-9, 1.0 / 60.0);
    REQUIRE(t.size() == 601);
    const double delta = (-5.0 + 1.0 / 60.0) - -5.0;
    CHECK(delta == 0.016666666666666607);
    CHECK(t[0] == -5.0);
    CHECK(t[1] == -5.0 + 1 * delta);
    CHECK(t[2] == -5.0 + 2 * delta);
    CHECK(t[300] == -5.0 + 300 * delta);
    CHECK(t[600] == -5.0 + 600 * delta);
    // The naive rule is measurably different: the last element is not the same double.
    CHECK(t[600] != -5.0 + 600 * (1.0 / 60.0));
    // The global_contacts grid of the Python: +/-5 h in 1-minute steps.
    CHECK_THAT(t.back(), WithinAbs(5.0, 1e-12));

    // Degenerate lengths.
    CHECK(num::arange(0.0, 0.0, 1.0).empty());
    CHECK(num::arange(0.0, -1.0, 1.0).empty());
    CHECK(num::arange(1.0, 1.5, 1.0) == std::vector<double>{1.0});
    CHECK(num::arange(0.0, 2.0, 1.0) == std::vector<double>{0.0, 1.0});
    // /central-line's default track: arange(-2, 2 + 1e-9, 2/60) has 121 samples.
    CHECK(num::arange(-2.0, 2.0 + 1e-9, 2.0 / 60.0).size() == 121);
}

TEST_CASE("bisect finds a bracketed root with exactly iterations + 1 objective calls") {
    Counting<double (*)(double)> f{[](double t) { return std::cos(t) - t; }};
    const double lo[] = {0.0, -1.0};
    const double hi[] = {1.0, 2.0};
    const std::vector<double> r = num::bisect(f, lo, hi);
    CHECK(f.calls == 31);  // f(t_lo) once + 30 midpoints
    REQUIRE(r.size() == 2);
    // The Dottie number, cos(t) = t. 30 halvings of a unit bracket: ~1e-9.
    CHECK_THAT(r[0], WithinAbs(0.7390851332151607, 2e-9));
    CHECK_THAT(r[1], WithinAbs(0.7390851332151607, 4e-9));
    // The iteration count is honoured (41 calls for 40), and a rising root
    // (f_lo < 0) is found as well as the falling one above.
    f.calls = 0;
    const double lo2[] = {0.0}, hi2[] = {1.0};
    const auto r2 = num::bisect(f, lo2, hi2, 40);
    CHECK(f.calls == 41);
    CHECK_THAT(r2[0], WithinAbs(0.7390851332151607, 2e-12));
    const double lo3[] = {-2.0}, hi3[] = {2.0};
    const auto r3 = num::bisect([](std::span<const double> t) {
        std::vector<double> o(t.size());
        for (size_t i = 0; i < t.size(); ++i) o[i] = t[i] * t[i] * t[i] - 0.5;
        return o;
    }, lo3, hi3, 50);
    CHECK_THAT(r3[0], WithinAbs(std::cbrt(0.5), 1e-13));
}

TEST_CASE("bisect: a NaN objective moves the high end (sign(NaN) compares unequal)") {
    // f_lo is finite, every midpoint NaN: `same` is always false, so lo stays
    // and hi walks down to lo -> result == lo (NumPy: np.sign(nan) == x is False).
    int calls = 0;
    auto f = [&calls](std::span<const double> t) {
        ++calls;
        std::vector<double> o(t.size(), kNaN);
        if (calls == 1)
            for (auto& v : o) v = -1.0;
        return o;
    };
    const double lo[] = {1.0}, hi[] = {3.0};
    const auto r = num::bisect(f, lo, hi, 60);
    CHECK(r[0] == 1.0);
}

TEST_CASE("sign_changes: exact zero, ordinary crossing, NaN") {
    const double t[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const double f[] = {1.0, 0.0, 2.0, -1.0, kNaN, -3.0, 4.0, 0.0};
    const auto c = num::sign_changes(t, f);
    // i=0: 1*0 = 0, not < 0 -> no; i=1: exact zero, next 2 > 0 -> rising;
    // i=2: 2*-1 < 0, falling; i=3/4: NaN products -> skipped; i=5: -3*4 < 0
    // rising; i=6: 4*0 = 0 -> no (the zero at the LAST sample is not visited).
    REQUIRE(c.size() == 3);
    CHECK((c[0].i == 1 && c[0].rising));
    CHECK((c[1].i == 2 && !c[1].rising));
    CHECK((c[2].i == 5 && c[2].rising));
    // An exact zero followed by a negative sample: counted, falling.
    const double t2[] = {0, 1}, f2[] = {0.0, -1.0};
    const auto c2 = num::sign_changes(t2, f2);
    REQUIRE(c2.size() == 1);
    CHECK((c2[0].i == 0 && !c2[0].rising));
    // Zero followed by zero: counted, not rising (0 > 0 is false).
    const double f3[] = {0.0, 0.0};
    const auto c3 = num::sign_changes(t2, f3);
    REQUIRE(c3.size() == 1);
    CHECK_FALSE(c3[0].rising);
    CHECK(num::sign_changes(std::span<const double>{}, std::span<const double>{}).empty());
}

TEST_CASE("parabolic_minimum converges on a shifted parabola in 10 calls") {
    // f(t) = (t - c)^2 + k: the vertex of the 3-point fit is exact, so one
    // clipped step lands on c to rounding; 10 iterations, 10 objective calls.
    const double c[] = {0.37, -1.25, 2.0};
    int calls = 0;
    auto obj = [&](std::span<const double> t) {
        ++calls;
        std::vector<double> o(t.size());
        const size_t n = t.size() / 3;
        for (size_t i = 0; i < t.size(); ++i) {
            const double cc = c[i % n];
            o[i] = (t[i] - cc) * (t[i] - cc) + 3.0;
        }
        return o;
    };
    const double t0[] = {0.0, -1.0, 2.5};
    const auto r = num::parabolic_minimum(obj, t0, 2.0);
    CHECK(calls == 10);
    REQUIRE(r.size() == 3);
    for (size_t i = 0; i < 3; ++i) CHECK_THAT(r[i], WithinAbs(c[i], 1e-12));
    // Vertex farther than half_width away: the first step is clipped to +/-h,
    // then h quarters, so the point cannot travel past h * (1 + 1/4 + ...) = 4h/3.
    const double t0b[] = {0.0};
    const auto rb = num::parabolic_minimum([](std::span<const double> t) {
        std::vector<double> o(t.size());
        for (size_t i = 0; i < t.size(); ++i) o[i] = (t[i] - 10.0) * (t[i] - 10.0);
        return o;
    }, t0b, 1.0, 10);
    CHECK(rb[0] < 4.0 / 3.0 + 1e-12);
    CHECK(rb[0] > 1.0);
    // A maximum (denom < 0) or a flat fit (denom == 0 -> NaN step) does not move t0.
    const auto rc = num::parabolic_minimum([](std::span<const double> t) {
        std::vector<double> o(t.size());
        for (size_t i = 0; i < t.size(); ++i) o[i] = -(t[i] * t[i]);
        return o;
    }, t0b, 1.0, 3);
    CHECK(rc[0] == 0.0);
    const auto rd = num::parabolic_minimum([](std::span<const double> t) {
        return std::vector<double>(t.size(), 5.0);
    }, t0b, 1.0, 3);
    CHECK(rd[0] == 0.0);
}

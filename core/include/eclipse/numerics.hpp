// Vectorized root-finding / extremum helpers and NumPy-semantics primitives —
// the C++ twin of app/numerics.py plus the handful of NumPy float rules
// (remainder, sign, clip, arange) the ports depend on for bit-level parity.
//
// These are numerical methods, not cited physics: bisection and three-point
// parabolic refinement (roadmap §4: "numerics.* — exact (same algorithm, same
// iteration count)"). Every function is written in the Python's exact
// floating-point operation order; do not reassociate (roadmap §7).
//
// Header-only. Objectives are *vectorized*: ``f(span) -> vector`` evaluates
// all brackets / all vertices in one call, exactly as the Python's
// ``f(t_array) -> array`` — the objective call count per iteration is one.
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace eclipse::numerics {

// --- NumPy float semantics ------------------------------------------------------

/// NumPy float remainder (``np.remainder`` / ``%``, ``npy_divmod``): ``fmod``, then
/// add the divisor when the signs differ, and ``+0`` with the divisor's sign for
/// an exact multiple. Used for ``(deg + 180) % 360 - 180``. (numerical)
inline double np_remainder(double a, double b) {
    double mod = std::fmod(a, b);
    if (mod != 0.0) {
        if ((b < 0.0) != (mod < 0.0)) mod += b;
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

/// ``(deg + 180.0) % 360.0 - 180.0`` with NumPy's ``%``: longitude / hour angle
/// wrapped into [-180, 180). (numerical)
inline double wrap_180(double deg) { return np_remainder(deg + 180.0, 360.0) - 180.0; }

/// ``np.sign`` for floats: -1, 0, +1, and NaN for NaN (so ``np_sign(a) ==
/// np_sign(b)`` is false whenever either is NaN, as in NumPy). (numerical)
inline double np_sign(double v) {
    if (std::isnan(v)) return v;
    return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
}

/// ``np.clip(v, lo, hi)`` for floats: ``min(max(v, lo), hi)`` with NaN in ``v``
/// propagated (NumPy's ``_NPY_CLIP``). (numerical)
inline double np_clip(double v, double lo, double hi) {
    if (std::isnan(v)) return v;
    return std::min(std::max(v, lo), hi);
}

/// ``np.arange(start, stop, step)`` for float64 — NumPy's exact fill rule
/// (``PyArray_ArangeObj`` + ``DOUBLE_fill``): ``n = ceil((stop - start) / step)``,
/// ``t[0] = start``, ``t[1] = start + step``, and for ``i >= 2``
/// ``t[i] = start + i * delta`` with ``delta = (start + step) - start`` — NOT
/// ``start + i * step``, which differs in the last bit (for
/// ``arange(-5, 5 + 1e-9, 1/60)`` delta is 0.016666666666666607). (numerical)
inline std::vector<double> arange(double start, double stop, double step) {
    const double len = std::ceil((stop - start) / step);
    if (std::isnan(len)) throw std::invalid_argument("arange: NaN length");
    if (!(len > 0.0)) return {};
    const auto n = static_cast<std::size_t>(len);
    std::vector<double> t(n);
    t[0] = start;
    if (n == 1) return t;
    t[1] = start + step;
    const double delta = t[1] - t[0];
    for (std::size_t i = 2; i < n; ++i) t[i] = start + static_cast<double>(i) * delta;
    return t;
}

// --- vectorized objectives ----------------------------------------------------------

/// A vectorized objective ``f(t_array) -> array`` (app.numerics.Objective):
/// one call evaluates every bracket / vertex; the result has ``t.size()``
/// elements.
template <class F>
concept VectorObjective = requires(F f, std::span<const double> t) {
    { f(t) } -> std::same_as<std::vector<double>>;
};

/// Mirrors ``app.numerics.bisect``: bisect ``f`` to a sign change inside each
/// bracket ``[t_lo[i], t_hi[i]]``. ``f(t_lo)`` is evaluated once, then each of
/// the ``iterations`` steps evaluates ``f`` once at every midpoint (31 calls
/// for the default 30): ``same = sign(f_mid) == sign(f_lo)`` (false for NaN,
/// which therefore moves ``hi``); where ``same`` the low end moves to the
/// midpoint, else the high end. Returns ``0.5 * (lo + hi)``. 30 iterations
/// shrink a 30-second bracket to ~30 ns. (numerical)
template <VectorObjective F>
std::vector<double> bisect(F&& f, std::span<const double> t_lo, std::span<const double> t_hi,
                           int iterations = 30) {
    const std::size_t n = t_lo.size();
    if (t_hi.size() != n) throw std::invalid_argument("bisect: bracket arrays differ in length");
    std::vector<double> lo(t_lo.begin(), t_lo.end()), hi(t_hi.begin(), t_hi.end());
    std::vector<double> f_lo = f(std::span<const double>(lo));
    std::vector<double> mid(n);
    for (int it = 0; it < iterations; ++it) {
        for (std::size_t i = 0; i < n; ++i) mid[i] = 0.5 * (lo[i] + hi[i]);
        const std::vector<double> f_mid = f(std::span<const double>(mid));
        for (std::size_t i = 0; i < n; ++i) {
            const bool same = np_sign(f_mid[i]) == np_sign(f_lo[i]);
            if (same) {
                lo[i] = mid[i];
                f_lo[i] = f_mid[i];
            } else {
                hi[i] = mid[i];
            }
        }
    }
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = 0.5 * (lo[i] + hi[i]);
    return out;
}

/// One sign change of a sampled function: between ``t[i]`` and ``t[i + 1]``.
struct SignChange {
    std::size_t i;
    bool rising;
};

/// Mirrors ``app.numerics.sign_changes``: indices ``i`` where ``f`` changes sign
/// between ``t[i]`` and ``t[i + 1]``. An exact zero at ``f[i]`` counts as a
/// crossing at ``i`` with ``rising = f[i + 1] > 0``; otherwise the pair counts
/// when ``f[i] * f[i + 1] < 0`` (false for NaN) with ``rising = f[i + 1] > f[i]``.
/// Only ``t.size()`` is used, as in the Python (``range(len(t) - 1)``). (numerical)
inline std::vector<SignChange> sign_changes(std::span<const double> t, std::span<const double> f) {
    std::vector<SignChange> out;
    if (t.empty()) return out;
    for (std::size_t i = 0; i + 1 < t.size(); ++i) {
        if (f[i] == 0.0) {
            out.push_back({i, f[i + 1] > 0.0});
        } else if (f[i] * f[i + 1] < 0.0) {
            out.push_back({i, f[i + 1] > f[i]});
        }
    }
    return out;
}

/// Mirrors ``app.numerics.parabolic_minimum``: refine minima of ``f`` near each
/// ``t0[i]`` by repeated three-point parabolic fits. Each iteration evaluates
/// ``f`` ONCE on the ``3n`` vector ``[t0 - h..., t0..., t0 + h...]`` (the
/// Python's ``np.stack([...]).ravel()`` order), then per element
/// ``denom = y0 - 2.0 * y1 + y2``, ``step = 0.5 * h * (y0 - y2) / denom``
/// (evaluated left to right: ``((0.5 * h) * (y0 - y2)) / denom``), the step is
/// ``np_clip(step, -h, h)`` where ``denom > 0`` and ``0.0`` otherwise (a
/// maximum or a degenerate fit does not move the point), ``t0 += step`` and
/// ``h = h / 4.0``. Ten iterations from a 2-hour ``h`` reach ~7 ms. (numerical)
template <VectorObjective F>
std::vector<double> parabolic_minimum(F&& f, std::span<const double> t0_in, double half_width,
                                      int iterations = 10) {
    const std::size_t n = t0_in.size();
    std::vector<double> t0(t0_in.begin(), t0_in.end());
    std::vector<double> h(n, half_width);
    std::vector<double> pts(3 * n);
    for (int it = 0; it < iterations; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            pts[i] = t0[i] - h[i];
            pts[n + i] = t0[i];
            pts[2 * n + i] = t0[i] + h[i];
        }
        const std::vector<double> y = f(std::span<const double>(pts));
        for (std::size_t i = 0; i < n; ++i) {
            const double y0 = y[i], y1 = y[n + i], y2 = y[2 * n + i];
            const double denom = y0 - 2.0 * y1 + y2;
            const double step = 0.5 * h[i] * (y0 - y2) / denom;
            t0[i] = t0[i] + (denom > 0.0 ? np_clip(step, -h[i], h[i]) : 0.0);
            h[i] = h[i] / 4.0;
        }
    }
    return t0;
}

}  // namespace eclipse::numerics

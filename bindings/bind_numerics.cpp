// Bindings for eclipse/numerics.hpp (app.numerics + NumPy-semantics helpers).
//
// ``bisect`` and ``parabolic_minimum`` take a Python callable so the parity
// test can drive the C++ iteration with the SAME objective the Python oracle
// uses and compare results and call counts bit-for-bit. They hold the GIL
// throughout (the objective is Python) and exist for tests only: production
// paths call the header templates with C++ objectives.
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "common.hpp"
#include "eclipse/numerics.hpp"

using namespace nb::literals;
namespace num = eclipse::numerics;

namespace {

// Wrap ``f: ndarray -> ndarray`` as a VectorObjective. The span is copied into
// a fresh ndarray (the callee may keep it), and the result is converted to a
// contiguous float64 array (``nb::cast`` converts dtype/layout if needed).
auto python_objective(nb::callable& f) {
    return [&f](std::span<const double> t) -> std::vector<double> {
        nb::object r = f(to_numpy(std::vector<double>(t.begin(), t.end())));
        const In1D y = nb::cast<In1D>(r);
        if (y.shape(0) != t.size()) throw std::invalid_argument("objective returned a wrong-length array");
        return std::vector<double>(y.data(), y.data() + y.shape(0));
    };
}

}  // namespace

void bind_numerics(nb::module_& m) {
    m.def("np_remainder", &num::np_remainder, "a"_a, "b"_a,
          "NumPy float remainder (np.remainder / %) for one pair, signed zero included.");
    m.def("py_round", &num::py_round, "x"_a, "ndigits"_a,
          "Python's round(x, ndigits) for a float, ndigits >= 0 (eclipse::numerics::py_round).\n\n"
          "TEST-ONLY parity hook: the catalog port rounds with it natively; Python code has "
          "round().");
    m.def(
        "np_arange",
        [](double start, double stop, double step) {
            return to_numpy(num::arange(start, stop, step));
        },
        "start"_a, "stop"_a, "step"_a,
        "np.arange(start, stop, step) for float64 with NumPy's exact fill rule.");
    m.def(
        "sign_changes",
        [](In1D t, In1D f) {
            std::vector<std::pair<std::size_t, bool>> out;
            for (const auto& c : num::sign_changes(as_span(t), as_span(f)))
                out.emplace_back(c.i, c.rising);
            return out;
        },
        "t"_a, "f"_a, "app.numerics.sign_changes: list of (i, rising).");
    m.def(
        "bisect",
        [](nb::callable f, In1D t_lo, In1D t_hi, int iterations) {
            return to_numpy(num::bisect(python_objective(f), as_span(t_lo), as_span(t_hi), iterations));
        },
        "f"_a, "t_lo"_a, "t_hi"_a, "iterations"_a = 30,
        "app.numerics.bisect driven by a Python objective (ndarray -> ndarray).\n\n"
        "TEST-ONLY parity hook: holds the GIL and calls back into Python once per "
        "iteration (plus once for f(t_lo)). Never use on a production path; the "
        "C++ callers use the header template with a native objective.");
    m.def(
        "parabolic_minimum",
        [](nb::callable f, In1D t0, double half_width, int iterations) {
            return to_numpy(
                num::parabolic_minimum(python_objective(f), as_span(t0), half_width, iterations));
        },
        "f"_a, "t0"_a, "half_width"_a, "iterations"_a = 10,
        "app.numerics.parabolic_minimum driven by a Python objective (ndarray -> ndarray).\n\n"
        "TEST-ONLY parity hook: holds the GIL and calls back into Python once per "
        "iteration on the stacked 3n array. Never use on a production path.");
}

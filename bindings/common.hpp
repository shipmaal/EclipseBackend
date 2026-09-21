// Shared nanobind glue for the ``_eclipse`` binding translation units:
// the NumPy float64 array views and the zero-copy vector -> ndarray hand-off.
// Each bind_*.cpp includes this; _eclipse.cpp only assembles the module.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <span>
#include <utility>
#include <vector>

namespace nb = nanobind;

using In1D = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Out1D = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

inline std::span<const double> as_span(const In1D& a) { return {a.data(), a.shape(0)}; }

// Hand a std::vector to NumPy without copying: the capsule owns the vector.
inline Out1D to_numpy(std::vector<double>&& v) {
    auto* heap = new std::vector<double>(std::move(v));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const size_t n = heap->size();
    return Out1D(heap->data(), {n}, owner);
}

// One binding unit per core header (roadmap §3), so later work packages stay
// file-disjoint. Declared here, defined in bind_<name>.cpp, called from NB_MODULE.
void bind_numerics(nb::module_& m);
void bind_ellipsoid(nb::module_& m);
void bind_geometry(nb::module_& m);

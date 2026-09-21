// nanobind module ``_eclipse`` — the Python face of libeclipse.
//
// Phase 0 (docs/CPP_ROADMAP.md §5): kernel management, time-scale and position
// calls through the checked SPICE wrapper, and the version pins. Every
// SPICE-touching call releases the GIL; the C++ side holds its own lock.
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include "eclipse/ephem.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_eclipse, m) {
    m.doc() = "libeclipse: native core of EclipseBackend (CSPICE + ERFA).";

    // ``SpiceError`` derives from RuntimeError so ``except RuntimeError`` still
    // catches it; ``str(e)`` is the spiceypy-style multi-line report.
    nb::exception<eclipse::spice_error>(m, "SpiceError", PyExc_RuntimeError);

    nb::class_<eclipse::ephem::Position>(m, "Position",
                                         "spkpos_c result: km vector + light time (s).")
        .def_ro("km", &eclipse::ephem::Position::km)
        .def_ro("light_time_s", &eclipse::ephem::Position::light_time_s);

    m.def("toolkit_version", &eclipse::ephem::toolkit_version,
          "tkvrsn_c('TOOLKIT') of the vendored CSPICE, e.g. 'CSPICE_N0067'.");
    m.def("erfa_version", &eclipse::ephem::erfa_version, "eraVersion() of the vendored liberfa.");
    m.def("sofa_version", &eclipse::ephem::sofa_version, "eraSofaVersion() of the vendored liberfa.");

    m.def("furnish", &eclipse::ephem::furnish, "path"_a,
          nb::call_guard<nb::gil_scoped_release>(), "furnsh_c: load a kernel or metakernel.");
    m.def("kclear", &eclipse::ephem::kclear, nb::call_guard<nb::gil_scoped_release>(),
          "kclear_c: unload all kernels.");
    m.def("kernel_count", &eclipse::ephem::kernel_count, "kind"_a = "ALL",
          nb::call_guard<nb::gil_scoped_release>(), "ktotal_c.");
    m.def("str_to_et", &eclipse::ephem::str_to_et, "time_string"_a,
          nb::call_guard<nb::gil_scoped_release>(), "str2et_c: time string -> TDB seconds past J2000.");
    m.def("et_to_utc_iso", &eclipse::ephem::et_to_utc_iso, "et"_a, "precision"_a = 6,
          nb::call_guard<nb::gil_scoped_release>(), "et2utc_c(..., 'ISOC', precision).");
    m.def("body_position", &eclipse::ephem::body_position, "target"_a, "et"_a, "frame"_a,
          "abcorr"_a, "observer"_a, nb::call_guard<nb::gil_scoped_release>(),
          "spkpos_c: apparent position (km) of target seen from observer in frame.");
}

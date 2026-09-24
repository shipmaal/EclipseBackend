// Bindings for eclipse/circumstances.hpp (app.circumstances local_circumstances
// and circumstances_grid), phase 3 of docs/CPP_ROADMAP.md.
//
// ``local_circumstances`` hands ``LocalRaw`` back as a tuple in the Python
// ``_LocalRaw`` NamedTuple's FIELD ORDER (geometric, central, c1, c4, c2, c3,
// t_max, magnitude, obscuration, L2_x, alt_deg, az_deg, below, eclipse), the
// five-element event arrays as tuples, so ``_LocalRaw(*raw)`` on the Python
// side reconstructs the oracle's values type for type (Python floats with NaN
// in the absent C2/C3 slots, Python bools). ``circumstances_grid`` returns the
// oracle's dict of length-P arrays under the same keys, float64 for the four
// numeric ones and ``bool`` (``to_numpy_bool``) for ``visible`` / ``central``.
// Both release the GIL around the compute: the elements are evaluated under
// the C++ SPICE lock and the observer loop is pure math (OpenMP in the grid).
// ``std::invalid_argument`` (equal-length checks; the oracle's ValueError
// where no C1/C4 pair exists) maps to ``ValueError`` through nanobind.
#include <nanobind/stl/string_view.h>

#include <array>
#include <string_view>
#include <utility>

#include "common.hpp"
#include "eclipse/circumstances.hpp"
#include "eclipse/elements.hpp"

using namespace nb::literals;
namespace circ = eclipse::circumstances;

namespace {

// The five event slots as a Python tuple (the NamedTuple's ``tuple[...]``
// fields), not the list nanobind's std::array caster would produce.
template <typename T>
nb::tuple events_tuple(const std::array<T, 5>& a) {
    return nb::make_tuple(a[0], a[1], a[2], a[3], a[4]);
}

}  // namespace

nb::tuple local_raw_tuple(const circ::LocalRaw& r) {
    return nb::make_tuple(r.geometric, r.central, r.c1, r.c4, r.c2, r.c3, r.t_max, r.magnitude,
                          r.obscuration, r.L2_x, events_tuple(r.alt_deg), events_tuple(r.az_deg),
                          events_tuple(r.below), r.eclipse);
}

void bind_circumstances(nb::module_& m) {
    m.def(
        "local_circumstances",
        [](double et0, std::string_view earth_frame, double half_window_hours, double lat_deg,
           double lon_deg, bool profile, double height_m) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            circ::LocalRaw r;
            {
                nb::gil_scoped_release nogil;
                r = circ::local_circumstances(et0, f, half_window_hours, lat_deg, lon_deg, profile,
                                              height_m);
            }
            return local_raw_tuple(r);
        },
        "et0"_a, "earth_frame"_a, "half_window_hours"_a, "lat_deg"_a, "lon_deg"_a,
        "profile"_a = false, "height_m"_a = 0.0,
        "app.circumstances._local_raw for a model at et0 [TDB s] in earth_frame with the\n"
        "given fit half-window [h]: the _LocalRaw fields, in order, as a tuple (geometric,\n"
        "central, c1, c4, c2, c3, t_max, magnitude, obscuration, L2_x, alt_deg[5], az_deg[5],\n"
        "below[5], eclipse). Times in hours from T0; NaN / False in absent slots.\n"
        "profile=True is limb=\"profile\" (the band must be installed: set_limb_band).\n"
        "height_m: the observer's height above the WGS-84 ellipsoid [m].");
    m.def(
        "profile_g",
        [](double et0, std::string_view earth_frame, double half_window_hours, double lat_deg,
           double lon_deg, In1D t_hours, double height_m) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            std::vector<double> g;
            {
                nb::gil_scoped_release nogil;
                g = circ::profile_g(circ::model_from_ephem(et0, f, half_window_hours), lat_deg,
                                    lon_deg, as_span(t_hours), height_m);
            }
            return to_numpy(std::move(g));
        },
        "et0"_a, "earth_frame"_a, "half_window_hours"_a, "lat_deg"_a, "lon_deg"_a, "t_hours"_a,
        "height_m"_a = 0.0,
        "app.circumstances._profile_g: the limb-profile contact function at t_hours\n"
        "(height_m [m] above the WGS-84 ellipsoid).");
    m.def(
        "circumstances_grid",
        [](double et0, std::string_view earth_frame, double half_window_hours, In1D lat_deg,
           In1D lon_deg, double step_minutes, int threads) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            circ::GridResult g;
            {
                nb::gil_scoped_release nogil;
                g = circ::circumstances_grid(et0, f, half_window_hours, as_span(lat_deg),
                                             as_span(lon_deg), step_minutes, threads);
            }
            nb::dict out;
            out["magnitude"] = to_numpy(std::move(g.magnitude));
            out["obscuration"] = to_numpy(std::move(g.obscuration));
            out["t_max_hours"] = to_numpy(std::move(g.t_max_hours));
            out["sun_alt"] = to_numpy(std::move(g.sun_alt));
            out["visible"] = to_numpy_bool(std::move(g.visible));
            out["central"] = to_numpy_bool(std::move(g.central));
            return out;
        },
        "et0"_a, "earth_frame"_a, "half_window_hours"_a, "lat_deg"_a, "lon_deg"_a,
        "step_minutes"_a = 2.0, "threads"_a = 0,
        "app.circumstances.circumstances_grid for a model at et0 [TDB s] in earth_frame:\n"
        "dict of length-P arrays magnitude, obscuration, t_max_hours [h], sun_alt [deg]\n"
        "(float64), visible, central (bool) for equal-length 1-D lat_deg / lon_deg [deg].\n"
        "threads: 0 = OpenMP default, 1 = serial, n = that many (results identical).");
}

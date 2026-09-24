// Bindings for eclipse/circumstances.hpp (local circumstances, the /map grid,
// the limb-profile contact function and search; formatted by
// app/circumstances.py).
//
// ``local_circumstances`` hands ``LocalRaw`` back as a tuple in the
// ``app.circumstances._LocalRaw`` NamedTuple's FIELD ORDER (geometric, central,
// c1, c4, c2, c3, t_max, magnitude, obscuration, L2_x, alt_deg, az_deg, below,
// eclipse), the five-element event arrays as tuples, so ``_LocalRaw(*raw)``
// rebuilds it type for type (Python floats with NaN in the absent C2/C3 slots,
// Python bools). ``circumstances_grid`` returns a dict of length-P arrays,
// float64 for the four numeric ones and ``bool`` (``to_numpy_bool``) for
// ``visible`` / ``central``.
// Both release the GIL around the compute: the elements are evaluated under
// the C++ SPICE lock and the observer loop is pure math (OpenMP in the grid).
// ``std::invalid_argument`` (equal-length checks; no C1/C4 pair) maps to
// ``ValueError`` through nanobind.
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
    m.def("overlap_area", &circ::overlap_area, "r"_a, "R"_a, "d"_a,
          "Area of intersection of two circles of radii r <= R at centre distance d.");
    m.def("obscuration", &circ::obscuration, "L1p"_a, "L2p"_a, "m"_a,
          "Covered fraction of the Sun's area from the reduced cone radii L1', L2' and the\n"
          "observer's axis distance m [Earth radii].");
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
        "Local circumstances for a model at et0 [TDB s] in earth_frame with the\n"
        "given fit half-window [h]: the app.circumstances._LocalRaw fields, in order, as a tuple (geometric,\n"
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
        "The limb-profile contact function G at t_hours (negative in the central phase)\n"
        "(height_m [m] above the WGS-84 ellipsoid).");
    m.def(
        "profile_contacts",
        [](double et0, std::string_view earth_frame, double half_window_hours, double lat_deg,
           double lon_deg, double t_lo, double t_hi, double height_m) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            circ::ProfileContacts pc;
            {
                nb::gil_scoped_release nogil;
                pc = circ::profile_contacts(circ::model_from_ephem(et0, f, half_window_hours),
                                            lat_deg, lon_deg, t_lo, t_hi, height_m);
            }
            return nb::make_tuple(pc.central, pc.c2, pc.c3);
        },
        "et0"_a, "earth_frame"_a, "half_window_hours"_a, "lat_deg"_a, "lon_deg"_a, "t_lo"_a,
        "t_hi"_a, "height_m"_a = 0.0,
        "The limb-profile central phase searched in [t_lo, t_hi] [h from et0]:\n"
        "(central, c2, c3), c2 / c3 NaN when there is none.");
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
        "Maximum-eclipse circumstances of sea-level observers for a model at et0 [TDB s] in earth_frame:\n"
        "dict of length-P arrays magnitude, obscuration, t_max_hours [h], sun_alt [deg]\n"
        "(float64), visible, central (bool) for equal-length 1-D lat_deg / lon_deg [deg].\n"
        "threads: 0 = OpenMP default, 1 = serial, n = that many (results identical).");
}

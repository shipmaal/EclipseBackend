// Bindings for eclipse/geometry.hpp (shadow radii, the great-circle helpers,
// the shadow-edge limits and the global contacts).
//
// Arrays cross as equal-length 1-D float64 (spans in, fresh ndarrays out); the
// core does not broadcast. The array forms and the SPICE-backed
// global_contacts release the GIL around the compute. Contacts come back as a
// list of (name, t_hours) pairs in a fixed order (P1, P4, U1, U4, U2, U3,
// absent ones omitted) -- never a mapping here, which would lose the order.
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <string>
#include <tuple>
#include <string_view>
#include <utility>
#include <vector>

#include "common.hpp"
#include "eclipse/elements.hpp"
#include "eclipse/geometry.hpp"

using namespace nb::literals;
namespace geo = eclipse::geometry;

void bind_geometry(nb::module_& m) {
    m.def(
        "shadow_radii",
        [](double x, double y, double d_deg, double l1, double l2, double tan_f1, double tan_f2) {
            const geo::ShadowRadii r = geo::shadow_radii(x, y, d_deg, l1, l2, tan_f1, tan_f2);
            return nb::make_tuple(r.penumbra_km, r.umbra_km, r.is_total);
        },
        "x"_a, "y"_a, "d_deg"_a, "l1"_a, "l2"_a, "tan_f1"_a, "tan_f2"_a,
        "(penumbra_km, umbra_km, is_total) at the central point\n"
        "(x, y in Earth equatorial radii; d in degrees) [ES92] eq. 8.353; (0, 0, False) off Earth.");
    m.def(
        "destination",
        [](double lat_deg, double lon_deg, double bearing_deg, double dist_km) {
            const geo::LatLon p = geo::destination(lat_deg, lon_deg, bearing_deg, dist_km);
            return nb::make_tuple(p.lat_deg, p.lon_deg);
        },
        "lat_deg"_a, "lon_deg"_a, "bearing_deg"_a, "dist_km"_a,
        "(lat_deg, lon_deg) dist_km along the great circle at\n"
        "bearing_deg from (lat_deg, lon_deg), on the IUGG mean-radius sphere.");
    m.def("haversine_km", &geo::haversine_km, "lat1_deg"_a, "lon1_deg"_a, "lat2_deg"_a,
          "lon2_deg"_a,
          "Great-circle distance [km] on the mean-radius sphere between two points [deg].");
    m.def(
        "bearing",
        [](In1D lat1, In1D lon1, In1D lat2, In1D lon2) {
            std::vector<double> b;
            {
                nb::gil_scoped_release nogil;
                b = geo::bearing_deg(as_span(lat1), as_span(lon1), as_span(lat2), as_span(lon2));
            }
            return to_numpy(std::move(b));
        },
        "lat1_deg"_a, "lon1_deg"_a, "lat2_deg"_a, "lon2_deg"_a,
        "On equal-length 1-D arrays: initial great-circle bearing\n"
        "[deg, [0, 360)] from point 1 to point 2.");
    m.def(
        "shadow_edge_limits",
        [](In1D x, In1D y, In1D d_deg, In1D mu_deg, In1D l, In1D tan_f, In1D path_bearing_deg,
           double max_km, bool sunlit_only, std::optional<std::tuple<In1D, In1D, In1D, In1D, In1D>> rates) {
            geo::EdgeLimits e;
            std::optional<geo::EdgeRates> r;
            if (rates) {
                auto& [dx, dy, dd, dmu, dl] = *rates;
                r = geo::EdgeRates{as_span(dx), as_span(dy), as_span(dd), as_span(dmu), as_span(dl)};
            }
            {
                nb::gil_scoped_release nogil;
                e = geo::shadow_edge_limits(as_span(x), as_span(y), as_span(d_deg), as_span(mu_deg),
                                            as_span(l), as_span(tan_f), as_span(path_bearing_deg),
                                            max_km, sunlit_only, r ? &*r : nullptr);
            }
            return nb::make_tuple(to_numpy(std::move(e.north_lat)), to_numpy(std::move(e.north_lon)),
                                  to_numpy(std::move(e.south_lat)), to_numpy(std::move(e.south_lon)),
                                  to_numpy(std::move(e.width_km)));
        },
        "x"_a, "y"_a, "d_deg"_a, "mu_deg"_a, "l"_a, "tan_f"_a, "path_bearing_deg"_a,
        "max_km"_a = 600.0, "sunlit_only"_a = true, "rates"_a = nb::none(),
        "Shadow-edge limits on equal-length 1-D arrays:\n"
        "(north_lat, north_lon, south_lat, south_lon, width_km); NaN / 0 where no edge.\n"
        "x, y, l in Earth equatorial radii; d, mu, bearing in degrees; max_km in km.\n"
        "rates: optional (dx, dy, dd_deg, dmu_deg, dl) per hour -> the path envelope.");
    m.def(
        "global_contacts",
        [](double et0, std::string_view earth_frame, double half_window_hours) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            std::vector<geo::Contact> c;
            {
                nb::gil_scoped_release nogil;
                c = geo::global_contacts(et0, f, half_window_hours);
            }
            std::vector<std::pair<std::string, double>> out;
            out.reserve(c.size());
            for (auto& k : c) out.emplace_back(std::move(k.name), k.t_hours);
            return out;
        },
        "et0"_a, "earth_frame"_a = "ITRS", "half_window_hours"_a = 5.0,
        "Global contacts for a model at et0 [TDB s] in earth_frame: list of\n"
        "(name, t_hours) in the order P1, P4, U1, U4, U2, U3 (absent ones omitted).");
}

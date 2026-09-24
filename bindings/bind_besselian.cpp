// Bindings for eclipse/besselian.hpp: the elements of one model (et0, frame)
// evaluated directly, their rates, the tabular polynomial fit and the central
// line with its limits.
#include <nanobind/stl/string_view.h>

#include <string_view>
#include <utility>
#include <vector>

#include "common.hpp"
#include "eclipse/besselian.hpp"

using namespace nb::literals;
namespace bs = eclipse::besselian;

namespace {

nb::dict elements_dict(eclipse::Elements&& e) {
    nb::dict out;
    out["x"] = to_numpy(std::move(e.x));
    out["y"] = to_numpy(std::move(e.y));
    out["z"] = to_numpy(std::move(e.z));
    out["d"] = to_numpy(std::move(e.d));
    out["mu"] = to_numpy(std::move(e.mu));
    out["l1"] = to_numpy(std::move(e.l1));
    out["l2"] = to_numpy(std::move(e.l2));
    out["tan_f1"] = to_numpy(std::move(e.tan_f1));
    out["tan_f2"] = to_numpy(std::move(e.tan_f2));
    return out;
}

template <std::size_t N>
nb::list coeffs(const std::array<double, N>& a) {
    nb::list l;
    for (const double v : a) l.append(v);
    return l;
}

}  // namespace

void bind_besselian(nb::module_& m) {
    m.def(
        "elements_direct",
        [](double et0, std::string_view earth_frame, In1D t_hours, double k1, double k2) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            eclipse::Elements e;
            {
                nb::gil_scoped_release nogil;
                e = bs::elements_direct(et0, f, as_span(t_hours), k1, k2);
            }
            return elements_dict(std::move(e));
        },
        "et0"_a, "earth_frame"_a, "t_hours"_a, "k1"_a = eclipse::constants::K_PENUMBRA,
        "k2"_a = eclipse::constants::K_UMBRA,
        "Elements at t_hours [h] from et0 [TDB s], mu unwrapped: dict of arrays x, y, z,\n"
        "l1, l2 [Earth radii], d, mu [deg], tan_f1, tan_f2 (k1 / k2: the cones' lunar radii).");
    m.def(
        "element_rates",
        [](double et0, std::string_view earth_frame, In1D t_hours) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            bs::Rates r;
            {
                nb::gil_scoped_release nogil;
                r = bs::element_rates(et0, f, as_span(t_hours));
            }
            nb::dict out;
            out["x"] = to_numpy(std::move(r.x));
            out["y"] = to_numpy(std::move(r.y));
            out["d"] = to_numpy(std::move(r.d));
            out["mu"] = to_numpy(std::move(r.mu));
            out["l1"] = to_numpy(std::move(r.l1));
            out["l2"] = to_numpy(std::move(r.l2));
            return out;
        },
        "et0"_a, "earth_frame"_a, "t_hours"_a,
        "Per-hour rates of x, y, l1, l2 [Earth radii/h] and d, mu [deg/h] at t_hours.");
    m.def(
        "fit_polynomials",
        [](double et0, std::string_view earth_frame, double half_window_hours, double step_hours) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            bs::Polynomials p;
            {
                nb::gil_scoped_release nogil;
                p = bs::fit_polynomials(et0, f, half_window_hours, step_hours);
            }
            nb::dict out;
            out["x"] = coeffs(p.x);
            out["y"] = coeffs(p.y);
            out["d"] = coeffs(p.d);
            out["l1"] = coeffs(p.l1);
            out["l2"] = coeffs(p.l2);
            out["mu"] = coeffs(p.mu);
            out["tan_f1"] = p.tan_f1;
            out["tan_f2"] = p.tan_f2;
            return out;
        },
        "et0"_a, "earth_frame"_a, "half_window_hours"_a = 2.0, "step_hours"_a = 1.0,
        "Tabular polynomials (increasing powers of t = hours from et0): x, y cubic; d, l1,\n"
        "l2 quadratic; mu linear; tan_f1, tan_f2 constants.");
    m.def(
        "polyfit",
        [](In1D t, In1D y, int deg) { return to_numpy(bs::polyfit(as_span(t), as_span(y), deg)); },
        "t"_a, "y"_a, "deg"_a, "Least-squares polynomial coefficients, increasing powers.");
    m.def(
        "central_line",
        [](double et0, std::string_view earth_frame, In1D t_hours) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            bs::CentralLine c;
            {
                nb::gil_scoped_release nogil;
                c = bs::central_line(et0, f, as_span(t_hours));
            }
            std::vector<std::uint8_t> tot(c.is_total.begin(), c.is_total.end());
            nb::dict out;
            out["t_hours"] = to_numpy(std::move(c.t_hours));
            out["lat"] = to_numpy(std::move(c.lat_deg));
            out["lon"] = to_numpy(std::move(c.lon_deg));
            out["bearing"] = to_numpy(std::move(c.bearing_deg));
            out["penumbra_km"] = to_numpy(std::move(c.penumbra_km));
            out["umbra_km"] = to_numpy(std::move(c.umbra_km));
            out["is_total"] = to_numpy_bool(std::move(tot));
            out["north_lat"] = to_numpy(std::move(c.north_lat));
            out["north_lon"] = to_numpy(std::move(c.north_lon));
            out["south_lat"] = to_numpy(std::move(c.south_lat));
            out["south_lon"] = to_numpy(std::move(c.south_lon));
            out["width_km"] = to_numpy(std::move(c.width_km));
            out["pen_north_lat"] = to_numpy(std::move(c.pen_north_lat));
            out["pen_north_lon"] = to_numpy(std::move(c.pen_north_lon));
            out["pen_south_lat"] = to_numpy(std::move(c.pen_south_lat));
            out["pen_south_lon"] = to_numpy(std::move(c.pen_south_lon));
            return out;
        },
        "et0"_a, "earth_frame"_a, "t_hours"_a,
        "The central line at t_hours [h] from et0: the on-Earth points (lat, lon, bearing\n"
        "[deg], penumbra_km, umbra_km, is_total), the umbral path limits + width_km (the\n"
        "time envelope; NaN / 0 where absent) and the penumbral limits (pen_*).");
}

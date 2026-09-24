// Bindings for eclipse/limb.hpp (app/limb.py) and ephem::limb_axes.
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"
#include "eclipse/elements.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/limb.hpp"

using namespace nb::literals;

namespace {

using InI32 = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using InI16 = nb::ndarray<const std::int16_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using In2D = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Out3D = nb::ndarray<nb::numpy, double, nb::ndim<3>>;
using Out2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

template <class T, class A>
std::span<const T> span_of(const A& a) {
    return {a.data(), a.shape(0)};
}

}  // namespace

void bind_limb(nb::module_& m) {
    m.def(
        "set_limb_band",
        [](InI32 line, InI32 first, InI32 count, InI16 dn, In1D cos_lat, In1D sin_lat, In1D cos_lon,
           In1D sin_lon, double offset_km, double scale_km, double band_deg) {
            eclipse::limb::set_band(span_of<std::int32_t>(line), span_of<std::int32_t>(first),
                                    span_of<std::int32_t>(count), span_of<std::int16_t>(dn),
                                    as_span(cos_lat), as_span(sin_lat), as_span(cos_lon),
                                    as_span(sin_lon), offset_km, scale_km, band_deg);
        },
        "line"_a, "first"_a, "count"_a, "dn"_a, "cos_lat"_a, "sin_lat"_a, "cos_lon"_a, "sin_lon"_a,
        "offset_km"_a, "scale_km"_a, "band_deg"_a,
        "Install a limb band given as arrays (synthetic bands; app.limb.install_native).");
    m.def(
        "load_limb_band",
        [](const std::string& path) {
            nb::gil_scoped_release nogil;
            eclipse::limb::load_band_file(path);
        },
        "path"_a,
        "Load the LOLA limb band from its file (kernels/limb_band.py format) into the core.");
    m.def("limb_band_source", &eclipse::limb::band_source,
          "The file the installed limb band came from ('' for none or set_limb_band).");
    m.def("has_limb_band", &eclipse::limb::has_band);
    m.def(
        "limb_silhouette",
        [](In2D axes, int n_bins, double distance_km) {
            if (axes.shape(0) != 3 || axes.shape(1) != 3)
                throw std::invalid_argument("limb_silhouette: axes must be 3x3");
            std::array<double, 9> a{};
            for (std::size_t i = 0; i < 9; ++i) a[i] = axes.data()[i];
            std::vector<double> out;
            {
                nb::gil_scoped_release nogil;
                out = eclipse::limb::silhouette(a, n_bins, distance_km);
            }
            return to_numpy(std::move(out));
        },
        "axes"_a, "n_bins"_a = eclipse::limb::N_BINS,
        "distance_km"_a = eclipse::limb::INF_DISTANCE,
        "app.limb.silhouette: delta_rho [km] per bin for view axes (rows x^, y^, z^),\n"
        "in perspective from distance_km (inf: orthographic).");
    m.def("limb_sphere_radius", &eclipse::limb::sphere_radius, "distance_km"_a,
          "app.limb.sphere_radius: the LOLA sphere's apparent radius [km at distance_km].");
    m.def(
        "limb_profiles_at",
        [](In1D et, std::string_view earth_frame, std::string_view moon_frame) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            std::vector<double> flat;
            {
                nb::gil_scoped_release nogil;
                flat = eclipse::limb::profiles_at(as_span(et), f, moon_frame);
            }
            const std::size_t n = et.shape(0);
            auto* heap = new std::vector<double>(std::move(flat));
            nb::capsule owner(heap,
                              [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
            return Out2D(heap->data(), {n, static_cast<std::size_t>(eclipse::limb::N_BINS)}, owner);
        },
        "et"_a, "earth_frame"_a = "ITRS", "moon_frame"_a = "MOON_ME",
        "app.limb.profiles_at: (n, N_BINS) profiles, linear between the cached nodes.");
    for (const char* name : {"limb_g_total", "limb_g_annular"}) {
        const bool total = std::string_view(name) == "limb_g_total";
        m.def(
            name,
            [total](In1D px, In1D py, In1D r_s, In1D r_m, In2D profiles) {
                const auto nb_ = static_cast<int>(profiles.shape(1));
                const std::span<const double> p(profiles.data(), profiles.shape(0) * profiles.shape(1));
                return to_numpy(total ? eclipse::limb::g_total(as_span(px), as_span(py), as_span(r_s),
                                                               as_span(r_m), p, nb_)
                                      : eclipse::limb::g_annular(as_span(px), as_span(py),
                                                                 as_span(r_s), as_span(r_m), p, nb_));
            },
            "px"_a, "py"_a, "r_s"_a, "r_m"_a, "profiles"_a,
            total ? "app.limb.g_total (totality <=> negative)."
                  : "app.limb.g_annular (annularity <=> negative).");
    }
    m.def(
        "limb_delta_rho_at",
        [](In1D profile, In1D psi) {
            return to_numpy(eclipse::limb::delta_rho_at(as_span(profile), as_span(psi)));
        },
        "profile"_a, "psi"_a, "app.limb.delta_rho_at: periodic linear interpolation.");
    m.def(
        "limb_axes",
        [](In1D et, std::string_view earth_frame, std::string_view moon_frame) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            eclipse::ephem::LimbAxes la;
            {
                nb::gil_scoped_release nogil;
                la = eclipse::ephem::limb_axes(as_span(et), f, moon_frame);
            }
            const auto& ax = la.axes;
            std::vector<double> flat;
            flat.reserve(9 * ax.size());
            for (const auto& a : ax) flat.insert(flat.end(), a.begin(), a.end());
            auto* heap = new std::vector<double>(std::move(flat));
            nb::capsule owner(heap,
                              [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
            return nb::make_tuple(Out3D(heap->data(), {ax.size(), 3, 3}, owner),
                                  to_numpy(std::move(la.distance_km)));
        },
        "et"_a, "earth_frame"_a, "moon_frame"_a,
        "app.ephemeris.limb_axes: ((n, 3, 3) rows x^, y^, z^ in moon_frame, (n,) distance_km).");
}

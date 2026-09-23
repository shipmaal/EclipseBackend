// Bindings for eclipse/limb.hpp (app/limb.py) and ephem::limb_axes.
#include <nanobind/stl/string_view.h>

#include <array>
#include <cstdint>
#include <span>
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

template <class T, class A>
std::span<const T> span_of(const A& a) {
    return {a.data(), a.shape(0)};
}

}  // namespace

void bind_limb(nb::module_& m) {
    m.def(
        "set_limb_band",
        [](InI32 line, InI32 first, InI32 count, InI16 dn, InI32 right, InI32 down, In1D cos_lat, In1D sin_lat,
           In1D cos_lon, In1D sin_lon, double offset_km, double scale_km, double band_deg) {
            eclipse::limb::set_band(span_of<std::int32_t>(line), span_of<std::int32_t>(first),
                                    span_of<std::int32_t>(count), span_of<std::int16_t>(dn),
                                    span_of<std::int32_t>(right), span_of<std::int32_t>(down),
                                    as_span(cos_lat), as_span(sin_lat), as_span(cos_lon),
                                    as_span(sin_lon), offset_km, scale_km, band_deg);
        },
        "line"_a, "first"_a, "count"_a, "dn"_a, "right"_a, "down"_a, "cos_lat"_a, "sin_lat"_a, "cos_lon"_a,
        "sin_lon"_a, "offset_km"_a, "scale_km"_a, "band_deg"_a,
        "Install the LOLA limb band (app.limb.install_native).");
    m.def("has_limb_band", &eclipse::limb::has_band);
    m.def(
        "limb_silhouette",
        [](In2D axes, int n_bins) {
            if (axes.shape(0) != 3 || axes.shape(1) != 3)
                throw std::invalid_argument("limb_silhouette: axes must be 3x3");
            std::array<double, 9> a{};
            for (std::size_t i = 0; i < 9; ++i) a[i] = axes.data()[i];
            std::vector<double> out;
            {
                nb::gil_scoped_release nogil;
                out = eclipse::limb::silhouette(a, n_bins);
            }
            return to_numpy(std::move(out));
        },
        "axes"_a, "n_bins"_a = eclipse::limb::N_BINS,
        "app.limb.silhouette: delta_rho [km] per bin for view axes (rows x^, y^, z^).");
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
            std::vector<std::array<double, 9>> ax;
            {
                nb::gil_scoped_release nogil;
                ax = eclipse::ephem::limb_axes(as_span(et), f, moon_frame);
            }
            std::vector<double> flat;
            flat.reserve(9 * ax.size());
            for (const auto& a : ax) flat.insert(flat.end(), a.begin(), a.end());
            auto* heap = new std::vector<double>(std::move(flat));
            nb::capsule owner(heap,
                              [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
            return Out3D(heap->data(), {ax.size(), 3, 3}, owner);
        },
        "et"_a, "earth_frame"_a, "moon_frame"_a,
        "app.ephemeris.limb_axes: (n, 3, 3) rows x^, y^, z^ in moon_frame.");
}

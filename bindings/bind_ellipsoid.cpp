// Bindings for eclipse/ellipsoid.hpp (app.geography reduction: _reduction_aux,
// fund_to_geo_v, geo_to_fund).
//
// Arrays cross as equal-length 1-D float64 (spans in, fresh ndarrays out);
// NumPy broadcasting of the Python signatures is the glue's job on the Python
// side, not the core's. The array forms release the GIL around the compute.
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <nanobind/stl/optional.h>

#include "common.hpp"
#include "eclipse/ellipsoid.hpp"

using namespace nb::literals;
namespace ell = eclipse::ellipsoid;

void bind_ellipsoid(nb::module_& m) {
    m.def(
        "reduction_aux",
        [](double d_rad) {
            const ell::ReductionAux a = ell::reduction_aux(d_rad);
            return nb::make_tuple(a.rho1, a.rho2, a.sin_d1, a.cos_d1, a.sin_d1_d2, a.cos_d1_d2);
        },
        "d_rad"_a,
        "app.geography._reduction_aux at axis declination d_rad [radians]:\n"
        "(rho1, rho2, sin_d1, cos_d1, sin_d1_d2, cos_d1_d2) [ES92] eq. 8.331.");
    m.def(
        "fund_to_geo",
        [](In1D x, In1D y, In1D d_deg, In1D mu_deg) {
            ell::Geographic g;
            {
                nb::gil_scoped_release nogil;
                g = ell::fund_to_geo(as_span(x), as_span(y), as_span(d_deg), as_span(mu_deg));
            }
            return nb::make_tuple(to_numpy(std::move(g.lon_deg)), to_numpy(std::move(g.lat_deg)));
        },
        "x"_a, "y"_a, "d_deg"_a, "mu_deg"_a,
        "app.geography.fund_to_geo_v on equal-length 1-D arrays: (lon_deg, lat_deg),\n"
        "NaN where the axis misses the Earth. x, y in Earth equatorial radii; d, mu in degrees.");
    m.def(
        "geo_to_fund",
        [](In1D lat_deg, In1D lon_deg, In1D d_deg, In1D mu_deg, std::optional<In1D> height_m) {
            ell::Fundamental f;
            {
                nb::gil_scoped_release nogil;
                f = ell::geo_to_fund(as_span(lat_deg), as_span(lon_deg), as_span(d_deg),
                                     as_span(mu_deg),
                                     height_m ? as_span(*height_m) : std::span<const double>{});
            }
            return nb::make_tuple(to_numpy(std::move(f.xi)), to_numpy(std::move(f.eta)),
                                  to_numpy(std::move(f.zeta)));
        },
        "lat_deg"_a, "lon_deg"_a, "d_deg"_a, "mu_deg"_a, "height_m"_a = nb::none(),
        "app.geography.geo_to_fund on equal-length 1-D arrays: (xi, eta, zeta) in Earth\n"
        "equatorial radii. lat, lon, d, mu in degrees; height_m [m] above the WGS-84\n"
        "ellipsoid (None = sea level).");
}

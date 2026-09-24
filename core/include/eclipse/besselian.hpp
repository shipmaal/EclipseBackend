// Besselian elements of one eclipse model (an epoch T0 and an Earth frame):
// direct evaluation at offsets from T0, their rates, the tabular polynomial
// fit that ``/besselian`` publishes, and the central line with its limits
// that ``/central-line`` publishes.
//
// Everything here is built on ``ephem::besselian_instants`` (the apparent,
// frame-rotated elements [ES92] eq. 8.322-8.323) and the ellipsoid / shadow
// geometry of ``ellipsoid.hpp`` / ``geometry.hpp``.
#pragma once

#include <array>
#include <span>
#include <vector>

#include "eclipse/constants.hpp"
#include "eclipse/elements.hpp"

namespace eclipse::besselian {

/// The elements at offsets ``t_hours`` [h] from ``et0`` [TDB s]: one
/// ``ephem::besselian_instants`` call on ``et0 + t * 3600`` with ``mu``
/// unwrapped over the vector (``unwrap_mu_deg``), so it is continuous across
/// a 360-degree wrap. Exact at every instant, inside or outside any fit
/// window. ``k1`` / ``k2``: the penumbral / umbral cones' lunar radii [Earth
/// radii] (default the [Espenak] pair).
Elements elements_direct(double et0, Frame frame, std::span<const double> t_hours,
                         double k1 = constants::K_PENUMBRA, double k2 = constants::K_UMBRA);

/// Per-hour rates of ``x, y, l1, l2`` [Earth radii / h] and ``d, mu``
/// [deg / h] at ``t_hours``: central differences over ``+/- RATE_DT_H``
/// (``geometry.hpp``) of one ``elements_direct`` call on ``[t - dt..., t +
/// dt...]``, the ``mu`` difference wrapped to (-180, 180] first. These are the
/// ``rates`` of the path-limit envelope (``geometry::shadow_edge_limits``).
/// (numerical)
struct Rates {
    std::vector<double> x, y, d, mu, l1, l2;
};
Rates element_rates(double et0, Frame frame, std::span<const double> t_hours);

/// The tabular polynomials of the elements in ``t`` = hours from T0,
/// coefficients in increasing powers (the Espenak / NASA bulletin form):
/// ``x, y`` cubic, ``d, l1, l2`` quadratic, ``mu`` linear, ``tan_f1, tan_f2``
/// constants (the samples' mean).
struct Polynomials {
    std::array<double, 4> x, y;
    std::array<double, 3> d, l1, l2;
    std::array<double, 2> mu;
    double tan_f1, tan_f2;
};

/// Least-squares fit of the elements sampled at ``t = -half_window_hours,
/// ..., +half_window_hours`` in steps of ``step_hours`` (``numerics::arange(
/// -hw, hw + step / 2, step)``), ``mu`` unwrapped before its fit. The fit is
/// the ordinary least-squares polynomial on column-scaled powers of ``t``,
/// solved by Householder QR (the problem ``np.polynomial.polynomial.polyfit``
/// solves; with the default 5 samples the cubic is near-interpolating and the
/// system is well conditioned). Valid only inside the sampling window; for any
/// computation use ``elements_direct``. Throws ``std::invalid_argument`` when
/// there are fewer samples than the cubic's 4 coefficients.
Polynomials fit_polynomials(double et0, Frame frame, double half_window_hours = 2.0,
                            double step_hours = 1.0);

/// Least-squares polynomial of degree ``deg`` through ``(t, y)``: coefficients
/// in increasing powers (Householder QR on the column-scaled Vandermonde
/// matrix). Needs ``t.size() > deg``. (numerical)
std::vector<double> polyfit(std::span<const double> t, std::span<const double> y, int deg);

/// The ``/central-line`` product at offsets ``t_hours`` [h] from ``et0``: the
/// points where the shadow axis meets the WGS-84 ellipsoid, in time order
/// (instants where the axis misses the Earth are dropped), each with
///   * ``t_hours``, ``lat_deg``, ``lon_deg`` (``ellipsoid::fund_to_geo``);
///   * ``bearing_deg``: the along-track great-circle bearing from the previous
///     to the next on-Earth point (clamped at the ends; 0 for a lone point);
///   * ``penumbra_km``, ``umbra_km``, ``is_total`` (``geometry::shadow_radii``);
///   * the umbral / antumbral path limits and ``width_km``: the envelope of
///     the shadow over time (``geometry::shadow_edge_limits`` with the
///     ``element_rates`` at each point, 600 km search), NaN / 0 where absent;
///   * the penumbral limits at the instant (10 000 km search, clipped to the
///     terminator), a visualization bound, NaN where absent.
struct CentralLine {
    std::vector<double> t_hours, lat_deg, lon_deg, bearing_deg;
    std::vector<double> penumbra_km, umbra_km;
    std::vector<char> is_total;
    std::vector<double> north_lat, north_lon, south_lat, south_lon, width_km;
    std::vector<double> pen_north_lat, pen_north_lon, pen_south_lat, pen_south_lon;
};
/// Search distances [km] of the umbral and penumbral limits.
inline constexpr double UMBRA_LIMIT_MAX_KM = 600.0;
inline constexpr double PENUMBRA_LIMIT_MAX_KM = 10000.0;
CentralLine central_line(double et0, Frame frame, std::span<const double> t_hours);

}  // namespace eclipse::besselian

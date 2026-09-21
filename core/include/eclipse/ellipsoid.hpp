// Ellipsoid reduction — the C++ twin of app/geography.py's ``_reduction_aux``,
// ``fund_to_geo_v`` and ``geo_to_fund`` (roadmap §3, §4: parity gate 1e-13).
//
// The reduction carries the WGS-84 flattening into the fundamental plane
// through the auxiliary declinations ``d1`` (and ``d1 - d2``) and the radii
// ``rho1``, ``rho2`` [ES92] eq. 8.331; a fundamental-plane point is mapped to
// the auxiliary sphere and back [ES92] eq. 8.331-8.334, and the parametric
// (reduced) latitude is exchanged with the geodetic one through
// ``tan(phi) = tan(phi1) / (1 - f)`` [Meeus98] ch. 11, eq. 11.1.
//
// Every expression is written in the Python's exact floating-point operation
// order (NumPy evaluates left to right; ``**2`` is ``v * v``), so a residual
// above the 1-ulp libm-vs-NumPy noise is a bug, not a tolerance to widen
// (roadmap §7). Header-only, pure math: no SPICE, no ERFA.
//
// Units: angles in **degrees** at this API (``d_deg``, ``mu_deg``, ``lat_deg``,
// ``lon_deg``), radians internally (``reduction_aux`` takes radians exactly as
// ``_reduction_aux`` does); ``x, y, xi, eta, zeta`` in Earth equatorial radii.
// Python-side broadcasting is the binding glue's job: the span forms take
// equal-length arrays only.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "eclipse/constants.hpp"
#include "eclipse/numerics.hpp"

namespace eclipse::ellipsoid {

/// Auxiliary quantities of the ellipsoid reduction at one axis declination
/// (``app.geography._ReductionAux``): the radii ``rho1``, ``rho2`` and the
/// sines/cosines of the auxiliary declinations ``d1`` and ``d1 - d2``
/// [ES92] eq. 8.331. Dimensionless.
struct ReductionAux {
    double rho1, rho2, sin_d1, cos_d1, sin_d1_d2, cos_d1_d2;
};

/// Mirrors ``_reduction_aux(d)``: ``d_rad`` is the axis declination in
/// **radians**. [ES92] eq. 8.331 with ``e^2`` of WGS-84 [WGS84]:
///   rho1 = sqrt(1 - e^2 cos^2 d),  rho2 = sqrt(1 - e^2 sin^2 d),
///   sin d1 = sin d / rho1,         cos d1 = sqrt(1 - e^2) cos d / rho1,
///   sin(d1 - d2) = e^2 sin d cos d / (rho1 rho2),
///   cos(d1 - d2) = sqrt(1 - e^2) / (rho1 rho2).
inline ReductionAux reduction_aux(double d_rad) {
    using constants::WGS84_E2;
    const double cos_d = std::cos(d_rad), sin_d = std::sin(d_rad);
    const double root = std::sqrt(1.0 - WGS84_E2);
    const double rho1 = std::sqrt(1.0 - WGS84_E2 * (cos_d * cos_d));
    const double rho2 = std::sqrt(1.0 - WGS84_E2 * (sin_d * sin_d));
    return ReductionAux{
        rho1,
        rho2,
        sin_d / rho1,
        root * cos_d / rho1,
        WGS84_E2 * sin_d * cos_d / (rho1 * rho2),
        root / (rho1 * rho2),
    };
}

/// Geographic position, degrees; longitude east positive in [-180, 180).
struct LonLat {
    double lon_deg, lat_deg;
};

/// Elementwise body of ``fund_to_geo_v``: fundamental-plane ``(x, y)`` [Earth
/// equatorial radii] at axis declination ``d_deg`` and Greenwich hour angle
/// ``mu_deg`` [degrees] -> ``(lon, lat)`` [degrees]. Both fields are NaN when
/// the axis misses the ellipsoid (``1 - x^2 - eta1^2 < 0``, the Python's
/// ``np.where(disc >= 0, disc, nan)``) and NaN inputs propagate; ``sin(phi1)``
/// is clipped to [-1, 1] as the vectorized Python does (the raising scalar
/// ``fund_to_geo`` is not ported).
///
/// Reduction to the auxiliary sphere [ES92] eq. 8.331-8.332 (``eta1 = y / rho1``,
/// ``zeta1 = sqrt(1 - x^2 - eta1^2)``), fundamental plane -> geocentric
/// direction at latitude ``d1`` [ES92] eq. 8.333 (``sin phi1``, hour angle
/// ``theta`` east of the axis meridian), parametric -> geodetic latitude
/// [Meeus98] ch. 11, eq. 11.1, and ``lon = theta - mu`` wrapped [ES92] eq. 8.334.
inline LonLat fund_to_geo_one(double x, double y, double d_deg, double mu_deg) {
    using constants::DEG_TO_RAD;
    using constants::RAD_TO_DEG;
    using constants::WGS84_F;
    const ReductionAux aux = reduction_aux(d_deg * DEG_TO_RAD);

    const double eta1 = y / aux.rho1;
    const double disc = 1.0 - x * x - eta1 * eta1;
    const double zeta1 = disc >= 0.0 ? std::sqrt(disc) : std::numeric_limits<double>::quiet_NaN();

    const double sin_phi1 = eta1 * aux.cos_d1 + zeta1 * aux.sin_d1;
    const double theta = std::atan2(x, zeta1 * aux.cos_d1 - eta1 * aux.sin_d1);
    const double phi1 = std::asin(numerics::np_clip(sin_phi1, -1.0, 1.0));
    const double phi = std::atan(std::tan(phi1) / (1.0 - WGS84_F));

    // (degrees(theta) - mu + 180) % 360 - 180 with NumPy's %.
    const double lon = numerics::wrap_180(theta * RAD_TO_DEG - mu_deg);
    return LonLat{lon, phi * RAD_TO_DEG};
}

/// Struct-of-arrays result of ``fund_to_geo`` [degrees].
struct Geographic {
    std::vector<double> lon_deg, lat_deg;
};

/// ``fund_to_geo_v`` over equal-length arrays (throws ``std::invalid_argument``
/// otherwise). Units as ``fund_to_geo_one``.
inline Geographic fund_to_geo(std::span<const double> x, std::span<const double> y,
                              std::span<const double> d_deg, std::span<const double> mu_deg) {
    const std::size_t n = x.size();
    if (y.size() != n || d_deg.size() != n || mu_deg.size() != n)
        throw std::invalid_argument("fund_to_geo: x, y, d_deg, mu_deg differ in length");
    Geographic out;
    out.lon_deg.resize(n);
    out.lat_deg.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const LonLat p = fund_to_geo_one(x[i], y[i], d_deg[i], mu_deg[i]);
        out.lon_deg[i] = p.lon_deg;
        out.lat_deg[i] = p.lat_deg;
    }
    return out;
}

/// Fundamental-plane coordinates of a point on the ellipsoid [Earth equatorial
/// radii]: ``xi, eta`` in the plane, ``zeta`` along the shadow axis (positive
/// towards the Moon).
struct Fund {
    double xi, eta, zeta;
};

/// Elementwise ``geo_to_fund``: geodetic ``(lat, lon)`` [degrees] on the WGS-84
/// ellipsoid at axis declination ``d_deg`` and Greenwich hour angle ``mu_deg``
/// [degrees] -> ``(xi, eta, zeta)`` [Earth equatorial radii]. The exact inverse
/// of ``fund_to_geo_one`` on the near side: geodetic -> parametric latitude
/// ``tan(beta) = (1 - f) tan(phi)`` [Meeus98] ch. 11, eq. 11.1, hour angle
/// ``theta = lon + mu`` east of the axis meridian, then the inverse ellipsoid
/// reduction [ES92] eq. 8.331 (``eta = eta1 rho1``,
/// ``zeta = rho2 (zeta1 cos(d1 - d2) - eta1 sin(d1 - d2))``).
inline Fund geo_to_fund_one(double lat_deg, double lon_deg, double d_deg, double mu_deg) {
    using constants::DEG_TO_RAD;
    using constants::WGS84_F;
    const ReductionAux aux = reduction_aux(d_deg * DEG_TO_RAD);

    const double beta = std::atan((1.0 - WGS84_F) * std::tan(lat_deg * DEG_TO_RAD));
    const double theta = (lon_deg + mu_deg) * DEG_TO_RAD;
    const double cb = std::cos(beta), sb = std::sin(beta);

    const double xi = cb * std::sin(theta);
    const double eta1 = sb * aux.cos_d1 - cb * std::cos(theta) * aux.sin_d1;
    const double zeta1 = sb * aux.sin_d1 + cb * std::cos(theta) * aux.cos_d1;
    const double eta = eta1 * aux.rho1;
    const double zeta = aux.rho2 * (zeta1 * aux.cos_d1_d2 - eta1 * aux.sin_d1_d2);
    return Fund{xi, eta, zeta};
}

/// Struct-of-arrays result of ``geo_to_fund`` [Earth equatorial radii].
struct Fundamental {
    std::vector<double> xi, eta, zeta;
};

/// ``geo_to_fund`` over equal-length arrays (throws ``std::invalid_argument``
/// otherwise). Units as ``geo_to_fund_one``.
inline Fundamental geo_to_fund(std::span<const double> lat_deg, std::span<const double> lon_deg,
                               std::span<const double> d_deg, std::span<const double> mu_deg) {
    const std::size_t n = lat_deg.size();
    if (lon_deg.size() != n || d_deg.size() != n || mu_deg.size() != n)
        throw std::invalid_argument("geo_to_fund: lat_deg, lon_deg, d_deg, mu_deg differ in length");
    Fundamental out;
    out.xi.resize(n);
    out.eta.resize(n);
    out.zeta.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Fund p = geo_to_fund_one(lat_deg[i], lon_deg[i], d_deg[i], mu_deg[i]);
        out.xi[i] = p.xi;
        out.eta[i] = p.eta;
        out.zeta[i] = p.zeta;
    }
    return out;
}

}  // namespace eclipse::ellipsoid

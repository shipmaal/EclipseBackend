// Shadow geometry — the C++ twin of app/geography.py's ``shadow_radii``, the
// great-circle helpers (``_destination``, ``_haversine_km``, ``bearing``),
// ``shadow_edge_limits_v`` and ``global_contacts`` (roadmap §3, §4: parity
// gates 1e-9 deg / 1e-6 km for the limits, 1e-9 h for the contacts).
//
// Pure math on the WGS-84 ellipsoid [WGS84] and, for the short perpendicular
// offsets of the limit search, on a sphere of the IUGG mean radius; the
// ellipsoid reduction itself is ``eclipse/ellipsoid.hpp`` ([ES92] eq.
// 8.331-8.334), never re-derived here. ``global_contacts`` evaluates the
// Besselian elements through an injected ``ElementsAt`` (kernel-free tests) or,
// in its second overload, through ``ephem::besselian_instants`` — the only
// path by which this unit reaches SPICE, and only through ``ephem``.
//
// Every expression is written in the Python's exact floating-point operation
// order (NumPy evaluates left to right; ``**2`` is ``v * v``; ``np.hypot`` is
// ``std::hypot``), so a residual above the 1-ulp libm-vs-NumPy noise is a bug,
// not a tolerance to widen (roadmap §7).
//
// Units: angles in **degrees** at this API (``d_deg``, ``mu_deg``, latitudes,
// longitudes, bearings), radians internally; ``x, y, l, l1, l2`` in Earth
// equatorial radii; distances in **km**; contact times in **hours from T0**.
// The span forms take equal-length arrays only (NumPy broadcasting is the
// binding glue's job) and throw ``std::invalid_argument`` otherwise.
#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "eclipse/elements.hpp"

namespace eclipse::geometry {

// ------------------------------------------------------------ shadow radii

/// Result of ``shadow_radii``: ground radii in **km** and the total/annular flag.
struct ShadowRadii {
    double penumbra_km;  ///< |L1| a, the penumbra radius at the observer [km]
    double umbra_km;     ///< |L2| a, the umbra / antumbra radius [km]
    bool is_total;       ///< L2 < 0: the umbra reaches the ground (total); else annular
};

/// Mirrors ``app.geography.shadow_radii``: approximate penumbra / umbra radii on
/// the ground at the central point ``(x, y)`` [Earth equatorial radii] for axis
/// declination ``d_deg`` [degrees], cone radii ``l1, l2`` [Earth radii] and cone
/// tangents ``tan_f1, tan_f2``. The cone radii are reduced to the observer's
/// distance below the fundamental plane, ``L = l - zeta tan f`` [ES92] eq.
/// 8.353, with ``zeta = sqrt(1 - x^2 - eta1^2)`` on the auxiliary sphere [ES92]
/// eq. 8.331-8.332; ``(0, 0, false)`` when the axis misses the Earth.
/// Adequate for visualization; the true limits are ``shadow_edge_limits``.
ShadowRadii shadow_radii(double x, double y, double d_deg, double l1, double l2, double tan_f1,
                         double tan_f2);

// ------------------------------------------------------ great-circle helpers
// Standard spherical trigonometry on a sphere of radius
// ``constants::EARTH_MEAN_RADIUS_KM``: destination point, haversine distance
// and initial bearing. Used only for the short (<~600 km) perpendicular
// offsets of the limit search; the reduction is on the full WGS-84 ellipsoid.

/// Geographic position [degrees]; longitude east positive in [-180, 180).
struct LatLon {
    double lat_deg, lon_deg;
};

/// Mirrors ``app.geography._destination``: the point ``dist_km`` [km] along the
/// great circle leaving ``(lat_deg, lon_deg)`` [degrees] at ``bearing_deg``
/// [degrees clockwise from north]. Longitude is wrapped with NumPy's ``%``;
/// the ``asin`` argument is NOT clipped (exactly as the Python).
LatLon destination(double lat_deg, double lon_deg, double bearing_deg, double dist_km);

/// Mirrors ``app.geography._haversine_km``: great-circle distance [km] between
/// ``(lat1, lon1)`` and ``(lat2, lon2)`` [degrees].
double haversine_km(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg);

/// Mirrors ``app.geography.bearing``: initial great-circle bearing [degrees,
/// [0, 360)] from point 1 to point 2 [degrees].
double bearing_deg(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg);

/// ``bearing`` over equal-length arrays [degrees].
std::vector<double> bearing_deg(std::span<const double> lat1_deg, std::span<const double> lon1_deg,
                                std::span<const double> lat2_deg, std::span<const double> lon2_deg);

// ------------------------------------------------------- shadow-edge limits

/// Struct-of-arrays result of ``shadow_edge_limits``: the northern and southern
/// edge points [degrees] (NaN where not found) and the width [km] (0 there).
struct EdgeLimits {
    std::vector<double> north_lat, north_lon, south_lat, south_lon, width_km;
};

/// Mirrors ``app.geography.shadow_edge_limits_v``: for each track instant, the
/// two shadow-edge points perpendicular to ``path_bearing_deg`` and the width
/// between them. ``x, y`` [Earth equatorial radii], ``d_deg, mu_deg`` [degrees],
/// ``l, tan_f`` the cone radius [Earth radii] and tangent of the shadow searched
/// (umbra: l2, tan_f2; penumbra: l1, tan_f1), ``path_bearing_deg`` the
/// along-track bearing [degrees]; all equal-length. Each search marches
/// ``max_km`` [km] from the central point at ``bearing -/+ 90`` and bisects
/// (50 fixed halvings, its own loop — not ``numerics::bisect``) the edge
/// condition ``|axis separation| - |l - zeta tan f| = 0`` [ES92] / [MeeusSE]
/// on the WGS-84 ellipsoid (``ellipsoid::geo_to_fund_one``). ``sunlit_only``
/// treats points beyond the terminator (``zeta <= 0``) as outside the shadow,
/// clipping a night-side penumbral limit to the terminator. Entries are NaN
/// (width 0) where the central point is outside the shadow, the axis misses the
/// Earth, or an edge is not found within ``max_km``; the northern point is the
/// one with the greater latitude.
EdgeLimits shadow_edge_limits(std::span<const double> x, std::span<const double> y,
                              std::span<const double> d_deg, std::span<const double> mu_deg,
                              std::span<const double> l, std::span<const double> tan_f,
                              std::span<const double> path_bearing_deg, double max_km = 600.0,
                              bool sunlit_only = true);

// ---------------------------------------------------------- global contacts

/// One global contact: ``name`` in {P1, P4, U1, U4, U2, U3}, ``t_hours`` from T0.
struct Contact {
    std::string name;
    double t_hours;
};

/// Vectorized element evaluation at offsets ``t_hours`` from T0 [hours]
/// (``BesselianModel.evaluate_direct``); only ``x, y, d, l1, l2`` are read.
using ElementsAt = std::function<Elements(std::span<const double> t_hours)>;

/// Mirrors ``app.geography.global_contacts``: the instants [hours from T0] the
/// penumbra (P1/P4) and umbra (U1-U4) first and last touch the Earth's outline.
/// In the fundamental plane the outline is the unit auxiliary circle in
/// ``(x, y/rho1)`` [ES92] eq. 8.331, so the external contacts are where the
/// axis distance equals ``1 + (cone radius)`` and the internal umbral contacts
/// where it equals ``1 - |l2|`` [ES92] sec. 8.34 (the cone radius unscaled by
/// rho1 is the standard bulletin approximation). Roots are bracketed on a
/// 1-minute grid over ``+/- half_window_hours`` (``numerics::arange``, first
/// falling / last rising sign change per condition) and refined with
/// ``numerics::bisect``. Returned in the Python dict's insertion order
/// ``P1, P4, U1, U4, U2, U3`` with absent contacts omitted — never sorted.
std::vector<Contact> global_contacts(const ElementsAt& elements_at, double half_window_hours = 5.0);

/// ``global_contacts`` with the elements evaluated by
/// ``ephem::besselian_instants(et0 + t * 3600.0, frame)`` — what
/// ``BesselianModel.evaluate_direct`` does for a model at ``et0`` [TDB s].
std::vector<Contact> global_contacts(double et0, Frame frame, double half_window_hours = 5.0);

}  // namespace eclipse::geometry

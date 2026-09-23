// Local circumstances — the C++ twin of app/circumstances.py (roadmap §3, §4:
// parity gates contacts 1e-9 h, magnitude 1e-12; grid 1e-12; §5 phase 3).
//
// From the Besselian elements evaluated directly from the ephemeris at each
// instant (``BesselianModel.evaluate_direct``) and an observer's geodetic
// position, what that observer sees: the partial contacts C1/C4, the central
// contacts C2/C3 and their duration, the time and magnitude of maximum
// eclipse, the obscuration (covered fraction of the Sun's area) and the Sun's
// altitude/azimuth.
//
// Method ([ES92] sec. 8.35, eq. 8.353-8.354; [MeeusSE]): the observer's
// fundamental-plane position ``(xi, eta, zeta)`` (``ellipsoid::geo_to_fund_one``,
// [ES92] eq. 8.331) gives the separation ``m`` from the shadow axis; the
// penumbral/umbral cone radii reduced to the observer are
// ``L1' = l1 - zeta tan f1`` and ``L2' = l2 - zeta tan f2``. Contacts are where
// ``m`` equals ``L1'`` (partial) or ``|L2'|`` (central); the eclipse magnitude
// is ``(L1' - m) / (L1' + L2')`` [Espenak].
//
// Horizon: the cone geometry is valid on the whole ellipsoid, including the
// night side (``zeta < 0``), where it describes the shadow cone continued
// through the Earth. Contacts are therefore always computed geometrically and
// then checked against the Sun's altitude: any event with the Sun below the
// rise/set altitude ``h0 = -0 deg 50'`` [Meeus98] ch. 15 is flagged ``below``,
// and an observer for whom C1, maximum and C4 are all below it sees no eclipse
// (``eclipse = false``). Geometric only otherwise: no refraction on the
// contacts; the mean lunar limb (folded into k2), or with ``profile`` the
// central phase from the LOLA limb profile (docs/LIMB_PROFILE.md).
//
// Every expression is written in the Python's exact floating-point operation
// order (NumPy evaluates left to right; ``np.hypot`` is ``std::hypot``), so a
// residual above the 1-ulp libm-vs-NumPy noise is a bug, not a tolerance to
// widen (roadmap §7). Pure math: no SPICE here. The ephemeris enters only
// through the injected ``Model`` callables, and ``model_from_ephem`` is the one
// place that binds them to ``ephem::besselian_instants`` /
// ``ephem::sub_solar_points`` — so ``circumstances_grid`` can run its observer
// loop in parallel (OpenMP when ``ECLIPSE_HAVE_OPENMP``) after evaluating the
// elements once, serially, under the SPICE lock.
//
// Units: angles in **degrees** at this API (latitudes, longitudes, altitudes,
// azimuths, ``d``, ``mu``), radians internally; ``x, y, xi, eta, zeta, m, l1,
// l2, L1', L2'`` in Earth equatorial radii; times in **hours from T0**
// (``t_hours``) except ``et0`` [TDB seconds past J2000] and ``step_minutes``.
// The span forms take equal-length arrays only and throw
// ``std::invalid_argument`` otherwise.
//
// Contract note: ``mu`` reaching this unit must ALREADY be the unwrapped value
// (``unwrap_mu_deg``, i.e. ``np.degrees(np.unwrap(np.radians(mu)))``) — the
// round trip is not bit-identity even without a wrap, and the Python oracle
// only ever sees the unwrapped series (elements.hpp).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "eclipse/elements.hpp"
#include "eclipse/ephem.hpp"
#include "eclipse/geometry.hpp"

namespace eclipse::circumstances {

// ---------------------------------------------------------------- constants

/// Sun rise/set altitude [deg]: refraction 34' + solar semidiameter 16'
/// [Meeus98] ch. 15 (``app.circumstances.HORIZON_ALT_DEG``).
inline constexpr double HORIZON_ALT_DEG = -0.8333;

/// Contact bracketing cap [hours]: the sampling half-window is widened (x1.5
/// per step) until the observer is outside the penumbra at both ends, up to
/// this ceiling — a partial eclipse lasts at most a few hours at one site
/// (``_MAX_HALF_WINDOW_HOURS``).
inline constexpr double MAX_HALF_WINDOW_HOURS = 6.0;

/// The 30-second contact grid [hours], written as the Python writes it
/// (``_GRID_STEP_HOURS = 0.5 / 60.0``) so ``numerics::arange`` sees the same
/// double.
inline constexpr double GRID_STEP_HOURS = 0.5 / 60.0;

// -------------------------------------------------------------------- model

/// Vectorized element evaluation at offsets ``t_hours`` from T0 [hours], the
/// ``BesselianModel.evaluate_direct`` contract: ``x, y, d, mu, l1, l2, tan_f1,
/// tan_f2`` all read, and ``mu`` ALREADY unwrapped (``unwrap_mu_deg``).
using geometry::ElementsAt;

/// Sub-solar point (lon, lat) [deg] at offsets ``t_hours`` from T0 [hours]:
/// ``sub_solar_points(et0 + t * 3600.0, frame)`` (``app.ephemeris``).
using SubSolarAt = std::function<ephem::SubSolar(std::span<const double> t_hours)>;

/// Limb profiles at offsets ``t_hours`` from T0 [hours]: row-major n x
/// ``limb::N_BINS`` (``app.limb.profiles_at(et0 + t * 3600, frame)``).
using ProfilesAt = std::function<std::vector<double>(std::span<const double> t_hours)>;

/// The slice of ``BesselianModel`` the circumstances read: direct element
/// evaluation, the sub-solar point (for the horizon check) and the fit
/// half-window [hours] that seeds ``bracketed_series`` and bounds the grid.
/// The polynomial fit itself stays in Python (roadmap §4). The limb-profile
/// mode also reads ``elements_ref_at`` (the elements with ``limb::K_REF``
/// for both cones, ``evaluate_direct(t, K_REF, K_REF)``) and ``profiles_at``.
struct Model {
    ElementsAt elements_at;
    SubSolarAt sub_solar_at;
    double half_window_hours;
    ElementsAt elements_ref_at = nullptr;
    ProfilesAt profiles_at = nullptr;
};

/// A ``Model`` bound to the ephemeris exactly as ``BesselianModel`` at
/// ``et0`` [TDB s] in ``frame`` evaluates: ``elements_at(t)`` is
/// ``ephem::besselian_instants(et0 + t * 3600.0, frame)`` followed by
/// ``unwrap_mu_deg`` (``evaluate_direct``), ``sub_solar_at(t)`` is
/// ``ephem::sub_solar_points(et0 + t * 3600.0, frame)``. Both take the SPICE
/// lock inside ``ephem``; nothing else in this unit does.
Model model_from_ephem(double et0, Frame frame, double half_window_hours);

// ------------------------------------------------------------------- series

/// ``_series``: per instant, the observer's axis separation ``m``, the reduced
/// cone radii ``L1'``, ``L2'`` [Earth equatorial radii] and the magnitude
/// ``(L1' - m) / (L1' + L2')`` [ES92] eq. 8.353-8.354, [Espenak].
struct Series {
    std::vector<double> m, L1p, L2p, mag;
};

/// ``_series`` for one observer ``(lat_deg, lon_deg)`` [deg] over the N
/// instants of ``e`` (``mu`` unwrapped): ``(xi, eta, zeta) =
/// geo_to_fund(lat, lon, d, mu)``, ``m = hypot(xi - x, eta - y)``,
/// ``L1' = l1 - zeta tan_f1``, ``L2' = l2 - zeta tan_f2``. N-vectors out.
Series series(const Elements& e, double lat_deg, double lon_deg);

/// ``_series`` for P observers over the N instants of ``e`` — the Python's
/// ``(P, 1) x (N,)`` broadcast — as P x N row-major vectors (observer-major:
/// element ``[p * N + i]``). ``lat_deg`` and ``lon_deg`` [deg] are
/// equal-length. Implementations hoist the per-observer parametric latitude
/// and the per-instant ``reduction_aux`` through the hoisted
/// ``ellipsoid::geo_to_fund_one`` overload (bit-identical to the scalar one).
Series series(const Elements& e, std::span<const double> lat_deg,
              std::span<const double> lon_deg);

// -------------------------------------------------------------- obscuration

/// ``_overlap_area``: area of intersection of two circles of radii ``r <= R``
/// at centre distance ``d`` (any consistent unit; area in its square). The
/// circular-segment (lens) area: ``0`` when ``d >= r + R``, ``pi r^2`` when
/// ``d <= R - r``, else ``r^2 acos(a) + R^2 acos(b) - tri`` with ``a``, ``b``
/// the clipped cosine terms and ``tri`` the kite half-area
/// ``0.5 sqrt(max(0, (-d + r + R)(d + r - R)(d - r + R)(d + r + R)))`` —
/// in the Python's ``np.where`` order (the lens is evaluated unconditionally;
/// NaN from the degenerate branches never reaches the result). (numerical)
double overlap_area(double r, double R, double d);

/// ``_obscuration``: covered fraction of the Sun's AREA [Espenak] (distinct
/// from the magnitude, a diameter ratio) from the reduced cone radii ``L1'``,
/// ``L2'`` and the separation ``m`` [Earth equatorial radii]: with
/// ``denom = L1' + L2'`` (``safe = denom > 0 ? denom : 1.0``), the Moon/Sun
/// radius ratio ``q = (L1' - L2') / safe`` and centre separation
/// ``sep = 2 m / safe`` in Sun radii, ``overlap_area(min(q, 1), max(q, 1),
/// sep) / pi``; ``0`` where ``denom <= 0`` (``np_minimum`` / ``np_maximum``).
double obscuration(double L1p, double L2p, double m);

// ------------------------------------------------------------------ horizon

/// Sun altitude and azimuth [deg].
struct AltAz {
    double alt_deg, az_deg;
};

/// ``_sun_altaz`` given the sub-solar point ``(ss_lon, ss_lat)`` [deg] at the
/// instant: geodetic altitude from the spherical
/// ``cos c = sin phi sin delta + cos phi cos delta cos H`` with
/// ``H = radians(ss_lon - lon)`` (ONE ``radians`` of the difference),
/// ``alt = 90 - degrees(acos(clip(cos c, -1, 1)))``; azimuth is
/// ``geometry::bearing_deg(lat, lon, ss_lat, ss_lon)``. The sub-solar latitude
/// is the Sun's declination and the ellipsoid normal the zenith, so geodetic
/// ``phi`` is exact (solar parallax < 9" neglected).
AltAz sun_altaz(double lat_deg, double lon_deg, double ss_lon_deg, double ss_lat_deg);

/// ``circumstances_grid``'s OWN altitude expression [deg], kept separate from
/// ``sun_altaz`` because its rounding order differs: the hour angle is
/// ``radians(ss_lon) - radians(lon)`` (two ``radians``, then the difference),
/// not ``radians(ss_lon - lon)``. Same ``cos c`` and ``90 - degrees(acos(
/// clip(...)))`` otherwise. No azimuth.
double sun_alt_grid(double lat_deg, double lon_deg, double ss_lon_deg, double ss_lat_deg);

// -------------------------------------------------------------------- roots

/// One linear-interpolated zero crossing of a sampled function
/// (``_roots``): ``t`` [hours], ``rising`` and the grid index ``i`` of the
/// bracket ``[t[i], t[i + 1]]``.
struct Root {
    double t;
    bool rising;
    std::size_t i;
};

/// ``_roots(t, f)``: for each ``numerics::sign_changes`` bracket ``i``, the
/// crossing ``t[i]`` itself when ``f[i] == 0``, else
/// ``t[i] - f[i] * (t[i + 1] - t[i]) / (f[i + 1] - f[i])`` (left to right:
/// ``t[i] - ((f[i] * dt) / df)``). Equal-length spans. (numerical)
std::vector<Root> roots(std::span<const double> t, std::span<const double> f);

// --------------------------------------------------------------- bracketing

/// ``_bracketed_series`` result: the 30-second offsets ``t`` [hours] and the
/// series on them.
struct Bracketed {
    std::vector<double> t;
    Series s;
};

/// ``_bracketed_series``: sample the series on ``numerics::arange(-hw, hw +
/// 1e-9, GRID_STEP_HOURS)`` starting at ``model.half_window_hours`` and widen
/// (``hw = min(hw * 1.5, MAX_HALF_WINDOW_HOURS)``, re-sampling each time) until
/// ``m - L1' > 0`` at both ends (observer outside the penumbra) or
/// ``hw >= MAX_HALF_WINDOW_HOURS``. Direct evaluation stays exact outside the
/// fit window, so C1/C4 are bracketed even when the partial phase runs past
/// it.
Bracketed bracketed_series(const Model& model, double lat_deg, double lon_deg);

/// ``_refine_contacts``: ``numerics::bisect`` (30 iterations) of the contact
/// conditions inside the 30-s brackets ``[t_lo[k], t_hi[k]]`` [hours]; where
/// ``central[k]`` the objective is ``m - |L2'|`` (C2/C3), else ``m - L1'``
/// (C1/C4). All brackets are refined together, one ``series`` per iteration.
/// Equal-length spans; returns the refined times [hours].
std::vector<double> refine_contacts(const Model& model, double lat_deg, double lon_deg,
                                    std::span<const double> t_lo, std::span<const double> t_hi,
                                    std::span<const std::uint8_t> central);

/// ``_refine_maximum``: parabolic refinement of the time of maximum [hours]
/// through the three grid points around ``imax`` — ``denom = y0 - 2 y1 + y2``
/// and, when ``denom < 0`` (a maximum), ``t[imax] + 0.5 * GRID_STEP_HOURS *
/// (y0 - y2) / denom`` (left to right); ``t[imax]`` at the ends or when the
/// fit is not a maximum. (numerical)
double refine_maximum(std::span<const double> t, std::span<const double> mag, std::size_t imax);

// -------------------------------------------------------- local circumstances

/// The raw numbers of ``local_circumstances`` — the Python ``_LocalRaw``
/// NamedTuple, SAME FIELD ORDER, before any rounding / clock formatting (which
/// stays in Python, ``_format_local``). Events are ordered C1, max, C4, C2, C3
/// in ``alt_deg`` / ``az_deg`` / ``below``.
///
/// ``geometric`` is false when fewer than two partial roots were found or the
/// peak magnitude is ``<= 0`` (the Python's early ``{"eclipse": False}``): then
/// every other field is a placeholder (NaN / false). Otherwise ``central`` says
/// the observer is inside the umbra/antumbra at maximum with BOTH C2 and C3
/// bracketed (``c2``, ``c3`` NaN and the C2/C3 event slots NaN / false when
/// not), ``magnitude`` is the diameter ratio ``(L1' - L2') / (L1' + L2')``
/// when central and ``(L1' - m) / (L1' + L2')`` otherwise [Espenak],
/// ``obscuration`` is ``obscuration(L1', L2', m)`` at the refined maximum,
/// ``L2_x`` is ``L2'`` there (its sign gives total / annular), ``below[k]`` is
/// ``alt_deg[k] <= HORIZON_ALT_DEG`` over the present events, and ``eclipse``
/// is false when every present event is below the horizon.
struct LocalRaw {
    bool geometric, central;
    double c1, c4, c2, c3, t_max, magnitude, obscuration, L2_x;
    std::array<double, 5> alt_deg, az_deg;
    std::array<bool, 5> below;
    bool eclipse;
};

/// Profile-mode search constants (``app.circumstances._PROFILE_*``) [hours, km].
inline constexpr double PROFILE_STEP_H = 0.5 / 3600.0;
inline constexpr double PROFILE_MARGIN_H = 30.0 / 3600.0;
inline constexpr double PROFILE_GRAZE_H = 3.0 / 60.0;
inline constexpr double PROFILE_NEAR_KM = 12.0;
inline constexpr int PROFILE_MAX_WIDEN = 10;

/// ``_profile_g``: the limb-profile contact function at offsets ``t_hours``
/// (``G_T`` where ``L2' < 0``, else ``G_A``; negative = central phase), from
/// ``model.elements_ref_at`` and ``model.profiles_at``.
std::vector<double> profile_g(const Model& model, double lat_deg, double lon_deg,
                              std::span<const double> t_hours);

/// ``_profile_contacts``: ``(central, c2, c3)`` [hours] in ``[t_lo, t_hi]``,
/// a central window end pushed out by ``PROFILE_MARGIN_H`` up to
/// ``PROFILE_MAX_WIDEN`` times.
struct ProfileContacts {
    bool central;
    double c2, c3;
};
ProfileContacts profile_contacts(const Model& model, double lat_deg, double lon_deg, double t_lo,
                                 double t_hi);

/// ``_local_raw``: local circumstances of the observer ``(lat_deg, lon_deg)``
/// [deg]. T0 should be near the observer's maximum. ``bracketed_series``,
/// ``roots`` of ``m - L1'`` (first falling -> C1, last rising -> C4),
/// ``argmax_first`` of the magnitude, ``roots`` of ``m - |L2'|`` when
/// ``m < |L2'|`` at that sample (last entering root at or before ``t[imax]``
/// -> C2, first exiting at or after -> C3), ``refine_contacts`` of all
/// brackets together, ``refine_maximum``, the series re-evaluated at the
/// refined maximum, and ``sun_altaz`` at every event.
/// ``profile`` (``limb="profile"``) takes the central phase from
/// ``profile_contacts`` as the Python does.
LocalRaw local_circumstances(const Model& model, double lat_deg, double lon_deg,
                             bool profile = false);

/// ``local_circumstances`` through ``model_from_ephem(et0, frame,
/// half_window_hours)`` — what ``/circumstances`` computes for a
/// ``BesselianModel`` at ``et0`` [TDB s].
LocalRaw local_circumstances(double et0, Frame frame, double half_window_hours, double lat_deg,
                             double lon_deg, bool profile = false);

// --------------------------------------------------------------------- grid

/// ``circumstances_grid`` result, one entry per observer: ``magnitude`` (0
/// where no eclipse; the diameter ratio where central), ``obscuration`` (0
/// where no eclipse), ``t_max_hours`` (the GRID instant of maximum, no
/// refinement) [hours], ``sun_alt`` at that instant [deg] (``sun_alt_grid``),
/// ``visible`` (eclipse and ``sun_alt > HORIZON_ALT_DEG``) and ``central``
/// (eclipse and ``m < |L2'|`` at maximum). Booleans as 0/1 bytes, handed to
/// NumPy as ``bool`` arrays.
struct GridResult {
    std::vector<double> magnitude, obscuration, t_max_hours, sun_alt;
    std::vector<std::uint8_t> visible, central;
};

/// ``circumstances_grid``: maximum-eclipse circumstances for P observers
/// (``lat_deg``, ``lon_deg`` [deg], equal-length) sampled on ``numerics::
/// arange(-hw, hw + 1e-9, step_minutes / 60.0)`` with ``hw =
/// model.half_window_hours`` (the model's window must cover the eclipse
/// globally, ~+/-3 h): per observer ``imax = argmax_first(mag)``, ``has = mag
/// > 0``, ``central = has && m < |L2'|`` at ``imax``. Same geometry as
/// ``local_circumstances`` minus the contacts and refinement. The elements and
/// sub-solar points are evaluated ONCE for the whole grid (serially, SPICE
/// lock), then the observer loop is parallel: ``threads`` 0 = the OpenMP
/// default, 1 = serial, n = that many; results are identical for any thread
/// count (no reductions across observers). Serial when built without OpenMP.
GridResult circumstances_grid(const Model& model, std::span<const double> lat_deg,
                              std::span<const double> lon_deg, double step_minutes = 2.0,
                              int threads = 0);

/// ``circumstances_grid`` through ``model_from_ephem(et0, frame,
/// half_window_hours)`` — what ``/map`` computes for a ``BesselianModel`` at
/// ``et0`` [TDB s].
GridResult circumstances_grid(double et0, Frame frame, double half_window_hours,
                              std::span<const double> lat_deg, std::span<const double> lon_deg,
                              double step_minutes = 2.0, int threads = 0);

}  // namespace eclipse::circumstances

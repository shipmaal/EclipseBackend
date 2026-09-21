// Eclipse catalog — the C++ twin of app/catalog.py (roadmap §3, §4: parity
// gate "identical rows (times to 1e-6 s)" against the 2019–2024 canon; §5
// phase 4: a century scan with detail in < 10 s).
//
// The search is the evaluation, over a coarse time grid, of the shadow axis'
// distance from the Earth's centre in the fundamental plane,
// ``rho = hypot(x, y)`` [ES92] eq. 8.322-6: it has a local minimum at every
// syzygy, and a solar eclipse occurs when that minimum is below ``1 + l1`` (the
// penumbra reaches the Earth) [ES92] sec. 8.34 AND the Moon is on the Sun's
// side of the Earth (``z > 0``; a full moon shares the minimum). ``rho`` and
// ``z`` are invariant under any rotation of the Earth-orientation frame, so
// the scan and the refinement read them from ``ephem::axis_separation`` (the
// same eq. 8.322-6 on the un-rotated J2000 vectors: no nutation, no EOP);
// only the elements AT greatest eclipse and everything after them use the
// configured ``Frame``. Each minimum is refined by ``numerics::
// parabolic_minimum`` to the instant of greatest eclipse (the axis' closest
// approach to the Earth's centre [Espenak]) and classified: ``gamma`` the
// signed ``rho`` (positive north of the centre) [Espenak]; ``central`` when
// the axis meets the ellipsoid (``ellipsoid::fund_to_geo``); the kind
// ``partial`` when the umbra never touches (``rho > 1 + |l2|``), else
// ``total`` / ``annular`` by the sign of the umbral radius at the ground
// ``L2' = l2 - zeta tan f2`` and ``hybrid`` when that sign changes along the
// central line [Espenak] (Canon codes P/T/A/H) — at the ends of the central
// line ``zeta = 0`` and ``L2' = l2``, so a path total at greatest eclipse is
// hybrid exactly when ``l2 > 0`` at either end; the magnitude at greatest
// eclipse is the Moon/Sun diameter ratio for a central eclipse and the
// covered diameter fraction at the closest limb point otherwise [Espenak].
//
// This unit is pure math plus the ``ephem`` calls named below; the ephemeris
// enters through the injected ``Sources`` callables so the scan can be tested
// kernel-free, and ``sources_from_ephem`` is the one place binding them.
//
// PORT, DON'T IMPROVE (roadmap §7). The Python oracle's operation order and
// its rules, which every implementation of this header keeps:
//
//  * grid: ``numerics::arange(et_a - COARSE_STEP_S, et_b + 2 * COARSE_STEP_S,
//    COARSE_STEP_S)`` — NumPy's fill rule, the oracle's exact expression
//    (``_coarse_grid``). The oracle evaluates ``rho`` in chunks of
//    ``COARSE_CHUNK`` instants; chunking cannot change a value.
//  * candidates: ``rho[i] < rho[i-1] && rho[i] <= rho[i+1] && rho[i] <
//    CANDIDATE_RHO`` for ``1 <= i <= n-2`` (``_local_minima``: strict below the
//    previous sample, at or below the next — a plateau's first sample — and
//    ``+inf`` far-side samples never qualify), in grid order, never sorted.
//  * refinement: ``numerics::parabolic_minimum(rho_at, cand, COARSE_STEP_S)``
//    (10 iterations, all candidates in one call) on the SAME ``RhoAt`` as the
//    scan, ``+inf`` where ``z <= 0`` (``np.where(z > 0, rho, inf)``).
//  * elements at ``et_g``: ``besselian_instants(et_g, frame)`` on the whole
//    vector, ``mu`` RAW and wrapped (NOT ``unwrap_mu_deg``); ``rho_g =
//    hypot(x, y)``; ``(lon_g, lat_g) = fund_to_geo(x, y, d, mu)`` (NaN = not
//    central).
//  * keep: ``et_a <= et_g && et_g <= et_b`` (inclusive) and NOT ``rho_g > 1 +
//    l1``; skipped candidates leave no trace.
//  * central: ``zeta`` from ``ellipsoid::geo_to_fund_one(lat_g, lon_g, d, mu)``
//    with the UNROUNDED point and the raw wrapped ``mu``; ``L1' = l1 - zeta *
//    tan_f1``, ``L2' = l2 - zeta * tan_f2``; kind ``L2' < 0 ? Total :
//    Annular``; magnitude ``(L1' - L2') / (L1' + L2')``.
//  * non-central: ``rho_g > 1.0 + |l2|`` -> Partial with magnitude ``(l1 -
//    (rho_g - 1.0)) / (l1 + l2)`` (that bracket order); else ``l2 < 0 ? Total
//    : Annular`` with ``(l1 - l2) / (l1 + l2)``.
//  * hybrid (central only, after the kind): false when ``L2' >= 0``; else the
//    elements at ``et_g + arange(-HYBRID_HALF_SPAN_H, HYBRID_HALF_SPAN_H +
//    1e-9, HYBRID_STEP_H)[i] * 3600.0`` in ``frame``, ``fund_to_geo`` on all,
//    ``on`` = the on-Earth indices; false when none; else ``max(l2[on.front()],
//    l2[on.back()]) > 0.0``.
//  * ``gamma = copysign(rho_g, y)``.
//  * detail (``add_detail``): ``et0 = utc_to_et(et_to_utc(et_g))`` — the
//    oracle re-parses the WHOLE-SECOND string the row reports, so ``et0`` is
//    not ``et_g``; contacts = ``geometry::global_contacts(et0, frame,
//    CONTACT_HALF_WINDOW_H)`` in insertion order; when central, ``local =
//    circumstances::local_circumstances(et0, frame, DETAIL_HALF_WINDOW_H,
//    py_round(lat_g, 2), py_round(lon_g, 2))`` (the ROUNDED point the row
//    reports) and the width from the oracle's three-point ``central_track``:
//    elements at ``et0 + {-TRACK_DT_H, 0, +TRACK_DT_H} * 3600`` WITH
//    ``unwrap_mu_deg`` (``evaluate_direct``), ``fund_to_geo`` on all, the
//    on-Earth points in order, each point's bearing =
//    ``geometry::bearing_deg`` from its previous on-Earth neighbour to its
//    next (indices clamped to the ends; ``0.0`` when only one point is on
//    Earth), then for the point with ``|t| < 1e-9`` (absent -> no width)
//    ``geometry::shadow_edge_limits`` of ONE instant with ``l2``, ``tan_f2``,
//    that bearing, ``UMBRA_LIMIT_MAX_KM``, ``sunlit_only = true``; the width
//    is ``0.0`` when the limits are NaN (the scalar oracle's ``(None, None,
//    0.0)``), never absent once a ``t = 0`` point exists.
//
// CONDITIONING OF ``et_g`` (measured in work package A, see app.catalog._rho_at):
// ``rho`` is quadratic and flat at its minimum, so an objective that differs
// by ``delta`` (1 ulp of ``rho``, ~4e-14, is what any two frames or libms give)
// moves the refined instant by ``~sqrt(2 delta / rho'')`` with ``rho'' ~ 8e-8``
// Earth radii/s^2, i.e. ~1 ms: the greatest-eclipse instant is determined
// only to the millisecond by double precision. Switching the Python scan from
// the ITRS elements to ``axis_separation`` moved ``et_g`` by <= 3.4 ms over
// 114 eclipses (2000-2050) and flipped ONE whole-second row of 226 in
// 2000-2100 (2000-07-31 02:13:03.4997 -> 03.5005). The native parity gate is
// therefore: rows identical, ``|et_g - et_g_oracle| <= 1e-2 s``, and a
// whole-second ``greatest_utc`` difference of exactly 1 s is admissible only
// when the oracle's ``et_g`` lies within that band of a rounding boundary —
// the roadmap's "times to 1e-6 s" is not attainable for ``et_g`` itself.
//
// Units: ``et`` [TDB seconds past J2000]; ``rho``, ``gamma``, ``l1``, ``l2``,
// ``zeta`` [Earth equatorial radii]; latitudes / longitudes [degrees]; contact
// and local times [hours from ``et0``]; ``width_km`` [km]. Every expression in
// the Python's exact floating-point operation order; a residual above the
// 1-ulp libm-vs-NumPy noise is a bug, not a tolerance to widen (roadmap §7).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "eclipse/circumstances.hpp"
#include "eclipse/elements.hpp"
#include "eclipse/geometry.hpp"

namespace eclipse::catalog {

// ---------------------------------------------------------------- constants
// Every value is the Python's (app/catalog.py), written as the Python writes
// it so both sides see the same double.

/// Coarse scan step [s] (``_COARSE_STEP_S``): rho changes by < 0.6 Earth radii
/// per hour, so a 2-hour grid cannot miss a minimum.
inline constexpr double COARSE_STEP_S = 2.0 * 3600.0;

/// Instants per ``rho_at`` call in the oracle's scan (``_COARSE_CHUNK``): a
/// memory bound only; values are independent of it.
inline constexpr std::size_t COARSE_CHUNK = 50000;

/// Candidate threshold [Earth radii] (``_CANDIDATE_RHO``): a coarse minimum is
/// at most the true minimum + 1.2, so keep it when below 1 + l1 + margin.
inline constexpr double CANDIDATE_RHO = 2.6;

/// ``_hybrid`` samples the elements over +/- this [h] around greatest eclipse ...
inline constexpr double HYBRID_HALF_SPAN_H = 3.5;
/// ... every 5 minutes [h] (``5.0 / 60.0``), through ``arange(-span, span +
/// 1e-9, step)``.
inline constexpr double HYBRID_STEP_H = 5.0 / 60.0;

/// Fit half-window [h] of the detail model (``BesselianModel(...,
/// half_window_hours=2.5)``): seeds ``circumstances::bracketed_series``.
inline constexpr double DETAIL_HALF_WINDOW_H = 2.5;

/// ``global_contacts`` window [h] (its default ``half_window_hours``).
inline constexpr double CONTACT_HALF_WINDOW_H = 5.0;

/// Offsets [h] of the three-point ``central_track`` whose middle point's
/// bearing orients the width search (``dt = 0.02``).
inline constexpr double TRACK_DT_H = 0.02;

/// ``shadow_edge_limits`` march distance [km] for the umbral width (its
/// default ``max_km``); ``sunlit_only`` is true there.
inline constexpr double UMBRA_LIMIT_MAX_KM = 600.0;

// ------------------------------------------------------------------ sources

/// The scan / refinement objective: ``et`` [TDB s] -> ``rho`` [Earth radii],
/// ``+inf`` where ``z <= 0`` (``_rho_at``). Vectorized: one call per
/// ``parabolic_minimum`` iteration on the stacked ``3n`` vector.
using RhoAt = std::function<std::vector<double>(std::span<const double> et)>;

/// Elements at absolute instants ``et`` [TDB s] in the catalog's frame —
/// ``besselian_instants(et, frame)`` with ``mu`` wrapped, NOT unwrapped (the
/// oracle never unwraps in the classification; only the detail's
/// ``evaluate_direct`` path does, and ``add_detail`` handles that itself).
using ElementsAtEt = std::function<Elements(std::span<const double> et)>;

/// What the catalog reads from the ephemeris: the frame-free objective, the
/// frame-bound elements, and the frame (for ``add_detail``, which builds the
/// detail model through ``ephem`` / ``geometry`` / ``circumstances``).
struct Sources {
    RhoAt rho_at;
    ElementsAtEt elements_at;
    Frame frame;
};

/// ``_rho_at`` through ``ephem::axis_separation``: ``z > 0 ? hypot(x, y) :
/// +inf`` per instant. The SPICE lock is taken inside ``ephem``.
std::vector<double> rho_at(std::span<const double> et);

/// ``Sources`` bound to the ephemeris exactly as ``app.catalog`` reads it:
/// ``rho_at`` above and ``ephem::besselian_instants(et, frame)``.
Sources sources_from_ephem(Frame frame);

// --------------------------------------------------------------------- scan

/// ``_coarse_grid``: ``numerics::arange(et_a - COARSE_STEP_S, et_b + 2 *
/// COARSE_STEP_S, COARSE_STEP_S)`` [TDB s].
std::vector<double> coarse_grid(double et_a, double et_b);

/// ``_local_minima``: indices ``i`` (``1 <= i <= n - 2``) with ``rho[i] <
/// rho[i - 1] && rho[i] <= rho[i + 1] && rho[i] < CANDIDATE_RHO``, in grid
/// order. Empty for ``n < 3``.
std::vector<std::size_t> local_minima(std::span<const double> rho);

/// ``_scan_candidates``: the coarse-grid instants [TDB s] of every candidate
/// minimum — ``coarse_grid`` evaluated through ``rho_at`` (in
/// ``COARSE_CHUNK`` pieces or not; same values) and ``local_minima``.
std::vector<double> scan_candidates(double et_a, double et_b, const RhoAt& rho_at);

/// The refined instants of greatest eclipse [TDB s]: ``numerics::
/// parabolic_minimum(rho_at, cand_et, COARSE_STEP_S)`` (10 iterations, one
/// objective call per iteration for all candidates).
std::vector<double> refine_greatest(std::span<const double> cand_et, const RhoAt& rho_at);

// ----------------------------------------------------------- classification

/// Canon type codes P / A / T / H (``EclipseEvent["type"]``).
enum class Kind : std::uint8_t { Partial, Annular, Total, Hybrid };

/// The oracle's row strings: "partial", "annular", "total", "hybrid".
inline const char* to_string(Kind k) {
    switch (k) {
    case Kind::Partial: return "partial";
    case Kind::Annular: return "annular";
    case Kind::Total: return "total";
    case Kind::Hybrid: return "hybrid";
    }
    return "?";
}

/// ``_classify`` result: the kind BEFORE the hybrid test (never ``Hybrid``),
/// the magnitude at greatest eclipse [Espenak], and ``L2'`` at the
/// greatest-eclipse ground point [Earth radii] (NaN when not central; the
/// input ``hybrid`` reads).
struct Classification {
    Kind kind;
    double magnitude;
    double L2p;
};

/// ``_classify(rho_g, l1, l2, tan_f1, tan_f2, central, zeta)`` — the rules
/// in the header comment ("central", "non-central"). ``zeta`` [Earth radii]
/// is read only when ``central``.
Classification classify(double rho_g, double l1, double l2, double tan_f1, double tan_f2,
                        bool central, double zeta);

/// ``_hybrid(et_g, L2p_greatest, frame)`` through ``elements_at`` (header
/// comment, "hybrid"): true when a path total at greatest eclipse is annular
/// at either end of the central line.
bool hybrid(double et_g, double L2p_greatest, const ElementsAtEt& elements_at);

// ------------------------------------------------------------------- detail

/// ``_detail_raw``'s numbers (the Python ``_EventRaw`` fields ``et0``,
/// ``contacts``, ``local``, ``width_km``): ``et0`` [TDB s] the detail model's
/// epoch, ``contacts`` in ``global_contacts`` insertion order [hours from
/// ``et0``], ``local`` the ``LocalRaw`` at the rounded greatest point (absent
/// when not central) and ``width_km`` the umbral path width there [km]
/// (absent when not central or when no track point is on Earth at ``t = 0``).
struct Detail {
    double et0;
    std::vector<geometry::Contact> contacts;
    std::optional<circumstances::LocalRaw> local;
    std::optional<double> width_km;
};

/// ``_detail_raw`` for one event (header comment, "detail"): reaches the
/// ephemeris through ``ephem::et_to_utc`` / ``ephem::utc_to_et`` (the
/// whole-second re-parse), ``geometry::global_contacts(et0, frame, ...)``,
/// ``circumstances::local_circumstances(et0, frame, ...)`` and
/// ``ephem::besselian_instants`` for the track. ``lat_g_deg`` / ``lon_g_deg``
/// are the UNROUNDED greatest point [deg] (ignored when not central).
Detail add_detail(double et_g, Frame frame, bool central, double lat_g_deg, double lon_g_deg);

// -------------------------------------------------------------------- event

/// One eclipse before any rounding / formatting — the Python ``_EventRaw``
/// (SAME FIELD ORDER: ``et_g, kind, central, gamma, magnitude, lat, lon``,
/// then the ``Detail`` fields). ``lat_deg`` / ``lon_deg`` are the UNROUNDED
/// greatest-eclipse point (NaN when not central); the row's ``round(.., 2)``,
/// ``round(gamma, 4)``, ``round(magnitude, 4)``, ``round(width, 1)``, the
/// clock strings and the varying dict shape stay in Python
/// (``app.catalog._format_event``).
struct Event {
    double et_g;
    Kind kind;
    bool central;
    double gamma;
    double magnitude;
    double lat_deg, lon_deg;
    std::optional<Detail> detail;
};

/// ``_catalog_raw(et_a, et_b, frame, detail)``: every eclipse with greatest
/// eclipse in ``[et_a, et_b]`` [TDB s] (inclusive), in scan order, through the
/// injected ``Sources`` (whose ``frame`` the detail uses). ``threads`` bounds
/// the OpenMP threads of the per-event hybrid / detail loop when built with
/// OpenMP (0 = the default, 1 = serial); the ephemeris calls inside serialize
/// on the SPICE lock and the results are identical for any thread count.
std::vector<Event> find_eclipses(double et_a, double et_b, const Sources& sources, bool detail,
                                 int threads = 0);

/// ``find_eclipses`` through ``sources_from_ephem(frame)`` — what
/// ``/eclipses`` computes (the range check and the UTC parsing stay in Python).
std::vector<Event> find_eclipses(double et_a, double et_b, Frame frame, bool detail,
                                 int threads = 0);

}  // namespace eclipse::catalog

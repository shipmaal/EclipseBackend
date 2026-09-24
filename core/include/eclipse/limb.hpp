// limb — the lunar limb profile from the LRO LOLA DEM (docs/LIMB_PROFILE.md;
// app/limb.py is the oracle, bit-identical). The core owns the DEM limb band:
// it reads the band file itself (``load_band_file``, 2 bytes a point; no
// per-point neighbour indices) and keeps it read-only, so every function here
// is lock-free and thread-safe.
//
// ``psi`` is the angle in the fundamental plane from x^ (east) toward y^
// (north) [rad]; a profile is ``n_bins`` values of delta_rho [km] (silhouette
// radius minus R_REF_KM), bin k covering [-pi + k w, -pi + (k + 1) w).
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "eclipse/elements.hpp"

namespace eclipse::limb {

/// LOLA reference sphere (LDEM label OFFSET) [LOLA].
inline constexpr double R_REF_KM = 1737.4;
/// 0.05 deg bins (docs/LIMB_PROFILE.md sec. 3.2).
inline constexpr int N_BINS = 7200;
/// A peak shows over the limb from <= 6.2 deg behind it (h = 10 km).
inline constexpr double VISIBLE_DEG = 8.0;
/// Points projecting further below R_REF_KM cannot be on the silhouette.
inline constexpr double FLOOR_KM = 20.0;

/// More empty bins than this fraction is an error (app.limb.MAX_EMPTY_FRACTION).
inline constexpr double MAX_EMPTY_FRACTION = 0.1;
/// One lunar radius for both cones in profile mode: the LOLA sphere in Earth
/// equatorial radii (``app.limb.K_REF``; [LOLA], [WGS84] a).
inline constexpr double K_REF = R_REF_KM / 6378.137;
/// Profile time lattice [s of TDB] (``app.limb.LATTICE_S``).
inline constexpr double LATTICE_S = 300.0;
inline constexpr double INF_DISTANCE = std::numeric_limits<double>::infinity();

/// Load the limb band from a file written by ``kernels/limb_band.py`` (format
/// ``ECLLIMB1``: header JSON, row runs, int16 DN), computing the pixel-centre
/// trig tables with libm in ``app.limb.band_from_file``'s order. This is how
/// the band normally reaches the core (``app.limb.ensure_native_band``): the
/// core owns it, 2 bytes a point plus the runs and tables. Replaces any
/// previous band; throws ``std::runtime_error`` for a missing or malformed file.
void load_band_file(const std::string& path);

/// The path the installed band was loaded from (empty: none, or ``set_band``).
std::string band_source();

/// Install a band given as arrays (synthetic bands in tests,
/// ``app.limb.install_native``): per run ``line``, ``first`` sample and
/// ``count`` (runs in grid order), DN per point in run order, per-line
/// cos/sin latitude and per-sample cos/sin longitude of pixel centres;
/// radius = offset_km + DN * scale_km. Grid neighbours are implied by the
/// runs. Replaces any previous band.
void set_band(std::span<const std::int32_t> line, std::span<const std::int32_t> first,
              std::span<const std::int32_t> count, std::span<const std::int16_t> dn,
              std::span<const double> cos_lat, std::span<const double> sin_lat,
              std::span<const double> cos_lon, std::span<const double> sin_lon,
              double offset_km, double scale_km, double band_deg);
bool has_band();

/// ``app.limb.silhouette``: delta_rho per bin for the view ``axes`` (row-major
/// rows x^, y^, z^ in the DEM frame, ``ephem::limb_axes``) in perspective from
/// ``distance_km`` along -z^ (apparent radius ``|q| / (1 + w / D)``,
/// ``w = p . z^``; infinity = orthographic): the max over the projected points
/// and grid edges (``app.limb._edge_crossings``), minus ``sphere_radius(D)``. Throws
/// ``std::invalid_argument`` if z^ leaves the band's coverage or more than
/// MAX_EMPTY_FRACTION of the bins are empty (fewer are filled by periodic
/// linear interpolation, app.limb._fill_empty).
/// OpenMP-parallel (per-thread maxima merged; identical for any thread count).
std::vector<double> silhouette(const std::array<double, 9>& axes, int n_bins = N_BINS,
                               double distance_km = INF_DISTANCE);

/// Fill the empty (non-finite) bins of a profile in place by periodic linear
/// interpolation between the nearest filled bins on either side,
/// ``a + (b - a) * (m / L)`` for the m-th of the L steps from bin a to bin b
/// (across the -pi/pi seam too). Throws ``std::invalid_argument`` when more than
/// ``MAX_EMPTY_FRACTION`` of the bins are empty. (numerical)
void fill_empty(std::vector<double>& rho);

/// ``app.limb.sphere_radius``: ``R / sqrt(1 - (R / D)^2)``, the LOLA sphere's
/// apparent radius in km at distance ``D`` (tangent cone).
double sphere_radius(double distance_km);

/// ``app.limb.profiles_at``: n profiles (row-major n x N_BINS) at TDB ``et``,
/// linear in time between cached ``LATTICE_S`` nodes (each node the
/// silhouette along ``ephem::limb_axes`` of ``node * LATTICE_S``). The cache
/// is keyed by node, frames, the installed band and the kernel-pool and EOP
/// generations (``ephem::kernel_generation``, ``eop::generation``), so a
/// reload of any of them is never served a stale profile; thread-safe.
std::vector<double> profiles_at(std::span<const double> et, Frame frame,
                                std::string_view moon_frame = "MOON_ME");

/// ``app.limb.g_total``: per instant i, ``max_phi |Q| - r_M(arg Q)`` with
/// ``Q = P + R_s e(phi)`` at the bin centres; ``profiles`` row-major n x
/// ``n_bins``. Totality <=> negative (docs/LIMB_PROFILE.md sec. 3.3).
std::vector<double> g_total(std::span<const double> px, std::span<const double> py,
                            std::span<const double> r_s, std::span<const double> r_m,
                            std::span<const double> profiles, int n_bins = N_BINS);

/// ``app.limb.g_annular``: per instant, ``max_psi |M(psi) - P| - R_s``,
/// ``M = r_M(psi) e(psi)`` at the bin centres. Annularity <=> negative.
std::vector<double> g_annular(std::span<const double> px, std::span<const double> py,
                              std::span<const double> r_s, std::span<const double> r_m,
                              std::span<const double> profiles, int n_bins = N_BINS);

/// ``app.limb.delta_rho_at``: periodic linear interpolation at bin centres.
std::vector<double> delta_rho_at(std::span<const double> profile, std::span<const double> psi);

}  // namespace eclipse::limb

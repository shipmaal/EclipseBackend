// limb — the lunar limb profile from the LRO LOLA DEM (port of app/limb.py;
// docs/LIMB_PROFILE.md). Pure math: the DEM limb band is injected once by
// Python (``set_band``, the same pattern as the EOP table) and is read-only
// afterwards, so every function here is lock-free and thread-safe.
//
// ``psi`` is the angle in the fundamental plane from x^ (east) toward y^
// (north) [rad]; a profile is ``n_bins`` values of delta_rho [km] (silhouette
// radius minus R_REF_KM), bin k covering [-pi + k w, -pi + (k + 1) w).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

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

/// Install the limb band (``app.limb.install_native``): per run ``line``,
/// ``first`` sample and ``count``; DN per point in run order; each point's
/// east (``j + 1`` mod samples) and southern (``i + 1``) neighbour index or
/// -1 (``app.limb._neighbours``); per-line
/// cos/sin latitude and per-sample cos/sin longitude of pixel centres;
/// radius = offset_km + DN * scale_km. Replaces any previous band.
void set_band(std::span<const std::int32_t> line, std::span<const std::int32_t> first,
              std::span<const std::int32_t> count, std::span<const std::int16_t> dn,
              std::span<const std::int32_t> right, std::span<const std::int32_t> down,
              std::span<const double> cos_lat, std::span<const double> sin_lat,
              std::span<const double> cos_lon, std::span<const double> sin_lon,
              double offset_km, double scale_km, double band_deg);

bool has_band();

/// ``app.limb.silhouette``: delta_rho per bin for the view ``axes`` (row-major
/// rows x^, y^, z^ in the DEM frame, ``ephem::limb_axes``): the max over the
/// projected points and grid edges (``app.limb._edge_crossings``). Throws
/// ``std::invalid_argument`` if z^ leaves the band's coverage or more than
/// MAX_EMPTY_FRACTION of the bins are empty (fewer are filled by periodic
/// linear interpolation, app.limb._fill_empty).
/// OpenMP-parallel (per-thread maxima merged; identical for any thread count).
std::vector<double> silhouette(const std::array<double, 9>& axes, int n_bins = N_BINS);

/// ``app.limb.delta_rho_at``: periodic linear interpolation at bin centres.
std::vector<double> delta_rho_at(std::span<const double> profile, std::span<const double> psi);

}  // namespace eclipse::limb

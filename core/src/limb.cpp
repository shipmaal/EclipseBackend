// See eclipse/limb.hpp. Arithmetic follows app/limb.py's NumPy evaluation
// order exactly (roadmap §7); the bin maximum is order-independent.
#include "eclipse/limb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "eclipse/constants.hpp"
#include "eclipse/eop.hpp"
#include "eclipse/ephem.hpp"

namespace eclipse::limb {

namespace {

// The band as the silhouette reads it: row runs, DN in run order and the
// pixel-centre trig tables. No per-point neighbour indices: two points are
// grid neighbours exactly when both are in the band, which the silhouette's
// row buffers already know (``Proj::line``), so the DEM costs 2 bytes a point.
struct Band {
    std::vector<std::int32_t> line, first, count;
    std::vector<std::size_t> offset;  // first point of each run
    std::vector<std::size_t> line_runs;  // runs of line i: [line_runs[i], line_runs[i + 1])
    std::vector<std::int16_t> dn;
    std::vector<double> cos_lat, sin_lat, cos_lon, sin_lon;
    double offset_km = 0.0, scale_km = 0.0, band_deg = 0.0;
    std::string source;  // the band file's path; empty for set_band
};

std::mutex& band_mutex() {
    static std::mutex m;
    return m;
}

std::shared_ptr<const Band>& band_slot() {
    static std::shared_ptr<const Band> b;
    return b;
}

// Bumped by every set_band (under band_mutex): the profile cache's key.
std::uint64_t& band_generation() {
    static std::uint64_t g = 0;
    return g;
}

std::shared_ptr<const Band> band() {
    std::scoped_lock lock(band_mutex());
    auto b = band_slot();
    if (!b) throw std::logic_error("eclipse::limb: limb band not set (call set_band first)");
    return b;
}

}  // namespace

void fill_empty(std::vector<double>& rho) {
    const std::size_t n = rho.size();
    std::vector<std::size_t> filled;
    for (std::size_t k = 0; k < n; ++k)
        if (std::isfinite(rho[k])) filled.push_back(k);
    if (filled.size() == n) return;
    if (static_cast<double>(n - filled.size()) > MAX_EMPTY_FRACTION * static_cast<double>(n) ||
        filled.empty())
        throw std::invalid_argument(
            "eclipse::limb::silhouette: limb profile has too many empty bins: n_bins too fine for "
            "this DEM");
    const std::vector<double> in = rho;
    for (std::size_t f = 0; f < filled.size(); ++f) {
        const std::size_t a_i = filled[f], b_i = filled[(f + 1) % filled.size()];
        const std::size_t gap = (b_i + n - a_i) % n;
        if (gap <= 1) continue;
        const double a = in[a_i], b = in[b_i];
        for (std::size_t m = 1; m < gap; ++m)
            rho[(a_i + m) % n] = a + (b - a) * (static_cast<double>(m) / static_cast<double>(gap));
    }
}


namespace {

// Validate the runs against the grid, index them and publish the band (bumps
// the generation, so cached profiles of a previous band are never reused).
void install(std::shared_ptr<Band> b) {
    b->offset.clear();
    b->offset.reserve(b->line.size());
    std::size_t total = 0;
    for (std::size_t r = 0; r < b->line.size(); ++r) {
        if (b->line[r] < 0 || static_cast<std::size_t>(b->line[r]) >= b->cos_lat.size() ||
            b->first[r] < 0 || b->count[r] < 0 ||
            static_cast<std::size_t>(b->first[r]) + static_cast<std::size_t>(b->count[r]) >
                b->cos_lon.size())
            throw std::invalid_argument("eclipse::limb: band run outside the DEM grid");
        if (r > 0 && (b->line[r] < b->line[r - 1] ||
                      (b->line[r] == b->line[r - 1] &&
                       b->first[r] < b->first[r - 1] + b->count[r - 1])))
            throw std::invalid_argument("eclipse::limb: band runs must be in grid order");
        b->offset.push_back(total);
        total += static_cast<std::size_t>(b->count[r]);
    }
    if (total != b->dn.size())
        throw std::invalid_argument("eclipse::limb: band runs do not cover the DN array");
    b->line_runs.assign(b->cos_lat.size() + 1, 0);
    for (const std::int32_t li : b->line) ++b->line_runs[static_cast<std::size_t>(li) + 1];
    for (std::size_t i = 1; i < b->line_runs.size(); ++i) b->line_runs[i] += b->line_runs[i - 1];
    std::scoped_lock lock(band_mutex());
    band_slot() = std::move(b);
    ++band_generation();
}

// ---- the limb-band file (kernels/limb_band.py; format ECLLIMB1) ----------

// Little-endian integers assembled from bytes: independent of the host's
// byte order and of alignment (the DN array starts at an odd offset).
template <class T>
T read_le(const std::vector<char>& raw, std::size_t& pos) {
    static_assert(std::is_integral_v<T>);
    if (pos + sizeof(T) > raw.size()) throw std::runtime_error("truncated");
    using U = std::make_unsigned_t<T>;
    U u = 0;
    for (std::size_t k = 0; k < sizeof(T); ++k)
        u = static_cast<U>(u | (static_cast<U>(static_cast<unsigned char>(raw[pos + k])) << (8 * k)));
    pos += sizeof(T);
    return static_cast<T>(u);
}

// A number from the header's flat JSON object (``json.dumps(sort_keys=True,
// separators=(",", ":"))``): ``"key":<number>``. Python writes floats with
// repr (shortest round trip), so strtod gives back the same double.
double header_number(const std::string& json, const std::string& key) {
    const std::string pat = "\"" + key + "\":";
    const std::size_t at = json.find(pat);
    if (at == std::string::npos) throw std::runtime_error("header has no " + key);
    const char* begin = json.c_str() + at + pat.size();
    char* end = nullptr;
    const double v = std::strtod(begin, &end);
    if (end == begin) throw std::runtime_error("header " + key + " is not a number");
    return v;
}

}  // namespace

void set_band(std::span<const std::int32_t> line, std::span<const std::int32_t> first,
              std::span<const std::int32_t> count, std::span<const std::int16_t> dn,
              std::span<const double> cos_lat, std::span<const double> sin_lat,
              std::span<const double> cos_lon, std::span<const double> sin_lon,
              double offset_km, double scale_km, double band_deg) {
    if (line.size() != first.size() || line.size() != count.size() ||
        cos_lat.size() != sin_lat.size() || cos_lon.size() != sin_lon.size())
        throw std::invalid_argument("eclipse::limb::set_band: mismatched array lengths");
    auto b = std::make_shared<Band>();
    b->line.assign(line.begin(), line.end());
    b->first.assign(first.begin(), first.end());
    b->count.assign(count.begin(), count.end());
    b->dn.assign(dn.begin(), dn.end());
    b->cos_lat.assign(cos_lat.begin(), cos_lat.end());
    b->sin_lat.assign(sin_lat.begin(), sin_lat.end());
    b->cos_lon.assign(cos_lon.begin(), cos_lon.end());
    b->sin_lon.assign(sin_lon.begin(), sin_lon.end());
    b->offset_km = offset_km;
    b->scale_km = scale_km;
    b->band_deg = band_deg;
    install(std::move(b));
}

void load_band_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("eclipse::limb: cannot open limb band " + path);
    const std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto b = std::make_shared<Band>();
    try {
        if (raw.size() < 8 || std::memcmp(raw.data(), "ECLLIMB1", 8) != 0)
            throw std::runtime_error("not a limb-band file");
        std::size_t pos = 8;
        const auto hlen = read_le<std::uint32_t>(raw, pos);
        if (pos + hlen > raw.size()) throw std::runtime_error("truncated");
        const std::string json(raw.data() + pos, hlen);
        pos += hlen;
        if (header_number(json, "format_version") != 1.0)
            throw std::runtime_error("unsupported limb-band format");
        const auto n_runs = read_le<std::uint32_t>(raw, pos);
        b->line.resize(n_runs);
        b->first.resize(n_runs);
        b->count.resize(n_runs);
        for (std::uint32_t r = 0; r < n_runs; ++r) {
            b->line[r] = read_le<std::int32_t>(raw, pos);
            b->first[r] = read_le<std::int32_t>(raw, pos);
            b->count[r] = read_le<std::int32_t>(raw, pos);
        }
        const auto n_pts = read_le<std::uint32_t>(raw, pos);
        if (pos + 2 * static_cast<std::size_t>(n_pts) != raw.size())
            throw std::runtime_error("truncated or inconsistent");
        b->dn.resize(n_pts);
        for (auto& v : b->dn) v = read_le<std::int16_t>(raw, pos);

        // Pixel centres, PDS map projection of the LDEM label (kernels/limb_band.py):
        // latitude (LINE_PROJECTION_OFFSET - i) / ppd, east longitude
        // (j - SAMPLE_PROJECTION_OFFSET) / ppd + CENTER_LONGITUDE, in degrees, then
        // radians and libm cos/sin -- app.limb.band_from_file's order.
        const double ppd = header_number(json, "map_resolution");
        const double lpo = header_number(json, "line_projection_offset");
        const double spo = header_number(json, "sample_projection_offset");
        const double clon = header_number(json, "center_longitude");
        const auto lines = static_cast<std::size_t>(header_number(json, "lines"));
        const auto samples = static_cast<std::size_t>(header_number(json, "line_samples"));
        b->cos_lat.resize(lines);
        b->sin_lat.resize(lines);
        for (std::size_t i = 0; i < lines; ++i) {
            const double lat = ((lpo - static_cast<double>(i)) / ppd) * constants::DEG_TO_RAD;
            b->cos_lat[i] = std::cos(lat);
            b->sin_lat[i] = std::sin(lat);
        }
        b->cos_lon.resize(samples);
        b->sin_lon.resize(samples);
        for (std::size_t j = 0; j < samples; ++j) {
            const double lon = ((static_cast<double>(j) - spo) / ppd + clon) * constants::DEG_TO_RAD;
            b->cos_lon[j] = std::cos(lon);
            b->sin_lon[j] = std::sin(lon);
        }
        // radius_m = DN * SCALING_FACTOR + OFFSET (label), here in km.
        b->offset_km = header_number(json, "offset") / 1000.0;
        b->scale_km = header_number(json, "scaling_factor") / 1000.0;
        b->band_deg = header_number(json, "band_deg");
    } catch (const std::runtime_error& e) {
        throw std::runtime_error("eclipse::limb: " + path + ": " + e.what());
    }
    b->source = path;
    install(std::move(b));
}

std::string band_source() {
    std::scoped_lock lock(band_mutex());
    return band_slot() ? band_slot()->source : std::string();
}

bool has_band() {
    std::scoped_lock lock(band_mutex());
    return static_cast<bool>(band_slot());
}

double sphere_radius(double distance_km) {
    return R_REF_KM / std::sqrt(1.0 - (R_REF_KM / distance_km) * (R_REF_KM / distance_km));
}

namespace {

// ``silhouette`` on a given band (the caller's snapshot, so a profile and the
// generation it is cached under come from the same band; item C4).
std::vector<double> silhouette_on(const std::shared_ptr<const Band>& b,
                                  const std::array<double, 9>& axes, int n_bins,
                                  double distance_km) {
    if (n_bins < 1) throw std::invalid_argument("eclipse::limb::silhouette: n_bins < 1");
    const double cover = std::cos((b->band_deg - VISIBLE_DEG) * constants::DEG_TO_RAD);
    if (std::abs(axes[6]) < cover)
        throw std::invalid_argument(
            "eclipse::limb::silhouette: the view axis leaves the limb band's coverage");

    const double x0 = axes[0], x1 = axes[1], x2 = axes[2];
    const double y0 = axes[3], y1 = axes[4], y2 = axes[5];
    const double z0 = axes[6], z1 = axes[7], z2 = axes[8];
    const double floor_km = R_REF_KM - FLOOR_KM;
    const double pi = std::numbers::pi;
    const double per_rad = static_cast<double>(n_bins) / (2.0 * pi);
    const double half = static_cast<double>(n_bins) / 2.0;
    const double nbd = static_cast<double>(n_bins);
    const auto nb = static_cast<std::size_t>(n_bins);
    const std::size_t samples = b->cos_lon.size();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    std::vector<double> rho(nb, neg_inf);

    // One DEM point -> (s, u): radius and continuous bin coordinate (u only if
    // s > floor), in app/limb.py silhouette's operation order.
    struct Proj {
        double s, u;
        bool near;
        std::ptrdiff_t line;  // the DEM line this slot holds (-1: none yet)
    };
    auto project = [&](std::size_t li, std::size_t j, std::size_t p) {
        const double cl = b->cos_lat[li];
        const double rad = b->offset_km + static_cast<double>(b->dn[p]) * b->scale_km;
        const double px = rad * (cl * b->cos_lon[j]);
        const double py = rad * (cl * b->sin_lon[j]);
        const double pz = rad * b->sin_lat[li];
        const double qx = (px * x0 + py * x1) + pz * x2;
        const double qy = (px * y0 + py * y1) + pz * y2;
        const double w = (px * z0 + py * z1) + pz * z2;
        // Perspective from distance D along -z^ (app.limb.silhouette).
        Proj r{std::sqrt(qx * qx + qy * qy) / (1.0 + w / distance_km), 0.0, false,
               static_cast<std::ptrdiff_t>(li)};
        if (r.s > floor_km) {
            r.near = true;
            r.u = (std::atan2(qy, qx) + pi) * per_rad;
        }
        return r;
    };
    auto wrap_bin = [&](std::ptrdiff_t i) {  // Python % n_bins
        auto k = i % static_cast<std::ptrdiff_t>(n_bins);
        if (k < 0) k += n_bins;
        return static_cast<std::size_t>(k);
    };

    const auto n_lines = static_cast<std::ptrdiff_t>(b->cos_lat.size());

    // Lines are processed in order with two row buffers (this line, the next),
    // so each point is projected once per block of lines: its right neighbour
    // is in the same row buffer and its down neighbour in the next one. A slot
    // is a band point of line L exactly when it was filled for L (``line``), so
    // grid adjacency needs no neighbour index (app.limb._neighbours: east
    // j + 1 mod samples, south i + 1, both in the band). The max over points
    // and edges is the same whatever the order.
#ifdef _OPENMP
#pragma omp parallel if (!omp_in_parallel())
#endif
    {
        std::vector<double> local(nb, neg_inf);
        std::vector<Proj> cur(samples, Proj{0.0, 0.0, false, -1}), nxt(cur);
        auto put = [&](std::size_t k, double v) {
            if (v > local[k]) local[k] = v;
        };
        // app.limb._edge_crossings for one edge.
        auto edge = [&](const Proj& a, const Proj& e) {
            double du = e.u - a.u;
            du = du > half ? du - nbd : (du <= -half ? du + nbd : du);
            const double ub = a.u + du;
            const double first = std::floor(std::min(a.u, ub)) + 1.0;
            const double last = std::floor(std::max(a.u, ub));
            const auto span = static_cast<std::ptrdiff_t>(last - first);
            for (std::ptrdiff_t m = 0; m <= span; ++m) {
                const double bnd = first + static_cast<double>(m);
                const double frac = (bnd - a.u) / du;
                const double rad = a.s + (e.s - a.s) * frac;
                const auto ib = static_cast<std::ptrdiff_t>(bnd);
                put(wrap_bin(ib - 1), rad);
                put(wrap_bin(ib), rad);
            }
        };
        auto fill_row = [&](std::size_t li, std::vector<Proj>& row) {
            for (std::size_t r = b->line_runs[li]; r < b->line_runs[li + 1]; ++r) {
                const auto j0 = static_cast<std::size_t>(b->first[r]);
                const auto cnt = static_cast<std::size_t>(b->count[r]);
                for (std::size_t c = 0; c < cnt; ++c)
                    row[j0 + c] = project(li, j0 + c, b->offset[r] + c);
            }
        };
        std::ptrdiff_t filled_next = -1;  // line held in ``nxt``, if any
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t ll = 0; ll < n_lines; ++ll) {
            const auto li = static_cast<std::size_t>(ll);
            if (b->line_runs[li] == b->line_runs[li + 1]) continue;
            if (filled_next == ll)
                std::swap(cur, nxt);
            else
                fill_row(li, cur);
            const bool has_below = ll + 1 < n_lines;
            if (has_below) {
                fill_row(li + 1, nxt);
                filled_next = ll + 1;
            }
            for (std::size_t r = b->line_runs[li]; r < b->line_runs[li + 1]; ++r) {
                const auto j0 = static_cast<std::size_t>(b->first[r]);
                const auto cnt = static_cast<std::size_t>(b->count[r]);
                for (std::size_t c = 0; c < cnt; ++c) {
                    const std::size_t j = j0 + c;
                    const Proj& a = cur[j];
                    if (!a.near) continue;
                    auto k = static_cast<std::ptrdiff_t>(std::floor(a.u));
                    k = std::min<std::ptrdiff_t>(k, n_bins - 1);
                    put(static_cast<std::size_t>(k), a.s);
                    const Proj& e_right = cur[(j + 1) % samples];
                    if (e_right.line == ll && e_right.near) edge(a, e_right);
                    if (has_below) {
                        const Proj& e_down = nxt[j];
                        if (e_down.line == ll + 1 && e_down.near) edge(a, e_down);
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp critical(eclipse_limb_merge)
#endif
        for (std::size_t k = 0; k < nb; ++k)
            if (local[k] > rho[k]) rho[k] = local[k];
    }
    fill_empty(rho);
    const double ref = sphere_radius(distance_km);
    for (double& v : rho) v = v - ref;
    return rho;
}

}  // namespace

std::vector<double> silhouette(const std::array<double, 9>& axes, int n_bins, double distance_km) {
    return silhouette_on(band(), axes, n_bins, distance_km);
}

std::vector<double> delta_rho_at(std::span<const double> profile, std::span<const double> psi) {
    const auto n = static_cast<std::ptrdiff_t>(profile.size());
    if (n < 1) throw std::invalid_argument("eclipse::limb::delta_rho_at: empty profile");
    const double pi = std::numbers::pi;
    const double per_rad = static_cast<double>(n) / (2.0 * pi);
    std::vector<double> out(psi.size());
    for (std::size_t i = 0; i < psi.size(); ++i) {
        const double u = (psi[i] + pi) * per_rad - 0.5;
        const double k0 = std::floor(u);
        const double w = u - k0;
        // Python's % on int64: non-negative for a positive modulus.
        auto i0 = static_cast<std::ptrdiff_t>(k0) % n;
        if (i0 < 0) i0 += n;
        const std::ptrdiff_t i1 = (i0 + 1) % n;
        const double a = profile[static_cast<std::size_t>(i0)];
        out[i] = a + w * (profile[static_cast<std::size_t>(i1)] - a);
    }
    return out;
}

// ------------------------------------------------------------ time lattice

namespace {

// (node, frame, moon frame, band generation, kernel-pool generation, EOP
// table generation): a profile depends on all of them (item C4).
using NodeKey =
    std::tuple<std::int64_t, int, std::string, std::uint64_t, std::uint64_t, std::uint64_t>;

std::mutex& cache_mutex() {
    static std::mutex m;
    return m;
}

std::map<NodeKey, std::shared_ptr<const std::vector<double>>>& cache() {
    static std::map<NodeKey, std::shared_ptr<const std::vector<double>>> c;
    return c;
}

constexpr std::size_t kMaxCachedNodes = 256;

std::shared_ptr<const std::vector<double>> node_profile(std::int64_t node, Frame frame,
                                                       std::string_view moon_frame) {
    // One snapshot of the band with its generation, and the kernel-pool / EOP
    // generations, taken before the computation; the result is cached only if
    // neither of those changed while it ran (a reload mid-computation could
    // otherwise file a new-kernel profile under the old key, or the reverse).
    std::shared_ptr<const Band> b;
    std::uint64_t generation;
    {
        std::scoped_lock lock(band_mutex());
        b = band_slot();
        generation = band_generation();
    }
    if (!b) throw std::logic_error("eclipse::limb: limb band not set (call set_band first)");
    const std::uint64_t kernels = ephem::kernel_generation();
    const std::uint64_t eops = eop::generation();
    NodeKey key{node, static_cast<int>(frame), std::string(moon_frame), generation, kernels, eops};
    {
        std::scoped_lock lock(cache_mutex());
        if (auto it = cache().find(key); it != cache().end()) return it->second;
    }
    // The silhouette along limb_axes(node * LATTICE_S).
    const double et[1] = {static_cast<double>(node) * LATTICE_S};
    const ephem::LimbAxes la = ephem::limb_axes(et, frame, moon_frame);
    auto prof = std::make_shared<const std::vector<double>>(
        silhouette_on(b, la.axes[0], N_BINS, la.distance_km[0]));
    if (ephem::kernel_generation() != kernels || eop::generation() != eops) return prof;
    std::scoped_lock lock(cache_mutex());
    if (cache().size() >= kMaxCachedNodes) cache().clear();
    cache().emplace(key, prof);
    return prof;
}

// cos / sin of the n bin centres -pi + (k + 0.5) 2 pi / n (app.limb._unit_circle).
struct UnitCircle {
    std::vector<double> c, s;
};
UnitCircle unit_circle(int n) {
    const double pi = std::numbers::pi;
    const double step = 2.0 * pi / static_cast<double>(n);
    UnitCircle u;
    u.c.resize(static_cast<std::size_t>(n));
    u.s.resize(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        const double phi = -pi + (static_cast<double>(k) + 0.5) * step;
        u.c[static_cast<std::size_t>(k)] = std::cos(phi);
        u.s[static_cast<std::size_t>(k)] = std::sin(phi);
    }
    return u;
}

// Running np.max: a NaN, once seen, is the result.
double np_max2(double best, double v) {
    if (std::isnan(best) || std::isnan(v)) return std::numeric_limits<double>::quiet_NaN();
    return v > best ? v : best;
}

void check_g(std::span<const double> px, std::span<const double> py, std::span<const double> r_s,
             std::span<const double> r_m, std::span<const double> profiles, int n_bins) {
    const std::size_t n = px.size();
    if (n_bins < 1 || py.size() != n || r_s.size() != n || r_m.size() != n ||
        profiles.size() != n * static_cast<std::size_t>(n_bins))
        throw std::invalid_argument("eclipse::limb: g_total/g_annular argument lengths differ");
}

}  // namespace

std::vector<double> profiles_at(std::span<const double> et, Frame frame,
                                std::string_view moon_frame) {
    const auto nb = static_cast<std::size_t>(N_BINS);
    std::vector<double> out(et.size() * nb);
    for (std::size_t i = 0; i < et.size(); ++i) {
        const double q = et[i] / LATTICE_S;
        const double k0 = std::floor(q);
        const double w = q - k0;
        const auto node = static_cast<std::int64_t>(k0);
        const auto p0 = node_profile(node, frame, moon_frame);
        const auto p1 = node_profile(node + 1, frame, moon_frame);
        for (std::size_t k = 0; k < nb; ++k)
            out[i * nb + k] = (*p0)[k] + w * ((*p1)[k] - (*p0)[k]);
    }
    return out;
}

std::vector<double> g_total(std::span<const double> px, std::span<const double> py,
                            std::span<const double> r_s, std::span<const double> r_m,
                            std::span<const double> profiles, int n_bins) {
    check_g(px, py, r_s, r_m, profiles, n_bins);
    const auto nb = static_cast<std::size_t>(n_bins);
    const UnitCircle u = unit_circle(n_bins);
    std::vector<double> out(px.size()), qx(nb), qy(nb), psi(nb);
    for (std::size_t i = 0; i < px.size(); ++i) {
        for (std::size_t k = 0; k < nb; ++k) {
            qx[k] = px[i] + r_s[i] * u.c[k];
            qy[k] = py[i] + r_s[i] * u.s[k];
            psi[k] = std::atan2(qy[k], qx[k]);
        }
        const std::vector<double> rho = delta_rho_at(profiles.subspan(i * nb, nb), psi);
        double best = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < nb; ++k) {
            const double v = std::sqrt(qx[k] * qx[k] + qy[k] * qy[k]) -
                             r_m[i] * (1.0 + rho[k] / R_REF_KM);
            best = np_max2(best, v);
        }
        out[i] = best;
    }
    return out;
}

std::vector<double> g_annular(std::span<const double> px, std::span<const double> py,
                              std::span<const double> r_s, std::span<const double> r_m,
                              std::span<const double> profiles, int n_bins) {
    check_g(px, py, r_s, r_m, profiles, n_bins);
    const auto nb = static_cast<std::size_t>(n_bins);
    const UnitCircle u = unit_circle(n_bins);
    std::vector<double> out(px.size());
    for (std::size_t i = 0; i < px.size(); ++i) {
        double best = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < nb; ++k) {
            const double rad = r_m[i] * (1.0 + profiles[i * nb + k] / R_REF_KM);
            const double dx = rad * u.c[k] - px[i];
            const double dy = rad * u.s[k] - py[i];
            best = np_max2(best, std::sqrt(dx * dx + dy * dy));
        }
        out[i] = best - r_s[i];
    }
    return out;
}

}  // namespace eclipse::limb

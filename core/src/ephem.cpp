// See eclipse/ephem.hpp. This translation unit is the only one that includes
// SpiceUsr.h / erfa.h. Arithmetic is written in the Python oracle's exact
// operation order (left-to-right as NumPy evaluates it) so that parity is
// bit-level, not "close" — do not reassociate (roadmap §7).
#include "eclipse/ephem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef _OPENMP  // CMake defines ECLIPSE_HAVE_OPENMP and adds -fopenmp together
#include <omp.h>
#endif

#include "eclipse/constants.hpp"
#include "eclipse/deltat.hpp"
#include "eclipse/eop.hpp"
#include "eclipse/numerics.hpp"

extern "C" {
#include "SpiceUsr.h"
#include "erfa.h"
#include "erfaextra.h"
}

namespace eclipse {

namespace {

using namespace constants;
using numerics::wrap_180;

// One lock around every CSPICE entry point: the kernel pool and the error
// subsystem are process-global state (roadmap §2).
std::mutex& spice_mutex() {
    static std::mutex m;
    return m;
}

// Kernel-pool generation, bumped by every ``furnish`` / ``kclear``; read and
// written only under ``spice_mutex``. ``spkpos_both`` releases the lock between
// batches, so it checks this to refuse a vector whose batches would come from
// different kernel sets (review item C6).
std::uint64_t& pool_generation() {
    static std::uint64_t g = 0;
    return g;
}

// Put the toolkit in RETURN mode with no device output, exactly as spiceypy
// does at import (``erract("set", 10, "return")``, ``errdev("set", 10, "null")``).
// Idempotent; runs once under the lock on the first call.
void ensure_return_mode() {
    static bool done = false;
    if (done) return;
    // These two never fail in RETURN/ABORT mode with valid arguments.
    erract_c("SET", 0, const_cast<SpiceChar*>("RETURN"));
    errdev_c("SET", 0, const_cast<SpiceChar*>("NULL"));
    done = true;
}

// Message buffer sizes: the maxima documented in getmsg_c.html / qcktrc_c.html
// (SHORT 25, EXPLAIN 80, LONG 1840 chars) — the same values spiceypy uses.
constexpr SpiceInt kShortLen = 26;
constexpr SpiceInt kExplainLen = 101;
constexpr SpiceInt kLongLen = 1841;
constexpr SpiceInt kTraceLen = 1841;

[[noreturn]] void throw_spice_error() {
    SpiceChar short_msg[kShortLen], explain[kExplainLen], long_msg[kLongLen],
        trace[kTraceLen];
    getmsg_c("SHORT", kShortLen, short_msg);
    getmsg_c("EXPLAIN", kExplainLen, explain);
    getmsg_c("LONG", kLongLen, long_msg);
    qcktrc_c(kTraceLen, trace);
    reset_c();
    throw spice_error(short_msg, explain, long_msg, trace);
}

// Run ``f`` under the SPICE lock in RETURN mode and convert ``failed_c()`` into
// an exception — the C++ form of spiceypy's ``@spice_error_check``. ``f`` may
// make several CSPICE calls; in RETURN mode the first failure short-circuits
// the rest, so one check at the end is exact.
template <class F>
auto spice_call(F&& f) {
    std::scoped_lock lock(spice_mutex());
    ensure_return_mode();
    if constexpr (std::is_void_v<decltype(f())>) {
        f();
        if (failed_c()) throw_spice_error();
    } else {
        auto result = f();
        if (failed_c()) throw_spice_error();
        return result;
    }
}

std::string strip(std::string s) {
    const auto last = s.find_last_not_of(" \n");
    return last == std::string::npos ? std::string{} : s.substr(0, last + 1);
}

// --- small vector helpers (NumPy evaluation order) ---------------------------

using Vec3 = std::array<double, 3>;

// np.linalg.norm(v, axis=-1): sqrt(add.reduce(v*v)) — ((x²+y²)+z²) for 3 terms.
double norm(const Vec3& v) { return std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]); }

// app.ephemeris._reclat: (radius, longitude, latitude).
struct Spherical {
    double r, lon, lat;
};
Spherical reclat(const Vec3& v) {
    return {norm(v), std::atan2(v[1], v[0]), std::atan2(v[2], std::hypot(v[0], v[1]))};
}

// --- geocentric vectors (app.ephemeris._geocentric_vectors) --------------------

struct Geocentric {
    std::vector<Vec3> moon, sun;  // Earth equatorial radii
    std::vector<double> gast;     // radians; 0 for an Earth-fixed frame
};

// Instants per SPICE-lock acquisition in ``spkpos_both``: bounds how long a
// long vector (a century catalog scan is ~4e5 instants) holds the pool lock.
// Each spkpos_c call is independent of its neighbours, so the batching cannot
// change a bit of the result -- provided the kernel pool does not change
// between batches, which ``pool_generation`` enforces.
constexpr size_t kSpkposBatch = 2048;

// Apparent (LT+S) Moon and Sun in ``frame`` for each et, km, the lock taken
// per batch of ``kSpkposBatch`` instants.
void spkpos_both(std::span<const double> et, const char* frame, std::vector<Vec3>& moon,
                 std::vector<Vec3>& sun) {
    const size_t n = et.size();
    moon.resize(n);
    sun.resize(n);
    std::uint64_t generation = 0;
    for (size_t b = 0; b < n; b += kSpkposBatch) {
        const size_t end = std::min(n, b + kSpkposBatch);
        spice_call([&] {
            if (b == 0) {
                generation = pool_generation();
            } else if (pool_generation() != generation) {
                // A furnish/kclear ran between two batches: the vector would
                // mix kernel sets. A server-side condition, not a bad request.
                throw std::runtime_error(
                    "spkpos_both: the SPICE kernel pool changed during a batched evaluation");
            }
            SpiceDouble lt;
            for (size_t i = b; i < end; ++i) {
                spkpos_c("MOON", et[i], frame, "LT+S", "EARTH", moon[i].data(), &lt);
                spkpos_c("SUN", et[i], frame, "LT+S", "EARTH", sun[i].data(), &lt);
            }
        });
    }
}

// --- fundamental-plane coordinates (app.ephemeris._fundamental_xyz) -------------

// The Moon's fundamental-plane coordinates ``x, y, z`` [ES92] eq. 8.322-6 and
// the shadow-axis direction (longitude ``a``, declination ``d``, Sun-Moon
// distance ``g_dist``) from geocentric Moon/Sun vectors in Earth radii, in ANY
// common equatorial frame. ONE formula, shared by ``besselian_instants``
// (Earth-fixed / true-of-date vectors) and ``axis_separation`` (J2000).
struct FundamentalXYZ {
    double x, y, z, a, d, g_dist;
};
FundamentalXYZ fundamental_xyz(const Vec3& moon, const Vec3& sun) {
    // Moon geocentric spherical coordinates.
    const Spherical m = reclat(moon);
    const double r_moon = m.r, alpha = m.lon, delta = m.lat;

    // Shadow-axis direction, Moon toward Sun (S - M) [ES92] ch. 8: longitude
    // ``a``, declination ``d``, length = Sun-Moon distance.
    const Vec3 axis = {sun[0] - moon[0], sun[1] - moon[1], sun[2] - moon[2]};
    const Spherical ax = reclat(axis);
    const double g_dist = ax.r, a = ax.lon, d = ax.lat;

    // Fundamental-plane coordinates of the Moon [ES92] eq. 8.322-6.
    const double ha = alpha - a;
    const double x = r_moon * std::cos(delta) * std::sin(ha);
    const double y = r_moon * (std::sin(delta) * std::cos(d) -
                               std::cos(delta) * std::sin(d) * std::cos(ha));
    const double z = r_moon * (std::sin(delta) * std::sin(d) +
                               std::cos(delta) * std::cos(d) * std::cos(ha));
    return {x, y, z, a, d, g_dist};
}

// ``np.einsum("nij,nj->ni", r, v)`` then ``/ a_e``: rotate, then scale.
Vec3 rotate_scale(const double r[3][3], const Vec3& v, double a_e) {
    Vec3 out;
    eraRxp(const_cast<double(*)[3]>(r), const_cast<double*>(v.data()), out.data());
    for (double& c : out) c = c / a_e;
    return out;
}

Geocentric geocentric_vectors(std::span<const double> et, Frame frame) {
    const double a_e = EARTH_EQUATORIAL_RADIUS_KM;
    const size_t n = et.size();
    Geocentric g;
    g.gast.assign(n, 0.0);

    if (frame == Frame::ITRS || frame == Frame::TOD) {
        spkpos_both(et, "J2000", g.moon, g.sun);
        const ephem::EarthRotation ert = ephem::earth_rotation_times(et);
        // From here the loop is pure ERFA (reentrant, no shared state) plus
        // one rotation per instant, with no SPICE call: it runs OpenMP-parallel
        // for long vectors. Each instant writes only its own slots and there
        // is no reduction, so the result is bit-identical for any thread
        // count. Nested inside another parallel region (the catalog's
        // per-event detail loop) it stays serial.
        const auto count = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 64 && !omp_in_parallel())
#endif
        for (std::ptrdiff_t ii = 0; ii < count; ++ii) {
            const auto i = static_cast<size_t>(ii);
            double r[3][3];
            if (frame == Frame::ITRS) {
                // ERFA c2t06a: full IAU 2006/2000A celestial-to-terrestrial matrix
                // [SOFA] (Wallace & Capitaine 2006) with IERS polar motion and
                // UT1-UTC [IERS2010].
                eraC2t06a(J2000_JD, ert.tt2[i], J2000_JD, ert.ut1[i], ert.xp[i], ert.yp[i], r);
            } else {
                // TOD: pnm06a precession-nutation matrix + gst06a GAST [SOFA].
                eraPnm06a(J2000_JD, ert.tt2[i], r);
                g.gast[i] = eraGst06a(J2000_JD, ert.ut1[i], J2000_JD, ert.tt2[i]);
            }
            g.moon[i] = rotate_scale(r, g.moon[i], a_e);
            g.sun[i] = rotate_scale(r, g.sun[i], a_e);
        }
        return g;
    }

    spkpos_both(et, to_string(frame), g.moon, g.sun);
    for (size_t i = 0; i < n; ++i) {
        for (double& c : g.moon[i]) c = c / a_e;
        for (double& c : g.sun[i]) c = c / a_e;
    }
    return g;
}

}  // namespace

spice_error::spice_error(std::string short_msg, std::string explain, std::string long_msg,
                         std::string traceback)
    : std::runtime_error(
          "\n================================================================================"
          "\n\nToolkit version: " + std::string(tkvrsn_c("TOOLKIT")) + "\n\n" + short_msg +
          " --\n" + strip(explain) + "\n" + strip(long_msg) + "\n\n" + strip(traceback) +
          "\n\n================================================================================"),
      short_(std::move(short_msg)),
      explain_(strip(std::move(explain))),
      long_(strip(std::move(long_msg))),
      traceback_(strip(std::move(traceback))) {}

namespace ephem {

// ------------------------------------------------------------ toolkit / pool

std::string toolkit_version() {
    return spice_call([] { return std::string(tkvrsn_c("TOOLKIT")); });
}

std::string erfa_version() { return eraVersion(); }
std::string sofa_version() { return eraSofaVersion(); }

void furnish(std::string_view path) {
    const std::string p(path);
    spice_call([&] {
        ++pool_generation();  // even a failed furnsh may have loaded part of the set
        furnsh_c(p.c_str());
    });
}

void kclear() {
    spice_call([] {
        ++pool_generation();
        kclear_c();
    });
}

std::uint64_t kernel_generation() {
    std::scoped_lock lock(spice_mutex());
    return pool_generation();
}

int kernel_count(std::string_view kind) {
    const std::string k(kind);
    return spice_call([&] {
        SpiceInt n = 0;
        ktotal_c(k.c_str(), &n);
        return static_cast<int>(n);
    });
}

double sun_radius_km() { return SUN_RADIUS_KM; }

// --------------------------------------------------------------- raw SPICE

double str_to_et(std::string_view time_string) {
    const std::string s(time_string);
    return spice_call([&] {
        SpiceDouble et = 0.0;
        str2et_c(s.c_str(), &et);
        return et;
    });
}

std::string et_to_utc_iso(double et, int precision) {
    return spice_call([&] {
        // "ISOC" with 6 digits is 26 chars + NUL; leave room for any precision.
        SpiceChar buf[64];
        et2utc_c(et, "ISOC", static_cast<SpiceInt>(precision), sizeof buf, buf);
        return std::string(buf);
    });
}

Position body_position(std::string_view target, double et, std::string_view frame,
                       std::string_view abcorr, std::string_view observer) {
    const std::string t(target), f(frame), a(abcorr), o(observer);
    return spice_call([&] {
        Position p{};
        spkpos_c(t.c_str(), et, f.c_str(), a.c_str(), o.c_str(), p.km.data(), &p.light_time_s);
        return p;
    });
}

// ------------------------------------------------------------- time scales

namespace {

// The end of the measured Delta-T record: the Bulletin A table's last row,
// as a decimal year and TT - UT1 there [s] = (ET - UTC) - (UT1 - UTC), with
// ET for TT exactly as ``earth_rotation_times`` uses it inside the era.
struct RecordEnd {
    double year;
    double dt_s;
};

RecordEnd record_end() {
    const double mjd_end = eop::mjd_range().second;
    const double jd_end = mjd_end + MJD_OFFSET;
    const double utc_end = (jd_end - J2000_JD) * SECONDS_PER_DAY;
    const double et_minus_utc = spice_call([&] {
        SpiceDouble d = 0.0;
        deltet_c(utc_end, "UTC", &d);
        return static_cast<double>(d);
    });
    return {deltat::decimal_year_from_jd(jd_end), et_minus_utc - eop::interpolate(mjd_end).dut1_s};
}

// Delta-T [s] outside the IERS era at decimal ``year``: the [Espenak] model
// before the table; after it the model joined continuously to the measured
// value at the table's end (``deltat::delta_t_after_record``, item C2).
double model_delta_t(double year, const RecordEnd& end) {
    return deltat::delta_t_after_record(year, end.year, end.dt_s);
}

double model_delta_t(double year) { return model_delta_t(year, record_end()); }

}  // namespace

double utc_to_et(std::string_view utc) {
    const std::string s(utc);
    // tparse_c: seconds past J2000 on the leap-second-free calendar. A parse
    // failure is reported as a message, not a SPICE error; spiceypy ignores it
    // (formal = 0 -> "in era") and lets str2et raise the proper error, so do
    // the same. A SPICE error proper (e.g. SPICE(EMPTYSTRING) from the
    // wrapper's string check, which returns before writing ``errmsg``) goes
    // through ``spice_call`` like every other call -- thrown and reset, never
    // left pending in the global error state -- as spiceypy's
    // ``@spice_error_check`` does for ``tparse`` (review item C5).
    SpiceChar errmsg[256];
    errmsg[0] = '\0';
    const double formal = spice_call([&] {
        SpiceDouble sp2000 = 0.0;
        tparse_c(s.c_str(), sizeof errmsg, &sp2000, errmsg);
        return static_cast<double>(sp2000);
    });
    const double mjd = J2000_JD - MJD_OFFSET + formal / SECONDS_PER_DAY;
    if (errmsg[0] != '\0' || eop::in_iers_era(mjd)) return str_to_et(s);
    return formal + model_delta_t(deltat::decimal_year_from_jd(formal / SECONDS_PER_DAY + J2000_JD));
}

std::string et_to_utc(double et) {
    if (eop::in_iers_era(J2000_JD - MJD_OFFSET + et / SECONDS_PER_DAY)) return et_to_utc_iso(et, 0);
    const double ut1 = et - model_delta_t(deltat::decimal_year_from_jd(J2000_JD + et / SECONDS_PER_DAY));
    return spice_call([&] {
        SpiceChar buf[64];
        timout_c(ut1, "YYYY-MM-DDTHR:MN:SC ::TDB ::RND", sizeof buf, buf);
        return std::string(buf);
    });
}

EarthRotation earth_rotation_times(std::span<const double> et) {
    const size_t n = et.size();
    EarthRotation r;
    r.tt2.resize(n);
    r.ut1.resize(n);
    r.xp.resize(n);
    r.yp.resize(n);

    // ET - UTC [s] at each et (deltet_c), under one lock.
    std::vector<double> delta_et(n);
    spice_call([&] {
        for (size_t i = 0; i < n; ++i) deltet_c(et[i], "ET", &delta_et[i]);
    });

    std::optional<RecordEnd> end;  // computed once, only if an instant needs it
    for (size_t i = 0; i < n; ++i) {
        const double tt2 = et[i] / SECONDS_PER_DAY;
        const double utc2 = (et[i] - delta_et[i]) / SECONDS_PER_DAY;
        const double mjd_utc = J2000_JD - MJD_OFFSET + utc2;
        r.tt2[i] = tt2;
        if (eop::in_iers_era(mjd_utc)) {
            // UT1 = UTC + (UT1-UTC), polar motion from Bulletin A [IERS2010].
            const eop::Eop e = eop::interpolate(mjd_utc);
            r.ut1[i] = utc2 + e.dut1_s / SECONDS_PER_DAY;
            r.xp[i] = e.xp_rad;
            r.yp[i] = e.yp_rad;
        } else {
            // UT1 = TT - delta-T (the [Espenak] polynomial, carried on from
            // the measured value after the table's end), no polar motion.
            if (!end) end = record_end();
            const double dt_model =
                model_delta_t(deltat::decimal_year_from_jd(J2000_JD + tt2), *end);
            r.ut1[i] = tt2 - dt_model / SECONDS_PER_DAY;
            r.xp[i] = 0.0;
            r.yp[i] = 0.0;
        }
    }
    return r;
}

std::vector<double> tt_minus_ut1(std::span<const double> et) {
    const EarthRotation r = earth_rotation_times(et);
    std::vector<double> out(et.size());
    for (size_t i = 0; i < et.size(); ++i) out[i] = (r.tt2[i] - r.ut1[i]) * SECONDS_PER_DAY;
    return out;
}

// ---------------------------------------------------------------- geometry

Elements besselian_instants(std::span<const double> et, Frame frame, double k1, double k2) {
    const double a_e = EARTH_EQUATORIAL_RADIUS_KM;
    const Geocentric g = geocentric_vectors(et, frame);
    const double d_s = sun_radius_km() / a_e;  // solar radius in Earth radii

    Elements out;
    out.reserve(et.size());
    for (size_t i = 0; i < et.size(); ++i) {
        // Fundamental-plane coordinates of the Moon and the axis direction
        // [ES92] eq. 8.322-6 (shared with axis_separation).
        const FundamentalXYZ f = fundamental_xyz(g.moon[i], g.sun[i]);
        const double x = f.x, y = f.y, z = f.z, a = f.a, d = f.d, g_dist = f.g_dist;

        const double mu = g.gast[i] - a;  // Greenwich hour angle of the axis

        // Penumbral (f1) and umbral (f2) cones [ES92] eq. 8.323-1, 8.323-6,
        // 8.323-7 with distinct lunar radii k1/k2 [Espenak].
        const double sin_f1 = (d_s + k1) / g_dist;
        const double sin_f2 = (d_s - k2) / g_dist;
        const double tan_f1 = std::tan(std::asin(sin_f1));
        const double tan_f2 = std::tan(std::asin(sin_f2));

        out.x.push_back(x);
        out.y.push_back(y);
        out.z.push_back(z);
        out.d.push_back(d * RAD_TO_DEG);
        out.mu.push_back(wrap_180(mu * RAD_TO_DEG));
        out.l1.push_back((z + k1 / sin_f1) * tan_f1);
        out.l2.push_back((z - k2 / sin_f2) * tan_f2);
        out.tan_f1.push_back(tan_f1);
        out.tan_f2.push_back(tan_f2);
    }
    return out;
}

AxisSeparation axis_separation(std::span<const double> et) {
    const double a_e = EARTH_EQUATORIAL_RADIUS_KM;
    const size_t n = et.size();
    std::vector<Vec3> moon, sun;
    spkpos_both(et, "J2000", moon, sun);
    AxisSeparation out;
    out.rho.resize(n);
    out.z.resize(n);
    for (size_t i = 0; i < n; ++i) {
        // ``moon / a_e``, ``sun / a_e``: km -> Earth radii, componentwise.
        for (double& c : moon[i]) c = c / a_e;
        for (double& c : sun[i]) c = c / a_e;
        const FundamentalXYZ f = fundamental_xyz(moon[i], sun[i]);
        out.rho[i] = std::hypot(f.x, f.y);
        out.z[i] = f.z;
    }
    return out;
}

SubSolar sub_solar_points(std::span<const double> et, Frame frame) {
    const Geocentric g = geocentric_vectors(et, frame);
    SubSolar s;
    s.lon_deg.resize(et.size());
    s.lat_deg.resize(et.size());
    for (size_t i = 0; i < et.size(); ++i) {
        // Latitude = the Sun's Earth-fixed declination (geodetic latitude of the
        // zenith point); longitude east-positive.
        const Spherical sun = reclat(g.sun[i]);
        s.lon_deg[i] = wrap_180((sun.lon - g.gast[i]) * RAD_TO_DEG);
        s.lat_deg[i] = sun.lat * RAD_TO_DEG;
    }
    return s;
}

// Row-wise dot product summed left to right (app.ephemeris._dot3).
static double dot3(const double* a, const double* b) {
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}

LimbAxes limb_axes(std::span<const double> et, Frame frame, std::string_view moon_frame) {
    const size_t n = et.size();
    const std::string mf(moon_frame);
    std::vector<Vec3> moon(n), sun(n);
    std::vector<std::array<double, 9>> rot(n);
    spice_call([&] {
        for (size_t i = 0; i < n; ++i) {
            SpiceDouble lt = 0.0, lt_sun = 0.0;
            spkpos_c("MOON", et[i], "J2000", "LT+S", "EARTH", moon[i].data(), &lt);
            spkpos_c("SUN", et[i], "J2000", "LT+S", "EARTH", sun[i].data(), &lt_sun);
            SpiceDouble m[3][3];
            pxform_c("J2000", mf.c_str(), et[i] - lt, m);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) rot[i][static_cast<size_t>(3 * r + c)] = m[r][c];
        }
    });

    // Earth-frame pole in J2000: row 2 of the J2000 -> frame matrix
    // (app.ephemeris._earth_pole_j2000).
    std::vector<Vec3> pole(n);
    if (frame == Frame::ITRS || frame == Frame::TOD) {
        const EarthRotation ert = earth_rotation_times(et);
        for (size_t i = 0; i < n; ++i) {
            double r[3][3];
            if (frame == Frame::ITRS)
                eraC2t06a(J2000_JD, ert.tt2[i], J2000_JD, ert.ut1[i], ert.xp[i], ert.yp[i], r);
            else
                eraPnm06a(J2000_JD, ert.tt2[i], r);
            pole[i] = {r[2][0], r[2][1], r[2][2]};
        }
    } else {
        const char* ef = to_string(frame);
        spice_call([&] {
            for (size_t i = 0; i < n; ++i) {
                SpiceDouble m[3][3];
                pxform_c("J2000", ef, et[i], m);
                pole[i] = {m[2][0], m[2][1], m[2][2]};
            }
        });
    }

    LimbAxes result;
    std::vector<std::array<double, 9>>& out = result.axes;
    out.resize(n);
    result.distance_km.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const Vec3 w = {sun[i][0] - moon[i][0], sun[i][1] - moon[i][1], sun[i][2] - moon[i][2]};
        const double wn = std::sqrt(dot3(w.data(), w.data()));
        const Vec3 zh = {w[0] / wn, w[1] / wn, w[2] / wn};
        const double pz = dot3(pole[i].data(), zh.data());
        const Vec3 yv = {pole[i][0] - pz * zh[0], pole[i][1] - pz * zh[1],
                         pole[i][2] - pz * zh[2]};
        const double yn = std::sqrt(dot3(yv.data(), yv.data()));
        const Vec3 yh = {yv[0] / yn, yv[1] / yn, yv[2] / yn};
        const Vec3 xh = {yh[1] * zh[2] - yh[2] * zh[1], yh[2] * zh[0] - yh[0] * zh[2],
                         yh[0] * zh[1] - yh[1] * zh[0]};
        const Vec3* rows[3] = {&xh, &yh, &zh};
        for (size_t k = 0; k < 3; ++k)
            for (size_t r = 0; r < 3; ++r)
                out[i][3 * k + r] = dot3(&rot[i][3 * r], rows[k]->data());
        result.distance_km[i] = dot3(moon[i].data(), zh.data());
    }
    return result;
}

}  // namespace ephem
}  // namespace eclipse

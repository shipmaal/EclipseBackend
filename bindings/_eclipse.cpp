// nanobind module ``_eclipse`` — the Python face of libeclipse.
//
// Phase 1 (docs/CPP_ROADMAP.md §5): kernel management, time scales, EOP
// injection, Besselian elements and sub-solar points, bound here. Arrays
// cross the boundary as NumPy float64 (spans in, fresh ndarrays out); the
// SPICE-touching calls release the GIL and the C++ side holds its own lock.
// Phase 2 and 3 units are bound one translation unit per core header
// (bind_numerics / bind_ellipsoid / bind_geometry / bind_circumstances /
// bind_catalog, see common.hpp) and assembled at the end of NB_MODULE.
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <span>
#include <utility>
#include <vector>

#include "common.hpp"
#include "eclipse/deltat.hpp"
#include "eclipse/eop.hpp"
#include "eclipse/ephem.hpp"

using namespace nb::literals;

namespace {

eclipse::Frame frame_arg(std::string_view s) { return eclipse::frame_from_string(s); }

}  // namespace

NB_MODULE(_eclipse, m) {
    m.doc() = "libeclipse: native core of EclipseBackend (CSPICE + ERFA).";

    // ``SpiceError`` derives from RuntimeError so ``except RuntimeError`` still
    // catches it; ``str(e)`` is the spiceypy-style multi-line report and the
    // four CSPICE message fields are attributes, so app/native.py can rebuild
    // the exact spiceypy exception class (e.g. SpiceFRAMEDATANOTFOUND) that the
    // Python backend would have raised.
    static PyObject* spice_error_type =
        nb::exception<eclipse::spice_error>(m, "SpiceError", PyExc_RuntimeError).inc_ref().ptr();
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const eclipse::spice_error& e) {
            nb::object exc = nb::borrow(spice_error_type)(e.what());
            exc.attr("short_message") = e.short_message();
            exc.attr("explanation") = e.explanation();
            exc.attr("long_message") = e.long_message();
            exc.attr("traceback_text") = e.traceback();
            PyErr_SetObject(spice_error_type, exc.ptr());
        }
    });

    nb::class_<eclipse::ephem::Position>(m, "Position",
                                         "spkpos_c result: km vector + light time (s).")
        .def_ro("km", &eclipse::ephem::Position::km)
        .def_ro("light_time_s", &eclipse::ephem::Position::light_time_s);

    // ---- toolkit / pool
    m.def("toolkit_version", &eclipse::ephem::toolkit_version,
          "tkvrsn_c('TOOLKIT') of the vendored CSPICE, e.g. 'CSPICE_N0067'.");
    m.def("erfa_version", &eclipse::ephem::erfa_version, "eraVersion() of the vendored liberfa.");
    m.def("sofa_version", &eclipse::ephem::sofa_version, "eraSofaVersion() of the vendored liberfa.");
    m.def("furnish", &eclipse::ephem::furnish, "path"_a,
          nb::call_guard<nb::gil_scoped_release>(), "furnsh_c: load a kernel or metakernel.");
    m.def("kclear", &eclipse::ephem::kclear, nb::call_guard<nb::gil_scoped_release>(),
          "kclear_c: unload all kernels.");
    m.def("kernel_count", &eclipse::ephem::kernel_count, "kind"_a = "ALL",
          nb::call_guard<nb::gil_scoped_release>(), "ktotal_c.");
    m.def("sun_radius_km", &eclipse::ephem::sun_radius_km,
          nb::call_guard<nb::gil_scoped_release>(), "Solar radius [km]: the IAU 1976 constant 696000 (app.ephemeris.sun_radius_km).");

    // ---- raw SPICE
    m.def("str_to_et", &eclipse::ephem::str_to_et, "time_string"_a,
          nb::call_guard<nb::gil_scoped_release>(), "str2et_c: time string -> TDB seconds past J2000.");
    m.def("et_to_utc_iso", &eclipse::ephem::et_to_utc_iso, "et"_a, "precision"_a = 6,
          nb::call_guard<nb::gil_scoped_release>(), "et2utc_c(..., 'ISOC', precision).");
    m.def("body_position", &eclipse::ephem::body_position, "target"_a, "et"_a, "frame"_a,
          "abcorr"_a, "observer"_a, nb::call_guard<nb::gil_scoped_release>(),
          "spkpos_c: apparent position (km) of target seen from observer in frame.");

    // ---- delta-T model [Espenak]
    m.def(
        "delta_t_seconds",
        [](In1D years) { return to_numpy(eclipse::deltat::delta_t_seconds(as_span(years))); },
        "years"_a, "Delta-T = TT - UT1 [s] at decimal years (Espenak & Meeus polynomials).");
    m.def(
        "decimal_year_from_jd",
        [](In1D jd) {
            std::vector<double> out;
            out.reserve(jd.shape(0));
            for (const double v : as_span(jd)) out.push_back(eclipse::deltat::decimal_year_from_jd(v));
            return to_numpy(std::move(out));
        },
        "jd"_a, "2000.0 + (jd - 2451545.0) / 365.25, elementwise.");

    // ---- EOP injection
    m.def(
        "set_eop_table",
        [](In1D mjd, In1D xp_arcsec, In1D yp_arcsec, In1D dut1_s) {
            eclipse::eop::set_table(as_span(mjd), as_span(xp_arcsec), as_span(yp_arcsec),
                                    as_span(dut1_s));
        },
        "mjd"_a, "xp_arcsec"_a, "yp_arcsec"_a, "dut1_s"_a,
        "Install the IERS Bulletin A table (app.eop._table() columns).");
    m.def("has_eop_table", &eclipse::eop::has_table);
    m.def("eop_mjd_range", &eclipse::eop::mjd_range, "(first, last) MJD of the table.");
    m.def(
        "eop",
        [](In1D mjd_utc) {
            const auto s = as_span(mjd_utc);
            std::vector<double> xp(s.size()), yp(s.size()), dut1(s.size());
            for (size_t i = 0; i < s.size(); ++i) {
                const auto e = eclipse::eop::interpolate(s[i]);
                xp[i] = e.xp_rad;
                yp[i] = e.yp_rad;
                dut1[i] = e.dut1_s;
            }
            return nb::make_tuple(to_numpy(std::move(xp)), to_numpy(std::move(yp)),
                                  to_numpy(std::move(dut1)));
        },
        "mjd_utc"_a, "(xp_rad, yp_rad, dut1_s) interpolated like app.eop.eop.");

    // ---- time scales (IERS-era rule)
    m.def("utc_to_et", &eclipse::ephem::utc_to_et, "utc"_a,
          nb::call_guard<nb::gil_scoped_release>(), "app.ephemeris.utc_to_et.");
    m.def("et_to_utc", &eclipse::ephem::et_to_utc, "et"_a,
          nb::call_guard<nb::gil_scoped_release>(), "app.ephemeris.et_to_utc.");
    m.def(
        "earth_rotation_times",
        [](In1D et) {
            eclipse::ephem::EarthRotation r;
            {
                nb::gil_scoped_release nogil;
                r = eclipse::ephem::earth_rotation_times(as_span(et));
            }
            return nb::make_tuple(to_numpy(std::move(r.tt2)), to_numpy(std::move(r.ut1)),
                                  to_numpy(std::move(r.xp)), to_numpy(std::move(r.yp)));
        },
        "et"_a, "(tt2, ut1_frac, xp, yp) as app.ephemeris.earth_rotation_times.");
    m.def(
        "tt_minus_ut1",
        [](In1D et) {
            std::vector<double> r;
            {
                nb::gil_scoped_release nogil;
                r = eclipse::ephem::tt_minus_ut1(as_span(et));
            }
            return to_numpy(std::move(r));
        },
        "et"_a, "Delta-T [s] actually used at et (measured or modelled).");

    // ---- geometry
    m.def(
        "besselian_instants",
        [](In1D et, std::string_view earth_frame) {
            const eclipse::Frame f = frame_arg(earth_frame);
            eclipse::Elements e;
            {
                nb::gil_scoped_release nogil;
                e = eclipse::ephem::besselian_instants(as_span(et), f);
            }
            nb::dict out;
            out["x"] = to_numpy(std::move(e.x));
            out["y"] = to_numpy(std::move(e.y));
            out["z"] = to_numpy(std::move(e.z));
            out["d"] = to_numpy(std::move(e.d));
            out["mu"] = to_numpy(std::move(e.mu));
            out["l1"] = to_numpy(std::move(e.l1));
            out["l2"] = to_numpy(std::move(e.l2));
            out["tan_f1"] = to_numpy(std::move(e.tan_f1));
            out["tan_f2"] = to_numpy(std::move(e.tan_f2));
            return out;
        },
        "et"_a, "earth_frame"_a = "ITRS",
        "Besselian elements at each et: dict of arrays as app.ephemeris.besselian_instants.");
    m.def(
        "axis_separation",
        [](In1D et) {
            eclipse::ephem::AxisSeparation s;
            {
                nb::gil_scoped_release nogil;
                s = eclipse::ephem::axis_separation(as_span(et));
            }
            return nb::make_tuple(to_numpy(std::move(s.rho)), to_numpy(std::move(s.z)));
        },
        "et"_a,
        "(rho, z) at each et as app.ephemeris.axis_separation: the axis distance hypot(x, y)\n"
        "and the Moon's distance along the axis [Earth radii], frame-free (J2000 vectors).");
    m.def(
        "sub_solar_points",
        [](In1D et, std::string_view earth_frame) {
            const eclipse::Frame f = frame_arg(earth_frame);
            eclipse::ephem::SubSolar s;
            {
                nb::gil_scoped_release nogil;
                s = eclipse::ephem::sub_solar_points(as_span(et), f);
            }
            return nb::make_tuple(to_numpy(std::move(s.lon_deg)), to_numpy(std::move(s.lat_deg)));
        },
        "et"_a, "earth_frame"_a = "ITRS", "(lon_deg, lat_deg) of the sub-solar point at each et.");

    // ---- phase 2 units (one binding TU per core header)
    bind_numerics(m);
    bind_ellipsoid(m);
    bind_geometry(m);
    bind_circumstances(m);  // phase 3
    bind_catalog(m);        // phase 4
}

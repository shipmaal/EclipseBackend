// Bindings for eclipse/catalog.hpp (app.catalog find_eclipses), phase 4 of
// docs/CPP_ROADMAP.md.
//
// ``find_eclipses`` hands each ``catalog::Event`` back as a tuple in the
// Python ``_EventRaw`` NamedTuple's FIELD ORDER (et_g, kind, central, gamma,
// magnitude, lat, lon, et0, contacts, local, width_km): ``kind`` the oracle's
// string, ``contacts`` a list of ``(name, t_hours)`` pairs in
// ``global_contacts`` insertion order, ``local`` the phase-3 ``_LocalRaw``
// tuple (``local_raw_tuple``, shared with bind_circumstances.cpp) and
// ``width_km`` / ``et0`` floats -- ``None`` in every detail slot that the
// oracle leaves ``None`` (no detail; not central; no on-Earth track point), so
// ``_EventRaw(*t)`` on the Python side reconstructs the oracle's row type for
// type. The GIL is released around the whole scan: the ephemeris calls take
// the C++ SPICE lock and the per-event hybrid / detail loop is OpenMP-parallel
// (``threads``: 0 = the default, 1 = serial; results identical).
// ``std::invalid_argument`` maps to ``ValueError`` through nanobind.
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "common.hpp"
#include "eclipse/catalog.hpp"
#include "eclipse/elements.hpp"

using namespace nb::literals;
namespace cat = eclipse::catalog;

namespace {

nb::object detail_or_none(const std::optional<cat::Detail>& d, auto&& field) {
    return d ? nb::object(field(*d)) : nb::none();
}

nb::object event_tuple(const cat::Event& ev) {
    nb::object et0 = detail_or_none(ev.detail, [](const cat::Detail& d) { return nb::float_(d.et0); });
    nb::object contacts = detail_or_none(ev.detail, [](const cat::Detail& d) {
        nb::list out;
        for (const auto& c : d.contacts) out.append(nb::make_tuple(c.name, c.t_hours));
        return out;
    });
    nb::object local = nb::none();
    nb::object width = nb::none();
    if (ev.detail) {
        if (ev.detail->local) local = local_raw_tuple(*ev.detail->local);
        if (ev.detail->width_km) width = nb::float_(*ev.detail->width_km);
    }
    return nb::make_tuple(ev.et_g, cat::to_string(ev.kind), ev.central, ev.gamma, ev.magnitude,
                          ev.lat_deg, ev.lon_deg, et0, contacts, local, width);
}

}  // namespace

void bind_catalog(nb::module_& m) {
    m.def(
        "find_eclipses",
        [](double et_a, double et_b, std::string_view earth_frame, bool detail, int threads) {
            const eclipse::Frame f = eclipse::frame_from_string(earth_frame);
            std::vector<cat::Event> events;
            {
                nb::gil_scoped_release nogil;
                events = cat::find_eclipses(et_a, et_b, f, detail, threads);
            }
            nb::list out;
            for (const auto& ev : events) out.append(event_tuple(ev));
            return out;
        },
        "et_a"_a, "et_b"_a, "earth_frame"_a, "detail"_a, "threads"_a = 0,
        "app.catalog._catalog_raw(et_a, et_b, earth_frame, detail): every eclipse with\n"
        "greatest eclipse in [et_a, et_b] (TDB s, inclusive) in scan order, each an _EventRaw\n"
        "tuple (et_g, kind, central, gamma, magnitude, lat, lon, et0, contacts, local,\n"
        "width_km); the detail slots are None where the oracle's are. threads: 0 = OpenMP\n"
        "default, 1 = serial, n = that many (results identical).");
}

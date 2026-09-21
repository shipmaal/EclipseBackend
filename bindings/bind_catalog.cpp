// Bindings for eclipse/catalog.hpp (app.catalog find_eclipses), phase 4 of
// docs/CPP_ROADMAP.md. Work package A declares the unit; work package C binds
// ``find_eclipses`` (the ``_EventRaw`` tuples, GIL released) and dispatches
// ``app.catalog._catalog_raw``. Nothing is bound yet.
#include "common.hpp"
#include "eclipse/catalog.hpp"

void bind_catalog(nb::module_& /*m*/) {}

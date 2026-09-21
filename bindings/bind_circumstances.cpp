// Bindings for eclipse/circumstances.hpp (app.circumstances local_circumstances
// and circumstances_grid). Phase 3, work package A: the translation unit is
// wired into the module and empty; work package B fills it (LocalRaw as a tuple
// in the Python _LocalRaw field order, GridResult as the dict of arrays with
// ``to_numpy_bool`` for ``visible`` / ``central``, GIL released around the
// compute). Package C adds the ``ECLIPSE_BACKEND=native`` dispatch in
// app/circumstances.py.
#include "common.hpp"
#include "eclipse/circumstances.hpp"

void bind_circumstances(nb::module_& /*m*/) {}

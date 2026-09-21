// Local circumstances (eclipse/circumstances.hpp) — phase 3, work package B
// adds the tests here: parity against the ``ovl`` / ``obsc`` / ``altaz`` /
// ``roots`` records of fixtures/circumstances_cases.txt (offline; the ``unw``
// records are checked in test_numerics.cpp), the ``local`` / ``grid`` records
// of the three modern per-eclipse fixtures (kernels required; skipped
// otherwise), kernel-free unit tests through the injectable ``Model``
// callables, the grid-vs-scalar test and the 0.5-degree global-grid benchmark
// (< 0.5 s, roadmap §5). Gates: contacts 1e-9 h, magnitude 1e-12, grid 1e-12
// (roadmap §4); measured residuals are to be recorded here, never widened.
//
// Placeholder: this translation unit is listed in CMakeLists.txt so the
// executable's source list does not change when the tests land.

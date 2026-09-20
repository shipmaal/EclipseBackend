# Handoff: native C++ core (`libeclipse`) with Python bindings — Path B

Decision: build the numeric core as a **C++ library linking CSPICE + ERFA**,
exposed to Python via bindings, with the existing Python `app/` becoming a thin
facade over it. This is a deliberate architectural choice (the code is *not*
compute-bound today), taken for the **capabilities** a native core unlocks, not
for single-eclipse speed.

Do this in a **full-access local session** (needs a C++ toolchain, CMake, and
NAIF access). Read `CLAUDE.md` and `docs/CODE_REVIEW_FOLLOWUPS.md` first — the
scientific-code rules (cite every equation, reuse shared math, validate against
Espenak, explicit units) apply to `.cpp`/`.hpp` exactly as to Python.

## Why (capabilities this opens)

- **Bulk grid computation**: local circumstances / obscuration / magnitude over a
  10⁵–10⁶-point lat-lon grid → global obscuration & magnitude **maps**. The
  per-point work is pure geometry (no SPICE call per point — see "Concurrency"),
  so it is embarrassingly parallel.
- **Eclipse catalogs**: scan centuries–millennia for eclipses and their
  circumstances (Five-Millennium-Canon scale).
- **Limb-corrected limits** over dense grids (the LRO/Watts limb frontier).
- A **reusable, distributable native library** others can link (C++/C ABI).
- A **WebAssembly build** so the Three.js frontend can compute client-side with no
  backend — exactly what ELP/MPP02 already does with its JS generator.

## Guardrails (do not skip)

1. **Parity-first, oracle = current Python.** Keep the 26-test Python suite green
   throughout. Port one function at a time and assert the C++ result matches the
   Python result to ~1e-9 on the three reference eclipses (2017-08-21,
   2024-04-08 total; 2023-10-14 annular) *before* moving on — then re-check the
   Espenak references (width ~1 km, duration <1 s, magnitude ~1e-3).
2. **Python API unchanged.** `app/` keeps its signatures; internally it calls the
   extension. Tests and the frontend must not change behavior. Consider keeping
   the pure-Python implementation behind a flag as a permanent cross-check oracle.
3. **Citations + units carry into C++.** Doxygen docstrings with the `CLAUDE.md`
   reference keys and equation numbers; units in every signature; shared math
   (ellipsoid reduction, constants) in one header, mirroring `constants.py` /
   `geography._reduction_aux`.

## Dependencies (all reachable with full access)

- **CSPICE** (NAIF C toolkit) → link `cspice.a`. Read license terms (permissive,
  not OSI). Alternatively keep SPICE calls in Python and pass positions into C++.
- **ERFA** (BSD-licensed SOFA derivative; what pyerfa already bundles) → vendor
  the C source or link `liberfa`. Prefer ERFA over SOFA C for licensing.
- **nanobind** (recommended — small, fast, excellent NumPy/array support) or
  **pybind11**. Build with **scikit-build-core** + **CMake** so `uv`/pip build the
  wheel. Add `clang-format`/`clang-tidy`.
- DE440 kernel + `earth_latest_high_prec.bpc` (ITRF93) via `kernels.bootstrap
  --source naif`.

## Proposed layout

```
core/                     # C++17/20, no Python in the numeric path
  constants.hpp           # WGS-84 (a, f -> b, e^2), unit radius, K_PENUMBRA/K_UMBRA  [WGS84][Espenak]
  ellipsoid.hpp           # _reduction_aux, fund<->geo, parametric<->geodetic         [ES92 8.331-8.334]
  besselian.{hpp,cpp}     # besselian_instant (uses ephemeris + frames)               [ES92 8.322/8.323]
  frames.{hpp,cpp}        # ITRS/TOD via ERFA c2t06a / pnm06a / gst06a                 [SOFA][IERS2010]
  ephemeris.{hpp,cpp}     # thin CSPICE spkpos wrappers (apparent LT+S)                [SPICE]
  geometry.{hpp,cpp}      # limits/width root-find, great-circle helpers               [ES92][MeeusSE]
  circumstances.{hpp,cpp} # contacts/duration/magnitude/obscuration                    [Meeus98 54][Espenak]
  batch.{hpp,cpp}         # vectorized/OpenMP grid + catalog entry points (NEW)
bindings/
  _eclipse.cpp            # nanobind module: scalars + array/batch overloads
CMakeLists.txt            # links cspice + erfa; builds _eclipse
pyproject.toml            # scikit-build-core backend
app/                      # unchanged signatures; delegates to _eclipse (+ python oracle behind a flag)
```

## Concurrency note (design-critical)

CSPICE is **not thread-safe** (global kernel pool). Design so SPICE is called only
a handful of times (the few Besselian sample epochs), after which every per-grid /
per-catalog point is **pure math** (no SPICE) — that inner loop parallelizes with
OpenMP/`std::execution` trivially. Furnish kernels once at startup; treat the pool
as read-only thereafter.

## Phased plan (each phase parity-gated)

- **Phase 0 — skeleton.** CMake + scikit-build-core + nanobind "hello" module
  importable via `uv run python -c "import _eclipse"`. CI builds the wheel.
- **Phase 1 — `besselian_instant`.** Port it (linking ERFA + CSPICE). Assert
  parity vs `app.ephemeris.besselian_instant` on the 3 eclipses (<1e-9).
- **Phase 2 — reduction + limits.** `fund_to_geo`/`geo_to_fund`/`shadow_edge_limits`;
  parity + Espenak width.
- **Phase 3 — circumstances.** contacts/duration/magnitude/obscuration; parity +
  Espenak durations/magnitudes.
- **Phase 4 — batch/new capabilities.** Vectorized grid + catalog entry points
  (OpenMP). New endpoints: `/map` (obscuration/magnitude grid → frontend heatmap)
  and `/catalog` (scan a date range). This is the payoff.
- **Phase 5 (optional).** WASM build for client-side compute; ELP/MPP02 as a
  kernel-free fallback Moon (see below).

## Bring over from the legacy trees (`c_src/`, `f_src/`)

- **ELP/MPP02** (`c_src/ElpMpp02`) as an **independent Moon validator** (not the
  engine): a kernel-gated test asserting the SPICE DE Moon agrees with the analytic
  ELP/MPP02 Moon within the theory's tolerance (~metres/century for the DE-fit
  params). **Frame gotcha**: ELP output is in the **mean ecliptic & equinox of
  J2000** — rotate to J2000 equatorial with `R_x(ε₀)`, ε₀ = 23.4392911°, and apply
  the ICRS↔dynamical-equinox frame bias before comparing. The old `f_src/test.py`
  skipped this and was wrong. Cite Chapront & Francou 2003, *A&A* 404, 735.
- **`ElpMpp_trim.h`** truncation-with-accuracy-estimate: the basis for a kernel-free
  fallback Moon (and useful for the WASM build).
- **`c_src/spice`** (hand-rolled CSPICE C extension): superseded by SpiceyPy — do
  not revive; use as a reference only.

## Also do (full-access, independent of the core)

- Verify DE440 + ITRF93 load; compare ITRS vs ITRF93 residuals at the reference
  eclipses; add a hybrid eclipse to the validation set.
- Still-open pure-computation features: rise/set + maximum-eclipse curves,
  penumbral (partial-region) limits.

## First task

Phase 0 skeleton + Phase 1 (`besselian_instant`) parity, on a branch off `main`.
Keep `uv run pytest` and `ruff` green; add a `parity` test module comparing the
extension to the Python oracle.

# C++ core (`libeclipse`) — approach review and roadmap

> **Status (2026-09-24):** historical. The phases below are done, and the
> Python oracle they relied on was retired after commit `fe1a64a`: the C++
> core is now the implementation (`docs/CPP_NATIVE.md`, the current plan and
> roadmap). References to `app/*.py` modules and parity gates describe that
> period.

Companion to `HANDOFF_CPP_CORE.md` (the *why*). This is the *how*: a review of
the choices already made in the local vertical slice (link CSPICE from C++;
C++20; Ninja; vendored SPICE sources), and a phased roadmap for the codebase
that follows from them. The Python core is the oracle throughout: it is
vectorized, validated against Espenak, and now covers the whole feature set
(elements → geometry → circumstances → contacts → catalog), so every C++ unit
has a reference implementation and a reference eclipse to be measured against.

Everything in `CLAUDE.md` applies to `.hpp`/`.cpp`: cite every equation with
the same keys, units in every signature, one home for shared math.

## 1. The native dependency surface (what the port actually needs)

Measured from `app/` — this is the *entire* CSPICE + ERFA usage:

| CSPICE (N0067) | used for | ERFA | used for |
| --- | --- | --- | --- |
| `furnsh_c`, `kclear_c` | kernel pool | `eraC2t06a` | ITRS: GCRS→ITRS matrix (IAU 2006/2000A + EOP) |
| `str2et_c`, `tparse_c`, `et2utc_c`, `timout_c`, `deltet_c` | time scales | `eraPnm06a` | TOD: bias-precession-nutation |
| `spkpos_c` (`LT+S`, `J2000` / body-fixed) | apparent Sun & Moon | `eraGst06a` | TOD: GAST |
| `bodvrd_c` (`SUN`, `RADII`) | solar radius (paired with k1/k2) | | |
| `reclat_c` | replaced by 3 lines of NumPy; do the same in C++ | | |

Thirteen entry points. Two consequences:

1. **Vendoring CSPICE is justified but the alternative is real.** With a surface
   this small, "SPICE stays in Python, C++ gets `(et, moon, sun, rc2t)` arrays"
   would also work. The reasons to still link CSPICE natively are the ones in
   the handoff: a distributable library, a WASM build, and catalogs that call
   `spkpos_c` millions of times without Python in the loop. Keep both doors
   open: the `ephem` layer below is the *only* place SPICE is called, so it
   can be swapped for an injected-positions implementation later.
2. **Bit-level parity is achievable.** ERFA is C; pyerfa bundles the same
   source. Vendor liberfa at pyerfa's version and `c2t06a` produces identical
   doubles. CSPICE is the same N0067 that spiceypy 8 wraps. So the elements
   layer should match Python to ~1e-15, and any larger difference is a bug,
   not "numerics".

## 2. Verdict on the vertical slice's choices

### C++20 — keep

- nanobind needs C++17+; GCC ≥ 10, Clang ≥ 12, MSVC 2019 16.10+, and
  Emscripten (for the WASM phase) all support what we need.
- Use: `std::span<const double>` as the array-view type at every API boundary
  (nanobind `ndarray` → span with zero copy); designated initializers for
  `Elements`/`ReductionAux`; `<numbers>` for π; `constexpr` constants;
  concepts for the vectorized objective passed to `bisect`.
- **Do not use C++20 modules.** CMake/Ninja module scanning is still fragile
  (needs CMake ≥ 3.28, Ninja ≥ 1.11, and per-compiler quirks) and buys
  nothing for a library this size. Headers + a few `.cpp` files.
- CSPICE is C89 (f2c output). Compile it as **C** in its own target
  (`enable_language(C)`); never as C++. `SpiceUsr.h` carries `extern "C"`
  guards, so including it from C++ is fine.

### Ninja — keep

- It is what scikit-build-core uses to build the wheel anyway (it pulls the
  `ninja` PyPI wheel if none is installed), so local builds and `uv build`
  run the same generator. Commit a `CMakePresets.json` with a `ninja`
  configure preset + `debug`/`release`/`asan` build presets so `cmake
  --preset release && cmake --build --preset release` is the whole story.
- CSPICE is ~2,100 translation units; first build ≈ 2–4 min. Enable
  `ccache` via `CMAKE_C_COMPILER_LAUNCHER` in the presets and cache it in CI.

### Vendored SPICE sources — keep, and treat as mandatory

- **NAIF is unreachable from restricted networks** (some sandboxes return 403
  for `naif.jpl.nasa.gov`; `kernels/bootstrap.py` exists for the same
  reason). A `FetchContent`/`ExternalProject` that downloads the
  toolkit at configure time would make the build unreproducible exactly where
  the Python build already had to add a mirror. Vendor the source.
- Vendor the **source tree**, not `lib/cspice.a`: NAIF's prebuilt archive is
  per-platform and not built with `-fPIC`, which a Python extension needs.
  Layout: `third_party/cspice/{src/cspice/*.c,include/*.h,LICENSE}`,
  unmodified from `cspice.tar.Z` (N0067), so a toolkit bump is a directory
  swap. ~25 MB; mark it `linguist-vendored` in `.gitattributes` and exclude
  it from ruff/clang-tidy/formatters.
- Build flags (from NAIF's `mkprodct.csh`, which is the citation for them):
  Linux/macOS `-DNON_UNIX_STDIO -O2 -fPIC -w`; MSVC `-DMSDOS
  -DNON_ANSI_STDIO -D_COMPLEX_DEFINED -DOMIT_BLANK_CC`. Add nothing else; in
  particular no `-ffast-math` anywhere in the project (breaks parity) and no
  sanitizers on the CSPICE target (f2c code trips UBSan constantly; sanitize
  only `libeclipse`).
- Record `tkvrsn_c("TOOLKIT")` in a test that asserts `"CSPICE_N0067"`, so
  the vendored version and the spiceypy oracle cannot silently diverge.
- Vendor **liberfa** the same way (`third_party/erfa`, BSD-3, ~90 files, tiny)
  at the version pyerfa ships (see `erfa.__version__` in the venv: pyerfa
  2.0.1.x ⇒ ERFA 2.0.1). Both licenses permit redistribution with notice:
  add `THIRD_PARTY_NOTICES.md` (NAIF's terms include a no-endorsement clause;
  ERFA's require keeping the SOFA-derivation notice).

### The one thing the slice must get right before anything else: SPICE errors

CSPICE's default error action **aborts the process**. spiceypy hides this by
setting `erract_c("SET", "RETURN")`, `errprt_c("SET", "NONE")`, and after
every call checking `failed_c()` → `getmsg_c("LONG")` → `reset_c()` → raise.
The C++ wrapper must do exactly that (an RAII `SpiceCall` guard or a
`check()` after each call), converting to a `spice_error` exception that
nanobind maps to a Python exception. Add a test that requests a body with no
loaded SPK and asserts an exception, not a crash. This is the acceptance test
for "link SPICE to C++".

Also from day one: **one global mutex around every CSPICE call** (the kernel
pool is process-global and not thread-safe — `app/ephemeris.SPICE_LOCK` is the
same rule in Python), and `furnsh_c` once at init. Everything else in the
library is pure math and must not touch SPICE, so it can run under
`std::execution::par_unseq` / OpenMP without the lock.

## 3. Architecture

Two layers, which is the split the Python vectorization already made explicit:

```
core/
  include/eclipse/
    constants.hpp      WGS-84 (a, f -> b, e^2), unit radius, K_PENUMBRA/K_UMBRA   [WGS84][Espenak]
    deltat.hpp         Espenak & Meeus delta-T polynomials                          [Espenak] 2.5
    numerics.hpp       bisect / sign_changes / parabolic_minimum over spans         (numerical)
    ellipsoid.hpp      ReductionAux, fund_to_geo, geo_to_fund, parametric<->geodetic [ES92 8.331-8.334][Meeus98 11]
    elements.hpp       Elements (SoA: x,y,z,d,mu,l1,l2,tan_f1,tan_f2), Frame enum
    ephem.hpp          furnish, utc_to_et, et_to_utc, earth_rotation_times,
                       besselian_instants, sub_solar_points      (the ONLY SPICE/ERFA users; mutex)
    geometry.hpp       shadow_edge_limits (vectorized bisection), shadow_radii,
                       great-circle helpers, central_track, global_contacts         [ES92 8.34][MeeusSE]
    circumstances.hpp  local_circumstances, circumstances_grid (parallel)           [ES92 8.35][Espenak][Meeus98 15]
    catalog.hpp        find_eclipses                                                [ES92 8.34][Espenak]
  src/                 one .cpp per header above that needs one (ephem, geometry, circumstances, catalog)
bindings/_eclipse.cpp  nanobind: spans <-> ndarray, exceptions, GIL released around compute
third_party/cspice, third_party/erfa
tests/cpp/             Catch2: unit + parity fixtures (JSON dumped from the Python oracle)
CMakeLists.txt, CMakePresets.json
```

Design rules:

- **Struct-of-arrays everywhere.** `Elements` holds `std::vector<double>` per
  element, exactly the dict `besselian_instants` returns; nanobind exposes
  each as an ndarray view. Grid results (`circumstances_grid`) likewise.
- **Spans in, vectors out.** Every public function takes `std::span<const
  double>` for arrays; no raw pointers, no Eigen dependency (the math is
  elementwise; Eigen would add a dependency for nothing).
- **Time enters as `et` (TDB seconds).** The era logic (IERS table vs
  delta-T model) lives in `ephem::earth_rotation_times`, ported verbatim.
- **EOP and kernels are injected, not parsed.** Python already reads
  `finals2000A.all` from `astropy-iers-data`; expose `set_eop_table(mjd, xp,
  yp, dut1)` and `furnish(path)` on the module. No C++ IERS parser, no second
  source of EOP truth.
- **The Python `app/` stays the API and the oracle.** `app/native.py` is the
  `ECLIPSE_BACKEND=python|native|auto` switch; since phase 4 the default is
  `auto`, which resolves to `native` when `_eclipse` imports and to `python`
  with one `RuntimeWarning` otherwise (an explicit `native` that cannot import
  is an `ImportError`). The pure-Python implementation is never deleted — it is
  the cross-check, and CI runs the whole suite through it explicitly.

## 4. Port map and parity gates

Each row is one PR: port, add the Catch2 unit test, add the parity test,
then the Espenak reference test already in `tests/` must pass through the
native path. Tolerances are what the *same* source/formulas justify; looser
means a bug.

| Python (oracle) | C++ unit | parity gate | reference test |
| --- | --- | --- | --- |
| `constants` | `constants.hpp` | exact | — |
| `deltat.delta_t_seconds` | `deltat.hpp` | 1e-12 s | `test_deltat.py` values (Meeus 10.A) |
| `numerics.*` | `numerics.hpp` | exact (same algorithm, same iteration count) | — |
| `geography._reduction_aux`, `fund_to_geo(_v)`, `geo_to_fund` | `ellipsoid.hpp` | 1e-13 | round-trip test |
| `ephemeris.earth_rotation_times`, `utc_to_et`, `et_to_utc` | `ephem` | 1e-9 s | 1919 / 2024 delta-T tests |
| `ephemeris.besselian_instants` | `ephem::besselian_instants` | 1e-13 (x, y, l1, l2), 1e-11 deg (d, mu) | 2017/2023/2024 elements + mu |
| `ephemeris.sub_solar_points` | `ephem` | 1e-11 deg | — |
| `geography.shadow_edge_limits_v` | `geometry` | 1e-9 deg / 1e-6 km | widths 114.7 / 197.5 / 187.4 km |
| `geography.global_contacts` | `geometry` | 1e-9 h | 2017/2024 P1–P4 |
| `circumstances.local_circumstances` | `circumstances` | contacts 1e-9 h, magnitude 1e-12 | durations 160/268/317 s; night-side; sunset |
| `circumstances.circumstances_grid` | `circumstances` (parallel) | 1e-12 | grid-vs-scalar test |
| `catalog.find_eclipses` | `catalog` | identical rows; `et_g` ≤ 1e-2 s (double-precision conditioning of the flat `rho` minimum, §5 phase 4), a whole-second `greatest_utc` flip admissible only at a rounding boundary inside that band; row gamma / magnitude 1e-12 and lat / lon 1e-11 deg at the same instant, plus the quantity's measured excursion over the band row-to-row; detail at the phase-2/3 gates | 2019–2024 canon |
| `besselian.BesselianModel` (polynomial fit) | **stay in Python** | — | it is a tabular product, not compute |

Parity harness: `tests/cpp/fixtures/*.txt` (whitespace records, no JSON
library needed) are dumped by `tools/dump_oracle.py` from the Python at the
three reference eclipses plus the 1919 epoch, with the IERS rows they need;
the Catch2 tests load them and skip when the pinned DE440s is not on disk.
That keeps C++ tests runnable without Python in the loop and pins the oracle
values in the repo. The stronger gate is `tests/test_native.py`, which
compares the live Python and native results over dense windows.

## 5. Phases and exit criteria

0. **Skeleton — DONE.** CMake + presets + vendored CSPICE/ERFA build as
   static libs; nanobind `_eclipse` importable via `uv run python -c "import
   _eclipse"`; `furnish` + `tkvrsn` + the *error-handling test* (§2) pass
   (`tests/cpp/test_ephem.cpp`, `tests/test_native.py`); CI builds on ubuntu +
   macos with ccache. Exit met: the wheel builds from a clean clone with no
   network beyond PyPI (`uv sync`). Measured: `str2et`/`et2utc`/`spkpos`
   bit-identical to spiceypy at the three reference instants; cold build
   ~55 s on 4 cores, ~10 s warm with ccache. Actual vendored size is 45 MB
   for CSPICE source (not the 25 MB estimated above) + 2.5 MB ERFA.
1. **Time + elements — DONE.** `constants`, `deltat`, `eop` (injected table),
   `elements`, and in `ephem`: `utc_to_et`/`et_to_utc` (IERS-era rule),
   `earth_rotation_times`, `besselian_instants`, `sub_solar_points`.
   `app/native.py` is the `ECLIPSE_BACKEND=python|native` switch; the five
   `app/ephemeris.py` functions dispatch at the top, the Python bodies stay
   as the oracle. Exit met: the whole suite passes with
   `ECLIPSE_BACKEND=native`. Measured parity (`tests/test_native.py`, 241
   instants × 4 eclipses × 4 frames): delta-T, EOP, time scales and
   `earth_rotation_times` bit-identical; elements x ≤ 5.5e-14, y ≤ 7e-15,
   d ≤ 4e-15 deg, mu ≤ 1.2e-13 deg, l1/l2 ≤ 2e-16 — the residual is NumPy's
   SIMD `arctan2`/`hypot`/`tan`/`arcsin` differing from libm by 1 ulp on
   AVX-512 hosts (verified directly), amplified by the Moon's 60-R⊕ distance.
   The oracle's `einsum` rotation was replaced by an order-defined
   multiply-and-sum (`app.ephemeris._rotate`) that is bit-identical to
   `eraRxp`; before that the vectors differed by 1 ulp (6e-11 km) and the
   oracle itself was CPU-dependent. Fixtures: `tools/dump_oracle.py` →
   `tests/cpp/fixtures/*.txt` (plain text, not JSON — no parser dependency).
2. **Ellipsoid + limits + contacts — DONE.** `numerics.hpp` (exact:
   `np.arange` fill rule, `np.remainder`, `sign_changes`, `bisect`,
   `parabolic_minimum` with the same call counts), `ellipsoid.hpp`
   (`reduction_aux`, `fund_to_geo`, `geo_to_fund`) and `geometry.hpp`
   (`shadow_radii`, great-circle helpers, `shadow_edge_limits`,
   `global_contacts`), bound as equal-length 1-D array functions; NumPy
   broadcasting stays in the `app/geography.py` glue. Six geography functions
   dispatch under `ECLIPSE_BACKEND=native` (`fund_to_geo_v`, `geo_to_fund`,
   `shadow_radii`, `bearing`, `shadow_edge_limits_v`, `global_contacts`);
   `global_contacts` is handed `(et0, frame)` and re-evaluates the elements
   natively. Exit met: the whole suite passes with `ECLIPSE_BACKEND=native`,
   so the Espenak widths 114.7 / 197.5 / 187.4 km, the 2017/2024 P1–P4 and the
   `/central-line` limits/contacts assertions run through C++, and
   `/central-line` parses to identical JSON through the two backends for four
   requests (test_api's and the default 4-hour tracks of the three reference
   eclipses). Measured parity (`tests/test_native.py`, x86-64):
   `_reduction_aux` and `global_contacts` bit-identical (0 residual, 200 001
   declinations; 2017/2023/2024/1919 contact sets in identical order);
   `fund_to_geo_v` lat 2.9e-14 deg, lon 5.7e-14 deg, `geo_to_fund` 2.8e-16
   over 234 256-point grids; `shadow_edge_limits_v` points 1.4e-13 deg, widths
   1.1e-11 km over 241-instant umbral and penumbral tracks of the three modern
   eclipses (gates 1e-9 deg / 1e-6 km). Residuals are 1-ulp libm-vs-NumPy
   SIMD trig differences, as in phase 1; at |mu| → 180 the longitude wrap's
   intermediate (~540 deg) makes that ulp 1.14e-13, which the parity test
   bounds separately from the 1e-13 gate it asserts at eclipse-realistic
   |mu| ≤ 120. Fixtures: `tests/cpp/fixtures/geometry_cases.txt` (+ the
   per-eclipse `contacts` records) from `tools/dump_oracle.py`.
3. **Circumstances — DONE.** `circumstances.hpp` (`series`, `roots`,
   `bracketed_series`, `refine_contacts`, `refine_maximum`, `overlap_area`,
   `obscuration`, `sun_altaz` / `sun_alt_grid`, `local_circumstances`,
   `circumstances_grid`), every expression in the Python's operation order;
   the ephemeris enters through an injected `Model` so the grid's observer
   loop runs OpenMP-parallel after one serial element evaluation under the
   SPICE lock (`ECLIPSE_OPENMP`, serial fallback without libomp; identical
   results for any thread count — no cross-observer reductions). Bound as
   `local_circumstances(et0, frame, hw, lat, lon)` → the `_LocalRaw` tuple
   and `circumstances_grid(et0, frame, hw, lats, lons, step_minutes,
   threads)` → the oracle's dict (`to_numpy_bool` for `visible` /
   `central`), GIL released; `app/circumstances.py` dispatches
   `local_circumstances` (numbers native, `_format_local` Python) and
   `circumstances_grid` (`chunk` ignored natively). Exit met: the whole suite
   passes with `ECLIPSE_BACKEND=native`, so the Espenak durations 160 / 268 /
   317 s, the magnitudes, the NYC partial, the night-side `eclipse False`,
   the Irish sunset `["max", "C4"]`, the grid-vs-scalar and the 1919
   duration assertions run through C++, and `/circumstances` and `/map` parse
   to identical JSON through the two backends for twelve requests
   (test_api's, the seven reference sites, the 10° map and the default 2°
   maps of the three eclipses). Measured parity (`tests/test_native.py`,
   x86-64, 39 sites × 3 eclipses incl. poles, antimeridian, a 1-hour window
   that forces the bracketing to widen): flags and NaN masks identical,
   contacts C1–C4 bit-identical (0.0), t_max 2.55e-12 h (the parabolic
   step amplifies 1 ulp of the magnitude samples), magnitude / obscuration /
   L2_x 1.5e-14, altitude 2.1e-14 deg, azimuth 5.7e-14 deg (gates 1e-9 h /
   1e-12 / 1e-11 deg); grid over 3 × 684 cells: `visible` / `central`
   identical, t_max_hours 0.0, magnitude 7.3e-14, obscuration 6.3e-14,
   sun_alt 5.7e-14 deg (gate 1e-12), `threads=1` equal to the OpenMP default.
   Benchmark (`test_circumstances_grid_benchmark`, `ECLIPSE_BENCH=1`; the
   0.5° grid exceeds `MAX_MAP_CELLS`, so it calls the function directly):
   361 × 720 observers × 181 instants in 0.38 s at the OpenMP default on
   the 4-core development host (1.39 s serial) against 4.0 s for the Python
   `circumstances_grid` (which it matches: bools identical, floats ≤ 9.4e-14
   on that grid too); gate 0.5 s there, 2.0 s in CI as a regression guard
   on shared 2–4 vCPU runners (`ECLIPSE_BENCH_MAX_S`). Offline fixtures:
   `tests/cpp/fixtures/*.txt` `local` / `grid` records from
   `tools/dump_oracle.py` (`tests/cpp/test_circumstances.cpp`). Operational
   note: libgomp is not fork-safe after its pool has started, so a forking
   server (gunicorn) must not `--preload` and warm the native grid in the
   parent. The oracle's ValueError (partial roots all one-signed) maps to
   ValueError natively; no site of a 5° global scan reaches it.
4. **Catalog + batch — DONE.** `catalog.hpp` (`coarse_grid`, `local_minima`,
   `scan_candidates`, `refine_greatest`, `classify`, `hybrid`, `add_detail`,
   `find_eclipses`), every expression in the Python's operation order, the
   ephemeris injected through `Sources` (kernel-free scan tests); the
   per-event hybrid / detail loop and `ephem`'s element loop are OpenMP
   (identical results for any thread count). One formula change, made in both
   languages: the scan / refinement objective is the frame-free
   `axis_separation` (`rho`, `z` from the un-rotated J2000 vectors, the same
   [ES92] eq. 8.322-6 — invariant under any Earth-orientation rotation) rather
   than `hypot(x, y)` of the ITRS elements, so the scan needs no ERFA / EOP
   per sample. Its consequence is the parity gate: `rho` is quadratic and flat
   at its minimum and carries ~3e-12 of sample-to-sample SPK evaluation jitter
   (identical in both backends), so the refined `et_g` is determined only to
   ~1 ms by double precision in *any* implementation — the roadmap's "times to
   1e-6 s" is not attainable for `et_g` itself and the gate is 1e-2 s with
   rows identical up to a whole-second `greatest_utc` flip that is admissible
   only when the oracle's `et_g` lies within |Δet_g| of the .5-s rounding
   boundary (asserted and warned per row, never silent). Bound as
   `find_eclipses(et_a, et_b, earth_frame, detail, threads)` → the `_EventRaw`
   tuples, GIL released; `app.catalog.find_eclipses` dispatches (the rows are
   formatted by the same `_format_event`). `app/native.py` gained the `auto`
   default (§3). Exit met: the whole suite passes with the default backend
   (native) and with `ECLIPSE_BACKEND=python`, so the 2019–2024 canon of
   `tests/test_catalog.py` and its greatest-eclipse rows run through C++, and
   `/eclipses` parses to identical JSON through the two backends for four
   requests (test_api's two, the canon window, 2023–2024 with detail);
   `/map` and `/eclipses` default to native. Measured parity
   (`tests/test_native.py`, x86-64, DE440s): 2019–2024 (detail off and on)
   and 2000–2050 (112 eclipses) rows identical with 0 boundary cases; `et_g`
   3.3e-3 s; the oracle replayed at the native's `et_g` agrees to gamma
   1.3e-14, magnitude 1.9e-14, lat 1.0e-12 deg, lon 5.7e-12 deg; row against
   row gamma 4.4e-12, magnitude 2.2e-10, lat 2.6e-5 deg, lon 2.3e-5 deg
   against sampled band excursions of 2.9e-11, 2.2e-9, 9.0e-5 deg, 3.6e-4 deg
   (the `et_g` shift times the quantity's rate); `et0` and the global
   contacts bit-identical, LocalRaw times 1.0e-14 h, magnitudes 0, altitudes /
   azimuths 5.7e-14 deg, width 9.9e-13 km; `threads=1` equal to the OpenMP
   default. Offline (`tests/cpp/test_catalog.cpp`, fixtures
   `catalog_cases.txt` / `catalog_2019_2024.txt`): grids and candidate sets
   identical, `et_g` 3.3e-3 s, at the oracle's `et_g` gamma 1.1e-14 /
   magnitude 2.8e-15 / lat 5.5e-14 deg / lon 1.5e-12 deg. Benchmark
   (`test_catalog_century_benchmark`, `ECLIPSE_BENCH=1`, gate
   `ECLIPSE_CATALOG_BENCH_MAX_S` = 10 s; 30 s in CI): 2000–2100, 226
   eclipses, through `app.catalog.find_eclipses` on the 4-core development
   host 9.3 s with detail and 4.6 s without (the C++ alone 8.5 s / 4.07 s;
   12.1 s serial) against 7.9 s for the Python oracle with detail *off* (its
   detail scan takes minutes). Docker builds the core in a two-stage image
   (wheel on `uv:python3.11-bookworm`, runtime slim + `libgomp1`); gunicorn
   must not `--preload` (libgomp is not fork-safe after first use).
5. **Optional.** (a) WASM via Emscripten: CSPICE builds under emcc; kernels
   go through the virtual FS — trim an SPK with `spkmerge` to the years the
   viewer needs (DE440s is 32 MB, a decade is ~1 MB). (b) ELP/MPP02 from
   `c_src/` as a kernel-free Moon *validator* (see the handoff for the
   J2000-ecliptic → equatorial frame gotcha). (c) A C ABI (`eclipse_c.h`) if
   other languages need it.

## 6. Tooling

- `pyproject.toml`: `build-backend = "scikit_build_core.build"`; keep `uv`.
  `cibuildwheel` for manylinux/macos wheels (CSPICE compiles slowly —
  ccache + one wheel per CI job).
- `clang-format` (LLVM style, 100 cols to match ruff) and `clang-tidy`
  (`modernize-*`, `bugprone-*`, `performance-*`) on `core/` and `bindings/`
  only; `third_party/` excluded.
- Presets: `release`, `debug`, `asan` (ASan+UBSan on `libeclipse`, not on
  CSPICE), `wasm`.
- Catch2 v3 via `FetchContent` is acceptable (GitHub is reachable where NAIF
  is not) — or vendor the single-header amalgamation to be safe.

## 7. Risks

- **Parity drift from "improvements" during the port.** Port first, improve
  after, in both languages, with the reference tests. Never fix a formula
  only on one side.
- **Floating-point reassociation.** Keep the operation order of the Python
  (e.g. `l1 = (z + k/sin f1) * tan f1`), no `-ffast-math`, `-ffp-contract=off`
  if FMA contraction ever shows up as a 1-ulp parity failure.
- **CSPICE global state + parallel grids.** Enforced structurally: only
  `ephem` includes `SpiceUsr.h`; a `static_assert`-style check is not
  possible, so make it a `clang-tidy` `misc-include-cleaner` rule plus a
  code-review rule.
- **Licensing.** CSPICE is permissive but not OSI; ERFA is BSD. Ship both
  notices; do not call the product "SPICE-based" in a way that implies NAIF
  endorsement.

## 8. Definition of done

The C++ core "replaces the Python compute" when: `uv run pytest` is green
with `ECLIPSE_BACKEND=native`, the parity suite is green at the tolerances in
§4, and the phase-3/4 benchmarks hold. The pure-Python path remains in the
tree, tested, as the oracle.

**Status (phase 4): met.** The default backend is native (`auto`); the whole
suite (116 tests with the NAIF kernel set: 114 pass and the two benchmarks
skip unless `ECLIPSE_BENCH=1`) is green both with the default and with
`ECLIPSE_BACKEND=python`; every
row of §4 dispatches natively except `BesselianModel` (a tabular product) and
the scalar / formatting helpers; the parity residuals are recorded per test in
`tests/test_native.py` and `tests/cpp/test_*.cpp`; the 0.5° grid runs in 0.4 s
and the century catalog with detail in 9 s on 4 cores; CI runs the oracle and
the native core as two explicit full-validation passes. The Python `app/`
remains the API and the oracle, and `tools/dump_oracle.py` regenerates the
offline fixtures from it (pinning `ECLIPSE_BACKEND=python` itself).

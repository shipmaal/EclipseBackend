# Native core: from port to C++-first

`docs/CPP_ROADMAP.md` built `libeclipse` as a line-by-line port of `app/`, so the
Python could be its bit-level oracle. Phases 0–4 are done. This document is the
next step, agreed on 2026-09-24: **C++ becomes the primary implementation,
designed as C++**. Layout, ownership, allocation, API shape, caching and
parallelism are free to change. **The per-value arithmetic and its operation
order are not.**

## The rule (replaces "Port, don't improve")

- **Same arithmetic, native structure.** Each value is computed with the
  same operations in the same order as `app/` (and so NumPy), so the parity
  tests stay bit-level. Everything around the arithmetic is C++'s to design:
  - buffers and workspaces instead of per-call vectors;
  - concrete types and concepts instead of `std::function` duck typing;
  - owned data instead of data injected from Python;
  - hoisting, batching and caching.
- **A formula change is still made in both languages** in the same PR, and
  `tools/dump_oracle.py` regenerates the fixtures. Python stays the oracle,
  and the API.
- **Batch boundaries can be arithmetic.** `unwrap_mu_deg` runs over the vector
  it is given (`circumstances.cpp`, `evaluate_direct`), so splitting or merging
  `elements_at` calls there changes results when a wrap falls inside a batch.
  Those boundaries stay as they are. The catalog (no unwrap) and geometry
  (`mu` unused there) can re-batch freely.
- **NumPy-shaped arithmetic that must stay:**
  - `numerics::arange`'s delta rule, `unwrap`'s sequential sum;
  - `np_remainder` / `wrap_180`, `np_clip` / `np_maximum` NaN rules,
    `py_round`;
  - `parabolic_minimum`'s `((0.5 h)(y0 − y2)) / denom`;
  - `norm` as `(x² + y²) + z²`, `dot3`;
  - `np.interp`'s fallback in `eop.cpp`, Horner in `deltat.cpp`;
  - `sun_altaz` vs `sun_alt_grid` (they differ on purpose);
  - the `mu` degrees → radians → degrees round trip.
- **libm, not NumPy SIMD,** wherever the two languages must agree on
  transcendental values: `app.limb._atan2` and `_libm`. The build keeps
  `-ffp-contract=off` and math-errno, so the compiler never merges or fuses
  anything behind our back. Repeated `sin` / `cos` calls are *not* merged
  automatically, and hoisting them by hand is a real, bit-identical saving.

## Done

- **Limb band ownership** (`docs/LIMB_PROFILE.md` §3.4, §9.11). The core reads
  the band file itself (`limb::load_band_file`), with no per-point neighbour
  indices (row-buffer stamps) and no NumPy copy under the native backend.
  `LDEM_64` 14°: 5.4 GB → 0.44 GB peak, 8–22 s → 1 s load. Silhouettes are
  bit-identical (fixtures unchanged).

## Plan (payoff vs risk; each item its own PR unless small)

From a survey of the core. Every item keeps bit parity.

1. **Profile-mode hot path** (`limb.cpp`, `circumstances.cpp::profile_g`).
   *Large CPU and memory payoff, small change.*
   - `g_total` / `g_annular` rebuild the 7200-point unit circle (14 400 trig
     calls) plus three scratch vectors *per instant*. Build it once per
     `n_bins`, and add a per-instant `g_*_one(…, span profile, scratch)`.
   - `profiles_at` materialises n × 7200 doubles (~35 MB for a 600-sample
     window). Interpolate `p0 + w (p1 − p0)` lazily per instant instead.
   - The node cache: a small LRU keyed by (node, frame, moon frame,
     generation) with in-flight de-duplication, instead of a `std::map` keyed
     by a string built per lookup and cleared wholesale at 256 entries.
2. **One geocentric evaluation per instant set** (`ephem.cpp`). *Halves
   SPICE/ERFA work in the grid and profile mode.* The grid computes the
   elements and the sub-solar points on the same `t` with two full
   `spkpos` + `c2t06a` passes. Profile mode does the same for the `K_REF`
   elements, and `limb_axes` repeats the pattern serially. One `Geocentric`
   should feed all of them.
3. **Innermost-loop hoisting** (`ellipsoid.hpp`, `geometry.cpp`). *Trivial.*
   - `geo_to_fund_one` evaluates `cos θ` 2–4× and `sin θ` 1–2×.
   - `shadow_edge_limits` recomputes `reduction_aux(d[i])` ~104× per instant
     and `sin` / `cos(lat0)` for all 52 steps.
4. **Allocation-free solvers** (`numerics.hpp`, `circumstances.cpp`).
   - Today a `VectorObjective` returns a `std::vector`, and each bisection
     iteration of `local_circumstances` allocates ~25 vectors (times, three
     geocentric, rotation, nine elements, unwrap temporaries, aux, series).
   - Native design: objectives write to a caller span, and one workspace per
     call. `sign_changes` / `roots` become scans for the first falling / last
     rising crossing.
5. **SPICE lock and EOP access** (`ephem.cpp`, `eop.cpp`). *Better parallel
   scaling in the catalog.*
   - `deltet` gets its own `spice_call` after `spkpos`.
   - Every instant locks the EOP table twice and runs three `upper_bound`
     searches for one abscissa.
   - Native design: batch `deltet` into the same locked call, take one
     `eop::Snapshot` per call, and share the interval index across columns.
6. **Concrete models** (`circumstances.hpp`, `geometry.hpp`, `catalog.hpp`).
   - `Model`, `Sources`, `ElementsAt` / `SubSolarAt` / `ProfilesAt` /
     `RhoAt` are `std::function` hooks. Production always binds them to
     `ephem`; the hooks exist only so tests can inject fakes.
   - Native design: a concrete `EphemerisModel` (et0, frame, scratch, shared
     geocentric cache) and templates constrained by a concept, so tests keep
     their fakes. This also retires the per-call `check_elements` contracts.
7. **Element storage** (`elements.hpp` and every struct-of-arrays result).
   - `Elements` is nine `std::vector`s filled with `push_back`, and `Series`,
     `SubSolar`, `EarthRotation`, `EdgeLimits` and `GridResult` follow suit.
   - Native design: one 9 × n block, or an array-of-structs per instant, in
     the hot loops. SoA only at the NumPy boundary, as one capsule. It is the
     broadest change, so it goes last.
8. **EOP file ownership** (`eop.cpp`). The core parses `finals2000A.all`
   itself, the fixed-width columns Python reads today. `float()` and
   `strtod` both round correctly, so the result is bit-identical. Small
   payoff; it mostly removes an injection step.
9. **Binding copies** (`bindings/`).
   - `limb_axes` copies `vector<array<9>>` to a flat vector.
   - `besselian_instants` returns nine capsules.
   - `limb_g_*` and `limb_delta_rho_at` hold the GIL.

## Verification (unchanged in kind)

- Live parity in `tests/test_native.py` and the offline fixtures in
  `tests/cpp/` stay bit-level where they are today. A layout change that moves
  a residual is a bug, not a tolerance to widen.
- Performance claims are measured and recorded in the PR: peak RSS, wall time
  for `/map`, one profile-mode `/circumstances`, and a 20-year catalog.

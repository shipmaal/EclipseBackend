# The C++ core: status and roadmap

`docs/CPP_ROADMAP.md` built `libeclipse` as a line-by-line port of the Python
`app/`, with the Python as its bit-level oracle (phases 0–4). On 2026-09-24 the
project went **C++-first**: the core is the implementation, designed as C++,
and the Python math was retired. This document records that step and plans
what follows.

## Done: the Python math is retired

**What moved into the core** (so `app/` computes nothing):

| was Python | now |
| --- | --- |
| `app/eop.py` (parse `finals2000A.all`) | `eop::load_file`; `app.core` passes the `astropy-iers-data` path |
| `BesselianModel._fit` (NumPy `polyfit`) | `besselian::fit_polynomials` (Householder QR on the column-scaled Vandermonde matrix) |
| `evaluate_direct` (the `mu` unwrap in NumPy) | `besselian::elements_direct` |
| `geography.central_track`, `element_rates`, the `/central-line` assembly in `main.py` | `besselian::central_line`, `besselian::element_rates` |
| the constants in `app/constants.py`, `app/limb.py` | `_eclipse` module attributes from `constants.hpp` / `limb.hpp` |

**What went:** `app/{ephemeris,geography,circumstances (math),catalog (math),
limb,eop,deltat,numerics,constants,native}.py`, the `ECLIPSE_BACKEND`
switch and the spiceypy exception translation, `tests/test_native.py`
(Python-vs-native parity), `tests/test_backend_switch.py`,
`tools/dump_oracle.py` and the parity-only bindings (`bind_numerics`).
`app/` is 1 000 lines, mostly docstrings and response shaping. At runtime it
needs `_eclipse`, NumPy, FastAPI, `astropy-iers-data` and `requests` (for
`kernels.bootstrap`). spiceypy, pyerfa and pandas are dev dependencies: the
independent test oracles and the reference-table loader use them.

**Equivalence, measured before deleting the oracle** (commit `fe1a64a` vs this
change, same machine):
- Every `/central-line`, `/circumstances` (mean and profile, at a height),
  `/map` and `/eclipses` response over 19 requests was JSON-identical. So were
  the error responses.
- `/besselian` coefficients differ at 1e-15 relative (QR vs NumPy's SVD least
  squares).
- `fit_polynomials`, `elements_direct`, `element_rates` and `central_line`
  against the Python on five reference eclipses × two frames: ≤ 3.5e-10 (deg /
  km / Earth radii). That is the existing NumPy-vs-libm ulp residual,
  amplified through the limit search.
- Timing is unchanged: the pre-change and new builds give the same grid and
  catalog benchmarks on one host.

**What guards the numbers now** (`CLAUDE.md`, "The core"):
- the reference-eclipse tests ([Espenak], [EB2024], [Irwin21]);
- the independent oracles (3D limb ray tracer, geodesy vectors), whose SPICE /
  ERFA come from spiceypy / pyerfa, not the core;
- the golden fixtures in `tests/cpp/fixtures/`, frozen from the last Python
  oracle run. A change that moves one re-baselines it in the same PR, says by
  how much, and keeps the reference tests green.

The "same arithmetic" rule is gone with the oracle: operation order, hoisting
and fusion (under `-ffp-contract=off`) are C++'s to choose. So is merging
`unwrap_mu_deg` batches. The price of a bit change is a golden re-baseline,
not a design constraint.

## Baseline (this change; 4 vCPU cloud VM, NAIF kernels, `LDEM_16` unless noted)

| scenario | first call | warm | peak RSS |
| --- | --- | --- | --- |
| `/map` (2°, the default) | 0.071 s | 0.045 s | 77 MB |
| `/circumstances`, mean limb | 0.046 s | 0.030 s | 72 MB |
| `/circumstances?limb=profile&elev=185` (Indianapolis 2024) | 0.70 s | 0.25 s | 154 MB |
| the same on `LDEM_64` (14° band) | 5.8 s | 0.26 s | 452 MB |
| `/central-line` (2-min step, ±2 h) | 0.078 s | 0.053 s | 72 MB |
| catalog 2000–2020, `detail=True` (46 eclipses) | 2.13 s | — | 74 MB |
| 0.5° global grid benchmark (259 920 observers) | — | 0.43–0.50 s | — |
| catalog 2000–2100, `detail=True` (226) | 10.3–10.9 s | — | — |

Reproduce by timing the endpoints through `fastapi.testclient` in a fresh
process, one scenario each. The grid and century-catalog lines are
`ECLIPSE_BENCH=1 uv run pytest tests/test_core.py -k benchmark`. On this VM
the old developer-host figures (0.38 s, 8.5 s) are not reached by the
pre-change build either.

Where the warm profile call's time goes (re-measured by the 2026-09-24
review; 0.29–0.35 s on this VM): the mean-limb part is 0.065 s. The rest is
`profile_contacts`' 0.5-s grid, about 580 instants, costing 0.20 s. In that
grid, elements take 0.02 s, `profiles_at` 0.01 s and the G loop 0.17 s. The
G loop is 4.2 M sin/cos pairs (≈ 0.055 s) plus 4.2 M `atan2` (≈ 0.075 s).
The number of instants is what dominates, not the unit circle.

## Roadmap (revised by the 2026-09-24 review)

A review of the whole core against the first roadmap (A1–A5, B1–B8) found
two things:
- correctness bugs that no item covered (track **C**);
- hygiene and CI gaps that let the goldens go unchecked (track **H**).

It also re-scoped several items. B1's target was not reachable as scoped. B3
was aimed at the wrong loop. A3's parabola-minimum design is ill-posed. B7
would cost more than it pays.

The tracks:
- **C**: correctness, first.
- **H**: hygiene and CI; must precede any golden move.
- **A**: lunar limb.
- **B**: structure and performance.

**Order:** C → H → A1+B1 (merged) → A2 → B3 → A3 → catalog fixes (C8) →
A4 → A5 → B4 → B2 → B5 → B6. B7 is dropped. B8 is folded into H.

Each item is its own PR unless noted. Each PR:
- records the baseline table's rows it can move, before and after;
- runs `pytest` on NAIF + mirror, `ctest` and `ruff`;
- re-baselines any golden it moves, with the size of the move.

### C. Correctness (the 2026-09-24 review)

Each fix gets a test that fails before it.

**Status: C1–C7 done** (one commit each, 2026-09-24). No golden moved. NAIF:
131 passed, 2 skipped, 2 xfailed. Mirror: 118 passed, 15 skipped, 2 xfailed.
`ctest` 86/86. Where the fixes differ from the plan below:
- **C2**: the offset from the model tapers linearly to zero at 2050, the end
  of [Espenak]'s 2005–2050 segment. This is a project rule, recorded in
  `deltat.hpp`.
- **C3**: the mean limb had the same fault as the profile. A central phase
  shorter than the 30-s grid step came back "partial" with magnitude > 1;
  it is now bracketed from the refined maximum.
  - A profile partial's magnitude is 1 − G_T/(2 R_s) at maximum, which is
    [Espenak]'s partial magnitude for a smooth limb.
  - Its obscuration stays the mean limb's: a bead is ~1e-6 of the disk.
- **C5**: the API bounds were kept. A fit window under 1.5 h is sampled at
  2 hw / 3 instead.
- **C6**: the nested-region guards now use `omp_get_level() == 0`.
- **C7**: the `et_to_utc` era test (it classifies by the ET MJD) remains
  open.

C8 remains, after A3.

**C1. UT1 across a leap second** (`eop.cpp`).
- UT1−UTC jumps by 1 s at each leap second, and `eop::interpolate` draws a
  straight line across the jump. On the day before the jump that is up to
  0.5 s wrong. Example: MJD 48803.5 gives −0.056 s, but the true value is
  ≈ −0.557 s.
- That day, 1992-06-30, is the day of a total eclipse (greatest ≈ 12:11 UT).
  The error there is about 7″ in `mu` and about 0.2 km on the ground.
- Fix: interpolate UT1−TAI (= UT1−UTC − (TAI−UTC)), the practice of the IERS
  interpolation routines. That is linear across the step.

**C2. ΔT continuity at the end of the EOP table** (`ephem.cpp`, `deltat`).
- The table's last row is MJD 61659 (2027-09-11). TT−UT1 is 69.29 s the day
  before and 76.13 s the day after, from the [Espenak] 2005–2050 polynomial.
- So every eclipse after the table's end moves by about 7 s and about 3 km,
  and the move depends on which `finals2000A` is installed.
- Fix: past the last row, carry the last measured ΔT and blend it into the
  model. The rule must be continuous and documented with its source. Add a
  test that asserts continuity at the boundary.

**C3. Magnitude when the profile removes totality**
(`circumstances.cpp::local_circumstances`).
- Case 1: the mean limb is central but the profile finds no central phase.
  The partial formula `(L1−m)/(L1+L2)` then gives a magnitude > 1 inside the
  mean umbra, labelled "partial", with obscuration 1.
- Case 2, the reverse: "total" is reported with the mean limb's obscuration
  below 1.
- Fix: make both consistent with the reported type, and test both.
- Related: `profile_g` picks G_T or G_A by the K_REF cone's L2′, but the type
  follows K_UMBRA's `L2_x`. Near the hybrid transition these can disagree.

**C4. Profile node-cache key and band snapshot** (`limb.cpp`).
- The key holds the band generation, but not the kernel-pool generation or
  the EOP table. A `furnish` / `kclear` or a new EOP table therefore serves
  stale profiles.
- The band is also read separately from its generation. Take the (band,
  generation) pair once per call.

**C5. Error model** (`app/main.py`, `besselian.cpp`).
- Example: `window_hours ≤ 1.25` passes validation but yields fewer than 4
  fit samples. The core's `ValueError` is then reported as 400 "invalid
  epoch".
- Fix: validate the real bound. Map core errors by kind, not all
  `ValueError` as a parse error.

**C6. OpenMP teams** (`app/core.py`, `ephem.cpp`).
- `PARALLEL_LOCK` does not cover profile mode (the silhouette team) or
  `/central-line`'s geocentric loop. `app/core.py` says it does.
- A cold node is not de-duplicated between concurrent requests.
- `threads=1` still opens a nested team in `ephem` (an inactive outer
  region).

**C7. Small items.**
- `numerics::arange`: step 0 or non-finite is UB (a cast from inf), and
  there is no length cap. Guard it in the core.
- `format_clock` truncates to the second; it should round.
- `spice_call` skips `reset_c` if a C++ exception follows a failed call.
- `eop.hpp` promises a one-day boundary tolerance that the code does not
  have.

**C8. Catalog and global contacts** (after A3; these are scientific changes
with golden moves).
- The keep test and the partial / non-central split use the sphere
  `hypot(x, y)`, while `global_contacts` uses the ellipse `hypot(x, y/ρ1)`.
  Near the thresholds (about 20 km) a "partial" row can carry U1/U4.
- `hybrid` reads `l2` at the first and last on-Earth 5-min samples. A central
  line shorter than 5 min is never "hybrid". Bisect the central line's ends
  instead.
- `global_contacts` brackets on a 1-min grid, so a shorter grazing umbral
  phase loses U2/U3. Add the `rho` minimum to the samples.
- `/map`'s "visible" uses the Sun's altitude at maximum, while
  `/circumstances` uses any event above the horizon. They disagree for
  sunrise and sunset eclipses.

### H. Hygiene and CI (before the first golden move)

- **Strict goldens in CI.**
  - The C++ golden tests `SKIP` without `de440s.bsp`. `native-tests`
    bootstraps with `--source auto`, whose fallback is the mirror (DE432s),
    so they can skip while the job stays green.
  - Fix: use `--source naif` there and fail on skips.
  - The ITRF93 rows always skip, and their comments cite the deleted
    `tests/test_native.py`. Retire or re-home them.
- **Sanitizers.** Add an `asan` (ASan + UBSan) `ctest` job; it passes today.
  Consider a TSan run over the mutex, cache and OpenMP paths.
- **Portability.** macOS runs no `ctest`, so "goldens stay portable" (arm64,
  Apple libm, no OpenMP) is unchecked.
- **Citations** (convention 1). About 70 `app.*` references remain.
  - `ephem.hpp` and `deltat.hpp` / `.cpp` name deleted Python files as the
    scientific reference.
  - Rewrite them as descriptions with the source keys, in one PR.
- **Retired rule.** Remove "PORT, DON'T IMPROVE / do not reassociate" from
  `numerics.hpp`, `ellipsoid.hpp`, `geometry.hpp` and `catalog.hpp`.
- **Fixture generator.** The fixture headers still name the deleted
  `tools/dump_oracle.py`. Add a dumper from `_eclipse` before any B item moves
  a golden, and record the writing commit in each header.
- **Binding copies** (old B8; negligible payoff, do on touch):
  - `limb_axes` flattens its output;
  - nine capsules from `besselian_instants`;
  - `limb_g_*` hold the GIL.

### A. Lunar limb (docs/LIMB_PROFILE.md)

**A1 + B1. Bead-safe, coarse-to-fine contact search and the G hot path**
(`circumstances.cpp::profile_contacts`, `limb.cpp`).
- The bead problem (§9.11): G is sampled every 0.5 s, so a sub-0.5 s bead is
  caught or missed depending on the grid phase. Vale at sea level on
  `LDEM_16` has a 0.38 s bead that moves C2 by 0.9 s.
- The test: an interval Δt whose ends a, b have the same sign can hide a sign
  change only if |a| + |b| < L·Δt.
- Search coarse-to-fine:
  - start at about 8–16 s;
  - subdivide intervals that fail the test or bracket a change, down to
    Δt_min ≈ 0.05 s.
  - Deep in totality |G| is tens of km against L ≈ 1–1.5 km/s. So this takes
    about 580 instants to 60–100, and that is the speed-up.
  - Refining down from the 0.5 s grid would only add cost.
- **L per lattice node:** the planar speed plus (R_m/R_s)·max|dδρ/dψ|·dψ/dt,
  taken from the node's profile. A constant cannot cover `LDEM_64` crater
  walls (≈ 0.3 km/s of slope term at dψ/dt ≈ 0.016°/s).
- **Beads:** a bead shorter than Δt_min is not a contact. Decide whether a
  seen bead is "C2", and report beads explicitly if so. The same rule defines
  A3's graze zone.
- **Hot path:**
  - build the unit circle once per `n_bins`;
  - replace `atan2` with arg Q = φ + asin(|P| sin(θ_P − φ)/|Q|), a short
    series since |P| ≪ R_s;
  - evaluate `p0 + w(p1 − p0)` per instant instead of materialising
    `profiles_at` (33 MB at 580 instants; memory, not time);
  - bisect about 15 iterations, not 30 (a 0.5 s bracket to sub-ns).
- Gates:
  - [EB2024] median / max no worse;
  - Vale matches [Irwin21];
  - warm profile time ≤ 0.15 s (from 0.30 s). The old 0.06 s target was
    below the mean-limb call alone.

**A2. Oracle test, bead-agnostic** (`tests/test_limb_oracle.py`). Unchanged.
- Replace the bracket check with the mismatch that
  `tools/limb_study.py oracle` computes. Take all roots of each method within
  ±2.5 s of the contact, then evaluate each method at the other's roots, in
  metres of limb height.
- Keep the 75 m gate. Exceedances are strict xfails with reasons.
- Measured maxima: `LDEM_16` / 7200 bins 82 m; `LDEM_64` / 7200 93 m;
  `LDEM_64` / 28800 57 m.
- The Vale-at-height xfail should then pass. Note the annular point's 22 /
  34 m G_A systematic.

**A3. Limb-corrected limits and catalog** (PR 3 of `LIMB_PROFILE.md` §6,
redesigned).
- The first design took the minimum over τ of G with a parabolic
  refinement, inside `shadow_edge_limits`' 50-halving bisection. It is
  ill-posed:
  - G is a max of piecewise-linear functions (kinked), and its minimum near a
    limit is a bead;
  - it costs about 50 × 480 × 52 × 2 evaluations of G.
- Instead, G depends on the observer only through P (and weakly ζ). So once
  per instant:
  - compute the **umbral outline in the fundamental plane**, the set of P
    where the solar disk fits inside r_M(ψ);
  - take its support distance perpendicular to the track;
  - form the time envelope in the plane, map it with `fund_to_geo`, and
    iterate once for ζ.
- Report a **graze zone**: an outer limit (any bead) and an inner limit
  (continuous totality), consistent with A1's bead rule. Otherwise a point
  inside the limit can come back non-central from `/circumstances`.
- Part of this PR, not B1:
  - an LRU node cache with in-flight de-duplication, replacing today's
    256-entry wholesale clear;
  - a lattice pre-build for the path. Cold nodes dominate: about 50 per
    ±2 h path, ≈ 3.5 s on `LDEM_16` and ≈ 60 s on `LDEM_64`;
  - the catalog's per-event loop runs `silhouette` serially.
- Add `limb=` to `/central-line` and `/eclipses?detail=true`.
- Gates:
  - the 3D oracle at the limits (≤ 50 m);
  - the SVS limb- and terrain-corrected limits
    (`docs/LIMB_VALIDATION_SOURCES.md`), at the sea-level equivalent or with
    the terrain noted.

**A4. `LDEM_64` at 0.05° bins.** Unchanged.
- Expected gains (§9.11): [EB2024] 36 mid-path 0.30 / 0.61 / 1.10 →
  0.28 / 0.59 / 0.98 s, bias halved, Vale inside [Irwin21].
- A 16° band (≈ 230 MB) covers the 1900–2100 libration maximum of 6.65°.
- Host it on `limb-data` as files under 100 MB, reassembled and
  SHA-checked. Update the pins and every test comment that states achieved
  numbers.

**A5. Validation gate and the grid** (PR 4).
- §5.4 external gate 0.3 s on the median. p90 (0.59 s) is a strict xfail for
  site-to-site scatter.
- The default stays `limb=mean`.
- Profile mode on `/map`: reuse A3's per-instant outline as a
  point-in-region test in P-space, shared by all observers. Per-observer G
  cannot fit ≤ 2× `/map`.
  - Only `central` changes. Document magnitude and obscuration as mean-limb.
- Update the `CLAUDE.md` accuracy ceiling.

### B. Structure and performance

**B3. The limit envelope's inner loop** (`geometry.cpp`, `ellipsoid.hpp`).
Before A3, which adds work to the same loop.
- Production always passes `rates`. Each envelope residual then does 10
  `geo_to_fund_one` calls: about 1 030 `reduction_aux` + `site()` calls per
  instant per limit, not ~104. And `d` varies with τ, so `reduction_aux`
  cannot be hoisted per instant.
- The wins:
  - `Site(lat)` once per trial point;
  - an early exit when the centre is off-Earth or no edge is found
    (bit-neutral);
  - about 32 halvings instead of 50 (sub-millimetre; moves goldens, so
    re-baseline).

**B4. The ephemeris out of the solvers** (`circumstances.cpp`,
`numerics.hpp`).
- Each bisection iteration of `local_circumstances` makes about 25
  allocations. But it also makes 2 `spkpos` and a `c2t06a` per instant,
  under the global mutex, and that dominates.
- Replace those with a local interpolant of the elements over the window,
  checked against the direct evaluation (moves goldens).
- Then the allocation-free spans, and `sign_changes` → first falling / last
  rising scans. Choose the rule for touching zeros deliberately.
- `bisect` should check its bracket and the objective's result length.

**B2. One geocentric evaluation per instant set.** The grid evaluates
elements and sub-solar points on the same `t` in two passes, but only about
181 instants per grid. Small; do it with B4's cache.

**B5. EOP snapshot** (`eop.cpp`, `ephem.cpp`).
- Take one `eop::Snapshot` per call:
  - for consistency first: one call can read two tables today;
  - then for speed: two locks and three `upper_bound`s per instant today.
- `deltet` is already one lock per vector. The catalog is limited by the
  SPICE mutex, not EOP, so expect little thread scaling from this.

**B6. Concrete models.**
- Replace the `std::function` `Model` / `Sources` / `*At` hooks with a
  concrete `EphemerisModel` and concept-constrained templates.
- The value is structure, not speed. The main one: `add_detail` hard-wires
  `ephem` and ignores the injected `Sources`, so detail cannot be tested
  kernel-free.

**B7. Dropped.**
- The hot loops already read contiguous SoA, which suits SIMD.
- AoS could lose SIMD, and it is the broadest change.
- Revisit only with a profile that shows a storage cost.

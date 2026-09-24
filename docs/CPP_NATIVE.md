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

The warm profile call is 5× the mean-limb one. `limb::g_total` rebuilds the
7 200-point unit circle (14 400 trig calls) for every instant, over ~600
instants. That is item B1.

## Roadmap

Two tracks: A finishes the lunar-limb work, now in C++ only; B is structural
and performance work from the survey of the core. They interleave because A3
and A4 multiply the profile evaluations that B1 makes cheap.

**Order:** A1 → A2 → B1 → A3 → A4 → B2 → A5 → B3 → B4 → B5 → B6 → B7 → B8.
Each is its own PR unless noted. Each PR:
- records the baseline table's rows it can move, before and after;
- runs `pytest` on NAIF + mirror, `ctest` and `ruff`;
- re-baselines any golden it moves, with the size of the move.

### A. Lunar limb (docs/LIMB_PROFILE.md)

**A1. Bead-safe contact search** (`circumstances.cpp::profile_contacts`).
- The problem: G is sampled every 0.5 s (`PROFILE_STEP_H`), so a sub-0.5 s
  bead (a brief entry into totality near a limit) is caught or missed
  depending on the grid phase. Vale at sea level on `LDEM_16` has a 0.38 s
  bead that moves C2 by 0.9 s (§9.11).
- The fix: G is Lipschitz in time with a constant L. An interval of length
  Δt whose ends a, b have the same sign can hide a sign change only if
  |a| + |b| < L·Δt. Subdivide such intervals, and those with a sign change,
  down to Δt_min ≈ 0.05 s, one batch of midpoints per level. Then take the
  first falling / last rising root.
- First measure L:
  - the planar speed |dP/dt| ≲ 1–1.5 km/s;
  - the limb-slope term ((R_m / R_s) · dδρ/dψ / R_ref) near a steep valley;
  - the lattice's time interpolation (small).
  Take L with margin and cite the measurement.
- Document the definition: a bead shorter than Δt_min is not a contact. Is a
  bead the observer sees "C2"? Report beads explicitly if the answer is yes.
- Gates:
  - [EB2024] median / max no worse;
  - Vale matches [Irwin21];
  - profile warm time (baseline table) ≤ 1.5×.

**A2. Oracle test, bead-agnostic** (`tests/test_limb_oracle.py`).
- Replace the bracket check with the mismatch that
  `tools/limb_study.py oracle` computes. Take all roots of each method within
  ±2.5 s of the contact, then evaluate each method at the other's roots, in
  metres of limb height.
- Keep the 75 m gate. Exceedances are strict xfails with reasons, never a
  wider gate. Measured maxima: `LDEM_16` / 7200 bins 82 m (Vale at sea level,
  the profile's bead); `LDEM_64` / 7200 93 m; `LDEM_64` / 28800 57 m.
- The Vale-at-height xfail should then pass (≤ 73 m).
- Note or investigate the annular point's 22 / 34 m G_A systematic.

**A3. Limb-corrected limits and catalog** (PR 3 of `LIMB_PROFILE.md` §6).
- Add the profile residual to `geometry::shadow_edge_limits`' envelope: the
  minimum over τ of G around closest approach (±2 min, parabolic
  refinement). The limit is the interior one, where totality just occurs.
- Add `limb=` to `/central-line` (`besselian::central_line`) and
  `/eclipses?detail=true` (duration, width).
- Gates:
  - the 3D oracle at the limits (≤ 50 m);
  - the SVS limb- and terrain-corrected limits
    (`docs/LIMB_VALIDATION_SOURCES.md`: 2024 at −97 / −86 / −80°, 2017 at
    −121 / −100 / −89.2 / −84.5°), compared at the sea-level equivalent or
    with the terrain noted.
- Cost: G per limit point per τ step. B1 first.

**A4. `LDEM_64` at 0.05° bins** (§9.11: [EB2024] 36 mid-path 0.30 / 0.61 /
1.10 → 0.28 / 0.59 / 0.98 s, bias halved, Vale inside [Irwin21]).
- Band width: 14° covers libration ≤ 6°, while the 1900–2100 maximum is 6.65°.
  So 16° (a view axis ≤ 8°) is needed for every eclipse; ~230 MB.
- Hosting: add a commit to `limb-data` (never rewrite it) with the band split
  into files under 100 MB. `kernels.bootstrap` reassembles them and checks the
  SHA-256. The `naif` source cuts `ldem_64.img` from PDS locally.
- Update the pins, `DEFAULT_BAND_FILE`, CI download size and every test
  comment that states achieved numbers.
- Cold start is 5.8 s (baseline table), mostly lattice nodes at 1.1–1.5 s
  each. Consider building the two nodes around T0 at band load, or a
  persistent node cache.

**A5. Validation gate and the grid** (PR 4).
- §5.4 external gate 0.3 s: gate the median; p90 (0.59 s) is a strict xfail
  whose reason is site-to-site scatter, not resolution or elevation.
- The default stays `limb=mean` (§8).
- Give `circumstances_grid` a profile mode within ≤ 2× `/map`, or document it
  as mean-only.
- Update the `CLAUDE.md` accuracy ceiling.

### B. Structure and performance (survey of the core)

**B1. Profile-mode hot path** (`limb.cpp`, `circumstances.cpp::profile_g`).
*Largest payoff; prerequisite for A3 and A4.*
- Build the unit circle once per `n_bins` (cached), and add a per-instant
  `g_total_one` / `g_annular_one(…, span profile, scratch)`.
- `profiles_at` materialises n × 7200 doubles (~35 MB for a 600-sample
  window). Interpolate `p0 + w (p1 − p0)` per instant instead.
- Node cache: an LRU keyed by (node, frame, moon frame, band generation) with
  in-flight de-duplication, instead of a `std::map` keyed by a string built
  per lookup and cleared wholesale at 256 entries.
- Target: warm profile `/circumstances` 0.25 s → ≤ 0.06 s; peak RSS down by
  the `profiles_at` buffer.

**B2. One geocentric evaluation per instant set** (`ephem.cpp`).
- The grid computes elements and sub-solar points on the same `t` with two
  full `spkpos` + `c2t06a` passes. Profile mode does the same for the `K_REF`
  elements, and `limb_axes` repeats it.
- One `Geocentric` (Moon, Sun, rotation) should feed all of them.
- Target: `/map` and profile mode, fewer SPICE / ERFA calls, measured.

**B3. Innermost-loop hoisting** (`ellipsoid.hpp`, `geometry.cpp`).
- `geo_to_fund_one` evaluates `cos θ` 2–4× and `sin θ` 1–2×.
- `shadow_edge_limits` recomputes `reduction_aux(d[i])` ~104× per instant and
  `sin` / `cos(lat0)` for all 52 steps.
- May move goldens in the last bit: re-baseline.

**B4. Allocation-free solvers** (`numerics.hpp`, `circumstances.cpp`).
- Objectives write to a caller span, with one workspace per call. Today each
  bisection iteration of `local_circumstances` allocates ~25 vectors.
- `sign_changes` / `roots` become scans for the first falling / last rising
  crossing.

**B5. SPICE lock and EOP access** (`ephem.cpp`, `eop.cpp`).
- Batch `deltet` into the `spkpos` locked call.
- Take one `eop::Snapshot` per call, sharing the interval index across
  columns. Today each instant locks the table twice and runs three
  `upper_bound` searches.
- Target: catalog scaling with threads (century benchmark, serial vs
  default).

**B6. Concrete models** (`circumstances.hpp`, `geometry.hpp`, `catalog.hpp`).
- `Model`, `Sources` and the `*At` hooks are `std::function`s that
  production always binds to `ephem`.
- Replace them with a concrete `EphemerisModel` (et0, frame, scratch, the
  shared geocentric cache of B2) and templates constrained by a concept, so
  the tests keep their fakes. This also retires the per-call
  `check_elements` contracts.

**B7. Element storage.** `Elements`, `Series`, `SubSolar`, `EarthRotation`,
`EdgeLimits` and `GridResult` are struct-of-arrays filled by `push_back`. Use
one block or AoS per instant in the hot loops, and SoA only at the NumPy
boundary. It is the broadest change, so it goes last.

**B8. Binding copies.**
- `limb_axes` copies `vector<array<9>>` to a flat vector.
- `besselian_instants` returns nine capsules.
- `limb_g_*` and `limb_delta_rho_at` hold the GIL.

### Housekeeping (any PR that touches the unit)

- Doc comments still name the Python function each unit was ported from
  (`app.geography.*` …, commit `fe1a64a`). Rewrite them as descriptions when
  the unit is next edited.
- The golden fixtures have no generator now. A PR that re-baselines adds a
  small dumper for the records it touches (from `_eclipse`), and the fixture
  headers say which commit wrote them.

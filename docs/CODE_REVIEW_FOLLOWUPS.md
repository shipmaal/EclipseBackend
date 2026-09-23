# Code review follow-ups

Scientific code review of the eclipse backend, tracked for follow-up work. Goals:
**scientific accuracy** and a **modern scientific codebase** — every non-trivial
equation cited (see `CLAUDE.md` References + conventions) and shared math factored
into reusable functions. The math is validated correct today (width ~1 km,
duration <1 s vs Espenak); these items are about traceability, consistency, and
structure, not wrong results.

Work them in the "Suggested order" below; validate against the reference eclipses
(2017-08-21, 2024-04-08 total; 2023-10-14 annular) after each step and commit per
item.

**Status: all items closed** (A1–A5, R1–R5, M1–M5, and the §2 citation pass).
Each was validated against the three reference eclipses with no regression
(greatest-eclipse lat/lon, path width, contact times and magnitudes unchanged to
< 1 km / < 1 s), the test suite grew from 17 to 26 (new pure-function tests), and
`ruff check` is clean. Details per item below.

## 1. Scientific accuracy

- **A1 — Ellipsoid constants duplicated / slightly inconsistent (medium). DONE.**
  The fundamental-plane unit was SPICE `a_e = 6378.1366 km` (`ephemeris.py`
  `earth_equatorial_radius_km`), but the reduction used WGS-84 `a = 6378.137`,
  `b = 6356.752` (`geography.py:30-31`). Fixed: `app/constants.py` now defines the
  WGS-84 ellipsoid from `a` + `f = 1/298.257223563` [WGS84] and derives `b, e²`;
  the same `a` is the fundamental-plane unit radius, used to scale the Sun/Moon
  vectors in `ephemeris.py` and as the reduction ellipsoid in `geography.py`.
  Reference eclipses unchanged (width Δ < 0.3 m, times/magnitudes identical).
- **A2 — Polynomial extrapolation / window-clipped contacts (medium; only
  accuracy-affecting item). DONE.** Added
  `BesselianModel.evaluate_direct(t)`, which recomputes elements per instant with
  `besselian_instant` (exact everywhere, no polynomial extrapolation). `/central-line`
  (`main.py`) and `local_circumstances` (`circumstances.py`) now use it; the
  polynomial `evaluate()` stays as the `/besselian` tabular product. Contacts are
  additionally bracketed: `_bracketed_series` widens the 30-second sampling grid
  (up to ±6 h) until the observer is outside the penumbra at both ends, so C1/C4
  are never clipped. Verified: a deliberately narrow ±1 h fit window (which
  previously clipped C4) now recovers the exact C1/C4/magnitude of the ±2.5 h
  reference for a partial observer (NYC, 2024-04-08). Reference-eclipse numbers
  unchanged (direct == polynomial inside the window).
- **A3 — Documented approximations (low). DONE.** Geometric-only caveat (no
  atmospheric refraction; mean lunar limb folded into `K_UMBRA`) now stated in the
  docstrings of `geography.fund_to_geo`, `geography.shadow_edge_limits` and the
  `circumstances` module.
- **A4 — TDB used as TT for ERFA (low). DONE.** Noted inline in
  `ephemeris._geocentric_vectors` (≤1.7 ms → sub-mas).
- **A5 — WGS-84 `b` rounded (low). DONE.** Folded into A1: `b` is now derived
  from `a` and `f` in `app/constants.py`, no longer hard-coded.

## 2. Citation coverage (core goal) — DONE

Applied the keyed references from `CLAUDE.md` with equation numbers across the
math modules (all rows of the table below):
- `ephemeris.py`: x,y,z [ES92] 8.322-6; cones [ES92] 8.323-1/6/7 + k1/k2 [Espenak];
  `c2t06a` [SOFA]+[IERS2010], `pnm06a`/`gst06a` [SOFA].
- `geography.py`: `_reduction_aux`/`fund_to_geo`/`geo_to_fund` [ES92] 8.331-8.334 +
  parametric↔geodetic [Meeus98] 54; great-circle helpers noted standard spherical
  trig; `shadow_edge_limits` noted numerical, limit defn [ES92]/[MeeusSE];
  `shadow_radii` [ES92] 8.353.
- `circumstances.py`: method [ES92] 8.353-8.354 / [Meeus98] 54; magnitude &
  obscuration [Espenak]; circle overlap = standard geometry.
- `eop.py`: EOP [IERS2010].

Original table (all rows now cited):

| Location | Formula | Citation |
| --- | --- | --- |
| `ephemeris.py` x,y,z | fundamental coords | [ES92] 8.322-6 |
| `ephemeris.py` cones | l1,l2,f1,f2 | [ES92] 8.323-x; k [Espenak] |
| `ephemeris.py` c2t06a/pnm06a/gst06a | Earth orientation | [SOFA], [IERS2010] |
| `geography.py` `fund_to_geo` | reduction + parametric→geodetic | [ES92] 8.33x / [Meeus98] 54 |
| `geography.py` `geo_to_fund` | inverse reduction (uncited) | [ES92] 8.331 |
| `geography.py` great-circle helpers | destination/haversine/bearing (uncited) | standard spherical trig |
| `geography.py` `shadow_edge_limits` | limit/width root-find | note numerical; limit defn [ES92]/[MeeusSE] |
| `circumstances.py` | magnitude/obscuration/contacts (uncited) | [Meeus98] 54; magnitude [Espenak]; circle overlap = geometry |
| `eop.py` | EOP interpolation | [IERS2010] |

## 3. Reusable functions (second north star)

- **R1 — DONE.** `ρ1/ρ2/d1/d2` auxiliaries were duplicated in `fund_to_geo`,
  `geo_to_fund`, `shadow_radii`. Extracted `_reduction_aux(d)` (a `_ReductionAux`
  NamedTuple) in `geography.py`, cited [ES92] 8.331; all three call it.
- **R2 — DONE.** WGS-84 ellipsoid + fundamental-plane unit radius + mean radius
  centralized in a cited `app/constants.py` and imported by `geography.py` and
  `ephemeris.py` (fixes A1). The lunar-radius constants `K_PENUMBRA`/`K_UMBRA`
  stay in `ephemeris.py`, which owns the shadow-cone geometry (per `CLAUDE.md`).
- **R3 — DONE.** Added `central_track(model, t) -> (elems, [TrackPoint(i, t, lat,
  lon, bearing)])` in `geography.py`; it evaluates directly (item A2), reduces to
  the central line, skips off-Earth instants, and tags each point with its
  along-track bearing. `/central-line` (`main.py`) and the integration-test
  greatest-eclipse helper both use it.
- **R4 — DONE.** Time formatting now lives only in `geography.py`: `dec_to_hms`
  (unchanged, tested), plus `format_offset(t)` (`±HH:MM:SS.s`, used by `main.py`)
  and `format_clock(t0_utc, t)` (absolute UTC `HH:MM:SS`, replaces
  `circumstances._clock`).
- **R5 — DONE.** Promoted `build_model_best_frame(t0_utc, ...)` and
  `best_earth_frame()` into `app/besselian.py`; the integration tests'
  `_build_model`/`_usable_frame` are now thin wrappers over them.

## 4. Modern codebase

- **M1 — DONE.** Added **ruff** (config in `pyproject.toml`; `c_src`/`f_src`
  vendored trees excluded) and fixed all findings in `app/` + `tests/`. Added
  `.github/workflows/ci.yml` with two jobs: (1) ruff + pure-function tests
  (kernels auto-skip), (2) `kernels.bootstrap --source mirror` + full
  reference-eclipse validation.
- **M2 — DONE.** `_build_model` (`main.py`) now catches
  `(spiceypy.utils.exceptions.SpiceyError, ValueError)` instead of
  `Exception`, so genuine bugs are no longer masked as HTTP 400.
- **M3 — DONE.** "JPL DE440" → "a JPL DE ephemeris (DE440 NAIF / DE432s mirror)"
  in `besselian.py`, `ephemeris.py`, `main.py`; `besselian_instant` and the
  `/besselian` `frame` description now list `ITRS` (and the full frame set).
- **M4 — DONE.** `local_circumstances` returns a `LocalCircumstances` TypedDict
  (`total=False`; documents the optional C2/C3/duration keys); `/central-line`
  points use a `CentralLinePoint`/`LimitPoint` TypedDict.
- **M5 — DONE.** Added pure-function tests (no kernels): `geo_to_fund`↔`fund_to_geo`
  round-trip (`tests/test_geography.py`) and `_overlap_area`/`_obscuration`
  analytic cases (`tests/test_circumstances_pure.py`), plus `format_offset`/
  `format_clock`. Suite is now 26 tests (17 → 26).

## Suggested order

1. `constants.py` (cited WGS-84 + unit radius; fixes A1, R2).
2. Direct-evaluation + contact bracketing (A2) — the only accuracy-affecting item.
3. Extract `_reduction_aux` and `central_track` (R1, R3); unify time format (R4).
4. Citation pass — apply §2 with equation numbers, per `CLAUDE.md`.
5. Tooling: ruff + CI + the two pure-function tests (M1, M5); doc/exception fixes
   (M2, M3, M4).

## 5. Native-core review (phases 0–4) — DONE

A second review, targeted at the C++ port (`14fb261..c92720a`, excluding
`third_party/` and the fixtures): concurrency and the CSPICE global state, the
binding boundary, port fidelity, and API hardening. **No finding changes an
eclipse number.** The parity residuals, the Espenak agreement and the
`axis_separation` change (the same in both languages) all hold. The findings
are about crashes, error mapping and behaviour under load. Each fix has a
regression test that fails on the old code, except C6 (timing-dependent, see
below) and C9 (configuration).

- **C1 — `/eclipses` failed on a fresh worker (high). DONE.** The range guard
  in `app/main.py` called `utc_to_et` before `find_eclipses` loaded the kernels,
  so a worker's first request had no leap-second kernel and got a 400
  "SPICE error". The handler now calls `load_kernels()` first.
  Test: `test_eclipses_on_a_fresh_worker_loads_the_kernels`.
- **C2 — Out-of-bounds read in `sign_changes` (medium). DONE.** The core loops
  over `t.size()` and reads `f[i + 1]`, and the binding did not check lengths.
  `numerics::sign_changes` now throws `std::invalid_argument` (→ `ValueError`)
  when `f` is shorter than `t`; the Python oracle raises `IndexError` there.
  Test: `test_numerics_parity`.
- **C3 — `/map` allocated before its cell guard (high). DONE.** `lat_step=1e-6`
  allocated gigabytes and `1e-300` gave a 500. `_arange_len` computes
  `ceil((stop − start)/step)` (the `np.arange` length) as a float, which is
  `inf` on overflow, and the guard runs before any array exists.
  Test: `test_map_tiny_step_is_a_400_before_any_allocation`.
- **C4 — An empty `ECLIPSE_BACKEND=` stopped the app starting (medium). DONE.**
  `app/native.py` now reads a set-but-empty variable as `auto`.
  Test: `test_empty_backend_behaves_like_unset`.
- **C5 — `utc_to_et` bypassed `spice_call` (medium). DONE.** `tparse_c` ran
  under a hand-taken lock with no `failed_c()` check or `reset_c()`. A SPICE
  error (e.g. `SPICE(EMPTYSTRING)`, raised before `errmsg` is written) was left
  pending in the global error state for the next caller, and `errmsg` was read
  uninitialized. It now goes through `spice_call` and raises exactly as
  `spiceypy.tparse("")` does; the result is unchanged for every valid string.
  Test: `test_utc_to_et_spice_error_is_raised_and_reset`.
- **C6 — The kernel pool could change mid-vector (low). DONE.** `spkpos_both`
  releases the SPICE mutex between 2048-instant batches, so a concurrent
  `furnish`/`kclear` could leave one vector using two kernel sets. `furnish`
  and `kclear` now bump a pool generation under the mutex, and `spkpos_both`
  raises `RuntimeError` (a 500: a server-side condition) if the generation
  changes between its batches. `test_kernel_pool_change_mid_vector_is_an_error_never_mixed`
  checks that every run is either the untouched result or that error. The
  race is rare in practice: glibc's mutex is unfair, so the batch loop re-takes
  it before a waiting `furnish` wakes, and 0 of 5 runs hit it here. The guard
  is therefore verified by inspection rather than by the test forcing it.
- **C7 — `/eclipses` reported compute errors as "invalid epoch" (medium).
  DONE.** Only epoch parsing is now caught as `ValueError` → 400. A
  `ValueError` from the computation (native invariant checks surface as that)
  is a 500. Test: `test_eclipses_compute_value_error_is_a_500_not_invalid_epoch`.
- **C8 — `/central-line` gave a 500 for a subnormal `step_minutes` (low).
  DONE.** The same `_arange_len` guard as C3.
  Test: `test_central_line_subnormal_step_is_a_400`.
- **C9 — Too many OpenMP threads under gunicorn (medium). DONE.** libgomp
  creates a team per calling thread, sized to the host's full core count (it
  ignores cgroup quotas), and every worker and request thread made its own.
  `native.PARALLEL_LOCK` now serializes the two long parallel entry points
  (`circumstances_grid`, `find_eclipses`) within a process. The Docker CMD
  sets `OMP_NUM_THREADS` to `nproc / WEB_CONCURRENCY` (at least 1) unless it
  is already set, and sets `OMP_WAIT_POLICY=PASSIVE` so idle teams sleep
  rather than spin. Results do not depend on either (they are identical for
  any thread count, as established in phase 3/4).
- **C10 — Native dispatch rebuilt a wrapper on every call (low). DONE.**
  `_Translating.__getattr__` built a new closure plus `functools.wraps` on
  every attribute access. It now caches each wrapper on the proxy.
  Test: `test_native_proxy_caches_wrapped_functions`.

Validation: `uv run pytest` gives 122 passed / 2 skipped (the benchmarks),
both with the default backend (native) and with `ECLIPSE_BACKEND=python`;
`ruff check` is clean.

Not fixed, pre-existing: `ctest` fails the three `ITRF93` rows of "Besselian
elements parity" with or without these changes (e.g. `mu` differs by 3.7e-10
deg against a 1e-11 gate). The offline fixtures were dumped with an older
`earth_latest_high_prec.bpc`, which NAIF regenerates daily, while the `ITRS` /
`TOD` / `IAU_EARTH` rows pass. Regenerate the fixtures
(`tools/dump_oracle.py`) against the current kernel, or pin the binary PCK
that the fixtures were dumped with.

## 6. Reference-eclipse coverage and two width findings

Four reference eclipses added to `tests/test_besselian_integration.py`, each
checked against F. Espenak's Canon values (`SEsearch/SEdata.php`) [Espenak]:
the 2023-04-20 hybrid, the 2021-12-04 polar grazing total (γ −0.95, Sun 17°),
and two outside the IERS era, 1919-05-29 and 1868-08-18. The latter two
exercise the `deltat.py` model path. The 1919 case replaces the old smoke test.
Measured agreement:

| quantity | gate | worst new case |
| --- | --- | --- |
| x, y at t0 | 1e-3 | 1e-5 |
| d at t0 | 1e-3 deg | 9.3e-5 deg |
| μ (ephemeris hour angle via ΔT) | 1e-4 deg | 1.9e-5 deg |
| greatest-eclipse lat / lon | 0.12 deg | 0.049 deg |
| central duration | 3 s | +2.4 s (all +1.7 to +2.4) |
| magnitude | 0.003 | +0.0005 |

The cases are anchored on Espenak's published TD instant, not the Canon's UT:
the Canon's UT uses its own ΔT (73.4 s in 2023, against 69.2 s measured). The
1868/1919 cases skip on the mirror kernel set (DE432s covers 1949–2050).
DE440s covers 1849–2150, not "1550–2650" (that is DE440); fixed in
`kernels/bootstrap.py`, `kernels/README.md` and `app/catalog.py`. A 2186 case
was dropped for the same reason.

Path widths are a new parametrized test over all seven eclipses at the
existing 2 km gate. They surfaced two real discrepancies, recorded as
**strict xfails**; the gate was not widened.

- **W1 — The umbral cone is ~0.04% wider than Espenak's (open, medium).**
  Signed width residuals (ours − Espenak): 2017 +1.64, 2024 +1.37,
  2023-04 hybrid +1.51, 1919 +2.70, 1868 +2.30 km, and the 2023-10 annular
  −1.63 km. Every total is wider and the annular narrower. The source is
  `tan f1` / `tan f2`, both ~1.9e-6 (0.04%) below Espenak's for every eclipse;
  via `l2 = z tan f2 − k / cos f2` [ES92] eq. 8.323-7 with z ≈ 60, that alone
  puts `l2` ~0.7 km more negative. `k2` is identical (0.272281), so the
  difference is in `sin f = (d_s ± k) / G`. It corresponds to ~0.39″ of solar
  semi-diameter or a G convention (apparent vs geometric Sun–Moon distance)
  and needs its own investigation before any formula changes (both languages,
  per the port rule). xfail: 1919, 1868 (> 2 km on ~245-km paths).
- **W2 — Limits and width are the instantaneous shadow section, not the path
  envelope (open, high for low-Sun eclipses).** `shadow_edge_limits` bisects
  the shadow edge perpendicular to the track *at one instant*. The path limits
  are the envelope of the shadow over time. For 2021-12-04 the points ever
  inside the umbra along the same perpendicular span 420.8 km (Espenak 418.7);
  our instantaneous width is 412.0 km (−6.7 km). For the high-Sun cases the two
  measures agree to ≤ 0.4 km, which is why this went unnoticed. It affects
  `/central-line`'s N/S limits and widths for grazing / low-Sun paths. The
  fix is a formula change in both languages. xfail: 2021-12-04.

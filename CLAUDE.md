# CLAUDE.md

Guidance for working in this repository. This is a **scientific codebase**: the
numbers must be right and every non-trivial formula must be traceable to a
citation. Read the "Scientific conventions" section before touching any math.

## What this is

EclipseBackend computes the **Besselian elements** of a solar eclipse and
everything that follows from them — the central line, the northern/southern
limits and path width, and per-observer local circumstances (contact times,
duration, magnitude, obscuration), the lunar-limb-profile contacts, a global
map and an eclipse catalog.

**The C++ core (`libeclipse`, `core/`) is the implementation.** It is exposed
to Python as the nanobind extension `_eclipse` (`bindings/`), and a thin
FastAPI service (`app/`) parses requests, calls the core and formats JSON; a
dependency-free Three.js globe (`frontend/`) visualizes it. There is no Python
implementation of the math (the pure-Python `app/` that the core was ported
from was retired after commit `fe1a64a`; see `docs/CPP_NATIVE.md`).

Positions come from **NASA/NAIF CSPICE** + a **JPL DE** ephemeris; Earth
orientation from **ERFA** (IAU 2006/2000A) with **IERS** EOP — both vendored in
`third_party/` and linked statically into the core. Packaging is **uv** +
scikit-build-core. The project deliberately does **not** use astropy (only its
`astropy-iers-data` file bundle).

## Setup & commands

```bash
uv sync                                  # install into .venv; builds the core (_eclipse)
uv run python -m kernels.bootstrap       # download SPICE kernels (auto: NAIF else GitHub mirror)
uv run python -m kernels.bootstrap --limb  # + lunar limb profile inputs (LOLA band, MOON_ME kernels)
uv run pytest                            # tests (kernel-dependent tests auto-skip without kernels)
uv run uvicorn app.main:app --reload     # API + viewer at http://localhost:8000/ui/
cmake --preset release && cmake --build --preset release && ctest --preset release  # C++ tests
```

Environment: `SPICE_EARTH_FRAME` overrides the default frame (`ITRS`);
`SPICE_METAKERNEL` overrides the kernel path (`app.core.metakernel()`; the
mirror kernel set is run this way); `ECLIPSE_LIMB_BAND` overrides the limb-band
file. `uv sync` rebuilds the core when `core/`, `bindings/`, `cmake/`,
`CMakeLists.txt` or `third_party/` change (`tool.uv.cache-keys`); install
`ccache` to make that seconds instead of a minute.

## Architecture

Data flows one direction: **ephemeris → elements → geometry → circumstances /
catalog → bindings → `app/` → frontend**.

The core, `core/include/eclipse/*.hpp` + `core/src/*.cpp`:

| Unit | Responsibility |
| --- | --- |
| `ephem` | The only CSPICE / ERFA user: kernels, time scales (UTC / UT1 / TDB, the IERS-era rule), apparent Sun/Moon (`LT+S`), `besselian_instants` (the 8 elements + `z`, [ES92] 8.322-8.323), the four Earth-orientation frames (`ITRS`/`TOD`/`ITRF93`/`IAU_EARTH`), `sub_solar_points`, `axis_separation` (frame-free `rho`, `z`), `limb_axes` (fundamental-plane axes in `MOON_ME`). |
| `eop` | IERS polar motion + UT1−UTC: parses `finals2000A.all` (`load_file`) and interpolates it. |
| `deltat` | ΔT = TT − UT1 polynomial model [Espenak] for epochs outside the IERS era. |
| `constants` | WGS-84, the lunar radii `K_PENUMBRA`/`K_UMBRA` [Espenak], the solar radius (IAU 1976), unit conversions — their one home (exposed to Python as `_eclipse` attributes). |
| `besselian` | One model (`et0`, frame): `elements_direct` (mu unwrapped), `element_rates`, the tabular polynomial fit `fit_polynomials` (`/besselian`), `central_line` (the whole `/central-line` product). |
| `ellipsoid` | The WGS-84 reduction: `fund_to_geo` (fundamental→geographic), `geo_to_fund` (its exact inverse, with the observer's height, [Meeus98] ch. 11). |
| `geometry` | `shadow_radii`, great-circle helpers, `geodesic_km` (Andoyer), `shadow_edge_limits` (N/S limits + width, umbral or penumbral, with the time envelope), `global_contacts` (P1–U4). |
| `circumstances` | Per-observer local circumstances (mean limb, or C2/C3 from the limb profile; `height_m` above the ellipsoid), the `/map` grid (OpenMP), the profile contact function `profile_g` and search `profile_contacts`. |
| `catalog` | `find_eclipses`: scan a date range, refine greatest eclipse, classify P/A/T/H, Canon-style rows, detail (OpenMP). |
| `limb` | Lunar limb profile from the LOLA DEM (`docs/LIMB_PROFILE.md`): reads the band file, `silhouette()` (δρ per 0.05° bin, perspective), `profiles_at()` (5-min cached lattice), `g_total()` / `g_annular()`. |
| `numerics` | Bisection, sign changes, parabolic extremum, `arange`/`unwrap`/`remainder` with NumPy semantics. |

The Python side:

| Module | Responsibility |
| --- | --- |
| `bindings/` | nanobind: one `bind_<unit>.cpp` per core header; arrays in and out as NumPy float64, the GIL released around compute; `_eclipse.SpiceError` carries the CSPICE message fields. |
| `app/core.py` | The process state the core needs: kernels (`load_kernels`), the EOP file, the limb band (`ensure_limb_band`), `PARALLEL_LOCK`, the frame list. |
| `app/besselian.py` | `normalize_utc` (the single epoch parser) and `BesselianModel`, a handle (`t0_utc`, `et0`, frame, fit window, the fitted polynomials). |
| `app/circumstances.py`, `app/catalog.py` | Format the core's raw tuples (`_LocalRaw`, `_EventRaw`) into the API's dicts. |
| `app/formatting.py` | Clock / offset strings. |
| `app/main.py` | FastAPI: `/health`, `/besselian`, `/central-line`, `/circumstances` (`limb=mean|profile`, `elev=`), `/map`, `/eclipses`; serves `frontend/` at `/ui`. |
| `kernels/bootstrap.py` | Downloads kernels (`--source auto|naif|mirror`, `--limb`) and writes `eclipse.tm`. |
| `kernels/limb_band.py` | Cuts LOLA `LDEM_*` to the lunar limb band (deterministic, SHA-pinned file) and reads it back. |

**New math goes into the core**, with a binding if the API or a test needs
it. `app/` holds no numerics beyond formatting (the one exception: evaluating
the published `/besselian` polynomials, as a reader of the table would).

## Scientific conventions — READ BEFORE EDITING MATH

1. **Every non-trivial equation and physical constant carries a citation** in
   the nearest doc comment or an inline comment, using the keys in "References"
   below (e.g. `// Explanatory Supplement eq. 11.323` or `[Espenak]`). New math
   without a source is not done. Prefer the equation number, not just the book.

2. **Reuse, don't re-derive.** Shared geometry lives in one place:
   - ellipsoid reduction → `ellipsoid.hpp`; great-circle math, limits → `geometry`
   - Earth orientation / apparent positions / time scales → `ephem`
   - EOP → `eop`; ΔT → `deltat`; constants → `constants.hpp`
   If you find the same formula (parametric latitude, ρ1/d1 auxiliaries, a
   root-find, a constant) written twice, factor it out rather than copying.

3. **Units are explicit and stated in every signature.** Conventions:
   angles in **degrees** at interfaces (radians internally), distances in
   **Earth equatorial radii** for fundamental-plane quantities and **km**
   otherwise, times in **seconds** (TDB past J2000) / **hours from T0** as
   labelled. There is no units library — the doc comment is the contract, so
   keep it accurate.

4. **Validate against an independent published source.** Every capability that
   produces a number has a test in `tests/test_besselian_integration.py`
   comparing to Fred Espenak / NASA values for real eclipses (2017-08-21 total,
   2024-04-08 total, 2023-10-14 annular, 2023-04-20 hybrid, 2021-12-04 polar
   total, and pre-IERS 1919-05-29 / 1868-08-18 totals). Add a reference eclipse
   when you add a capability; state the achieved agreement in the test comment.
   Known, open discrepancies are strict xfails, never widened gates
   (`docs/CODE_REVIEW_FOLLOWUPS.md` §6). Path limits/widths are the time
   envelope of the shadow (`shadow_edge_limits` with `element_rates`),
   measured with `geodesic_km`; the solar radius is the constant
   `constants::SUN_RADIUS_KM` (IAU 1976), never the PCK's. Limb-profile
   contacts are checked against an independent 3D ray-traced oracle
   (`tests/test_limb_oracle.py`, gated in limb-height metres) and against
   [EB2024] and [Irwin21] at the sites' heights (`docs/LIMB_VALIDATION_SOURCES.md`).
   Independent checks must stay independent: the test oracles
   (`tests/geodesy_oracle.py`, `tests/test_limb_oracle.py`) take their
   constants from the sources and their SPICE / ERFA from spiceypy / pyerfa
   (separate builds, dev dependencies), never from the core.

5. **Frames & epochs.** Input epochs are **UTC** and must be ISO-8601
   (`app.besselian.normalize_utc` is the single parser). ΔT = TT − UT1 is
   measured (leap-second kernel + IERS UT1−UTC) **only inside the IERS era,
   1973 – ~1 yr ahead**; outside it the epoch is read as UT1 and ΔT comes from
   the Espenak & Meeus polynomial model in `deltat` [Espenak] — never rely on
   SPICE's leap-second table there (it silently holds the first/last count).
   Inside the era, UT1−UTC is interpolated as UT1−TAI across a leap second.
   Published Besselian elements are tabulated in **TDT/TT** (use
   `_eclipse.str_to_et(".. TDT")` to compare). Published `mu` is the
   *ephemeris hour angle* (Earth rotation evaluated as if TT were UT); ours is
   the true Greenwich hour angle, so
   `mu_published = mu_ours + ΔT·1.002738·15″/s` [ES92] 8.36 — see the mu test.

6. **Horizon.** The cone geometry is valid on the whole ellipsoid, including
   the night side; every local circumstance must be checked against the Sun's
   altitude (`circumstances::HORIZON_ALT_DEG`, [Meeus98] ch. 15) before being
   reported.

7. **No unsourced scientific rules.** Every model choice that changes a
   reported number (a formula, a definition, a data source, a blend or
   extrapolation, a physical constant) must come from a published source,
   cited by key. A result that looks wrong is presumed to be a bug in our
   code until an independent check (a published value, the test oracles)
   says otherwise; do not paper over it with a rule of our own. Where the
   published sources disagree or run out (e.g. ΔT past the last published
   prediction), keep the published behaviour and record the gap as a known
   open discrepancy (a strict xfail and a note), rather than inventing a
   join. A new rule of our own is a research result: it needs a validation
   of its own (e.g. a hindcast against measurements and the published
   alternatives) before it goes in, and is labelled `[project rule]` at its
   definition. Purely numerical choices (step sizes, tolerances, bracket
   widths) are exempt, but must be shown not to move a validated result.

## The core (C++)

`docs/CPP_NATIVE.md` is the plan and the roadmap. Rules that are structural,
not stylistic:

- Only `core/src/ephem.cpp` includes `SpiceUsr.h`/`erfa.h`; every CSPICE call
  goes through its `spice_call` (global mutex + RETURN mode + `failed_c()` →
  `eclipse::spice_error`). Everything else is pure math and lock-free.
- `third_party/{cspice,erfa}` are vendored **unmodified** — never edit; see
  `THIRD_PARTY_NOTICES.md`. `_eclipse` links its own CSPICE statically, so
  its kernel pool is separate from spiceypy's (which the tests use as an
  independent reference).
- **Arithmetic is C++'s to design** (layout, ownership, allocation,
  hoisting, caching, parallelism, and the operation order itself). What
  guards the numbers:
  - the reference-eclipse and oracle tests (the physics), which must hold;
  - the golden fixtures in `tests/cpp/fixtures/` (written by the retired
    Python oracle; `tests/cpp/fixtures.hpp`), which pin today's results. A
    change that moves a golden value re-baselines those records in the same
    PR, says by how much, and shows the reference tests still hold. A
    tolerance is never widened to hide a change nobody explained.
- No `-ffast-math`; `-ffp-contract=off` is set, so results do not depend on
  the compiler's FMA choices across platforms (goldens stay portable).
- Parallelism: the grid's observer loop, the catalog's per-event loop and the
  limb silhouette are OpenMP-parallel (`ECLIPSE_OPENMP` CMake option, default
  ON; serial fallback when no OpenMP is found, e.g. Apple Clang without
  libomp); results are identical for any thread count. libgomp is **not
  fork-safe once it has started its thread pool**, so a forking server must
  not warm the grid in the parent: run gunicorn without `--preload` (the
  Dockerfile's CMD does not). `app.core.PARALLEL_LOCK` holds one OpenMP team
  per process; the Docker CMD sizes it with `OMP_NUM_THREADS` = cores /
  `WEB_CONCURRENCY`.
- The catalog's scan and refinement objective is the frame-free
  `axis_separation` (`rho`, `z` from the un-rotated J2000 vectors, [ES92] eq.
  8.322-6); only the elements *at* greatest eclipse use the configured frame.
  `rho` is flat at its minimum, so `et_g` is conditioned to ~1 ms
  (`catalog.hpp` "CONDITIONING OF et_g").
- The limb band: the core reads the band file itself
  (`limb::load_band_file`, via `app.core.ensure_limb_band`; 2 bytes a point;
  `LDEM_64` 14° in 0.44 GB); `set_limb_band` remains for in-memory
  (synthetic) bands. Profiles are cached on a 5-min lattice keyed by the band
  generation.
- Observer height (`docs/LIMB_PROFILE.md` §9.10): `ellipsoid::geo_to_fund_one`
  takes a per-observer `ellipsoid::Site` and adds the rotated normal only when
  `h != 0`, so sea level is exactly the reduction alone. `height_m` rides
  through `series` / `profile_g` / `local_circumstances`; the grid (`/map`)
  stays sea level.
- Doc comments in older units still name the Python function each was ported
  from (`app.geography.*` etc.); those names refer to commit `fe1a64a` and are
  rewritten as each unit is next touched.

## Testing

- `uv run pytest`: the reference eclipses (`test_besselian_integration.py`),
  the catalog, the API, frame consistency, the limb profile (synthetic DEMs in
  closed form, the real band), the 3D limb oracle, the geodesy oracle, and the
  core's behaviour as an extension (`test_core.py`: errors, kernel pool, EOP,
  thread-count independence; `ECLIPSE_BENCH=1` adds the grid / catalog
  benchmarks). Kernel-dependent tests skip cleanly without kernels.
- `ctest --preset release`: the Catch2 unit tests and the golden fixtures.
- Before pushing: both on the NAIF kernels and, for CI parity, on the mirror
  set (`SPICE_METAKERNEL=<dir>/eclipse.tm ECLIPSE_LIMB_BAND=<dir>/lola_ldem16_limb20.bin
  uv run pytest`, the set from `kernels.bootstrap --source mirror --limb` run in a
  short-path directory: SPICE `PATH_VALUES` ≤ 80 chars); `ruff check .`.
- Prefer adding a reference-eclipse assertion over a hand-picked expected
  value.

## Gotchas

- **Kernels are gitignored** (large, NAIF-versioned); regenerate with
  `kernels.bootstrap`. In restricted networks NAIF is blocked but PyPI and
  `raw.githubusercontent.com` work — the `mirror` source uses a pinned GitHub
  copy (DE432s + IAU_EARTH; the ERFA `ITRS`/`TOD` frames need no binary PCK).
- **No model identifier** goes into commits/PRs/code.
- **There is no pure-Python fallback**: an unbuilt or stale core is an
  `ImportError` / a missing attribute; `uv sync` rebuilds it.
- **Accuracy ceiling today**: DE432s (mirror) vs DE440 (both sub-km for Sun/Moon);
  the mean limb (`K_UMBRA`) everywhere except `/circumstances?limb=profile`
  (C2/C3 from the LOLA profile; limits, catalog and maps are PRs 3–4); and
  **only `/circumstances` has an observer height** (`elev=`, metres above the
  WGS-84 *ellipsoid* — a map/GPS height above the geoid needs + N, about
  −30 m in the US Midwest [EGM96]). `/map`, `/central-line` limits and the
  catalog are sea-level, as published limits conventionally are, and the
  horizon test ignores the dip of the horizon from a height. Near a limit a
  few hundred metres move C2/C3 by seconds (`docs/LIMB_PROFILE.md` §9.10).
- **Limb data** (`--limb`): `lola_ldem16_limb20.bin` + `moon_pa_de440_200625.bpc`
  + `moon_de440_250416.tf`, SHA-256-pinned in `kernels/bootstrap.py`. The
  `mirror` source fetches them from this repo's data-only `limb-data` branch at
  a pinned commit — never rewrite that branch; add a commit and move the pin.

## References

Cite these by key in code.

- **[ES92]** *Explanatory Supplement to the Astronomical Almanac*, 2nd ed.,
  P. K. Seidelmann (ed.), University Science Books, 1992 — ch. 8 (the `8.32x`/`8.33x`
  equation numbers currently cited).
- **[ES13]** *Explanatory Supplement to the Astronomical Almanac*, 3rd ed.,
  S. Urban & P. K. Seidelmann (eds.), 2013 — ch. 11 (eclipses).
- **[Meeus98]** J. Meeus, *Astronomical Algorithms*, 2nd ed., Willmann-Bell, 1998 —
  ch. 10 (ΔT, Table 10.A), ch. 11 (the Earth's globe: parametric latitude, eq. 11.1),
  ch. 15 (rise/set altitude h0 = −0°50′). Note ch. 54 (Eclipses) covers only *general*
  circumstances and defers local circumstances to [MeeusSE] — do not cite it for them.
- **[MeeusSE]** J. Meeus, *Elements of Solar Eclipses 1951–2200*, Willmann-Bell, 1989.
- **[Espenak]** F. Espenak & J. Meeus, *Five Millennium Canon of Solar Eclipses:
  −1999 to +3000*, NASA/TP-2006-214141, 2006 — methodology, the `k1`/`k2` lunar radii,
  and sec. 2.5 the polynomial ΔT expressions (also
  https://eclipse.gsfc.nasa.gov/SEhelp/deltatpoly2004.html). Per-eclipse Besselian
  elements: NASA Eclipse Web Site, e.g. `SEgoogle/SEgoogle2001/SE2024Apr08Tgoogle.html`.
- **[SOFA]** IAU SOFA / ERFA: `pnm06a`, `gst06a`, `c2t06a`; P. T. Wallace &
  N. Capitaine (2006), *A&A* 459, 981.
- **[IERS2010]** G. Petit & B. Luzum (eds.), *IERS Conventions (2010)*, IERS TN 36.
- **[WGS84]** NIMA TR8350.2, *Department of Defense World Geodetic System 1984*,
  3rd ed., 2000 — a = 6378137 m, f = 1/298.257223563.
- **[DE440]** R. S. Park et al. (2021), *AJ* 161, 105 — JPL DE440/DE441.
- **[SPICE]** C. H. Acton (1996), *Planet. Space Sci.* 44, 65 — NAIF SPICE.
- **[LOLA]** D. E. Smith et al. (2010), *GRL* 37, L18204; D. E. Smith et al.
  (2017), *Icarus* 283, 70 — LRO Lunar Orbiter Laser Altimeter; gridded DEMs
  `LDEM_*` in PDS `LRO-L-LOLA-3-RDR-V1` (reference sphere 1737.4 km, frame
  Mean Earth/Polar Axis of DE421 = SPICE `MOON_ME`).
- **[EB2024]** F. Espenak & J. Anderson, *Eclipse Bulletin: Total Solar Eclipse of
  2024 April 08*, Table 2–3 (limb-profile-corrected local contacts; sample page
  eclipsewise.com/pubs/images/EB2024-sample2.pdf). All rights reserved: cite values.
- **[Irwin21]** J. Irwin et al. (2021), *ApJS*, arXiv:2107.09416 — limb-corrected
  contact predictions at a 2017 near-limit site (Table 3).
- **[EGM96]** F. G. Lemoine et al. (1998), *The Development of the Joint NASA
  GSFC and NIMA Geopotential Model EGM96*, NASA/TP-1998-206861 — the geoid
  undulation N, and the ellipsoidal height h = H + N of an orthometric
  (map / SRTM) height H.
- **[SRTM]** T. G. Farr et al. (2007), *Rev. Geophys.* 45, RG2004 — Shuttle
  Radar Topography Mission heights (orthometric, EGM96).
- **[NASA-LC]** F. Espenak, NASA "Solar Eclipse Local Circumstances"
  calculator, eclipse.gsfc.nasa.gov/SEgoogle/SEcirc.js (`readdata`: Meeus's
  ρ sin φ′ / ρ cos φ′ with the observer's height; `timelocdependent`: ξ, η, ζ).
  Also the Eclipse Bulletins' local-circumstance tables use the site's
  elevation "if known. Otherwise … sea-level" (NASA/TP-1999-209484).
- **[NASA-limb]** F. Espenak, "The Lunar Limb Profile and Eclipse Predictions",
  eclipse.gsfc.nasa.gov/SEhelp/limb.html; "Limb Corrections to the Path Limits:
  Graze Zones", eclipse.gsfc.nasa.gov/SEmono/reference/graze.html.

# CLAUDE.md

Guidance for working in this repository. This is a **scientific codebase**: the
numbers must be right and every non-trivial formula must be traceable to a
citation. Read the "Scientific conventions" section before touching any math.

## What this is

EclipseBackend computes the **Besselian elements** of a solar eclipse and
everything that follows from them — the central line, the northern/southern
limits and path width, and per-observer local circumstances (contact times,
duration, magnitude, obscuration). A FastAPI service exposes it and a
dependency-free Three.js globe (`frontend/`) visualizes it.

Positions come from **NASA/NAIF SPICE** (SpiceyPy) + a **JPL DE** ephemeris;
Earth orientation from **ERFA** (IAU 2006/2000A) with **IERS** EOP. Packaging is
**uv**. The project deliberately does **not** use astropy (it was removed; SPICE
+ ERFA + NumPy cover the need with a smaller, more explicit surface).

## Setup & commands

```bash
uv sync                                  # install into .venv (also builds the native core)
uv run python -m kernels.bootstrap       # download SPICE kernels (auto: NAIF else GitHub mirror)
uv run python -m kernels.bootstrap --limb  # + lunar limb profile inputs (LOLA band, MOON_ME kernels)
uv run pytest                            # tests (integration tests auto-skip without kernels)
uv run uvicorn app.main:app --reload     # API + viewer at http://localhost:8000/ui/
cmake --preset release && cmake --build --preset release && ctest --preset release  # C++ tests
```

Environment: `SPICE_EARTH_FRAME` overrides the default frame (`ITRS`);
`SPICE_METAKERNEL` overrides the kernel path; `ECLIPSE_BACKEND=python|native|auto`
selects the compute backend (`app/native.py`; default `auto` = `native` when the
`_eclipse` extension is built, else `python` with one `RuntimeWarning`; an
explicit `native` that cannot import is an `ImportError`).

## Architecture (`app/`)

Data flows one direction: **ephemeris → besselian → geography/circumstances → main/frontend**.

| Module | Responsibility |
| --- | --- |
| `ephemeris.py` | SPICE apparent Sun/Moon positions; `besselian_instant()` builds the 8 elements at one instant; the four Earth-orientation frames (`ITRS`/`TOD`/`ITRF93`/`IAU_EARTH`); `sub_solar_point()`; `limb_axes()` (fundamental-plane axes in `MOON_ME`). Owns the lunar-radius constants `K_PENUMBRA`/`K_UMBRA`. |
| `eop.py` | IERS polar motion + UT1−UTC, interpolated from the `astropy-iers-data` bundle. |
| `besselian.py` | Samples the elements around T0 and polynomial-fits them (`BesselianModel`). |
| `geography.py` | The ellipsoid geometry: `fund_to_geo` (fundamental→geographic), `geo_to_fund` (its exact inverse; `height_m` adds the observer's height above the ellipsoid, exactly, [Meeus98] ch. 11), `shadow_edge_limits` (N/S limits + true width, umbral or penumbral), `global_contacts` (P1–U4), `shadow_radii`, great-circle helpers. |
| `deltat.py` | ΔT = TT − UT1 polynomial model [Espenak] for epochs outside the IERS era. |
| `circumstances.py` | Per-observer local circumstances from a `BesselianModel` (`limb="mean"` k2, or `"profile"`: C2/C3 from the LOLA limb profile; `height_m` above the WGS-84 ellipsoid); `circumstances_grid` for maps (sea level). |
| `catalog.py` | `find_eclipses`: scan a date range, refine greatest eclipse, classify P/A/T/H, Canon-style rows. |
| `limb.py` | Lunar limb profile from the LOLA DEM (`docs/LIMB_PROFILE.md`): loads the limb band, `silhouette()` (δρ per 0.05° bin, points + grid edges, in perspective from the Moon's distance), `profiles_at()` (5-min cached lattice), the contact functions `g_total()` / `g_annular()`. Used by `/circumstances?limb=profile`. |
| `numerics.py` | Vectorized bisection / parabolic-extremum helpers shared by circumstances, contacts and the catalog. |
| `reference.py` | Parses `app/data.txt` (a reference central-line track). |
| `main.py` | FastAPI: `/health`, `/besselian`, `/central-line`, `/circumstances` (`limb=mean|profile`, `elev=` m above the ellipsoid), `/map`, `/eclipses`; serves `frontend/` at `/ui`. |
| `kernels/bootstrap.py` | Downloads kernels (`--source auto|naif|mirror`, `--limb`) and writes `eclipse.tm`. |
| `kernels/limb_band.py` | Cuts LOLA `LDEM_*` to the lunar limb band (deterministic, SHA-pinned file) and reads it back. |

## Scientific conventions — READ BEFORE EDITING MATH

1. **Every non-trivial equation and physical constant carries a citation** in
   the nearest docstring or an inline comment, using the keys in "References"
   below (e.g. `# Explanatory Supplement eq. 11.323` or `[Espenak]`). New math
   without a source is not done. Prefer the equation number, not just the book.

2. **Reuse, don't re-derive.** Shared geometry lives in one place:
   - ellipsoid reduction and great-circle math → `geography.py`
   - Earth-orientation / apparent positions → `ephemeris.py`
   - EOP → `eop.py`
   If you find the same formula (parametric latitude, ρ1/d1 auxiliaries, a
   root-find, a constant) written twice, factor it out rather than copying.

3. **Units are explicit and stated in every signature.** Conventions:
   angles in **degrees** at module boundaries (radians internally), distances in
   **Earth equatorial radii** for fundamental-plane quantities and **km**
   otherwise, times in **seconds**/**hours-from-T0** as labelled. There is no
   units library — the docstring is the contract, so keep it accurate.

4. **Validate against an independent published source.** Every capability that
   produces a number has a test in `tests/test_besselian_integration.py`
   comparing to Fred Espenak / NASA values for real eclipses (2017-08-21 total,
   2024-04-08 total, 2023-10-14 annular, 2023-04-20 hybrid, 2021-12-04 polar
   total, and pre-IERS 1919-05-29 / 1868-08-18 totals). Add a reference eclipse
   when you add a capability; state the achieved agreement in the test comment.
   Known, open discrepancies are strict xfails, never widened gates
   (`docs/CODE_REVIEW_FOLLOWUPS.md` §6). Path limits/widths are the time
   envelope of the shadow (`shadow_edge_limits_v(..., rates=element_rates(...))`),
   measured with `geodesic_km`; the solar radius is the constant
   `constants.SUN_RADIUS_KM` (IAU 1976), never the PCK's. Limb-profile
   contacts are checked against an independent 3D ray-traced oracle
   (`tests/test_limb_oracle.py`, gated in limb-height metres) and against
   [EB2024] and [Irwin21] at the sites' heights (`docs/LIMB_VALIDATION_SOURCES.md`).

5. **Frames & epochs.** Input epochs are **UTC** and must be ISO-8601
   (`besselian.normalize_utc` is the single parser). ΔT = TT − UT1 is measured
   (leap-second kernel + IERS UT1−UTC) **only inside the IERS era, 1973 – ~1 yr
   ahead**; outside it the epoch is read as UT1 and ΔT comes from the
   Espenak & Meeus polynomial model in `deltat.py` [Espenak] — never rely on
   SPICE's leap-second table there (it silently holds the first/last count).
   Published Besselian elements are tabulated in **TDT/TT** (use `str2et(".. TDT")`
   to compare). Published `mu` is the *ephemeris hour angle* (Earth rotation
   evaluated as if TT were UT); ours is the true Greenwich hour angle, so
   `mu_published = mu_ours + ΔT·1.002738·15″/s` [ES92] 8.36 — see
   `ephemeris.py` and the mu test.
6. **Horizon.** The cone geometry is valid on the whole ellipsoid, including
   the night side; every local circumstance must be checked against the Sun's
   altitude (`circumstances.HORIZON_ALT_DEG`, [Meeus98] ch. 15) before being
   reported. Vectorize new geometry (array-native NumPy, broadcasting over
   observers × instants) rather than looping; the scalar functions are thin
   wrappers over the array ones.

## Native core (`libeclipse`, phases 0–4 done; now C++-first)

`docs/CPP_ROADMAP.md` built the C++20 `libeclipse` core as a port of `app/`;
`docs/CPP_NATIVE.md` is the current plan: **C++ is the primary
implementation, designed as C++**, and the Python `app/` stays the API *and
the bit-level oracle*. Every C++ unit is parity-tested against it at the
tolerances listed in the roadmap before a reference-eclipse test is routed
through it. Layout: `core/include/eclipse/*.hpp` + `core/src/*.cpp`
(library), `bindings/_eclipse.cpp` (nanobind), `tests/cpp/` (Catch2),
`cmake/` (build of the vendored libs), `third_party/{cspice,erfa}` (vendored,
**unmodified** — never edit; see `THIRD_PARTY_NOTICES.md`). Rules that are
structural, not stylistic:

- Only `core/src/ephem.cpp` includes `SpiceUsr.h`/`erfa.h`; every CSPICE call
  goes through its `spice_call` (global mutex + RETURN mode + `failed_c()` →
  `eclipse::spice_error`). Everything else is pure math and lock-free.
- **Same arithmetic, native structure** (`docs/CPP_NATIVE.md`). Layout,
  ownership, allocation, API shape, caching and parallelism are C++'s to
  design (workspaces over per-call vectors, concrete types over
  `std::function`, owned data over data injected from Python). The per-value
  arithmetic keeps the Python's operation order (left-to-right as NumPy
  evaluates; `np.polyval` = Horner; `np.interp` semantics in `eop.cpp`;
  `np.remainder` in `wrap_180`), and so do batch boundaries where they are
  arithmetic (the `mu` unwrap in `elements_at`). A change to a formula goes
  into both languages in the same PR, then `uv run python tools/dump_oracle.py`
  regenerates `tests/cpp/fixtures/`.
- Parity is checked live in `tests/test_native.py` (Python vs native over
  dense windows; `tests/test_backend_switch.py` covers the backend resolution
  in subprocesses) and offline in `tests/cpp/test_*.cpp` (fixtures).
  Residuals are documented there; do not widen a tolerance to make a test
  pass. `tools/dump_oracle.py` pins `ECLIPSE_BACKEND=python` itself.
- The EOP table is injected from `app.eop` (`set_eop_table`); the C++ does
  not parse `finals2000A.all` yet (`docs/CPP_NATIVE.md` item 8).
- Dispatch under the native backend (`native.is_native()`, read at call
  time): the five `app/ephemeris.py` functions plus `axis_separation` (phases
  1 / 4) and, in `app/geography.py` (phase 2), `fund_to_geo_v`,
  `geo_to_fund`, `shadow_radii`, `bearing`, `shadow_edge_limits_v` and
  `global_contacts` — a dispatch line plus broadcast/ravel/reshape glue at the
  top of each, the Python body untouched. The scalar `fund_to_geo`,
  `_reduction_aux`, the private great-circle helpers, `central_track` and the
  formatters stay Python.
- Phase 3, in `app/circumstances.py`: `local_circumstances` (the numbers —
  `_local_raw` — come from C++ as a `_LocalRaw` tuple; `_format_local` stays
  Python) and `circumstances_grid` (same keys/dtypes; `chunk` bounds only the
  Python path's memory and is ignored natively). The grid's observer loop is
  OpenMP-parallel (`ECLIPSE_OPENMP` CMake option, default ON; serial fallback
  when no OpenMP is found, e.g. Apple Clang without libomp; results are
  identical for any thread count). libgomp is **not fork-safe once it has
  started its thread pool**, so a forking server must not warm the native
  grid in the parent: run gunicorn without `--preload` (the Dockerfile's CMD
  does not).  `native.PARALLEL_LOCK` holds one grid/catalog OpenMP team per process;
  the Docker CMD sizes it with `OMP_NUM_THREADS` = cores / `WEB_CONCURRENCY`.
- Phase 4, in `app/catalog.py`: `find_eclipses` takes the core's `_EventRaw`
  tuples (`_eclipse.find_eclipses`, `local` rebuilt as `_LocalRaw`) in place
  of `_catalog_raw` and formats them with the same `_format_event`; the
  per-event hybrid / detail loop is OpenMP-parallel (identical for any thread
  count). The catalog's one formula change, made in both languages: the scan
  and refinement objective is the frame-free `ephemeris.axis_separation`
  (`rho`, `z` from the un-rotated J2000 vectors — the same [ES92] eq. 8.322-6,
  invariant under any Earth-orientation rotation) instead of `hypot(x, y)` of
  the ITRS elements, so the C++ scan needs no ERFA / EOP per sample; only the
  elements *at* greatest eclipse use the configured frame. Parity consequence:
  `rho` is flat at its minimum (and carries ~3e-12 of SPK evaluation jitter,
  identical in both backends), so `et_g` is conditioned to ~1 ms and its gate
  is 1e-2 s, with rows required identical up to a whole-second `greatest_utc`
  flip at a rounding boundary inside that band (`catalog.hpp` "CONDITIONING
  OF et_g", `app/catalog.py::_rho_at`, `tests/test_native.py` phase 4).
- Lunar limb profile (`docs/LIMB_PROFILE.md`, PRs 1-2): `ephem::limb_axes`
  (adds `pxform_c`; also returns the viewing distance), `core/src/limb.cpp`
  (`silhouette`, `delta_rho_at`, `profiles_at` with a mutex-guarded node cache
  keyed by the band generation, `g_total` / `g_annular`) and, in
  `circumstances.cpp`, `profile_g` / `profile_contacts` / the `profile` flag of
  `local_circumstances` (the `Model` carries `elements_ref_at` = the elements
  with `limb::K_REF` for both cones, and `profiles_at`); `besselian_instants`
  takes the cones' `k1`/`k2`. Profiles and contact functions are
  bit-identical to `app.ephemeris.limb_axes` / `app.limb`; profile-mode
  local circumstances are gated like the mean-limb ones (1e-9 h). The core
  **owns the band**: it reads the band file itself (`limb::load_band_file`
  via `app.limb.ensure_native_band`, 2 bytes a point, no neighbour indices,
  no NumPy copy; `LDEM_64` 14° in 0.44 GB) and builds the pixel-centre trig
  tables with libm; `set_limb_band` remains for in-memory (synthetic) bands.
  The Python oracle uses libm for those tables (`app.limb._libm`) and for
  `atan2` (`app.limb._atan2`), not NumPy's SIMD kernels (1 ulp off), to keep
  the parity exact; the native silhouette holds `native.PARALLEL_LOCK`.
- Observer height (`docs/LIMB_PROFILE.md` §9.10): `ellipsoid::geo_to_fund_one`
  takes a per-observer `ellipsoid::Site` (parametric-latitude factors, `h = H/a`,
  geodetic cos/sin) and adds the rotated normal only when `h != 0`, so sea level
  is bit-identical to the reduction alone; `ReductionAux` also carries `cos_d` /
  `sin_d` for it (the Python recomputes `np.cos(d)` / `np.sin(d)`, the same
  values, so `_ReductionAux` and the `red` fixtures are unchanged).
  `height_m` rides through `series` / `profile_g` / `local_circumstances` and
  the bindings; the grid (`/map`) stays sea level.
- No `-ffast-math`; `-ffp-contract=off` is set. Keep the Python's operation
  order so parity is bit-level, not "close".
- `_eclipse` links its own CSPICE statically: its kernel pool is separate from
  spiceypy's in the same process.
- `uv sync` rebuilds it when `core/`, `bindings/`, `cmake/`, `CMakeLists.txt`
  or `third_party/` change (`tool.uv.cache-keys`); install `ccache` to make
  that seconds instead of a minute.

## Testing

`uv run pytest`. Pure-function tests (geography, reference parsing) always run;
the numerical-validation tests require kernels and skip cleanly without them.
Keep the suite green and prefer adding a reference-eclipse assertion over a
hand-picked expected value.

## Gotchas

- **Kernels are gitignored** (large, NAIF-versioned); regenerate with
  `kernels.bootstrap`. In restricted networks NAIF is blocked but PyPI and
  `raw.githubusercontent.com` work — the `mirror` source uses a pinned GitHub
  copy (DE432s + IAU_EARTH; the ERFA `ITRS`/`TOD` frames need no binary PCK).
- **No model identifier** goes into commits/PRs/code.
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

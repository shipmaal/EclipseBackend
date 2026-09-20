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
- **A3 — Documented approximations (low).** Geometric only: no atmospheric
  refraction (contacts, low-Sun central line) and only the mean lunar limb
  (in `K_UMBRA`). State per-docstring.
- **A4 — TDB used as TT for ERFA (low).** ≤1.7 ms → sub-mas; note it.
- **A5 — WGS-84 `b` rounded (low). DONE.** Folded into A1: `b` is now derived
  from `a` and `f` in `app/constants.py`, no longer hard-coded.

## 2. Citation coverage (core goal)

Citations exist but are inconsistent (edition not labelled; equation numbers
missing in `geography.py`/`circumstances.py`; several formulas uncited). Apply
the keyed references from `CLAUDE.md` with equation numbers:

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
- **R5** — Frame-fallback loop duplicated in tests (`_build_model`,
  `_usable_frame`). Promote `build_model_best_frame()` into `app/`.

## 4. Modern codebase

- **M1** — No linter/formatter or CI. Add **ruff** (config in `pyproject.toml`) +
  a GitHub Actions workflow running pure-function tests (kernels auto-skip), ideally
  a second job with `kernels.bootstrap --source mirror` for full validation.
- **M2** — Over-broad `except Exception` in `main.py:47` masks non-SPICE bugs;
  narrow to `(spiceypy.utils.exceptions.SpiceyError, ValueError)`.
- **M3** — Stale docstrings: "JPL DE440" in `besselian.py:6`, `ephemeris.py:8`,
  `main.py:4` (mirror ships DE432s → say "JPL DE"); `besselian_instant` frame list
  omits `ITRS` (`ephemeris.py:178`); `/besselian` `frame` description omits `ITRS`
  (`main.py:60`).
- **M4** — Typed results: `local_circumstances` returns a variable-key `dict`
  (C2/C3 only sometimes); use a `TypedDict`/dataclass. Same for `/central-line`
  point shape.
- **M5** — Missing pure-function tests (run without kernels): `geo_to_fund`
  round-trip and `_obscuration`/`_overlap_area` (analytic cases).

## Suggested order

1. `constants.py` (cited WGS-84 + unit radius; fixes A1, R2).
2. Direct-evaluation + contact bracketing (A2) — the only accuracy-affecting item.
3. Extract `_reduction_aux` and `central_track` (R1, R3); unify time format (R4).
4. Citation pass — apply §2 with equation numbers, per `CLAUDE.md`.
5. Tooling: ruff + CI + the two pure-function tests (M1, M5); doc/exception fixes
   (M2, M3, M4).

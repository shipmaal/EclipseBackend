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
uv sync                                  # install into .venv
uv run python -m kernels.bootstrap       # download SPICE kernels (auto: NAIF else GitHub mirror)
uv run pytest                            # tests (integration tests auto-skip without kernels)
uv run uvicorn app.main:app --reload     # API + viewer at http://localhost:8000/ui/
```

Environment: `SPICE_EARTH_FRAME` overrides the default frame (`ITRS`);
`SPICE_METAKERNEL` overrides the kernel path.

## Architecture (`app/`)

Data flows one direction: **ephemeris → besselian → geography/circumstances → main/frontend**.

| Module | Responsibility |
| --- | --- |
| `ephemeris.py` | SPICE apparent Sun/Moon positions; `besselian_instant()` builds the 8 elements at one instant; the four Earth-orientation frames (`ITRS`/`TOD`/`ITRF93`/`IAU_EARTH`); `sub_solar_point()`. Owns the lunar-radius constants `K_PENUMBRA`/`K_UMBRA`. |
| `eop.py` | IERS polar motion + UT1−UTC, interpolated from the `astropy-iers-data` bundle. |
| `besselian.py` | Samples the elements around T0 and polynomial-fits them (`BesselianModel`). |
| `geography.py` | The ellipsoid geometry: `fund_to_geo` (fundamental→geographic), `geo_to_fund` (its exact inverse), `shadow_edge_limits` (N/S limits + true width, umbral or penumbral), `global_contacts` (P1–U4), `shadow_radii`, great-circle helpers. |
| `deltat.py` | ΔT = TT − UT1 polynomial model [Espenak] for epochs outside the IERS era. |
| `circumstances.py` | Per-observer local circumstances from a `BesselianModel`; `circumstances_grid` for maps. |
| `catalog.py` | `find_eclipses`: scan a date range, refine greatest eclipse, classify P/A/T/H, Canon-style rows. |
| `numerics.py` | Vectorized bisection / parabolic-extremum helpers shared by circumstances, contacts and the catalog. |
| `reference.py` | Parses `app/data.txt` (a reference central-line track). |
| `main.py` | FastAPI: `/health`, `/besselian`, `/central-line`, `/circumstances`, `/map`, `/eclipses`; serves `frontend/` at `/ui`. |
| `kernels/bootstrap.py` | Downloads kernels (`--source auto|naif|mirror`) and writes `eclipse.tm`. |

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
   2024-04-08 total, 2023-10-14 annular). Add a reference eclipse when you add a
   capability; state the achieved agreement in the test comment.

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
- **Accuracy ceiling today**: DE432s (mirror) vs DE440 (both sub-km for Sun/Moon)
  and no per-position lunar-limb profile (the mean limb is folded into `K_UMBRA`).

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

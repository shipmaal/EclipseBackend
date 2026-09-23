# SPICE kernels

These are **not committed** (they are large and versioned by NAIF). Fetch them:

```bash
uv run python -m kernels.bootstrap
```

The generated `eclipse.tm` metakernel loads the set below. The engine
(`app/ephemeris.py`) furnishes `eclipse.tm` by default, overridable with the
`SPICE_METAKERNEL` environment variable.

| Kernel | Type | Why it's needed |
| --- | --- | --- |
| `naif0012.tls` | LSK (leapseconds) | UTC ↔ ET (TDB) conversion in `str2et`. |
| `de440s.bsp` | SPK (ephemeris) | Apparent positions of Sun, Moon and Earth (JPL DE440s, 1849–2150). Use `de440.bsp` (1550–2650) for a wider span. |
| `pck00011.tpc` | text PCK | Body radii (`bodvrd`) and the low-precision `IAU_EARTH` frame. |
| `earth_latest_high_prec.bpc` | binary PCK | High-precision Earth orientation — defines the **ITRF93** frame with precession, nutation and true (UT1/EOP) rotation. Optional: the default **`TOD`** frame (pyerfa true-of-date) is already high precision without it. |

## Earth-orientation frames (`SPICE_EARTH_FRAME` / `frame=` query)

| Frame | Needs | Notes |
| --- | --- | --- |
| `ITRS` (default) | PyPI only (`astropy-iers-data`) | full IAU 2006/2000A + IERS EOP (polar motion, UT1) |
| `TOD` | nothing extra | true-of-date, UT1≈UTC (no EOP) |
| `ITRF93` | `earth_latest_high_prec.bpc` (NAIF only) | SPICE binary PCK; equivalent to ITRS |
| `IAU_EARTH` | text PCK | coarse fallback |

Validated against independent Espenak elements/points for 2017-08-21 and
2024-04-08: Besselian `x,y` match to ~1e-5, greatest-eclipse position to within
the published rounding, and **path width to ~1 km** (114.7 km and 197.4 km).

## Why this set, and not a lunar theory (ELP2000-82B / ELP/MPP02)

Those theories compute the **Moon only**, in their own frame, and require a
separate solar ephemeris plus manual light-time / aberration / frame handling —
exactly the bookkeeping that produced the earlier errors. SPICE + DE440 supplies
the Sun, Moon and Earth in one consistent, apparent, high-precision framework.
The lunar-theory sources remain in `f_src/` and `c_src/` for cross-checking only.

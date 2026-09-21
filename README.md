# EclipseBackend

Backend for a solar-eclipse visualizer. Given a date/time and two bodies
(Sun + Moon), it computes the **Besselian elements** of the eclipse and the
geographic track of the shadow axis.

## Ephemeris engine

Positions of the Sun, Moon and Earth come from **NASA/NAIF SPICE**
([SpiceyPy](https://github.com/AndrewAnnex/SpiceyPy)) with the **JPL DE440**
planetary ephemeris. SPICE is used because a solar eclipse needs the Sun *and*
Moon in one consistent framework, with **apparent** positions (light-time +
stellar aberration, `LT+S`) and a high-precision Earth-rotation model — exactly
the corrections the Besselian method depends on. (The earlier astropy /
lunar-theory experiments — ELP2000-82B, ELP/MPP02 — computed the Moon only and
are kept under `f_src/` and `c_src/` for reference / cross-checking, not as the
primary engine.)

## Setup (uv)

```bash
# install deps into a local venv; also builds the native core (`_eclipse`:
# vendored CSPICE + ERFA via CMake — needs a C/C++ compiler, ~1 min cold)
uv sync

# download the SPICE kernels
uv run python -m kernels.bootstrap            # auto: NAIF, else GitHub mirror
# uv run python -m kernels.bootstrap --source mirror   # restricted networks

# run the API + frontend
uv run uvicorn app.main:app --reload
```

Then open the 3D viewer at **http://localhost:8000/ui/**.

Kernels are written to `kernels/` and loaded through the metakernel
`kernels/eclipse.tm`. See `kernels/README.md` for the exact kernel set and why
each is needed. The mirror set has no high-precision Earth PCK, so the `ITRF93`
frame is unavailable there; the default `ITRS` frame (ERFA + IERS EOP from
PyPI) needs none and is verified equivalent.

### Native core (`libeclipse`)

The C++20 core under `core/` + `bindings/` links the **vendored** NAIF CSPICE
N0067 and liberfa 2.0.1 (`third_party/`, unmodified; see
`THIRD_PARTY_NOTICES.md`) and is exposed to Python as `_eclipse`. It is at
roadmap phase 3 (`docs/CPP_ROADMAP.md`): kernel management, time scales,
delta-T, IERS EOP, the Besselian elements, the ellipsoid reduction, the
shadow-edge limits/path width, the global contacts P1–P4 and the local
circumstances (contacts, duration, magnitude, obscuration, horizon flags and
the OpenMP-parallel global map grid), parity-tested against the Python oracle
to ~1e-13 (`/central-line`, `/circumstances` and `/map` are byte-identical
through either backend; the 0.5° global map grid takes 0.4 s against 4 s in
Python). Set `ECLIPSE_BACKEND=native` to route the ephemeris, geometry and
circumstances layers through it (the default stays `python` until phase 4).

```bash
cmake --preset release && cmake --build --preset release   # standalone build
ctest --preset release                                     # Catch2 tests
```

## Frontend

`frontend/` is a dependency-free Three.js globe (Three.js is vendored under
`frontend/vendor/`, so it works offline). It calls `/central-line` and:

* draws the shadow-axis central line and the **northern/southern limit curves**,
  shading the **path of totality** between them;
* lights the globe from the **sub-solar point** (`sun` in the response), giving a
  real day/night terminator;
* casts the Moon's **umbra + penumbra** as shadow blobs (umbra sized by the true
  `width_km`), animated along the track with a play/scrub control.

Served by the backend at `/ui`, or standalone with any static server
(`python -m http.server` in `frontend/`, then set the API base URL in the panel).
It ships a bundled `sample.json` so it renders even without a running backend
("Load sample").

## HTTP API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness check. |
| `GET /besselian?epoch=…` | Besselian element polynomials for the eclipse. |
| `GET /central-line?epoch=…&start_hours=…&end_hours=…&step_minutes=…` | Geographic shadow-axis track. |
| `GET /circumstances?epoch=…&lat=…&lon=…` | Local circumstances for an observer. |
| `GET /map?epoch=…&lat_step=…&lon_step=…` | Global maximum-eclipse grid: magnitude, obscuration, visibility. |
| `GET /eclipses?start=…&end=…` | Catalog of every solar eclipse in a range: type, gamma, magnitude, greatest-eclipse point, duration, width, global contacts. |

`epoch` is a UTC ISO-8601 time, e.g. `2024-04-08T18:00:00`. `/central-line`
points include the central `lat/lon`, the umbral `north_limit`/`south_limit`,
the true `width_km`, `is_total`, and the `penumbra_north/south_limit` (the edge of
the partial-eclipse region, clipped to the terminator); the response also carries
the global `contacts` P1/U1/U2/U3/U4/P4 (when the penumbra and umbra first/last
touch the Earth). `/circumstances` returns the observer's
C1–C4 contact times, maximum-eclipse time, `magnitude`, `obscuration`,
`central_duration_s` (when total/annular), the Sun's altitude at each contact and
`below_horizon` (the events the observer cannot see; `eclipse` is `false` when
that is all of them) — click anywhere on the globe in the frontend to see it.
`/map` evaluates the same geometry for a whole lat/lon grid at once (vectorized:
measured 0.6 s for a 2° global grid, 1.4 s at 1°, 5 s at 0.5°, with 2-minute time
sampling).

`/eclipses` scans a date range for eclipses (one vectorized element evaluation
per 2 h, refined to greatest eclipse) and classifies them partial / annular /
total / hybrid; 2019–2024 reproduces the NASA canon exactly, with gamma to 1e-4
and greatest-eclipse instants within ~5 s.

`epoch` must be ISO-8601 (`YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM]`); anything else is
a 400.

## Accuracy

Positions are apparent (light-time + stellar aberration); Earth orientation is
the full IAU 2006/2000A transform with IERS EOP (default `ITRS` frame); the
penumbral/umbral lunar radii use the standard `k1`/`k2` split; latitudes use the
parametric→geodetic conversion; and limits/widths are found by root-finding the
shadow-cone edge on the WGS-84 ellipsoid. Validated against independent Espenak
elements and greatest-eclipse points for 2017-08-21, 2024-04-08 (total) and
2023-10-14 (annular): Besselian `x,y` to ~1e-5, positions within the published
rounding, **path width to ~1 km** (114.7 / 197.4 / 187.4 km), and local
**central-eclipse duration to <1 s** and **magnitude to ~0.001** at the
greatest-eclipse points. Remaining approximations: DE432s rather than DE440
(sub-km for Sun/Moon), and no per-position lunar-limb profile (the mean limb is
folded into `k2`).

**Time scales.** Inside the IERS era (1973 – about a year ahead) the epoch is
UTC and ΔT = TT − UT1 comes from the leap-second kernel plus the measured
UT1−UTC. Outside it the epoch is read as UT1 and ΔT comes from the Espenak &
Meeus polynomial model (`app/deltat.py`, the ΔT of the *Five Millennium Canon*),
with zero polar motion; historical/future geometry is only as good as that
model (a few seconds in the 20th century, minutes by the 16th).

## Tests

```bash
uv run pytest            # pure-function tests always run;
                         # the end-to-end path validation is skipped
                         # automatically unless the SPICE kernels are present.
ctest --preset release   # C++ unit/parity tests (after `cmake --preset release`)
```

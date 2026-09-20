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
# install deps into a local venv
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
each is needed. If only the mirror kernel set is available (no high-precision
Earth PCK), set `SPICE_EARTH_FRAME=IAU_EARTH`.

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

`epoch` is a UTC ISO-8601 time, e.g. `2024-04-08T18:00:00`. `/central-line`
points include the central `lat/lon`, the umbral `north_limit`/`south_limit`,
the true `width_km`, and `is_total`.

## Accuracy

Positions are apparent (light-time + stellar aberration); Earth orientation is
the full IAU 2006/2000A transform with IERS EOP (default `ITRS` frame); the
penumbral/umbral lunar radii use the standard `k1`/`k2` split; latitudes use the
parametric→geodetic conversion; and limits/widths are found by root-finding the
shadow-cone edge on the WGS-84 ellipsoid. Validated against independent Espenak
elements and greatest-eclipse points for 2017-08-21 and 2024-04-08: Besselian
`x,y` to ~1e-5, positions within the published rounding, and **path width to
~1 km** (114.7 km and 197.4 km). Remaining approximations: DE432s rather than
DE440 (sub-km for Sun/Moon), and no per-position lunar-limb profile (the mean
limb is folded into `k2`).

## Tests

```bash
uv run pytest            # pure-function tests always run;
                         # the end-to-end path validation is skipped
                         # automatically unless the SPICE kernels are present.
```

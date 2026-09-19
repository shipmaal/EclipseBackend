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

# download the SPICE kernels (needs network access to naif.jpl.nasa.gov)
uv run python -m kernels.bootstrap

# run the API
uv run uvicorn app.main:app --reload
```

Kernels are written to `kernels/` and loaded through the metakernel
`kernels/eclipse.tm`. See `kernels/README.md` for the exact kernel set and why
each is needed.

## Tests

```bash
uv run pytest            # pure-function tests always run;
                         # the end-to-end path validation is skipped
                         # automatically unless the SPICE kernels are present.
```

# Third-party notices

The native core (`libeclipse`, see `docs/CPP_ROADMAP.md`) statically links two
vendored libraries under `third_party/`. Both trees are copied **unmodified**
from their upstream releases so that a version bump is a directory swap; the
only build-time additions are in `cmake/`.

## NAIF SPICE Toolkit — CSPICE N0067

- Source: `third_party/cspice/` — `src/*.c` and `include/*.h` from
  `cspice.tar.Z` (PC_Linux_GCC_64bit package, toolkit version N0067),
  https://naif.jpl.nasa.gov/naif/toolkit_C.html. `mkprodct.csh` is kept as
  the citation for the compiler flags used in `cmake/cspice.cmake`.
- Author: NASA's Navigation and Ancillary Information Facility (NAIF),
  Jet Propulsion Laboratory, California Institute of Technology.
- Terms: NAIF's "Rules Regarding Use of SPICE",
  https://naif.jpl.nasa.gov/naif/rules.html. In particular:
  - Including SPICE Toolkit modules (source and object code) as part of a
    package supporting a customer-built SPICE-based tool is permitted;
    redistribution of the *unmodified toolkit as such* (a mirror) is not.
    This repository includes the modules as part of `libeclipse`, a
    third-party SPICE-based library, and is not a toolkit mirror.
  - The SPICE code here is **not modified**. Per NAIF's rules, NAIF provides
    no support for third-party SPICE-based software; questions about
    `libeclipse` go to this project, not to NAIF.
  - SPICE is designated "Technology and Software Publicly Available" (TSPA)
    for export purposes; a derived product must obtain its own designation.
  - Use of the name "SPICE" or "NAIF" here does not imply NASA/JPL/NAIF
    endorsement of this project.
- Credit (per https://naif.jpl.nasa.gov/naif/credit.html): this project uses
  the SPICE system developed by NASA's NAIF, and JPL Development Ephemerides
  (DE440) [DE440]; C. H. Acton (1996), *Planet. Space Sci.* 44, 65 [SPICE].

## ERFA (Essential Routines for Fundamental Astronomy) — 2.0.1

- Source: `third_party/erfa/` — `src/*.c`, `src/*.h` from
  https://github.com/liberfa/erfa tag `v2.0.1` (the version bundled by
  `pyerfa` 2.0.1.x, so results are bit-identical to the Python oracle).
  The upstream self-test sources `t_erfa_c*.c` are omitted.
- License: 3-clause BSD, `third_party/erfa/LICENSE` (Copyright (C) 2013-2021,
  NumFOCUS Foundation). The license requires that the derivation notice be
  kept: ERFA is derived, with permission, from the International Astronomical
  Union's *Standards of Fundamental Astronomy* (SOFA) library
  (http://www.iausofa.org), SOFA release 2023-10-11. ERFA is **not** SOFA;
  the IAU SOFA Board bears no responsibility for it or for this project.

## nanobind

- Build dependency (from PyPI, not vendored): https://github.com/wjakob/nanobind,
  BSD-3-Clause, Wenzel Jakob. Linked statically into the `_eclipse` module.

## Catch2

- Test-only dependency fetched at configure time when `ECLIPSE_BUILD_TESTS=ON`
  (never part of the wheel): https://github.com/catchorg/Catch2, BSL-1.0.

"""Frame-consistency guard: the binary-PCK ITRF93 frame must agree with the
ERFA IAU 2006/2000A ITRS frame (both high precision) to well below our
accuracy ceiling.

This turns the claim (``ephem.hpp``) -- ITRF93 is "equivalent
accuracy to ITRS" -- into a CI-guarded invariant.  Measured agreement across
the three reference eclipses (DE440s): x, y < 0.1 m; d, mu < 0.01 arcsec;
greatest-eclipse ground point < 0.1 m (see docs; the tolerances below carry
~10x margin so a genuine frame regression trips but noise does not).

Requires the SPICE kernels AND the high-precision binary Earth PCK
(``earth_latest_high_prec.bpc``); skips cleanly when either is absent (the
GitHub-mirror kernel set ships no binary PCK, so ITRF93 is unavailable there).
"""

from __future__ import annotations

import _eclipse as E
import numpy as np
import pytest

from app.core import SpiceError, default_metakernel, load_kernels

pytestmark = pytest.mark.skipif(
    not default_metakernel().exists(),
    reason="SPICE kernels not downloaded (run `python -m kernels.bootstrap`)",
)

# Published element epochs (TDT) for the three reference eclipses.
_EPOCHS = [
    "2017-08-21 18:00:00 TDT",
    "2024-04-08 18:00:00 TDT",
    "2023-10-14 18:00:00 TDT",
]

# Tolerances (~10x the measured residual): x, y in Earth radii; d, mu in degrees.
_XY_TOL_RE = 1.0e-7           # ~0.6 m on the ground
_ANG_TOL_DEG = 0.05 / 3600.0  # 0.05 arcsec


def besselian_instant(et: float, frame: str) -> dict[str, float]:
    return {k: float(v[0]) for k, v in E.besselian_instants(np.array([et]), frame).items()}


def _itrf93_available() -> bool:
    """True if the furnished kernels support the ITRF93 binary-PCK frame."""
    if not default_metakernel().exists():
        return False
    load_kernels()
    try:
        besselian_instant(0.0, "ITRF93")  # et = 0 -> J2000
        return True
    except SpiceError:
        return False


requires_itrf93 = pytest.mark.skipif(
    not _itrf93_available(),
    reason="ITRF93 needs the high-precision binary Earth PCK (NAIF kernel set only)",
)


@requires_itrf93
@pytest.mark.parametrize("epoch", _EPOCHS)
def test_itrf93_matches_itrs(epoch):
    """ITRF93 (binary PCK) and ITRS (ERFA c2t06a + IERS EOP) agree to < ceiling.

    Both are full IAU 2006/2000A Earth-orientation models driven by the same
    EOP, so they must be interchangeable for eclipse geometry.  Confirms the
    default ITRS frame loses no accuracy by not requiring a binary PCK.
    """
    et = E.str_to_et(epoch)
    itrs = besselian_instant(et, "ITRS")
    itrf = besselian_instant(et, "ITRF93")

    assert itrf["x"] == pytest.approx(itrs["x"], abs=_XY_TOL_RE), "x"
    assert itrf["y"] == pytest.approx(itrs["y"], abs=_XY_TOL_RE), "y"
    assert itrf["d"] == pytest.approx(itrs["d"], abs=_ANG_TOL_DEG), "d"
    # mu carries the Greenwich rotation; both frames share it, so it agrees too.
    assert itrf["mu"] == pytest.approx(itrs["mu"], abs=_ANG_TOL_DEG), "mu"
    # Shadow-cone half-angles are frame-independent (depend only on distances).
    assert itrf["tan_f1"] == pytest.approx(itrs["tan_f1"], rel=1e-9), "tan_f1"
    assert itrf["tan_f2"] == pytest.approx(itrs["tan_f2"], rel=1e-9), "tan_f2"

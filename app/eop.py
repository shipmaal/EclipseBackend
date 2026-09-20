"""Earth orientation parameters (polar motion, UT1-UTC) from IERS.

Reads the IERS ``finals2000A.all`` bundle shipped by the ``astropy-iers-data``
PyPI package (updated weekly) and interpolates the Bulletin A values.  These
feed ERFA's celestial-to-terrestrial transform for the high-precision ``ITRS``
Earth-orientation frame, giving true (EOP-corrected) Earth rotation without the
NAIF binary Earth PCK.

Polar motion (xp, yp) and UT1-UTC are the Earth-orientation parameters defined by
the IERS Conventions [IERS2010]; linear interpolation between daily tabulated
values is the standard use of the Bulletin A series.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    from astropy_iers_data import IERS_A_FILE
except Exception:  # pragma: no cover - dependency always present in practice
    IERS_A_FILE = None

_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


@lru_cache(maxsize=1)
def _table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mjd, xp_arcsec, yp_arcsec, dut1_sec) arrays from finals2000A.all.

    Uses the Bulletin A columns; rows without a polar-motion value (far-future
    padding) are dropped.
    """
    if IERS_A_FILE is None:
        raise RuntimeError("astropy-iers-data is not installed")
    mjd, xp, yp, dut1 = [], [], [], []
    with open(IERS_A_FILE) as fh:
        for line in fh:
            pm_x = line[18:27].strip()
            if not pm_x:  # no Bulletin A polar motion beyond this row
                break
            mjd.append(float(line[7:15]))
            xp.append(float(pm_x))
            yp.append(float(line[37:46]))
            dut1.append(float(line[58:68]))
    return (np.array(mjd), np.array(xp), np.array(yp), np.array(dut1))


def eop(mjd_utc: float) -> tuple[float, float, float]:
    """Interpolate (xp, yp, dut1) at a UTC MJD.

    Returns polar motion ``xp, yp`` in **radians** and ``UT1-UTC`` in **seconds**.
    Outside the table range the endpoint values are held (np.interp clamps).
    """
    m, xp, yp, dut1 = _table()
    return (
        float(np.interp(mjd_utc, m, xp) * _ARCSEC_TO_RAD),
        float(np.interp(mjd_utc, m, yp) * _ARCSEC_TO_RAD),
        float(np.interp(mjd_utc, m, dut1)),
    )

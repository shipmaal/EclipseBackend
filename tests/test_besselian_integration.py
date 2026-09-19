"""End-to-end validation against the published 2024-04-08 central line.

Requires the SPICE kernels (run ``python -m kernels.bootstrap``); the whole
module is skipped automatically when they are absent, so the rest of the suite
still runs in CI without network access.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ephemeris import default_metakernel
from app.reference import central_line

pytestmark = pytest.mark.skipif(
    not default_metakernel().exists(),
    reason="SPICE kernels not downloaded (run `python -m kernels.bootstrap`)",
)

T0 = "2024-04-08T18:00:00"


def test_central_line_matches_published_track():
    from app.besselian import BesselianModel
    from app.geography import fund_to_geo

    ref = central_line()
    model = BesselianModel(t0_utc=T0, half_window_hours=2.0)

    # Reference rows are every 2 minutes starting at T0 (18:00 UT).
    t = np.arange(len(ref)) * 2 / 60.0
    elems = model.evaluate(t)

    dlat, dlon = [], []
    for i in range(len(ref)):
        lon, lat = fund_to_geo(elems["x"][i], elems["y"][i], elems["d"][i], elems["mu"][i])
        dlat.append(abs(lat - ref.lat[i]))
        dlon.append(abs(lon - ref.lon[i]))

    # Tolerance is generous for a first cut; tighten once verified locally.
    # (A correct SPICE/DE440 computation should land well inside this.)
    assert max(dlat) < 0.25, f"max latitude error {max(dlat):.3f} deg"
    assert max(dlon) < 0.25, f"max longitude error {max(dlon):.3f} deg"

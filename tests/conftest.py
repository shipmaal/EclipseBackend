"""Shared fixtures.

``spice`` is spiceypy with its own kernel pool, furnished from the same
metakernel as the core: the independent checks (the 3D limb oracle, the limb
axes) read positions and frames through it, so they share nothing with the
core's SPICE calls but the kernel files.
"""

from __future__ import annotations

import pytest

from app.core import metakernel


@pytest.fixture(scope="session")
def spice():
    import spiceypy

    spiceypy.furnsh(str(metakernel()))
    yield spiceypy
    spiceypy.kclear()

"""Backend switch for the native core (``libeclipse``, module ``_eclipse``).

``ECLIPSE_BACKEND`` selects who computes the ephemeris layer:

* ``python`` (default) -- the pure-Python/NumPy implementation in
  :mod:`app.ephemeris`, which is the **oracle** and is never deleted.
* ``native`` -- the C++ port (``core/``), reached through the nanobind module
  ``_eclipse`` built by ``uv sync``.  Its results are parity-tested against the
  Python at the tolerances of ``docs/CPP_ROADMAP.md`` §4
  (``tests/test_native.py``).

The extension links its own CSPICE, so it has its *own* kernel pool: this
module furnishes the metakernel into it alongside spiceypy's, and injects the
IERS table from :mod:`app.eop` (the C++ side parses nothing -- one source of
EOP truth).  Everything here is idempotent and takes
:data:`app.ephemeris.SPICE_LOCK` for the same reason the Python does.
"""

from __future__ import annotations

import os
from functools import lru_cache

BACKEND = os.environ.get("ECLIPSE_BACKEND", "python").lower()
BACKENDS = ("python", "native")

if BACKEND not in BACKENDS:
    raise RuntimeError(f"ECLIPSE_BACKEND={BACKEND!r}; expected one of {BACKENDS}")


def is_native() -> bool:
    """True when ``ECLIPSE_BACKEND=native``."""
    return BACKEND == "native"


@lru_cache(maxsize=1)
def module():
    """The ``_eclipse`` module with the IERS table installed (import on first use)."""
    import _eclipse

    from .eop import _table

    mjd, xp, yp, dut1 = _table()
    _eclipse.set_eop_table(mjd, xp, yp, dut1)
    return _eclipse


def furnish(metakernel: str) -> None:
    """Load the metakernel into the native pool (mirror of ``spice.furnsh``)."""
    module().furnish(metakernel)


def kclear() -> None:
    module().kclear()

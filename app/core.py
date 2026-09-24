"""The native core ``_eclipse`` (``libeclipse``, C++) as the API uses it.

Every number the service returns is computed by the core (``core/``, bound in
``bindings/``); ``app/`` only parses requests, calls the core and formats the
results.  This module owns the process-wide state the core needs:

* the SPICE kernels, furnished into the core's own kernel pool from the
  metakernel (``$SPICE_METAKERNEL``, default ``kernels/eclipse.tm``, written by
  ``python -m kernels.bootstrap``);
* the IERS Earth-orientation table, which the core parses from the
  ``finals2000A.all`` bundled by the ``astropy-iers-data`` package
  [IERS2010];
* the lunar limb band (``$ECLIPSE_LIMB_BAND``, default
  ``kernels/lola_ldem16_limb20.bin``, from ``kernels.bootstrap --limb``),
  which the core reads from its file on first use (docs/LIMB_PROFILE.md).

The core serializes its own CSPICE calls (a global mutex in ``ephem.cpp``), so
the API handlers run in FastAPI's thread pool without further locking, except
:data:`PARALLEL_LOCK` around the two OpenMP-parallel entry points.

``_eclipse`` is built by ``uv sync`` (scikit-build-core + CMake); there is no
pure-Python fallback, so a missing build is an ``ImportError`` here.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import _eclipse
from astropy_iers_data import IERS_A_FILE

ROOT = Path(__file__).resolve().parent.parent

# A CSPICE failure inside the core: a RuntimeError subclass carrying the four
# CSPICE message fields (short_message, explanation, long_message,
# traceback_text); short_message is e.g. "SPICE(SPKINSUFFDATA)".
SpiceError = _eclipse.SpiceError

# Earth-orientation frames the core supports (docs: CLAUDE.md "Frames"); ITRS
# (ERFA IAU 2006/2000A + IERS EOP) is the default and needs no binary PCK.
EARTH_FRAMES = ("ITRS", "TOD", "ITRF93", "IAU_EARTH")
DEFAULT_EARTH_FRAME = os.environ.get("SPICE_EARTH_FRAME", "ITRS")

# Serializes the core's OpenMP-parallel entry points within a process: the
# grid (``circumstances_grid``), the catalog (``find_eclipses``), the element
# loop of ``local_circumstances`` and ``central_line`` (>= 64 instants) and the
# cold-node silhouette of the limb profile (item C6).  Each already uses every core it is
# given, and libgomp starts a separate team -- sized ``OMP_NUM_THREADS``,
# default the host's core count -- for every calling thread, so K concurrent
# requests on FastAPI's thread pool would run K x cores threads.  Holding this
# lock bounds one worker process to one team; the Docker CMD sizes that team
# to cores / workers (review item C9).  Results do not depend on it.  The limb
# band load takes it too.  It is not re-entrant: take it around a single core
# call, never around code that takes it again (``ensure_limb_band``).
PARALLEL_LOCK = threading.Lock()

_STATE_LOCK = threading.Lock()
_kernels_loaded = False

# The limb-band file the core reads by default (kernels.bootstrap --limb).
DEFAULT_BAND_FILE = "lola_ldem16_limb20.bin"


def default_metakernel() -> Path:
    return ROOT / "kernels" / "eclipse.tm"


def metakernel() -> Path:
    """The metakernel in use: ``$SPICE_METAKERNEL``, else :func:`default_metakernel`."""
    return Path(os.environ.get("SPICE_METAKERNEL") or default_metakernel())


def load_kernels(path: str | os.PathLike | None = None) -> None:
    """Furnish the core's kernel pool and install the EOP table (idempotent).

    Resolution order: explicit argument, then :func:`metakernel`
    (``$SPICE_METAKERNEL``, then the repo default ``kernels/eclipse.tm``).
    ``FileNotFoundError`` when absent.
    """
    global _kernels_loaded
    with _STATE_LOCK:
        if _kernels_loaded:
            return
        mk = Path(path) if path else metakernel()
        if not mk.exists():
            raise FileNotFoundError(
                f"SPICE metakernel not found at {mk}. Run `python -m kernels.bootstrap` "
                "to download the required kernels."
            )
        if not _eclipse.has_eop_table():
            _eclipse.load_eop_file(str(IERS_A_FILE))
        _eclipse.furnish(str(mk))
        _kernels_loaded = True


def unload_kernels() -> None:
    """Unload every kernel from the core's pool (the EOP table stays)."""
    global _kernels_loaded
    with _STATE_LOCK:
        _eclipse.kclear()
        _kernels_loaded = False


def default_band_path() -> Path:
    """The limb-band file: ``$ECLIPSE_LIMB_BAND`` or ``kernels/`` + :data:`DEFAULT_BAND_FILE`."""
    env = os.environ.get("ECLIPSE_LIMB_BAND")
    return Path(env) if env else ROOT / "kernels" / DEFAULT_BAND_FILE


def ensure_limb_band(path: str | os.PathLike | None = None) -> str:
    """Make the core hold the limb band at ``path`` (default
    :func:`default_band_path`), reading the file there unless it already holds
    it; returns the resolved path.  ``FileNotFoundError`` when absent."""
    p = Path(path) if path else default_band_path()
    if not p.exists():
        raise FileNotFoundError(
            f"lunar limb band not found at {p}. Run `python -m kernels.bootstrap --limb`.")
    resolved = str(p.resolve())
    with PARALLEL_LOCK:
        if _eclipse.limb_band_source() != resolved:
            _eclipse.load_limb_band(resolved)
    return resolved

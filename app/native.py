"""Backend switch for the native core (``libeclipse``, module ``_eclipse``).

``ECLIPSE_BACKEND`` selects who computes the ephemeris, geometry, circumstances
and catalog layers:

* ``python`` -- the pure-Python/NumPy implementation in :mod:`app.ephemeris`,
  :mod:`app.geography`, :mod:`app.circumstances` and :mod:`app.catalog`, which
  is the **oracle** and is never deleted.
* ``native`` -- the C++ port (``core/``), reached through the nanobind module
  ``_eclipse`` built by ``uv sync``.  Its results are parity-tested against the
  Python at the tolerances of ``docs/CPP_ROADMAP.md`` §4
  (``tests/test_native.py``).  Explicitly requested and not importable is an
  ``ImportError`` at import time.
* ``auto`` (the default since phase 4) -- ``native`` when ``_eclipse`` imports,
  else ``python`` with one :class:`RuntimeWarning`: the API never fails for a
  missing extension.

:data:`BACKEND` holds the resolved choice (``python`` or ``native``);
:func:`is_native` reads it at call time, so tests can monkeypatch it.

The extension links its own CSPICE, so it has its *own* kernel pool: this
module furnishes the metakernel into it alongside spiceypy's, and injects the
IERS table from :mod:`app.eop` (the C++ side parses nothing -- one source of
EOP truth).  Everything here is idempotent and takes
:data:`app.ephemeris.SPICE_LOCK` for the same reason the Python does.
"""

from __future__ import annotations

import functools
import os
import warnings
from functools import lru_cache

import spiceypy.utils.exceptions as spice_exc

BACKENDS = ("python", "native", "auto")


def _resolve(requested: str) -> str:
    """``python`` or ``native`` for the ``ECLIPSE_BACKEND`` value ``requested``.

    ``auto`` probes ``import _eclipse`` (no kernel furnish, no EOP table: that
    happens on first use in :func:`raw_module`) and falls back to ``python``
    with a warning; ``native`` lets the ``ImportError`` propagate.
    """
    if requested not in BACKENDS:
        raise RuntimeError(f"ECLIPSE_BACKEND={requested!r}; expected one of {BACKENDS}")
    if requested == "native":
        import _eclipse  # noqa: F401  -- explicit request: an unbuilt core is an error
        return "native"
    if requested == "auto":
        try:
            import _eclipse  # noqa: F401
        except ImportError as exc:
            warnings.warn(
                "ECLIPSE_BACKEND is unset and the native core `_eclipse` is not importable "
                f"({exc}); computing with the pure-Python backend (run `uv sync` to build it, "
                "or set ECLIPSE_BACKEND=python to silence this).",
                RuntimeWarning,
                stacklevel=2,
            )
            return "python"
        return "native"
    return "python"


BACKEND = _resolve(os.environ.get("ECLIPSE_BACKEND", "auto").lower())


def is_native() -> bool:
    """True when the resolved backend is ``native`` (read at call time)."""
    return BACKEND == "native"


@lru_cache(maxsize=1)
def raw_module():
    """The ``_eclipse`` module itself, with the IERS table installed (import on first use).

    Raises the module's own ``SpiceError``; use :func:`module` from ``app`` code.
    """
    import _eclipse

    from .eop import _table

    mjd, xp, yp, dut1 = _table()
    _eclipse.set_eop_table(mjd, xp, yp, dut1)
    return _eclipse


def _as_spiceypy_error(err) -> spice_exc.SpiceyError:
    """Rebuild the spiceypy exception the Python backend would have raised.

    ``_eclipse.SpiceError`` carries the four CSPICE message fields; spiceypy
    derives its exception *class* from the short message (e.g.
    ``SpiceFRAMEDATANOTFOUND``), so ``except spiceypy.utils.exceptions.SpiceyError``
    in :mod:`app.main` / :mod:`app.besselian` and the specific subclasses in
    the tests behave identically on both backends.
    """
    return spice_exc.dynamically_instantiate_spiceyerror(
        short=err.short_message,
        explain=err.explanation,
        long=err.long_message,
        traceback=err.traceback_text,
    )


class _Translating:
    """Proxy over ``_eclipse`` that re-raises ``SpiceError`` as spiceypy's classes."""

    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, name):
        attr = getattr(self._mod, name)
        if not callable(attr) or isinstance(attr, type):
            return attr

        @functools.wraps(attr)
        def call(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except self._mod.SpiceError as err:
                raise _as_spiceypy_error(err) from err

        return call


@lru_cache(maxsize=1)
def module() -> _Translating:
    """``_eclipse`` for ``app`` code: same calls, spiceypy exception types."""
    return _Translating(raw_module())


def furnish(metakernel: str) -> None:
    """Load the metakernel into the native pool (mirror of ``spice.furnsh``)."""
    module().furnish(metakernel)


def kclear() -> None:
    module().kclear()

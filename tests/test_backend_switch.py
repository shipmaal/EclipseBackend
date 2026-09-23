"""The ``ECLIPSE_BACKEND`` resolution of :mod:`app.native` (no kernels, no
native core needed): each case is a fresh interpreter, with ``_eclipse`` made
unimportable by ``sys.modules["_eclipse"] = None`` where the case needs it.

* unset (``auto``) and unimportable -> ``python`` with one RuntimeWarning;
* ``native`` and unimportable -> ImportError at import;
* ``python`` -> ``python``, silently, whether or not the core is built;
* anything else -> RuntimeError.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BLOCK = "import sys; sys.modules['_eclipse'] = None\n"


def _run(code: str, backend: str | None):
    env = {k: v for k, v in os.environ.items() if k != "ECLIPSE_BACKEND"}
    env["PYTHONPATH"] = str(_ROOT)
    if backend is not None:
        env["ECLIPSE_BACKEND"] = backend
    return subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                          capture_output=True, text=True)


def test_auto_falls_back_to_python_with_a_warning_when_unbuilt():
    out = _run(_BLOCK + """
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from app import native
assert native.BACKEND == "python", native.BACKEND
assert not native.is_native()
hits = [x for x in w if issubclass(x.category, RuntimeWarning) and "_eclipse" in str(x.message)]
assert len(hits) == 1, [str(x.message) for x in w]
print("ok")
""", backend=None)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_explicit_auto_behaves_like_unset():
    out = _run(_BLOCK + """
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from app import native
assert native.BACKEND == "python" and len(w) == 1
print("ok")
""", backend="auto")
    assert out.returncode == 0, out.stderr


def test_explicit_native_is_an_import_error_when_unbuilt():
    out = _run(_BLOCK + """
try:
    from app import native
except ImportError as exc:
    print("ImportError", exc)
else:
    print("no error", native.BACKEND)
""", backend="native")
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ImportError")


def test_explicit_python_is_silent():
    out = _run("""
import warnings
warnings.simplefilter("error")
from app import native
assert native.BACKEND == "python" and not native.is_native()
print("ok")
""", backend="python")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_unknown_backend_is_a_runtime_error():
    out = _run("""
try:
    from app import native
except RuntimeError as exc:
    print("RuntimeError", exc)
""", backend="bogus")
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("RuntimeError") and "bogus" in out.stdout


def test_empty_backend_behaves_like_unset():
    # ``ECLIPSE_BACKEND=`` (a blank .env entry) used to be a RuntimeError at
    # import, taking the whole app down (review item C4).
    out = _run(_BLOCK + """
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from app import native
assert native.BACKEND == "python" and len(w) == 1
print("ok")
""", backend="")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"

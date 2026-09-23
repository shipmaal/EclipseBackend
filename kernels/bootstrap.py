"""Download the SPICE kernels needed for eclipse computation and write a metakernel.

Two sources are supported::

    uv run python -m kernels.bootstrap                # auto: NAIF, else mirror
    uv run python -m kernels.bootstrap --source naif   # official NAIF (best kernels)
    uv run python -m kernels.bootstrap --source mirror # GitHub mirror (restricted networks)

Some sandboxes block ``naif.jpl.nasa.gov`` at the egress proxy but allow
``raw.githubusercontent.com``; the ``mirror`` source fetches a working kernel
set from a pinned commit of the public ``alfonsogonzalez/AWP`` repository.

Downloads land in this directory and are loaded via the generated
``eclipse.tm`` metakernel.  Files already present are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
METAKERNEL = HERE / "eclipse.tm"

NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

# Official NAIF set: full-precision ephemeris + high-precision Earth orientation
# (ITRF93). Preferred when reachable.
NAIF_KERNELS: list[tuple[str, str]] = [
    (f"{NAIF}/lsk/naif0012.tls", "leapseconds (UTC <-> ET)"),
    (f"{NAIF}/spk/planets/de440s.bsp", "JPL DE440s ephemeris (1849-2150)"),
    (f"{NAIF}/pck/pck00011.tpc", "planetary constants (radii, IAU_EARTH)"),
    (f"{NAIF}/pck/earth_latest_high_prec.bpc", "high-precision Earth orientation (ITRF93)"),
]

# GitHub mirror, pinned for reproducibility. No binary Earth PCK here, so the
# ITRF93 frame is unavailable; the default ITRS frame (ERFA + IERS EOP from
# PyPI) needs no PCK and loses nothing (see tests/test_frame_consistency.py).
_AWP = "https://raw.githubusercontent.com/alfonsogonzalez/AWP/f90a593c76095cb1e94c609a947a1430458cefe2/data/spice"
MIRROR_KERNELS: list[tuple[str, str]] = [
    (f"{_AWP}/lsk/naif0012.tls", "leapseconds (UTC <-> ET)"),
    (f"{_AWP}/spk/de432s.bsp", "JPL DE432s ephemeris (1949-2050)"),
    (f"{_AWP}/pck/pck00010.tpc", "planetary constants (radii, IAU_EARTH)"),
]


def _reachable(host_url: str) -> bool:
    try:
        requests.head(host_url, timeout=10, allow_redirects=True)
        return True
    except requests.RequestException:
        return False


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip  {dest.name} (already present)")
        return
    print(f"  get   {dest.name} ...", end="", flush=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        tmp.replace(dest)
    print(f" done ({dest.stat().st_size / 1e6:.1f} MB)")


def write_metakernel(filenames: list[str]) -> None:
    listed = "\n".join(f"    '$KERNELS/{name}'" for name in filenames)
    METAKERNEL.write_text(
        "\\begindata\n"
        f"PATH_VALUES  = ( '{HERE}' )\n"
        "PATH_SYMBOLS = ( 'KERNELS' )\n"
        "KERNELS_TO_LOAD = (\n"
        f"{listed}\n"
        ")\n"
        "\\begintext\n"
    )
    print(f"wrote {METAKERNEL}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "naif", "mirror"], default="auto")
    args = parser.parse_args()

    source = args.source
    if source == "auto":
        source = "naif" if _reachable(NAIF + "/lsk/naif0012.tls") else "mirror"
        print(f"auto-selected source: {source}")

    kernels = NAIF_KERNELS if source == "naif" else MIRROR_KERNELS
    print(f"Downloading SPICE kernels into {HERE}")
    names = []
    for url, purpose in kernels:
        name = url.split("/")[-1]
        print(f"- {name}: {purpose}")
        download(url, HERE / name)
        names.append(name)
    write_metakernel(names)

    if source == "mirror":
        print(
            "\nNote: the mirror has no binary Earth PCK, so the ITRF93 frame is "
            "unavailable. The default 'ITRS' frame (pyerfa + IERS EOP) is high "
            "precision and needs no binary PCK, so no change is required."
        )
    print("\nDone. The API and tests will find the kernels via kernels/eclipse.tm.")


if __name__ == "__main__":
    main()

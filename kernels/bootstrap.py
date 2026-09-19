"""Download the SPICE kernels needed for eclipse computation and write a metakernel.

Run once (needs network access to ``naif.jpl.nasa.gov``)::

    uv run python -m kernels.bootstrap

Downloads land in this directory and are loaded via the generated
``eclipse.tm`` metakernel.  Files are skipped if already present.
"""

from __future__ import annotations

from pathlib import Path

import requests

NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

# (relative URL under NAIF, purpose) -- see kernels/README.md for the rationale.
KERNELS: list[tuple[str, str]] = [
    ("lsk/naif0012.tls", "leapseconds (UTC <-> ET)"),
    ("spk/planets/de440s.bsp", "JPL DE440s planetary ephemeris (Sun/Moon/Earth)"),
    ("pck/pck00011.tpc", "planetary constants (body radii, IAU_EARTH)"),
    ("pck/earth_latest_high_prec.bpc", "high-precision Earth orientation (ITRF93)"),
]

HERE = Path(__file__).resolve().parent
METAKERNEL = HERE / "eclipse.tm"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip  {dest.name} (already present)")
        return
    print(f"  get   {dest.name} ...", end="", flush=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        tmp.replace(dest)
    print(f" done ({dest.stat().st_size/1e6:.1f} MB)")


def write_metakernel(filenames: list[str]) -> None:
    # Absolute PATH_VALUES so `furnsh` works regardless of the process CWD.
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
    print(f"Downloading SPICE kernels into {HERE}")
    names = []
    for rel, purpose in KERNELS:
        name = rel.split("/")[-1]
        print(f"- {name}: {purpose}")
        download(f"{NAIF}/{rel}", HERE / name)
        names.append(name)
    write_metakernel(names)
    print("\nDone. The API and tests will now find the kernels via kernels/eclipse.tm.")


if __name__ == "__main__":
    main()

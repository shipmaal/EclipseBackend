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

``--limb`` also installs the lunar limb profile inputs (``docs/LIMB_PROFILE.md``):
the DE440 lunar orientation kernels (binary PCK ``moon_pa_de440_200625.bpc`` +
frame kernel ``moon_de440_250416.tf``, which define ``MOON_ME``) and the LOLA
limb band ``lola_ldem16_limb20.bin``.  From NAIF/PDS the band is cut locally
from ``LDEM_16`` (:mod:`kernels.limb_band`); the ``mirror`` source downloads the
same pre-cut bytes.  Every limb file is SHA-256 pinned.
"""

from __future__ import annotations

import argparse
import hashlib
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

# Lunar limb profile inputs (``--limb``), each SHA-256 pinned.
PDS_LOLA = ("https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/"
            "data/lola_gdr/cylindrical/img")
LIMB_BAND = "lola_ldem16_limb20.bin"
LIMB_SHA256 = {
    "moon_pa_de440_200625.bpc": "60cd55aa401ea2ea97360636f567554bfe4e37bb829f901b4460a455dfaf783f",
    "moon_de440_250416.tf": "a47c71e9c9f33796bdafb2c9d69a7ee447b6016ecad80f71cd6f3e479f9cf768",
    "ldem_16.img": "a511e40d7a3ea3275945b4da2a1df377133264fab0be94b7434b1cf8907254cb",
    LIMB_BAND: "97f0c52ce9d11bfc134b571184fc1b3d3e09fcb2b600ec5c9cc6cfeb5d042d55",
}
NAIF_MOON_KERNELS = [
    (f"{NAIF}/pck/moon_pa_de440_200625.bpc", "DE440 lunar orientation (MOON_PA)"),
    (f"{NAIF}/fk/satellites/moon_de440_250416.tf", "lunar frames (MOON_ME = LOLA's frame)"),
]
# Pre-cut limb band + the two lunar kernels, pinned, for networks without
# NAIF/PDS: the ``limb-data`` branch of this repository.
LIMB_MIRROR = ("https://raw.githubusercontent.com/shipmaal/EclipseBackend/"
               "e726f9aa6e93521dbc5519945d18b6f47a8fc077/limb")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> None:
    """Raise if ``path`` has a pin in ``LIMB_SHA256`` and does not match it."""
    want = LIMB_SHA256.get(path.name)
    if want is not None and sha256(path) != want:
        raise SystemExit(f"{path.name}: SHA-256 mismatch (expected {want}); delete it and retry")


def install_limb(source: str) -> list[str]:
    """Fetch the limb-profile inputs; returns the kernel names for the metakernel."""
    print("Lunar limb profile inputs")
    if source == "naif":
        kernels = NAIF_MOON_KERNELS
    else:
        kernels = [(f"{LIMB_MIRROR}/{u.split('/')[-1]}", p) for u, p in NAIF_MOON_KERNELS]
    names = []
    for url, purpose in kernels:
        name = url.split("/")[-1]
        print(f"- {name}: {purpose}")
        download(url, HERE / name)
        verify(HERE / name)
        names.append(name)
    band = HERE / LIMB_BAND
    print(f"- {LIMB_BAND}: LOLA limb band")
    if source == "naif" and _reachable(f"{PDS_LOLA}/ldem_16.lbl"):
        if not band.exists():
            from kernels.limb_band import cut

            for ext in ("lbl", "img"):
                download(f"{PDS_LOLA}/ldem_16.{ext}", HERE / f"ldem_16.{ext}")
            verify(HERE / "ldem_16.img")
            cut(HERE / "ldem_16.img", HERE / "ldem_16.lbl", band)
    else:
        download(f"{LIMB_MIRROR}/{LIMB_BAND}", band)
    verify(band)
    return names


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
    parser.add_argument("--limb", action="store_true",
                        help="also install the lunar limb profile inputs (~30 MB)")
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
    if args.limb:
        names += install_limb(source)
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

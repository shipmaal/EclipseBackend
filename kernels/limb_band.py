"""Cut the LRO LOLA gridded DEM down to the lunar limb band, and read the result.

The limb profile (``app/limb.py``, ``docs/LIMB_PROFILE.md``) needs only the
part of the Moon that can ever lie on the limb seen from the Earth.  The
Earth-facing direction in the mean-Earth frame (``MOON_ME``) is ``+x``; libration
moves the view axis off it (optical libration up to ~7.9 deg in longitude and
~6.9 deg in latitude [Meeus98] ch. 53; at eclipses 1900-2100 the shadow axis
stayed within 6.65 deg, docs/LIMB_PROFILE.md sec. 9.4), and a peak can show
over the limb from up to ~6-8 deg behind it (``cos a > R / (R + h)``,
h <= 10 km).  So the band kept is every pixel whose unit vector ``n``
satisfies ``|n . x| <= sin(BAND_DEG)`` with ``BAND_DEG`` = 20.

Source: LOLA ``LDEM_<ppd>`` [LOLA], PDS ``LRO-L-LOLA-3-RDR-V1``, simple
cylindrical, int16 LSB, ``radius_m = DN * SCALING_FACTOR + OFFSET`` (label),
line ``i`` at latitude ``(LINE_PROJECTION_OFFSET - i) / ppd`` and sample ``j``
at east longitude ``(j - SAMPLE_PROJECTION_OFFSET) / ppd + CENTER_LONGITUDE``
(PDS map-projection convention, 0-based indices), in the Mean Earth/Polar
Axis frame of DE421 = SPICE ``MOON_ME`` with ``moon_de440_*.tf``.

File layout (little-endian, deterministic: the same DEM gives the same bytes,
so the SHA-256 of a local cut equals the hosted file's)::

    b"ECLLIMB1"
    uint32 header_len, header_len bytes of JSON (sorted keys)   -- metadata
    uint32 n_runs, int32[n_runs, 3] (line, first_sample, count) -- row runs
    uint32 n_points, int16[n_points]                            -- DN, run order

Usage::

    uv run python -m kernels.limb_band kernels/ldem_16.img kernels/ldem_16.lbl OUT.bin
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"ECLLIMB1"
BAND_DEG = 20.0
FORMAT_VERSION = 1

_LABEL_KEYS = {
    "PRODUCT_ID": str,
    "LINES": int,
    "LINE_SAMPLES": int,
    "SAMPLE_BITS": int,
    "SAMPLE_TYPE": str,
    "SCALING_FACTOR": float,
    "OFFSET": float,
    "MAP_RESOLUTION": float,
    "LINE_PROJECTION_OFFSET": float,
    "SAMPLE_PROJECTION_OFFSET": float,
    "CENTER_LONGITUDE": float,
    "COORDINATE_SYSTEM_NAME": str,
}


def parse_label(text: str) -> dict:
    """The PDS3 keywords of ``_LABEL_KEYS`` from an LDEM ``.lbl`` (units dropped)."""
    out = {}
    for key, cast in _LABEL_KEYS.items():
        m = re.search(rf"^\s*{key}\s*=\s*(.+?)\s*$", text, re.MULTILINE)
        if m is None:
            raise ValueError(f"LDEM label has no {key}")
        value = re.sub(r"<[^>]*>", "", m.group(1)).strip().strip('"').strip()
        out[key] = cast(value)
    if (out["SAMPLE_BITS"], out["SAMPLE_TYPE"]) != (16, "LSB_INTEGER"):
        raise ValueError("expected 16-bit LSB_INTEGER LDEM samples")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cut(img: Path, lbl: Path, out: Path, band_deg: float = BAND_DEG) -> str:
    """Write the limb band of LDEM ``img`` to ``out``; returns ``out``'s SHA-256."""
    lab = parse_label(Path(lbl).read_text())
    lines, samples, ppd = lab["LINES"], lab["LINE_SAMPLES"], lab["MAP_RESOLUTION"]
    dem = np.memmap(img, dtype="<i2", mode="r", shape=(lines, samples))
    lat = np.radians((lab["LINE_PROJECTION_OFFSET"] - np.arange(lines)) / ppd)
    lon = np.radians((np.arange(samples) - lab["SAMPLE_PROJECTION_OFFSET"]) / ppd
                     + lab["CENTER_LONGITUDE"])
    cos_lon = np.cos(lon)
    limit = np.sin(np.radians(band_deg))

    runs, chunks = [], []
    for i in range(lines):
        keep = np.abs(np.cos(lat[i]) * cos_lon) <= limit
        edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.view(np.int8), [0]))))
        for j0, j1 in zip(edges[0::2], edges[1::2], strict=True):
            runs.append((i, int(j0), int(j1 - j0)))
            chunks.append(np.asarray(dem[i, j0:j1]))
    heights = np.concatenate(chunks).astype("<i2")
    header = {
        "format_version": FORMAT_VERSION,
        "band_deg": band_deg,
        "source_sha256": _sha256(Path(img)),
        **{k.lower(): v for k, v in lab.items()},
    }
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    run_arr = np.asarray(runs, dtype="<i4")
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(blob)))
        f.write(blob)
        f.write(struct.pack("<I", len(run_arr)))
        f.write(run_arr.tobytes())
        f.write(struct.pack("<I", len(heights)))
        f.write(heights.tobytes())
    return _sha256(Path(out))


@dataclass(frozen=True)
class BandFile:
    """A limb-band file as arrays: ``header`` (dict), ``runs`` (n, 3) int32 of
    ``(line, first_sample, count)`` and ``dn`` (int16, run order)."""

    header: dict
    runs: np.ndarray
    dn: np.ndarray


def read(path: str | Path) -> BandFile:
    """Parse a file written by :func:`cut` (validates magic, version and sizes)."""
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not a limb-band file")
    pos = 8
    (hlen,) = struct.unpack_from("<I", raw, pos)
    pos += 4
    header = json.loads(raw[pos:pos + hlen])
    pos += hlen
    if header.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported limb-band format {header.get('format_version')}")
    (n_runs,) = struct.unpack_from("<I", raw, pos)
    pos += 4
    runs = np.frombuffer(raw, dtype="<i4", count=3 * n_runs, offset=pos).reshape(n_runs, 3)
    pos += 12 * n_runs
    (n_pts,) = struct.unpack_from("<I", raw, pos)
    pos += 4
    dn = np.frombuffer(raw, dtype="<i2", count=n_pts, offset=pos)
    if pos + 2 * n_pts != len(raw) or int(runs[:, 2].sum()) != n_pts:
        raise ValueError(f"{path}: truncated or inconsistent limb-band file")
    return BandFile(header=header, runs=runs.astype(np.int32), dn=dn.astype(np.int16))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    print(cut(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))

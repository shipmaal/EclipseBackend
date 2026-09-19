"""Loader for the reference eclipse central-line track (``data.txt``).

``data.txt`` is a published prediction table (time + northern/southern limits,
central line, etc.) used to validate our computed path.  This module only reads
the file when called -- unlike the old ``test.py``, it does no work at import
time and resolves the path relative to this file rather than the process CWD.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "data.txt"

_COLUMNS = [
    "Universal",
    "Northern Limit",
    "Southern Limit",
    "Central Line",
    "Diam. Sun",
    "Sun Path",
    "Line",
    "Duration",
]


def _dm_to_degrees(deg: str, minute_hemi: str) -> float:
    """Parse a "degrees" token and a "minutes+hemisphere" token, e.g. ("20", "19.2N")."""
    hemisphere = minute_hemi[-1]
    value = float(deg) + float(minute_hemi[:-1]) / 60.0
    return -value if hemisphere in ("S", "W") else value


def central_line(data_file: Path | str = DEFAULT_DATA_FILE) -> pd.DataFrame:
    """Return the reference central line as a DataFrame [time, lat, lon] (degrees)."""
    rows = Path(data_file).read_text().strip().splitlines()

    records = []
    for row in rows:
        f = row.split()
        # Tokens 9-12 are the central line: lat_deg, lat_min+hemi, lon_deg, lon_min+hemi
        # e.g. "20 19.2N 108 45.8W".
        time = f[0]
        lat = _dm_to_degrees(f[9], f[10])
        lon = _dm_to_degrees(f[11], f[12])
        records.append({"time": time, "lat": lat, "lon": lon})

    return pd.DataFrame.from_records(records)

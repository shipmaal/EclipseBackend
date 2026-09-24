"""Tests for the reference central-line loader (no kernels needed)."""

from __future__ import annotations

from reference import central_line


def test_central_line_shape_and_bounds():
    df = central_line()
    assert list(df.columns) == ["time", "lat", "lon"]
    assert len(df) == 57
    # 2024-04-08 path enters over the Pacific/Mexico and exits over the Atlantic.
    assert df.lat.iloc[0] == 20.32
    assert df.lon.iloc[0] < -100  # western hemisphere
    # Track runs broadly south-west -> north-east across North America.
    assert df.lat.iloc[-1] > df.lat.iloc[0]
    assert df.lon.iloc[-1] > df.lon.iloc[0]

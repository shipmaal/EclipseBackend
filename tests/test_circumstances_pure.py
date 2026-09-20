"""Pure-function tests for the circle-overlap obscuration math (item M5).

These run without SPICE kernels: they exercise the geometry helpers
``_overlap_area`` and ``_obscuration`` against closed-form values.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.circumstances import _obscuration, _overlap_area


def test_overlap_area_disjoint():
    # Circles too far apart to touch -> no overlap.
    assert _overlap_area(1.0, 2.0, 5.0) == 0.0


def test_overlap_area_fully_contained():
    # Small circle entirely inside the large one (d <= R - r) -> its whole area.
    assert _overlap_area(1.0, 3.0, 0.5) == pytest.approx(np.pi)


def test_overlap_area_two_unit_circles():
    # Two radius-1 circles whose centres are 1 apart: the lens area has the closed
    # form 2 cos^-1(d/2) - (d/2) sqrt(4 - d^2) = 2*pi/3 - sqrt(3)/2.
    expected = 2.0 * np.pi / 3.0 - np.sqrt(3.0) / 2.0
    assert _overlap_area(1.0, 1.0, 1.0) == pytest.approx(expected)


def test_obscuration_total_is_one():
    # Total eclipse (umbral L2' < 0), observer on the axis (m = 0): the Moon disk
    # covers the whole Sun -> obscuration 1.0.
    assert _obscuration(0.005, -0.003, 0.0) == pytest.approx(1.0)


def test_obscuration_annular_is_ratio_squared():
    # Annular, observer on the axis: the covered fraction of the Sun's *area*
    # equals the (Moon/Sun apparent radius)^2 = q^2.
    l1p, l2p = 0.005, 0.003
    q = (l1p - l2p) / (l1p + l2p)
    assert _obscuration(l1p, l2p, 0.0) == pytest.approx(q * q)


def test_obscuration_no_overlap_is_zero():
    # Separation beyond both disk radii -> the Sun is uncovered.
    assert _obscuration(0.005, 0.003, 0.02) == 0.0

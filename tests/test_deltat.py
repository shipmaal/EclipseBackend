"""The core's delta-T polynomial model [Espenak] (``deltat.hpp``; no kernels needed)."""

from __future__ import annotations

import _eclipse as E
import numpy as np
import pytest


def delta_t_seconds(year):
    """Scalar or array years -> delta-T [s] (``_eclipse.delta_t_seconds``)."""
    out = E.delta_t_seconds(np.atleast_1d(np.asarray(year, dtype=float)))
    return out if np.ndim(year) else float(out[0])


def decimal_year_from_jd(jd: float) -> float:
    return float(E.decimal_year_from_jd(np.array([jd]))[0])

# Observed delta-T [s] from [Meeus98] Table 10.A (rounded to 0.1 s there).
_TABLE_10A = {1900: -2.8, 1920: 21.2, 1950: 29.1, 1960: 33.1, 1980: 50.5, 2000: 63.8}


@pytest.mark.parametrize("year,expected", sorted(_TABLE_10A.items()))
def test_matches_meeus_table_10a(year, expected):
    assert delta_t_seconds(year) == pytest.approx(expected, abs=0.3)


def test_segments_are_continuous():
    # The piecewise fit joins its segments to about a second [Espenak].
    for edge in (-500, 500, 1600, 1700, 1800, 1860, 1900, 1920, 1941, 1961, 1986, 2005, 2050, 2150):
        lo, hi = delta_t_seconds(edge - 1e-6), delta_t_seconds(edge + 1e-6)
        assert abs(hi - lo) < 1.5, edge


def test_vectorized_equals_scalar():
    years = np.array([-1000.0, 0.0, 1000.0, 1919.41, 2024.3, 2500.0])
    assert np.allclose(delta_t_seconds(years), [delta_t_seconds(y) for y in years])


def test_decimal_year_from_jd():
    assert decimal_year_from_jd(2451545.0) == pytest.approx(2000.0)
    assert decimal_year_from_jd(2451545.0 + 365.25) == pytest.approx(2001.0)

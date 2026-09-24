# USNO ΔT predictions (`deltat.preds`)

Unmodified copy of the U.S. Naval Observatory's published ΔT = TT − UT1
predictions, Earth Orientation Department, series 7:

- URL: https://maia.usno.navy.mil/ser7/deltat.preds
- Retrieved: 2026-09-24 (HTTP `Last-Modified: Thu, 18 Jun 2026 17:26:25 GMT`)
- SHA-256: `5d864fddd30b2c64d2a86d3debbb25604eb5de44370c96bccf2abd5463f3db08`
  (asserted by `tests/test_core.py`)

Columns (tab-separated): MJD, decimal year, TT−UT [s], UT1−UTC [s] (only in
the rows near the issue date), and the prediction's error estimate [s]
(0.031 s at the first row, growing to 1 s from 2032.75). Quarterly rows,
2022.50 – 2033.75.

Content age: despite the 2026 `Last-Modified`, the first row (2022.50) is
itself a prediction: its UT1−UTC (−0.104 s) differs from the IERS final value
for that day (−0.0672 s, Bulletin A and B in `finals2000A.all`) by 37 ms. So
the table was issued before mid-2022. Measured against the IERS values over
2022.50 – 2026.50, the predictions agree to ≤ 0.1 s, but by up to 2.1× their
own error column (`tests/test_core.py`). Treat that column as optimistic.

Use (CLAUDE.md "Frames & epochs", `eclipse::deltat::load_predictions`): past
the last row of the IERS Bulletin A table, ΔT is interpolated linearly in
this table; past its last row, the [Espenak] polynomial model applies. The
two joins are recorded, not smoothed (CLAUDE.md convention 7).

To update: download the file unmodified, update the date, `Last-Modified`
and SHA-256 above and in the test, and record in the PR how ΔT moved at the
reference instants.

A work of the U.S. Government (U.S. Naval Observatory), not subject to
copyright in the United States.

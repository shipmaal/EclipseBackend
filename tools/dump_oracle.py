"""Dump Python-oracle values at the reference eclipses for the C++ parity tests.

Writes ``tests/cpp/fixtures/*.txt`` (roadmap §4, "Parity harness"): plain
whitespace-separated records so the Catch2 tests need no JSON library and run
without Python in the loop.  Regenerate after any *intended* change to the
oracle (``uv run python tools/dump_oracle.py``) and commit the result; the
diff is then the review record of what moved.

Record types::

    eop  <mjd> <xp_arcsec> <yp_arcsec> <dut1_s>            # eop_subset.txt
    case <label> <frame> <et> x y z d mu l1 l2 tan_f1 tan_f2 tt2 ut1 xp yp lon lat
    utc  <string> <et> <round_trip_string>
    contacts <label> <frame> <et0> <half_window_h> <name> <t_hours> ...   # se*.txt
    axis <et> <rho> <z>                                    # se*.txt, phase 4: axis_separation
                                                           #   at the case instants (frame-free)

and, in ``geometry_cases.txt`` (phase 2: app.geography; inputs are literal so
these need no kernel to replay), all angles in degrees, lengths in Earth
radii / km as the Python signatures say::

    red <d_deg> rho1 rho2 sin_d1 cos_d1 sin_d1_d2 cos_d1_d2   # _reduction_aux
    f2g x y d mu lon lat                                       # fund_to_geo_v (nan = off Earth)
    g2f lat lon d mu xi eta zeta                               # geo_to_fund
    g2fh lat lon d mu h xi eta zeta                            # geo_to_fund at height h [m]
    gc  lat lon brg dist lat2 lon2 hav brg12                   # _destination/_haversine_km/bearing
    lim x y d mu l tan_f bearing max_km sunlit n_lat n_lon s_lat s_lon width
                                                               # shadow_edge_limits_v (sunlit 0/1)
    limr x y d mu l tan_f bearing dx dy dd dmu dl n_lat n_lon s_lat s_lon width
                                                               # ... with element rates (per hour):
                                                               #   the umbral path envelope (W2)
    geod lat1 lon1 lat2 lon2 km                                # geodesic_km (Andoyer)
    rad x y d l1 l2 tf1 tf2 pen umb is_total                   # shadow_radii (is_total 0/1)

and, in ``circumstances_cases.txt`` (phase 3: app.circumstances helpers at
literal inputs, kernel-free; angles in degrees, radii in Earth radii or, for
``ovl``, any consistent unit)::

    ovl  r R d area                      # _overlap_area (r <= R)
    obsc L1p L2p m obs                   # _obscuration
    unw  n v1..vn u1..un                 # np.degrees(np.unwrap(np.radians(v))), the
                                         #   mu unwrap of BesselianModel.evaluate_direct
    altaz lat lon ss_lon ss_lat alt az alt_grid
                                         # _sun_altaz (alt, az) at a given sub-solar
                                         #   point, and circumstances_grid's own
                                         #   altitude expression (alt_grid)
    roots n t1..tn f1..fn k tc1 rising1 i1 ... tck risingk ik   # _roots (rising 0/1)

and, appended to the three modern ``se*.txt`` (kernel-backed; models built as
``/circumstances`` and ``/map`` build them, ``lat``/``lon`` in degrees, times
in hours from T0, ``et0`` in TDB seconds)::

    local <label> <frame> <et0> <half_window_h> <lat> <lon> <_LocalRaw fields>
                                         # _local_raw: geometric central c1 c4 c2 c3
                                         #   t_max magnitude obscuration L2_x
                                         #   alt_deg[5] az_deg[5] below[5] eclipse
                                         #   (bools 0/1, nan where absent; events
                                         #   C1, max, C4, C2, C3)
    localh <label> <frame> <et0> <half_window_h> <lat> <lon> <height_m> <_LocalRaw fields>
                                         # _local_raw at a height above the ellipsoid [m]
    grid <label> <frame> <et0> <half_window_h> <step_min> <lat> <lon>
         magnitude obscuration t_max_hours sun_alt visible central
                                         # circumstances_grid on the single point

and, for phase 4 (app.catalog), in ``catalog_cases.txt`` (literal inputs:
replayable kernel-free)::

    ismin n rho1..rhon k idx1..idxk      # _local_minima: the k candidate indices into
                                         #   rho (1 <= idx <= n - 2), grid order
    classify rho_g l1 l2 tf1 tf2 central zeta kind magnitude L2p
                                         # _classify (central 0/1; kind a word; L2p
                                         #   nan when not central; the pre-hybrid kind)
    pyround x n r                        # Python's round(x, n) == r (signed zero kept)

and in ``catalog_2019_2024.txt`` (kernel-backed; ``label`` names the window,
``frame`` is ITRS, times in TDB seconds, ``kind`` a word, ``central`` 0/1,
``lat``/``lon`` the UNROUNDED greatest point in degrees or ``nan``)::

    scan <label> <et_a> <et_b> <n_grid> <grid0> <grid1> <grid_last> <n_cand> <cand_et>...
                                         # _coarse_grid / _scan_candidates over the window
    event <label> <frame> <detail> <et_g> <kind> <central> <gamma> <magnitude> <lat> <lon>
          <greatest_utc>                 # one _EventRaw of _catalog_raw (detail 0), or
    event ... <greatest_utc> <et0> <n_contacts> <name> <t_hours>... <local> <width>
                                         # (detail 1): contacts in global_contacts
                                         #   insertion order [h from et0]; <local> the
                                         #   26 _LocalRaw fields (as ``local``) or ``-``
                                         #   when not central; <width> km, ``0.0``
                                         #   when the limits are NaN, ``nan`` when
                                         #   absent (not central / no t = 0 track point)

Numbers are written with ``repr`` (shortest round-trip), so the fixture holds
the oracle's doubles exactly (``nan`` / ``inf`` for NaN and infinities;
``strtod`` parses them).  The
fixtures pin the **NAIF DE440s** kernel set (header line); the C++ test skips
when that SPK is not the one on disk.  The ``ITRF93`` rows also depend on the
binary Earth PCK, which NAIF regenerates daily under one name
(``earth_latest_high_prec.bpc``) and whose dated copies it deletes, so it
cannot be pinned by URL: the header records its identity instead,

    pck <name> <size_bytes> <crc32_hex>       # absent when no binary PCK is loaded

and the C++ test skips the ITRF93 rows (with a warning) unless the file on disk
is byte-identical.  The live test (``tests/test_native.py``) still checks ITRF93
parity with whatever PCK is loaded, since both backends then read the same file.
"""

from __future__ import annotations

import os
import sys
import warnings
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The fixtures are the *Python* oracle: the default backend is ``auto`` (native
# when built), so pin it before ``app`` resolves it.
os.environ["ECLIPSE_BACKEND"] = "python"

from app import catalog as cat  # noqa: E402
from app import circumstances as circ  # noqa: E402
from app import ephemeris as ep  # noqa: E402
from app import native  # noqa: E402
from app.besselian import BesselianModel  # noqa: E402
from app.circumstances import (  # noqa: E402
    _local_raw,
    _obscuration,
    _overlap_area,
    _roots,
    _sun_altaz,
    circumstances_grid,
)
from app.eop import _table  # noqa: E402
from app.geography import (  # noqa: E402
    _destination,
    _haversine_km,
    _reduction_aux,
    bearing,
    central_track,
    element_rates,
    fund_to_geo_v,
    geo_to_fund,
    geodesic_km,
    global_contacts,
    shadow_edge_limits_v,
    shadow_radii,
)
from tests.test_geography import POLY, _ev  # noqa: E402  (the pure-function test's instants)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "cpp" / "fixtures"
SPK = "de440s.bsp"
PCK = "earth_latest_high_prec.bpc"  # the ITRF93 frame's binary PCK (NAIF set only)


def _pck_identity() -> str | None:
    """``<name> <size> <crc32>`` of the binary Earth PCK on disk, or None if absent."""
    path = ROOT / "kernels" / PCK
    if not path.exists():
        return None
    data = path.read_bytes()
    return f"{PCK} {len(data)} {zlib.crc32(data):08x}"

# label, UTC start, UTC end, samples: ±3 h around greatest eclipse, 13 samples
# (the pytest parity test uses 241; the fixture is a compact pin).
ECLIPSES = [
    ("se2017aug21", "2017-08-21T15:26:40", "2017-08-21T21:26:40"),
    ("se2023oct14", "2023-10-14T15:00:41", "2023-10-14T21:00:41"),
    ("se2024apr08", "2024-04-08T15:17:20", "2024-04-08T21:17:20"),
    ("se1919may29", "1919-05-29T10:08:00", "1919-05-29T16:08:00"),  # outside the IERS era
]
FRAMES = ("ITRS", "TOD", "ITRF93", "IAU_EARTH")
N = 13

UTC_CASES = [
    "2017-08-21T18:26:40",
    "2023-10-14T18:00:41.5",
    "2024-04-08T18:17:20",
    "2016-12-31T23:59:60",  # leap second
    "1919-05-29T13:08:00",  # pre-IERS: read as UT1 + model delta-T
    "2200-01-01T00:00:00",  # post-table: same rule
]

# Greatest-eclipse epochs of tests/test_catalog.py (_GREATEST, [Espenak]): the
# ``contacts`` records replay global_contacts exactly as test_global_contacts_p1_p4
# builds it (half_window_hours=2.5 model, default 5 h contact window).
CONTACT_EPOCHS = {
    "se2017aug21": "2017-08-21T18:25:30",
    "se2023oct14": "2023-10-14T17:59:27",
    "se2024apr08": "2024-04-08T18:17:15",
}
CONTACT_FRAME = "ITRS"
CONTACT_HALF_WINDOW_H = 5.0  # global_contacts default

# geometry_cases.txt inputs. The central-line samples are the modern eclipses'
# 13 instants over +/-3 h, evaluated the way /central-line does
# (central_track -> BesselianModel.evaluate_direct, mu unwrapped), so ``lim``
# bearings are the along-track ones the endpoint uses.
TRACK_T_HOURS = np.linspace(-3.0, 3.0, N)
UMBRA_MAX_KM = 600.0      # shadow_edge_limits_v default (main.py umbral run)
PENUMBRA_MAX_KM = 10_000.0  # main.py penumbral run
OFF_EARTH = [  # |x| or |y/rho1| > 1: fund_to_geo_v -> NaN, limits -> NaN / width 0
    (2.0, 0.0, 7.5, 90.0),       # tests/test_geography.py: axis misses the Earth
    (-1.5, 0.3, -10.0, 45.0),
    (0.2, 1.2, 20.0, -100.0),
]
G2F_LATS = np.array([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
G2F_LONS = np.array([-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0])
# Observer heights above the ellipsoid [m] for the g2fh / localh records: the
# Dead Sea shore, Effingham IL, Vale OR (docs/LIMB_PROFILE.md sec. 9.10), Everest.
HEIGHTS_M = (-400.0, 149.4, 693.8, 8848.0)
LOCAL_HEIGHTS_M = (693.8, 3000.0)
GEOD_CASES = [  # (lat1, lon1, lat2, lon2): Andoyer geodesic_km, incl. the antimeridian
    (-75.0, -46.0, -78.5, -46.5), (4.4, -16.7, 6.5, -17.3), (25.3, -104.1, 26.2, -105.3),
    (0.0, 179.5, 0.5, -179.8), (48.8, 2.35, 52.5, 13.4), (-10.0, 125.0, -9.8, 125.3),
]
GC_CASES = [  # (lat, lon, bearing, dist_km): equator, mid-latitudes, poles, antimeridian
    (0.0, 0.0, 0.0, 100.0),
    (0.0, 0.0, 90.0, 100.0),
    (0.0, 0.0, 45.0, 20015.0),        # ~half the circumference
    (45.0, -100.0, 270.0, 600.0),
    (45.0, -100.0, 90.0, 600.0),
    (-33.9, 151.2, 45.0, 1000.0),
    (89.9, 0.0, 0.0, 50.0),           # over the pole
    (90.0, 0.0, 180.0, 100.0),        # from the north pole
    (-90.0, 0.0, 0.0, 100.0),         # from the south pole
    (-90.0, 45.0, 90.0, 300.0),
    (60.0, 179.9, 90.0, 200.0),       # across the antimeridian eastward
    (60.0, -179.9, 270.0, 200.0),     # ... westward
    (0.0, 180.0, 90.0, 1000.0),
    (0.0, -180.0, 270.0, 1000.0),
    (25.3, -104.1, 131.5, 10000.0),   # 2024 greatest eclipse, penumbral reach
    (25.3, -104.1, 311.5, 10000.0),
    (10.0, 20.0, 0.0, 0.0),           # zero distance
    (37.0, -87.7, 225.0, 600.0),
    (-60.0, -60.0, 135.0, 5000.0),
    (80.0, 170.0, 30.0, 3000.0),
    (11.4, -83.1, 359.999, 187.4),
]

# circumstances_cases.txt inputs (kernel-free).
OVL_CASES = [  # (r, R, d), r <= R: tests/test_circumstances_pure.py + the degenerate branches
    (1.0, 2.0, 5.0),      # disjoint -> 0
    (1.0, 3.0, 0.5),      # contained -> pi r^2
    (1.0, 1.0, 1.0),      # two unit circles -> 2 pi/3 - sqrt(3)/2
    (1.0, 1.0, 0.0),      # d = 0 with r = R: the contained branch (d <= R - r = 0)
    (1.0, 2.0, 3.0),      # external tangency d == r + R -> 0
    (1.0, 3.0, 2.0),      # internal tangency d == R - r -> pi r^2
    (0.7, 1.0, 0.9),
    (0.95, 1.0, 1.9),     # barely overlapping
    (0.9, 1.0, 0.15),     # deep overlap, not contained
    (1.0, 1.0, 0.5),
    (0.0, 1.0, 0.5),      # zero radius
    (1.0, 1.0, 1.9999),   # near tangency, r = R
    (0.98, 1.0, 0.02),    # d one ulp below R - r (0.020000000000000018): the contained branch
]
OBSC_CASES = [  # (L1p, L2p, m) in Earth radii
    (0.005, -0.003, 0.0),     # total, on axis -> 1
    (0.005, 0.003, 0.0),      # annular, on axis -> q^2
    (0.005, 0.003, 0.02),     # no overlap -> 0
    (0.005, -0.005, 0.0),     # degenerate L1p + L2p == 0 -> 0
    (0.001, -0.003, 0.0),     # degenerate L1p + L2p < 0 -> 0
    (0.5352, -0.0092, 0.3),   # realistic partial (2024-like L1', L2')
    (0.5352, -0.0092, 0.0),   # realistic total, on axis
    (0.5352, -0.0092, 0.005), # inside the umbra, off axis
    (0.5352, -0.0092, 0.0095),  # near the umbral edge
    (0.5352, -0.0092, 0.53),  # near the penumbral edge
    (0.5352, -0.0092, 0.5352),  # m == L1': the last contact
    (0.5645, 0.0184, 0.0),    # realistic annular (2023-like)
    (0.5645, 0.0184, 0.01),
    (0.5645, 0.0184, 0.25),
]
UNW_CASES = [  # degrees, as mu series reaching evaluate_direct's unwrap
    [170.0, 179.5, -179.5, -170.0],          # crossing +180 eastward
    [-170.0, -179.5, 179.5, 170.0],          # crossing -180
    [0.0, 180.0],                             # an exact +180 step (boundary_ambiguous)
    [-90.0, 90.0, -90.0],                     # exact +-180 steps both ways
    [175.0, -175.0, 175.0, -175.0],           # non-chronological 4-vector
    [10.0, 20.0, 30.0],                       # no wrap: the round trip alone
    [150.0, 165.0, 179.99, -165.0, -150.0],
    list(np.linspace(44.6, 134.6, 13)),       # a realistic +/-3 h hour-angle series
    [((v + 180.0) % 360.0) - 180.0 for v in np.linspace(150.0, 210.0, 13)],  # ... wrapping
]
ALTAZ_CASES = [  # (lat, lon, ss_lon, ss_lat)
    (37.0, -87.7, -87.9, 11.9),     # 2017 greatest: Sun near the zenith
    (25.3, -104.1, -95.6, 7.6),     # 2024 greatest
    (11.4, -83.1, -90.3, -8.2),     # 2023 greatest
    (40.71, -74.01, -95.6, 7.6),    # NYC, afternoon Sun
    (53.3, -9.0, -95.6, 7.6),       # W Ireland, sunset
    (20.0, 77.0, -95.6, 7.6),       # night side
    (-30.0, -60.0, -95.6, 7.6),
    (89.9, 0.0, -95.6, 7.6),        # poles
    (-89.9, 0.0, -95.6, 7.6),
    (90.0, 0.0, 0.0, 23.44),
    (0.0, 179.99, -95.6, 7.6),      # antimeridian, both sides
    (0.0, -179.99, -95.6, 7.6),
    (0.0, 179.99, 179.99, 0.0),     # sub-solar point exactly: alt 90, cos_c = 1
    (0.0, 0.0, 180.0, 0.0),         # antipode: alt -90
    (7.6, -95.6, -95.6, 7.6),       # observer at the sub-solar point
    (7.6, -95.6, 84.4, -7.6),       # ... and at its antipode
]
ROOTS_CASES = [  # (t, f)
    (np.arange(-2.0, 2.0 + 1e-9, 0.5 / 60.0)[:5], [1.0, 0.5, -0.25, -1.0, 0.0]),  # one falling
    ([0.0, 1.0, 2.0, 3.0], [-1.0, 0.0, 1.0, -2.0]),        # an exact zero (rising), then falling
    ([0.0, 1.0, 2.0], [0.0, -1.0, 0.0]),                    # exact zero first, last unvisited
    ([0.0, 0.5, 1.0, 1.5, 2.0], [2.0, 1.0, 0.5, 0.25, 0.125]),  # no crossing
    ([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, -1.0, float("nan"), -1.0, 1.0]),  # NaN skipped
    (list(np.arange(-2.5, 2.5 + 1e-9, 0.5 / 60.0)[100:106]),
     [0.0123, 0.0051, -0.0020, -0.0090, -0.0158, -0.0225]),  # a 30-s grid contact
    ([0.0, 1.0, 2.0, 3.0], [-0.5, 0.0, 0.0, 0.5]),           # consecutive zeros
]

# Kernel-backed circumstances sites per modern eclipse, as (lat, lon): the
# greatest-eclipse point (tests/test_besselian_integration.py reference values),
# a partial site (NYC), a sunset-in-progress site (W Ireland, 2024), the night
# side (central India), outside the penumbra (S America), the poles and both
# sides of the antimeridian. ``local`` records use the /circumstances defaults
# (half_window_hours 2.5) plus the greatest point at half_window_hours 1.0,
# where _bracketed_series must widen; ``grid`` records use /map's defaults
# (half_window_hours 3.0, step_minutes 2.0).
GREATEST_SITE = {
    "se2017aug21": (37.0, -87.7),
    "se2023oct14": (11.4, -83.1),
    "se2024apr08": (25.3, -104.1),
}
CIRC_SITES = [
    (40.71, -74.01), (53.3, -9.0), (20.0, 77.0), (-30.0, -60.0),
    (89.9, 0.0), (-89.9, 0.0), (0.0, 179.99), (0.0, -179.99),
]
LOCAL_HALF_WINDOW_H = 2.5   # /circumstances window_hours default
LOCAL_NARROW_HALF_WINDOW_H = 1.0
MAP_HALF_WINDOW_H = 3.0     # /map window_hours default
MAP_STEP_MIN = 2.0          # /map step_minutes default

# catalog_2019_2024.txt windows (tests/test_catalog.py's canon range and the
# one-year 2024 window of its greatest-eclipse rows), as find_eclipses parses them.
CATALOG_WINDOWS = [
    ("y2019_2024", "2019-01-01", "2024-12-31"),
    ("y2024", "2024-01-01", "2024-12-31"),
]
INF = float("inf")
ISMIN_CASES = [  # literal rho arrays for _local_minima (Earth radii; inf = far side)
    [3.0, 2.0, 1.0, 2.0, 3.0],                       # one V
    [3.0, 1.0, 1.0, 3.0],                            # a two-sample plateau: first wins
    [3.0, 1.0, 1.0, 1.0, 3.0],                       # three-sample plateau
    [INF, 2.0, INF, 1.0, INF],                       # far-side samples never qualify
    [3.0, 2.7, 3.0, 2.6, 3.0],                       # at / above CANDIDATE_RHO
    [3.0, 2.5999999999999996, 3.0],                  # one ulp below it
    [1.0, 2.0, 3.0],                                 # a minimum at the first sample: never
    [3.0, 2.0, 1.0],                                 # ... or the last
    [2.0, float("nan"), 1.0, 2.0],                   # NaN compares false
    [2.0, 1.0, 2.0, 0.5, 0.4, 0.4, 0.9, 2.0, INF, 2.5, 2.4, 2.5],
    [1.0],
    [1.0, 0.5],
    [],
]
CLASSIFY_CASES = [  # (rho_g, l1, l2, tf1, tf2, central, zeta), Earth radii
    (0.34, 0.5359, -0.0102, 0.004656, 0.004632, True, 0.94),     # central total (2024-like)
    (0.375, 0.5645, 0.0184, 0.004681, 0.004657, True, 0.93),     # central annular (2023-like)
    (0.1, 0.5359, 0.5 * 0.0046, 0.004656, 0.0046, True, 0.5),    # L2' == 0 exactly -> annular
    (0.9, 0.5359, 0.0040, 0.004656, 0.0046, True, 0.9),          # L2' slightly < 0 (hybrid-like)
    (0.99, 0.5359, -0.0102, 0.004656, 0.004632, True, 0.0),      # zeta = 0: L2' = l2
    (1.2166, 0.5359, -0.0102, 0.004656, 0.004632, False, 0.0),   # partial
    (1.0 + 0.5359, 0.5359, -0.0102, 0.004656, 0.004632, False, 0.0),  # rho_g == 1 + l1
    (1.0 + 0.0102, 0.5359, -0.0102, 0.004656, 0.004632, False, 0.0),  # rho_g == 1 + |l2|: grazing
    (1.0 + 0.0184, 0.5645, 0.0184, 0.004681, 0.004657, False, 0.0),   # ... annular
    (1.005, 0.5359, -0.0102, 0.004656, 0.004632, False, 0.0),    # non-central total
    (1.005, 0.5645, 0.0184, 0.004681, 0.004657, False, 0.0),     # non-central annular
    (1.0102000000000002, 0.5359, -0.0102, 0.004656, 0.004632, False, 0.0),  # one ulp above
    (0.0, 0.5359, -0.0102, 0.004656, 0.004632, True, 1.0),       # on axis
]
PYROUND_CASES = [  # (x, n): the hard cases of float.__round__ plus catalog-like values
    (2.675, 2), (0.125, 2), (0.375, 2), (1.005, 2), (0.005, 2), (-0.005, 2), (-0.001, 2),
    (0.001, 2), (-0.0, 2), (0.0, 4), (1e16, 2), (40.71, 2), (-104.1, 2), (25.3, 2),
    (0.5, 0), (1.5, 0), (2.5, 0), (-2.5, 0), (float("nan"), 2), (INF, 1), (-INF, 4),
    (0.43671234, 4), (1.0306, 4), (0.4367499999, 4), (0.43675, 4), (197.54999, 1),
    (114.65, 1), (114.75, 1), (1e-5, 4), (5e-5, 4), (-5e-5, 4), (2.5e-5, 4),
    (123456789.987654321, 4), (0.30000000000000004, 4), (0.9999999999999999, 4),
    (1.7976931348623157e308, 2), (5e-324, 2), (1234.5, 3), (37.00000000000001, 2),
    (-87.69999999999999, 2), (11.4, 2), (-83.1, 2), (0.34314159, 4), (1.0566, 4),
]


def header(fh, title: str) -> None:
    fh.write(f"# {title}\n# generated by tools/dump_oracle.py; do not edit, regenerate\n")
    fh.write(f"spk {SPK}\n")
    pck = _pck_identity()
    if pck is not None:
        fh.write(f"pck {pck}\n")


def row(*vals) -> str:
    """``repr`` of each value (bools as 0/1) joined by single spaces."""
    return " ".join("1" if v is True else "0" if v is False else repr(float(v)) for v in vals)


def contact_records(label: str) -> str:
    """``contacts`` record for ``label``: global_contacts at its greatest-eclipse epoch."""
    model = BesselianModel(t0_utc=CONTACT_EPOCHS[label], earth_frame=CONTACT_FRAME,
                           half_window_hours=2.5)
    contacts = global_contacts(model, CONTACT_HALF_WINDOW_H)  # insertion order = P1 P4 U1 U4 U2 U3
    names = " ".join(f"{name} {t!r}" for name, t in contacts.items())
    return f"contacts {label} {CONTACT_FRAME} {model.et0!r} {CONTACT_HALF_WINDOW_H!r} {names}\n"


def _quiet_model(**kw) -> BesselianModel:
    """BesselianModel without the RankWarning a 3-sample cubic fit emits (hw = 1 h)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        return BesselianModel(**kw)


def circumstances_records(label: str) -> str:
    """``local`` and ``grid`` records for ``label`` at its greatest-eclipse epoch."""
    lat0, lon0 = GREATEST_SITE[label]
    sites = [(lat0, lon0), *CIRC_SITES]
    out = []
    local = _quiet_model(t0_utc=CONTACT_EPOCHS[label], earth_frame=CONTACT_FRAME,
                         half_window_hours=LOCAL_HALF_WINDOW_H)
    narrow = _quiet_model(t0_utc=CONTACT_EPOCHS[label], earth_frame=CONTACT_FRAME,
                          half_window_hours=LOCAL_NARROW_HALF_WINDOW_H)
    for model, pts in ((local, sites), (narrow, [(lat0, lon0)])):
        for lat, lon in pts:
            raw = _local_raw(model, lat, lon)
            vals = [raw.geometric, raw.central, raw.c1, raw.c4, raw.c2, raw.c3, raw.t_max,
                    raw.magnitude, raw.obscuration, raw.L2_x, *raw.alt_deg, *raw.az_deg,
                    *raw.below, raw.eclipse]
            out.append(f"local {label} {CONTACT_FRAME} {model.et0!r} {model.half_window_hours!r} "
                       f"{lat!r} {lon!r} {row(*vals)}\n")
    for h in LOCAL_HEIGHTS_M:
        raw = _local_raw(local, lat0, lon0, height_m=h)
        vals = [raw.geometric, raw.central, raw.c1, raw.c4, raw.c2, raw.c3, raw.t_max,
                raw.magnitude, raw.obscuration, raw.L2_x, *raw.alt_deg, *raw.az_deg,
                *raw.below, raw.eclipse]
        out.append(f"localh {label} {CONTACT_FRAME} {local.et0!r} {local.half_window_hours!r} "
                   f"{lat0!r} {lon0!r} {h!r} {row(*vals)}\n")
    grid = _quiet_model(t0_utc=CONTACT_EPOCHS[label], earth_frame=CONTACT_FRAME,
                        half_window_hours=MAP_HALF_WINDOW_H)
    for lat, lon in sites:
        g = circumstances_grid(grid, [lat], [lon], step_minutes=MAP_STEP_MIN)
        vals = [g["magnitude"][0], g["obscuration"][0], g["t_max_hours"][0], g["sun_alt"][0],
                bool(g["visible"][0]), bool(g["central"][0])]
        out.append(f"grid {label} {CONTACT_FRAME} {grid.et0!r} {grid.half_window_hours!r} "
                   f"{MAP_STEP_MIN!r} {lat!r} {lon!r} {row(*vals)}\n")
    return "".join(out)


class _SubSolarStub:
    """The slice of BesselianModel the altitude helpers read, with a fixed sub-solar
    point patched into ``app.circumstances.sub_solar_points`` so ``_sun_altaz`` and
    ``circumstances_grid`` run their real expressions kernel-free. ``evaluate_direct``
    returns constant elements (irrelevant to ``sun_alt``); half_window_hours 0 makes
    the grid's ``np.arange`` a single instant."""

    et0 = 0.0
    earth_frame = CONTACT_FRAME
    half_window_hours = 0.0

    def __init__(self, ss_lon: float, ss_lat: float) -> None:
        self.ss_lon, self.ss_lat = ss_lon, ss_lat

    def evaluate_direct(self, t):
        one = np.ones_like(np.atleast_1d(np.asarray(t, dtype=float)))
        return {"x": 0.0 * one, "y": 0.0 * one, "d": 0.0 * one, "mu": 0.0 * one,
                "l1": 0.5 * one, "l2": -0.01 * one, "tan_f1": 0.0046 * one,
                "tan_f2": 0.0046 * one}

    def sub_solar_points(self, et, earth_frame):
        et = np.atleast_1d(np.asarray(et, dtype=float))
        return np.full(et.shape, self.ss_lon), np.full(et.shape, self.ss_lat)


def write_circumstances_cases(path: Path) -> int:
    """circumstances_cases.txt: the app.circumstances helpers at literal inputs
    (roadmap §4 row circumstances, kernel-free part). Returns the record count."""
    n_rec = 0
    with open(path, "w") as fh:
        header(fh, "app.circumstances oracle at literal inputs (phase 3 circumstances parity)")

        fh.write("# ovl r R d area  (_overlap_area)\n")
        for r, R, d in OVL_CASES:
            fh.write(f"ovl {row(r, R, d, _overlap_area(r, R, d))}\n")
            n_rec += 1

        fh.write("# obsc L1p L2p m obs  (_obscuration)\n")
        for L1p, L2p, m in OBSC_CASES:
            fh.write(f"obsc {row(L1p, L2p, m, _obscuration(L1p, L2p, m))}\n")
            n_rec += 1

        fh.write("# unw n v1..vn u1..un  (np.degrees(np.unwrap(np.radians(v))), degrees)\n")
        for v in UNW_CASES:
            v = np.asarray(v, dtype=float)
            u = np.degrees(np.unwrap(np.radians(v)))
            fh.write(f"unw {len(v)} {row(*v)} {row(*u)}\n")
            n_rec += 1

        fh.write("# altaz lat lon ss_lon ss_lat alt az alt_grid  "
                 "(_sun_altaz; circumstances_grid's sun_alt)\n")
        for lat, lon, ss_lon, ss_lat in ALTAZ_CASES:
            stub = _SubSolarStub(ss_lon, ss_lat)
            with mock.patch.object(circ, "sub_solar_points", stub.sub_solar_points):
                alt, az = _sun_altaz(stub, lat, lon, 0.0)
                alt_grid = circumstances_grid(stub, [lat], [lon], step_minutes=MAP_STEP_MIN)
            vals = (lat, lon, ss_lon, ss_lat, alt[0], az[0], alt_grid["sun_alt"][0])
            fh.write(f"altaz {row(*vals)}\n")
            n_rec += 1

        fh.write("# roots n t1..tn f1..fn k tc1 rising1 i1 ...  (_roots; rising 0/1)\n")
        for t, f in ROOTS_CASES:
            t, f = np.asarray(t, dtype=float), np.asarray(f, dtype=float)
            found = _roots(t, f)
            flat = [v for tc, rising, i in found for v in (tc, rising, float(i))]
            fh.write(f"roots {len(t)} {row(*t)} {row(*f)} {len(found)}"
                     f"{' ' + row(*flat) if flat else ''}\n")
            n_rec += 1
    return n_rec


def write_catalog_cases(path: Path) -> int:
    """catalog_cases.txt: the kernel-free app.catalog helpers and Python's round at
    literal inputs (roadmap §4 row catalog). Returns the record count."""
    n_rec = 0
    rng = np.random.default_rng(20240408)
    with open(path, "w") as fh:
        header(fh, "app.catalog oracle at literal inputs (phase 4 catalog parity)")

        fh.write("# ismin n rho1..rhon k idx1..idxk  (_local_minima; indices into rho)\n")
        # ... plus a real 20-day slice of the 2024 coarse grid (the March full
        # moon's far-side inf samples and the April new-moon minimum).
        real = cat._rho_at(cat._coarse_grid(ep.utc_to_et("2024-03-20T00:00:00"),
                                            ep.utc_to_et("2024-04-09T00:00:00")))
        for rho in [*ISMIN_CASES, list(real)]:
            rho = np.asarray(rho, dtype=float)
            idx = cat._local_minima(rho)
            fh.write(f"ismin {len(rho)}{' ' + row(*rho) if len(rho) else ''} {len(idx)}"
                     f"{' ' + row(*idx) if len(idx) else ''}\n")
            n_rec += 1

        fh.write("# classify rho_g l1 l2 tf1 tf2 central zeta kind magnitude L2p  (_classify)\n")
        for rho_g, l1, l2, tf1, tf2, central, zeta in CLASSIFY_CASES:
            kind, mag, L2p = cat._classify(rho_g, l1, l2, tf1, tf2, central, zeta)
            fh.write(f"classify {row(rho_g, l1, l2, tf1, tf2, central, zeta)} {kind} "
                     f"{row(mag, L2p)}\n")
            n_rec += 1

        fh.write("# pyround x n r  (Python round(x, n))\n")
        cases = list(PYROUND_CASES)
        for n in (1, 2, 4):
            cases += [(float(v), n) for v in rng.uniform(-200.0, 200.0, 8)]
            cases += [(float(v), n) for v in rng.uniform(-1.0, 1.0, 8) * 10.0 ** (-n)]
        for x, n in cases:
            fh.write(f"pyround {x!r} {n} {round(x, n)!r}\n")
            n_rec += 1
    return n_rec


def catalog_records(label: str, start: str, end: str) -> tuple[str, list[float]]:
    """``scan`` + ``event`` records of one window and the events' et_g (for the EOP
    subset). The events are ``_catalog_raw`` at detail False and True."""
    from app.besselian import normalize_utc

    et_a, et_b = ep.utc_to_et(normalize_utc(start)), ep.utc_to_et(normalize_utc(end))
    grid = cat._coarse_grid(et_a, et_b)
    cand = cat._scan_candidates(et_a, et_b)
    out = [f"scan {label} {row(et_a, et_b)} {len(grid)} {row(grid[0], grid[1], grid[-1])} "
           f"{len(cand)} {row(*cand)}\n"]
    et_gs: list[float] = []
    for detail in (False, True):
        for ev in cat._catalog_raw(et_a, et_b, CONTACT_FRAME, detail):
            head = (f"event {label} {CONTACT_FRAME} {int(detail)} {ev.et_g!r} {ev.kind} "
                    f"{row(ev.central, ev.gamma, ev.magnitude, ev.lat, ev.lon)} "
                    f"{ep.et_to_utc(ev.et_g)}")
            if not detail:
                out.append(head + "\n")
                et_gs.append(ev.et_g)
                continue
            contacts = " ".join(f"{name} {t!r}" for name, t in ev.contacts)
            if ev.local is None:
                local = "-"
            else:
                raw = ev.local
                local = row(raw.geometric, raw.central, raw.c1, raw.c4, raw.c2, raw.c3,
                            raw.t_max, raw.magnitude, raw.obscuration, raw.L2_x,
                            *raw.alt_deg, *raw.az_deg, *raw.below, raw.eclipse)
            width = "nan" if ev.width_km is None else repr(float(ev.width_km))
            out.append(f"{head} {ev.et0!r} {len(ev.contacts)} {contacts} {local} {width}\n")
    return "".join(out), et_gs


def write_geometry_cases(path: Path) -> int:
    """geometry_cases.txt: the app.geography oracle at literal inputs (roadmap §4 rows
    ellipsoid / shadow_edge_limits_v / great-circle helpers). Returns the record count."""
    n_rec = 0
    tracks = {}  # label -> (elems, track) from central_track over TRACK_T_HOURS
    track_rates = {}  # label -> element_rates at the track points (the W2 envelope)
    for label, utc0, _utc1 in ECLIPSES:
        if label not in CONTACT_EPOCHS:
            continue  # modern eclipses only (the 1919 SPK coverage is not needed here)
        t0 = (datetime.fromisoformat(utc0) + timedelta(hours=3)).isoformat()
        model = BesselianModel(t0_utc=t0, earth_frame=CONTACT_FRAME)
        tracks[label] = central_track(model, TRACK_T_HOURS)
        track_rates[label] = element_rates(
            model, np.array([tp.t_hours for tp in tracks[label][1]]))

    with open(path, "w") as fh:
        header(fh, "app.geography oracle at literal inputs (phase 2 ellipsoid + geometry parity)")

        fh.write("# red <d_deg> rho1 rho2 sin_d1 cos_d1 sin_d1_d2 cos_d1_d2  (_reduction_aux)\n")
        d_set = list(np.linspace(-30.0, 30.0, 25)) + [
            float(elems["d"][N // 2]) for elems, _ in tracks.values()
        ]
        for d in d_set:
            fh.write(f"red {row(d, *_reduction_aux(np.radians(d)))}\n")
            n_rec += 1

        fh.write("# f2g x y d mu lon lat  (fund_to_geo_v; nan where the axis misses the Earth)\n")
        f2g_inputs = []
        for i in range(0, 12, 3):  # tests/test_geography.py POLY instants
            t = i * 2 / 60.0
            f2g_inputs.append((_ev(POLY["x"], t), _ev(POLY["y"], t),
                               _ev(POLY["d"], t), _ev(POLY["mu"], t)))
        for elems, _ in tracks.values():
            f2g_inputs.extend(zip(elems["x"], elems["y"], elems["d"], elems["mu"], strict=True))
        f2g_inputs.extend(OFF_EARTH)
        for x, y, d, mu in f2g_inputs:
            lon, lat = fund_to_geo_v(x, y, d, mu)
            fh.write(f"f2g {row(x, y, d, mu, lon, lat)}\n")
            n_rec += 1

        fh.write("# g2f lat lon d mu xi eta zeta  (geo_to_fund)\n")
        lat_g, lon_g = np.meshgrid(G2F_LATS, G2F_LONS, indexing="ij")
        for elems, _ in tracks.values():
            d, mu = float(elems["d"][N // 2]), float(elems["mu"][N // 2])
            xi, eta, zeta = geo_to_fund(lat_g, lon_g, d, mu)
            for k in range(lat_g.size):
                vals = (lat_g.flat[k], lon_g.flat[k], d, mu, xi.flat[k], eta.flat[k], zeta.flat[k])
                fh.write(f"g2f {row(*vals)}\n")
                n_rec += 1

        fh.write("# g2fh lat lon d mu h xi eta zeta  (geo_to_fund at height h [m])\n")
        for elems, _ in tracks.values():
            d, mu = float(elems["d"][N // 2]), float(elems["mu"][N // 2])
            for h in HEIGHTS_M:
                xi, eta, zeta = geo_to_fund(lat_g, lon_g, d, mu, h)
                for k in range(lat_g.size):
                    vals = (lat_g.flat[k], lon_g.flat[k], d, mu, h,
                            xi.flat[k], eta.flat[k], zeta.flat[k])
                    fh.write(f"g2fh {row(*vals)}\n")
                    n_rec += 1

        fh.write("# gc lat lon brg dist lat2 lon2 hav brg12  "
                 "(_destination, _haversine_km, bearing)\n")
        for lat, lon, brg, dist in GC_CASES:
            lat2, lon2 = _destination(lat, lon, brg, dist)
            hav, brg12 = _haversine_km(lat, lon, lat2, lon2), bearing(lat, lon, lat2, lon2)
            fh.write(f"gc {row(lat, lon, brg, dist, lat2, lon2, hav, brg12)}\n")
            n_rec += 1

        fh.write("# lim x y d mu l tan_f bearing max_km sunlit n_lat n_lon s_lat s_lon width  "
                 "(shadow_edge_limits_v)\n")
        for elems, track in tracks.values():
            idx = np.array([tp.i for tp in track], dtype=int)
            brg = np.array([tp.bearing for tp in track])
            x, y, d, mu = (elems[k][idx] for k in ("x", "y", "d", "mu"))
            for l_key, tf_key, max_km, sunlit in (("l2", "tan_f2", UMBRA_MAX_KM, True),
                                                  ("l1", "tan_f1", PENUMBRA_MAX_KM, True),
                                                  ("l1", "tan_f1", PENUMBRA_MAX_KM, False)):
                l, tf = elems[l_key][idx], elems[tf_key][idx]
                out = shadow_edge_limits_v(x, y, d, mu, l, tf, brg, max_km=max_km,
                                           sunlit_only=sunlit)
                for k in range(len(idx)):
                    inputs = (x[k], y[k], d[k], mu[k], l[k], tf[k], brg[k], max_km, sunlit)
                    fh.write(f"lim {row(*inputs, *(o[k] for o in out))}\n")
                    n_rec += 1
        fh.write("# limr x y d mu l tan_f bearing dx dy dd dmu dl n_lat n_lon s_lat s_lon width  "
                 "(shadow_edge_limits_v with rates: the umbral path envelope, 600 km, sunlit)\n")
        for label, (elems, track) in tracks.items():
            idx = np.array([tp.i for tp in track], dtype=int)
            brg = np.array([tp.bearing for tp in track])
            x, y, d, mu = (elems[k][idx] for k in ("x", "y", "d", "mu"))
            l, tf = elems["l2"][idx], elems["tan_f2"][idx]
            r = track_rates[label]
            rates = (r["x"], r["y"], r["d"], r["mu"], r["l2"])
            out = shadow_edge_limits_v(x, y, d, mu, l, tf, brg, rates=rates)
            for k in range(len(idx)):
                inputs = (x[k], y[k], d[k], mu[k], l[k], tf[k], brg[k], *(v[k] for v in rates))
                fh.write(f"limr {row(*inputs, *(o[k] for o in out))}\n")
                n_rec += 1

        fh.write("# geod lat1 lon1 lat2 lon2 km  (geodesic_km)\n")
        for lat1, lon1, lat2, lon2 in GEOD_CASES:
            fh.write(f"geod {row(lat1, lon1, lat2, lon2, geodesic_km(lat1, lon1, lat2, lon2))}\n")
            n_rec += 1

        x, y, d, mu = OFF_EARTH[0]
        l, tf, brg = -0.0106, 0.00464, 45.0
        out = shadow_edge_limits_v(x, y, d, mu, l, tf, brg)
        fh.write(f"lim {row(x, y, d, mu, l, tf, brg, UMBRA_MAX_KM, True, *(o[0] for o in out))}\n")
        n_rec += 1

        fh.write("# rad x y d l1 l2 tf1 tf2 pen umb is_total  (shadow_radii)\n")
        for elems, _ in tracks.values():
            for i in range(N):
                args = [float(elems[k][i]) for k in ("x", "y", "d", "l1", "l2", "tan_f1", "tan_f2")]
                fh.write(f"rad {row(*args, *shadow_radii(*args))}\n")
                n_rec += 1
    return n_rec


def write_limb_cases(path: Path) -> int:
    """limb_cases.txt: app.ephemeris.limb_axes / app.limb oracle (docs/LIMB_PROFILE.md
    PR 1). A 1-ppd random synthetic limb band, written as ``limb_band_syn.bin``
    next to it (the C++ reads the file itself), with its silhouettes along
    tilted axes, delta_rho_at cases, and -- when the lunar kernels are loaded --
    the axes at the reference instants."""
    import tempfile

    import spiceypy

    from app import limb
    from kernels import limb_band

    n_rec = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lines, samples = 180, 360
        dn = np.random.default_rng(3).integers(-8000, 8000, (lines, samples)).astype("<i2")
        dn.tofile(tmp / "s.img")
        (tmp / "s.lbl").write_text(
            f'PRODUCT_ID = "SYN1"\nLINES = {lines}\nLINE_SAMPLES = {samples}\n'
            "SAMPLE_BITS = 16\nSAMPLE_TYPE = LSB_INTEGER\nSCALING_FACTOR = 0.5\n"
            "OFFSET = 1737400.\nMAP_RESOLUTION = 1\nLINE_PROJECTION_OFFSET = 89.5\n"
            "SAMPLE_PROJECTION_OFFSET = 179.5\nCENTER_LONGITUDE = 180\n"
            'COORDINATE_SYSTEM_NAME = "SYN"\n')
        band_file = path.parent / "limb_band_syn.bin"
        limb_band.cut(tmp / "s.img", tmp / "s.lbl", band_file)
        band = limb.band_from_file(limb_band.read(band_file))

    with open(path, "w") as fh:
        header(fh, "app.limb / app.ephemeris.limb_axes oracle (lunar limb profile, PRs 1-2)")
        fh.write("# the synthetic band is limb_band_syn.bin (kernels/limb_band.py format);\n"
                 "# the C++ loads it with limb::load_band_file\n")

        fh.write("# lsil <n_bins> <distance_km> <9 axes> <n_bins delta_rho>\n")
        rng = np.random.default_rng(5)
        for _ in range(4):
            z = np.array([-1.0, *rng.uniform(-0.14, 0.14, 2)])
            z /= np.linalg.norm(z)
            y = np.cross(z, rng.normal(size=3))
            y /= np.linalg.norm(y)
            axes = np.stack([np.cross(y, z), y, z])
            for n_bins in (90, 360):
                for dist in (np.inf, 370000.0):
                    prof = limb.silhouette(band, axes, n_bins, dist)
                    fh.write(f"lsil {n_bins} {row(dist, *axes.ravel(), *prof)}\n")
                    n_rec += 1

        fh.write("# ldra <n> <profile..> <m> <psi..> <out..>\n")
        prof = np.random.default_rng(9).normal(size=16)
        psi = np.linspace(-7.0, 7.0, 29)
        out = limb.delta_rho_at(prof, psi)
        fh.write(f"ldra {len(prof)} {row(*prof)} {len(psi)} {row(*psi)} {row(*out)}\n")
        n_rec += 1

        fh.write("# lg <total|annular> <n_bins> <px> <py> <r_s> <r_m> <profile..> <G>\n")
        rng = np.random.default_rng(13)
        for kind in ("total", "annular"):
            for _ in range(4):
                px, py = rng.normal(scale=0.005, size=2)
                r_s = rng.uniform(0.26, 0.28)
                r_m = r_s + rng.uniform(-0.01, 0.01)
                prof = rng.normal(scale=2.0, size=90)
                g = getattr(limb, f"g_{kind}")([px], [py], [r_s], [r_m], prof[None, :])[0]
                fh.write(f"lg {kind} 90 {row(px, py, r_s, r_m, *prof, g)}\n")
                n_rec += 1

        try:
            spiceypy.pxform("J2000", "MOON_ME", 0.0)
            have_moon = True
        except spiceypy.utils.exceptions.SpiceyError:
            have_moon = False
        if not have_moon:
            fh.write("# lax: skipped (MOON_ME not loaded; kernels.bootstrap --limb)\n")
            return n_rec
        fh.write("# lax <et> <earth_frame> <moon_frame> <9 axes, rows x^ y^ z^> <distance_km>\n")
        for label, utc0, utc1 in ECLIPSES:
            if label not in CONTACT_EPOCHS:
                continue
            et = np.linspace(ep.utc_to_et(utc0), ep.utc_to_et(utc1), 3)
            for frame in ("ITRS", "TOD", "IAU_EARTH"):
                for moon_frame in ("MOON_ME", "IAU_MOON"):
                    ax, dist = ep.limb_axes(et, frame, moon_frame)
                    for i in range(len(et)):
                        vals = row(*ax[i].ravel(), dist[i])
                        fh.write(f"lax {float(et[i])!r} {frame} {moon_frame} {vals}\n")
                        n_rec += 1
    return n_rec


def main() -> None:
    assert not native.is_native(), "dump the *Python* oracle: ECLIPSE_BACKEND=python"
    if not (ROOT / "kernels" / SPK).exists():
        sys.exit(f"fixtures pin {SPK}: run `python -m kernels.bootstrap --source naif`")
    ep.load_kernels()
    OUT.mkdir(parents=True, exist_ok=True)

    mjd, xp, yp, dut1 = _table()
    keep = np.zeros(len(mjd), dtype=bool)

    for label, utc0, utc1 in ECLIPSES:
        et = np.linspace(ep.utc_to_et(utc0), ep.utc_to_et(utc1), N)
        tt2, ut1, pxp, pyp = ep.earth_rotation_times(et)
        mjd_utc = ep._J2000_JD - ep._MJD_OFFSET + tt2  # ~UTC to a minute; ±5 d window
        for m in mjd_utc:
            keep |= np.abs(mjd - m) <= 5.0
        with open(OUT / f"{label}.txt", "w") as fh:
            header(fh, f"{label}: oracle Besselian elements, {utc0} .. {utc1}, {N} samples")
            for frame in FRAMES:
                try:
                    e = ep.besselian_instants(et, frame)
                    lon, lat = ep.sub_solar_points(et, frame)
                except Exception as exc:  # ITRF93 PCK has no 1919 coverage
                    fh.write(f"# {frame}: skipped ({type(exc).__name__})\n")
                    continue
                for i in range(N):
                    vals = [et[i], e["x"][i], e["y"][i], e["z"][i], e["d"][i], e["mu"][i],
                            e["l1"][i], e["l2"][i], e["tan_f1"][i], e["tan_f2"][i],
                            tt2[i], ut1[i], pxp[i], pyp[i], lon[i], lat[i]]
                    fh.write(f"case {label}[{i}] {frame} {row(*vals)}\n")
            rho, z = ep.axis_separation(et)
            for i in range(N):
                fh.write(f"axis {row(et[i], rho[i], z[i])}\n")
            if label in CONTACT_EPOCHS:
                fh.write(contact_records(label))
                fh.write(circumstances_records(label))

    with open(OUT / "utc_cases.txt", "w") as fh:
        header(fh, "utc_to_et / et_to_utc oracle cases (IERS-era rule)")
        for s in UTC_CASES:
            et = ep.utc_to_et(s)
            keep |= np.abs(mjd - (ep._J2000_JD - ep._MJD_OFFSET + et / 86400.0)) <= 5.0
            fh.write(f"utc {s} {et!r} {ep.et_to_utc(et)}\n")

    with open(OUT / "catalog_2019_2024.txt", "w") as fh:
        header(fh, "app.catalog oracle: scan candidates and _catalog_raw events, 2019-2024 "
                   "and 2024, ITRS, detail 0 and 1 (phase 4 catalog parity)")
        for label, start, end in CATALOG_WINDOWS:
            text, et_gs = catalog_records(label, start, end)
            fh.write(text)
            for et_g in et_gs:  # EOP rows for the elements / detail at each event
                keep |= np.abs(mjd - (ep._J2000_JD - ep._MJD_OFFSET + et_g / 86400.0)) <= 5.0

    with open(OUT / "eop_subset.txt", "w") as fh:
        header(fh, "IERS Bulletin A rows (astropy-iers-data) within 5 d of the cases")
        for i in np.flatnonzero(keep):
            cols = " ".join(repr(float(v)) for v in (mjd[i], xp[i], yp[i], dut1[i]))
            fh.write(f"eop {cols}\n")
    n_geo = write_geometry_cases(OUT / "geometry_cases.txt")
    n_circ = write_circumstances_cases(OUT / "circumstances_cases.txt")
    n_cat = write_catalog_cases(OUT / "catalog_cases.txt")
    n_limb = write_limb_cases(OUT / "limb_cases.txt")

    n_rows = int(keep.sum())
    print(f"wrote {len(ECLIPSES) + 7} fixtures to {OUT.relative_to(ROOT)}, {n_rows} EOP rows, "
          f"{n_geo} geometry records, {n_circ} circumstances records, {n_cat} catalog records, "
          f"{n_limb} limb records")


if __name__ == "__main__":
    main()

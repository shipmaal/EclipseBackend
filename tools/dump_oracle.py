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

and, in ``geometry_cases.txt`` (phase 2: app.geography; inputs are literal so
these need no kernel to replay), all angles in degrees, lengths in Earth
radii / km as the Python signatures say::

    red <d_deg> rho1 rho2 sin_d1 cos_d1 sin_d1_d2 cos_d1_d2   # _reduction_aux
    f2g x y d mu lon lat                                       # fund_to_geo_v (nan = off Earth)
    g2f lat lon d mu xi eta zeta                               # geo_to_fund
    gc  lat lon brg dist lat2 lon2 hav brg12                   # _destination/_haversine_km/bearing
    lim x y d mu l tan_f bearing max_km sunlit n_lat n_lon s_lat s_lon width
                                                               # shadow_edge_limits_v (sunlit 0/1)
    rad x y d l1 l2 tf1 tf2 pen umb is_total                   # shadow_radii (is_total 0/1)

Numbers are written with ``repr`` (shortest round-trip), so the fixture holds
the oracle's doubles exactly (``nan`` for NaN; ``std::stod`` parses it).  The
fixtures pin the **NAIF DE440s** kernel set (header line); the C++ test skips
when that SPK is not the one on disk.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ephemeris as ep  # noqa: E402
from app import native  # noqa: E402
from app.besselian import BesselianModel  # noqa: E402
from app.eop import _table  # noqa: E402
from app.geography import (  # noqa: E402
    _destination,
    _haversine_km,
    _reduction_aux,
    bearing,
    central_track,
    fund_to_geo_v,
    geo_to_fund,
    global_contacts,
    shadow_edge_limits_v,
    shadow_radii,
)
from tests.test_geography import POLY, _ev  # noqa: E402  (the pure-function test's instants)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "cpp" / "fixtures"
SPK = "de440s.bsp"

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


def header(fh, title: str) -> None:
    fh.write(f"# {title}\n# generated by tools/dump_oracle.py; do not edit, regenerate\n")
    fh.write(f"spk {SPK}\n")


def row(*vals) -> str:
    """``repr`` of each value (bools as 0/1) joined by single spaces."""
    return " ".join("1" if v is True else "0" if v is False else repr(float(v)) for v in vals)


def contact_records(label: str) -> str:
    """``contacts`` record for ``label``: global_contacts at its greatest-eclipse epoch."""
    model = BesselianModel(t0_utc=CONTACT_EPOCHS[label], earth_frame=CONTACT_FRAME,
                           half_window_hours=2.5)
    contacts = global_contacts(model, CONTACT_HALF_WINDOW_H)  # insertion order = P1 U1 U2 U3 U4 P4
    names = " ".join(f"{name} {t!r}" for name, t in contacts.items())
    return f"contacts {label} {CONTACT_FRAME} {model.et0!r} {CONTACT_HALF_WINDOW_H!r} {names}\n"


def write_geometry_cases(path: Path) -> int:
    """geometry_cases.txt: the app.geography oracle at literal inputs (roadmap §4 rows
    ellipsoid / shadow_edge_limits_v / great-circle helpers). Returns the record count."""
    n_rec = 0
    tracks = {}  # label -> (elems, track) from central_track over TRACK_T_HOURS
    for label, utc0, _utc1 in ECLIPSES:
        if label not in CONTACT_EPOCHS:
            continue  # modern eclipses only (the 1919 SPK coverage is not needed here)
        t0 = (datetime.fromisoformat(utc0) + timedelta(hours=3)).isoformat()
        model = BesselianModel(t0_utc=t0, earth_frame=CONTACT_FRAME)
        tracks[label] = central_track(model, TRACK_T_HOURS)

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


def main() -> None:
    assert not native.is_native(), "dump the *Python* oracle: unset ECLIPSE_BACKEND"
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
            if label in CONTACT_EPOCHS:
                fh.write(contact_records(label))

    with open(OUT / "utc_cases.txt", "w") as fh:
        header(fh, "utc_to_et / et_to_utc oracle cases (IERS-era rule)")
        for s in UTC_CASES:
            et = ep.utc_to_et(s)
            keep |= np.abs(mjd - (ep._J2000_JD - ep._MJD_OFFSET + et / 86400.0)) <= 5.0
            fh.write(f"utc {s} {et!r} {ep.et_to_utc(et)}\n")

    with open(OUT / "eop_subset.txt", "w") as fh:
        header(fh, "IERS Bulletin A rows (astropy-iers-data) within 5 d of the cases")
        for i in np.flatnonzero(keep):
            cols = " ".join(repr(float(v)) for v in (mjd[i], xp[i], yp[i], dut1[i]))
            fh.write(f"eop {cols}\n")
    n_geo = write_geometry_cases(OUT / "geometry_cases.txt")

    n_rows = int(keep.sum())
    print(f"wrote {len(ECLIPSES) + 3} fixtures to {OUT.relative_to(ROOT)}, {n_rows} EOP rows, "
          f"{n_geo} geometry records")


if __name__ == "__main__":
    main()

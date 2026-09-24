"""Lunar-limb resolution study (docs/LIMB_PROFILE.md sec. 9.11): reproduce the
LDEM_64 / bin-size measurements from repository data.

Two subcommands, both on the band in ``--band`` (a file cut by
``kernels/limb_band.py``, e.g. ``kernels/lola_ldem64_limb14.bin``; see the
docs for how to cut it from PDS ``LDEM_64``) and the native core::

    uv run python tools/limb_study.py profiles --band A.bin --band B.bin
        # profiles at 2017 / 2024 greatest eclipse: |delta_rho_B - delta_rho_A|
        # quantiles, mean delta_rho, points, load time, silhouette time, peak RSS

    uv run python tools/limb_study.py oracle --band A.bin --bins 7200,28800
        # at tests/test_limb_oracle.SITES: every root of the profile contact
        # function G (the EXACT profile at each instant, any bin count) and of
        # the 3D ray-traced oracle H within +-2.5 s of each production
        # contact, then the bead-agnostic mismatch: H at every G root and G at
        # every H root, in metres of limb height

Why roots and not "the" contact: near a limit the limbs close at 0.03-0.1
km/s and sub-second beads open and close (G < 0 briefly, then > 0 again), so
two methods a few tens of metres apart in limb height can disagree on which
root is "first entry" by seconds.  Evaluating each method at the other's roots
measures that limb-height difference directly.  The oracle needs a dense
height grid (``lines x samples`` float64: 2.1 GB for LDEM_64).
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import _eclipse as E
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app import core  # noqa: E402
from app.besselian import BesselianModel  # noqa: E402

EARTH_KM = 6378.137  # fundamental-plane unit [WGS84] a; G * EARTH_KM ~ limb km (R_m ~ k)
SAMPLE_S = 0.1       # root search: sampling step [s] ...
WINDOW_S = 2.5       # ... over +- this around each production contact
BISECT = 8           # bisections per root (0.4 ms)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _native_profile(et: float, n_bins: int, frame: str = "ITRS") -> np.ndarray:
    axes, dist = E.limb_axes(np.array([et]), frame, "MOON_ME")
    ax = np.ascontiguousarray(np.asarray(axes).reshape(3, 3))
    with core.PARALLEL_LOCK:
        return E.limb_silhouette(ax, int(n_bins), float(np.atleast_1d(dist)[0]))


def cmd_profiles(args) -> None:
    for t0 in ("2017-08-21T18:25:30", "2024-04-08T18:17:15"):
        et = BesselianModel(t0_utc=t0).et0
        profs = []
        for band in args.band:
            t = time.time()
            core.ensure_limb_band(band)
            t_load = time.time() - t
            t = time.time()
            p = _native_profile(et, E.LIMB_N_BINS)
            profs.append(p)
            print(f"{t0} {Path(band).name}: load {t_load:.1f} s, silhouette "
                  f"{time.time() - t:.2f} s, mean delta_rho {p.mean():.3f} km, "
                  f"peak RSS {_rss_gb():.2f} GB")
        for p in profs[1:]:
            d = np.abs(p - profs[0])
            q = np.percentile(d, [50, 90, 99])
            print(f"  |B - A| km: p50 {q[0]:.3f} p90 {q[1]:.3f} p99 {q[2]:.3f} max {d.max():.3f}")


def _g(model, lat, lon, h, t_s, n_bins) -> float:
    """The profile contact function [km of limb] at t_s [s from T0], with the
    exact profile of that instant (the core's ``profile_g`` without the
    lattice)."""
    k = E.LIMB_K_REF
    e = model.evaluate_direct(np.array([t_s / 3600.0]), k, k)
    xi, eta, zeta = E.geo_to_fund(np.array([lat]), np.array([lon]), e["d"], e["mu"],
                                  np.array([h]))
    px, py = xi - e["x"], eta - e["y"]
    L1 = e["l1"] - zeta * e["tan_f1"]
    L2 = e["l2"] - zeta * e["tan_f2"]
    prof = _native_profile(model.et0 + t_s, n_bins, model.earth_frame)[None, :]
    f = E.limb_g_total if L2[0] < 0 else E.limb_g_annular
    return float(f(px, py, (L1 + L2) / 2.0, (L1 - L2) / 2.0, prof)[0]) * EARTH_KM


def _fmt(roots) -> str:
    """Roots as ``+t.tttv`` (entering the central phase) / ``^`` (leaving)."""
    return " ".join(f"{t:+.3f}{'v' if e else '^'}" for t, e in roots)


def _roots(f, centre: float) -> list[tuple[float, bool]]:
    """(t, entering) for every sign change of f within +-WINDOW_S of centre."""
    ts = np.arange(centre - WINDOW_S, centre + WINDOW_S + 1e-9, SAMPLE_S)
    fs = np.array([f(t) for t in ts])
    out = []
    for i in np.flatnonzero(np.sign(fs[:-1]) != np.sign(fs[1:])):
        lo, hi, flo = ts[i], ts[i + 1], fs[i]
        for _ in range(BISECT):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if np.sign(fm) == np.sign(flo):
                lo, flo = mid, fm
            else:
                hi = mid
        out.append((0.5 * (lo + hi), bool(fs[i] > 0)))
    return out


def cmd_oracle(args) -> None:
    import spiceypy
    from test_limb_oracle import SITES, Dem, oracle_h

    from app.circumstances import local_raw
    from kernels import limb_band

    spiceypy.furnsh(str(core.metakernel()))  # the oracle's own SPICE pool
    band_path = core.ensure_limb_band(args.band)
    dem = Dem(limb_band.read(band_path))
    bins = [int(b) for b in args.bins.split(",")]
    print(f"band {Path(band_path).name}, dem ready, peak RSS {_rss_gb():.1f} GB")
    for label, t0, lat, lon, h in SITES:
        model = BesselianModel(t0_utc=t0)
        raw = local_raw(model, lat, lon, "profile", h)

        def oracle(t, model=model, lat=lat, lon=lon, h=h, annular=raw.L2_x > 0):
            return oracle_h(dem, model.et0 + t, lat, lon, annular, height_m=h)

        for name, c in (("C2", raw.c2 * 3600.0), ("C3", raw.c3 * 3600.0)):
            r_o = _roots(oracle, c)
            parts = [f"oracle roots [{_fmt(r_o)}]"]
            for nb in bins:
                def prof(t, nb=nb, model=model, lat=lat, lon=lon, h=h):
                    return _g(model, lat, lon, h, t, nb)
                r_p = _roots(prof, c)
                mism = [oracle(t) for t, _ in r_p] + [prof(t) for t, _ in r_o]
                worst = max((abs(v) for v in mism), default=float("nan")) * 1000.0
                parts.append(f"n{nb}: roots [{_fmt(r_p)}] mismatch {worst:.0f} m")
            print(f"{label[:34]:34s} {name} h={h:.1f}: " + " | ".join(parts), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profiles")
    p.add_argument("--band", action="append", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--band", required=True)
    o.add_argument("--bins", default="7200,28800")
    args = ap.parse_args()
    core.load_kernels()
    {"profiles": cmd_profiles, "oracle": cmd_oracle}[args.cmd](args)


if __name__ == "__main__":
    main()

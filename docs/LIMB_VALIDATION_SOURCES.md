# Limb-corrected contact times: external sources

Provenance for the external checks of the lunar limb profile
(`docs/LIMB_PROFILE.md` §5.4 and §9.7). The sources were found by web research
on 2026-09-23. Every value used in a test was read from a fetched document,
and the test says where from. Times are UTC and longitudes are east-positive.

## Sources

| key | source | limb data | what it gives | licence | used in |
| --- | --- | --- | --- | --- | --- |
| **[EB2024]** | F. Espenak & J. Anderson, *Eclipse Bulletin: Total Solar Eclipse of 2024 April 08*, Table 2–3 (free sample page: eclipsewise.com/pubs/images/EB2024-sample2.pdf; book: eclipsewise.com/pubs/EB2024.html) | not named on the sample; the book page says "These tables include the effect of the Moon's profile on the contact times" | 38 Illinois/Indiana sites: lat/lon to 0.01°, C2/C3 to 0.1 s in local daylight time (IL CDT = UT−5; IN EDT = UT−4, Evansville CDT) | © Espenak, all rights reserved | 16 sites cited in `tests/test_besselian_integration.py` (14 mid-path, 2 near a limit); comparison statistics over all 38 in §9.7 |
| **[Irwin21]** | J. Irwin et al. (2021), "Estimation of the Eclipse Solar Radius by Flash Spectrum Video Analysis", *ApJS*; arXiv:2107.09416, Table 3 | own model (LOLA + Kaguya SLDEM, MOON_ME DE421, DE430), plus Solar Eclipse Maestro (Jubier) and Occult (Herald) | one 2017 site near Vale OR, ~1.5 km inside the southern limit (43°57′10.9″ N, 117°13′09.8″ W, h = 711 m); four predictions of C2/C3 at S☉ = 959.63″ | journal article | site in `tests/test_limb_oracle.py` and `test_limb_profile_contacts_at_vale_match_irwin_2021`; quoted below |
| **SVS** | NASA Scientific Visualization Studio (E. Wright): 2024 svs.gsfc.nasa.gov/5123 (`umbra_hi.shp`, `upath_hi.shp`); 2017 svs.gsfc.nasa.gov/4518 (`umbra17_1s.shp`, `upath17_1s.shp`) | "lunar topography from LRO" (2024); LRO + Kaguya (2017); DE421; umbra drawn on SRTM terrain | umbra outline every 1 s (brackets C2/C3 to 1 s); limb- and terrain-corrected path limits | public domain; credit NASA SVS | statistics in §9.7; limits below (for PR 3) |
| **[SRTM]** | T. G. Farr et al. (2007), *Rev. Geophys.* 45, RG2004; SRTM90 served by opentopodata.org (`/v1/srtm90m`, queried 2026-09-23) | — | ground height H at a site, orthometric (above the EGM96 geoid), 90 m posts | public domain (NASA/USGS) | the 16 [EB2024] sites' heights in `tests/test_besselian_integration.py`, and the 36-site statistics |
| **[EGM96]** | F. G. Lemoine et al. (1998), NASA/TP-1998-206861; the 5′ grid as GeographicLib's `egm96-5.pgm` (geographiclib.sourceforge.io, `geoids-distrib`), bilinear | — | geoid undulation N, so the ellipsoidal height is h = H + N | public (NGA) | N for every site with a height in the tests |
| **Jubier** | X. Jubier's online calculators (xjubier.free.fr, `TSE_2024_GoogleMapFull.html` etc.) and their limb-correction service (`/php/php85/WattsChartCorrections.php`, reply "Kaguya/LRO Done") | Kaguya/LRO | the mean-limb C2/C3 **and** the per-contact limb correction for any site | © Jubier, no reuse licence stated | statistics only (§9.7); no values committed |

## Caveats

- **[EB2024]**
  - Coordinates are rounded to 0.01° (±0.5 km), which matters near a limit.
  - No elevation or ΔT is given on the sample page (Table 2–3 has no
    elevation column); heights are ours (below).
  - Its FDCL column (fraction of the distance from the central line, north
    positive) puts Effingham at +0.981 and Crawfordsville at +0.968: both
    are near the *northern* limit.
  - Near the limits, the printed "Durat. Total" column is not C3 − C2
    (Effingham: 00m48s printed, 25.7 s from the contacts); it looks like the
    mean-limb duration. Validate against the contact times.
  - The local-to-UT hour offsets were checked against the SVS brackets.
- **Jubier's web tool**
  - It applies one correction per contact, at the mean-limb contact
    angle, not the full limb solution of his desktop Solar Eclipse Maestro.
  - A correction of exactly ±51.2 s is a saturation value, not a number.
  - ΔT is fixed per page (2017: 68.8 s, 2024: 69.1 s).
  - Its (profile − mean) correction is still the cleanest check of the
    limb effect alone.
- **SVS**
  - Only 1-s brackets are available.
  - The outlines are drawn on SRTM terrain, so they include the observer's
    elevation; we don't yet.

## [Irwin21] Table 3 (Vale OR, 2017)

| predictor | C2 | C3 |
| --- | --- | --- |
| Solar Eclipse Maestro | 17:25:31.6 | 17:26:07.7 |
| Occult (main page) | 17:25:33.6 | 17:26:06.9 |
| Occult (Baily's beads tool) | 17:25:32.9 | 17:26:07.0 |
| Irwin et al. model | 17:25:34.3 | 17:26:06.9 |

Ours, for a sea-level observer: 17:25:41.3 / 17:26:05.3. At the site's
height, 711 m taken as orthometric plus N = −17.2 m [EGM96]: **17:25:34.5 /
17:26:07.0** (`docs/LIMB_PROFILE.md` §9.10). Used as ellipsoidal, 711 m gives
17:25:34.4 / 17:26:07.0. SRTM90 has 725 m at the published point.

## Site heights

The Bulletin gives no elevations, so each site's height comes from two
sources:
- H: SRTM90 ground height at the printed (0.01°) coordinates, from
  opentopodata (`srtm90m`). These match the heights recorded during the
  limb research.
- N: the EGM96 undulation from the `egm96-5` grid, bilinear. The grid
  reproduces GeographicLib's documented check point (16°46′33″ N, 3°00′34″ W:
  28.7054 m bilinear, against 28.7068 m cubic).

Observer height above the ellipsoid: h = H + N.

| site | H (m) | N (m) | site | H (m) | N (m) |
| --- | --- | --- | --- | --- | --- |
| Carbondale | 130 | −29.8 | Bloomington | 240 | −33.7 |
| Herrin | 129 | −29.9 | Indianapolis | 219 | −34.3 |
| Mount Vernon | 155 | −30.8 | Columbus | 197 | −34.7 |
| Evansville | 108 | −31.6 | Anderson | 271 | −35.0 |
| Vincennes | 130 | −32.4 | Muncie | 289 | −34.8 |
| Terre Haute | 142 | −33.3 | Richmond | 281 | −34.5 |
| Jasper | 156 | −33.4 | Effingham | 182 | −32.6 |
| Bedford | 217 | −33.9 | Crawfordsville | 246 | −34.1 |
| Vale OR (2017) | 711 ([Irwin21]) | −17.2 | 2023 annular (11.4° N, 83.1° W, sea) | 0 | +5.4 |

The other 22 mid-path Bulletin sites in the §9.10 statistics were looked up
the same way. Their values are not committed.

**Caveat.** The Bulletin's coordinates are 0.01°, so H is the ground height
somewhere within ±0.5 km of the actual site. Near a limit, the position
error is far larger than the height effect (`docs/LIMB_PROFILE.md` §9.10,
Effingham).

## SVS limb- and terrain-corrected path limits

Crossing latitude on a meridian, from `upath*.shp`, with the central line
from `center.shp` / `ucenter17_1s.shp`. Kept here for PR 3 (limits).

| eclipse | lon (°E) | S limit (°N) | central line (°N) | N limit (°N) |
| --- | --- | --- | --- | --- |
| 2024-04-08 | −97.0 | 30.801288 | 32.043827 | 33.244401 |
| 2024-04-08 | −86.0 | 38.523218 | 39.542232 | 40.542418 |
| 2024-04-08 | −80.0 | 41.461377 | 42.391150 | 43.312287 |
| 2017-08-21 | −121.0 | 44.242889 | 44.692747 | 45.152032 |
| 2017-08-21 | −100.0 | 40.788224 | 41.318717 | 41.854828 |
| 2017-08-21 | −89.2 | 37.033385 | 37.613274 | 38.199689 |
| 2017-08-21 | −84.5 | 34.930419 | 35.530133 | 36.133864 |

## Dead ends

- EclipseWise *Road Atlas 2024*: 1-s local times, no coordinates, no limb
  statement.
- *Eclipse Bulletin 2017* sample: whole seconds, no limb statement.
- EclipseWise circumstances calculators and the NASA eclipse pages: say they
  don't include the limb.
- nationaleclipse.com, greatamericaneclipse.com (limb-corrected maps, but no
  per-site tables), timeanddate.com.
- IOTA / JOA edge-of-path timings: none found with coordinates. A human could
  check the JOA index for 2018 and 2024–25.
- besselianelements.com: its 2024 Stephenville observation (observed
  18:39:06.6 / 18:39:20.3) has no coordinates.
- A Cloudy Nights thread comparing Jubier with Eclipse Orchestrator returned
  403 to the fetcher.

# Limb-corrected contact times: external sources

Provenance for the external checks of the lunar limb profile
(`docs/LIMB_PROFILE.md` §5.4 and §9.7). The sources were found by web research
on 2026-09-23. Every value used in a test was read from a fetched document,
and the test says where from. Times are UTC and longitudes are east-positive.

## Sources

| key | source | limb data | what it gives | licence | used in |
| --- | --- | --- | --- | --- | --- |
| **[EB2024]** | F. Espenak & J. Anderson, *Eclipse Bulletin: Total Solar Eclipse of 2024 April 08*, Table 2–3 (free sample page: eclipsewise.com/pubs/images/EB2024-sample2.pdf; book: eclipsewise.com/pubs/EB2024.html) | not named on the sample; the book page says "These tables include the effect of the Moon's profile on the contact times" | 38 Illinois/Indiana sites: lat/lon to 0.01°, C2/C3 to 0.1 s in local daylight time (IL CDT = UT−5; IN EDT = UT−4, Evansville CDT) | © Espenak, all rights reserved | 16 sites cited in `tests/test_besselian_integration.py` (14 mid-path, 2 near a limit); comparison statistics over all 38 in §9.7 |
| **[Irwin21]** | J. Irwin et al. (2021), "Estimation of the Eclipse Solar Radius by Flash Spectrum Video Analysis", *ApJS*; arXiv:2107.09416, Table 3 | own model (LOLA + Kaguya SLDEM, MOON_ME DE421, DE430), plus Solar Eclipse Maestro (Jubier) and Occult (Herald) | one 2017 site near Vale OR, ~1.5 km inside the southern limit (43°57′10.9″ N, 117°13′09.8″ W, h = 711 m); four predictions of C2/C3 at S☉ = 959.63″ | journal article | site used in `tests/test_limb_oracle.py`; quoted below |
| **SVS** | NASA Scientific Visualization Studio (E. Wright): 2024 svs.gsfc.nasa.gov/5123 (`umbra_hi.shp`, `upath_hi.shp`); 2017 svs.gsfc.nasa.gov/4518 (`umbra17_1s.shp`, `upath17_1s.shp`) | "lunar topography from LRO" (2024); LRO + Kaguya (2017); DE421; umbra drawn on SRTM terrain | umbra outline every 1 s (brackets C2/C3 to 1 s); limb- and terrain-corrected path limits | public domain; credit NASA SVS | statistics in §9.7; limits below (for PR 3) |
| **Jubier** | X. Jubier's online calculators (xjubier.free.fr, `TSE_2024_GoogleMapFull.html` etc.) and their limb-correction service (`/php/php85/WattsChartCorrections.php`, reply "Kaguya/LRO Done") | Kaguya/LRO | the mean-limb C2/C3 **and** the per-contact limb correction for any site | © Jubier, no reuse licence stated | statistics only (§9.7); no values committed |

## Caveats

- **[EB2024]**
  - Coordinates are rounded to 0.01° (±0.5 km), which matters near a limit.
  - No elevation or ΔT is given on the sample page.
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

Ours, for a sea-level observer: 17:25:41.3 / 17:26:05.3. The site is at
711 m, and elevation is not modelled yet (`docs/LIMB_PROFILE.md` §8, open
item 0).

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

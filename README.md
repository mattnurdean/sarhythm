# SARhythm — Instrument-Grade Radar Water-Loss Atlas (Lake Tisza, 2026)

**NASA Space Apps Challenge 2026 • Challenge: Dancing with the SARs**

SARhythm turns multi-temporal **NASA–ISRO NISAR L-band radar** imagery into an auditable, offline-capable surface-water contraction and drought-risk monitor for government water agencies and parametric insurers.

## Key Verified Findings (Tiszavalki Basin, Lake Tisza, Hungary — Summer/Autumn 2026)
- **Summer Open-Water Loss (`2026-06-18` → `2026-08-29`):** Open water contracted by **−6.98 km² (−15.9%)** (from `43.83 km²` to `36.84 km²`, roughly 980 football fields), with `10.21 km²` of gross water-to-land transition.
- **Hydrological Context:** Coincided with a **−140 cm** reservoir level drop at Kisköre-felső (`749 cm` → `609 cm`, losing `149.1 million m³` of stored volume), followed by a partial October recovery (`625–643 cm`) after KÖTIVIZIG suspended routine autumn/winter draining on October 22, 2026 (*"Vízvisszatartás a Tisza-tóban"*).
- **Cross-Sensor Validation:** Independently matches cloud-free **NASA OPERA DSWx-HLS** (`Tile T34TDT`) water contraction (`33.22 km²` on Jun 19 → `26.28 km²` on Aug 20 = **−6.94 km²**) within **0.04 km²**, while capturing flooded zones obscured by **66.6% cloud cover** on June 17 and ~24% floating macrophyte cover.
- **Metrology & Opposite-Sign Physics:** Calibrates a statistical guard band of $T_{\text{final}} = \pm 5.33\text{ dB}$ ($L = 9.6$ looks, $\alpha = 0.01$ $F$-distribution bound) alongside separate urban ($\hat{\sigma}_{\text{MAD}} = 1.119\text{ dB}$) and pelagial water ($\hat{\sigma}_{\text{MAD}} = 0.452\text{ dB}$) stability controls. Stratifies open water (`+1.66 dB` brightening in Sector 19; `+4.74 dB` in Sector 05) from emergent *Phragmites* reedbeds (`−1.48 dB` darkening in Sector 04).

## Repository Structure
- `index.html` — Self-contained interactive decision viewer (zero external CDNs, works 100% offline).
- `scenes.json`, `water_area_timeseries.json`, `change_polygons.geojson`, `validation_summary.json` — Precomputed static datasets and automated self-test verdicts (5/5 passed).
- `polygons/` — Per-sector JSON time series fetched on click.
- `rasters/` — Pre-rendered 4-class water-change and per-date radar PNG overlays.
- `pipeline/` — Reproducible Python extraction, calibration, and validation scripts (`build_local_aoi_cache.py`, `run_sarhythm_pipeline.py`, `gauge_timeseries_2026.csv`).

## Data Sources & Space Agency Partners
1. **NASA–ISRO Synthetic Aperture Radar (NISAR):** L2 Geocoded Polarimetric Covariance (`L2-GCOV`, Track 057 / Frame 026, Ascending, 20 m, Provisional `CRID P05023` & Validated `CRID P05036`) via NASA ASF DAAC.
2. **NASA OPERA Dynamic Surface Water Extent (`OPERA_L3_DSWX-HLS_V1`):** Harmonized Landsat-8/9 & Sentinel-2 water classification (`Tile T34TDT`) via NASA PO.DAAC.
3. **Hungarian Water Authorities (KÖTIVIZIG / OVF / Vízügy):** Official 2026 reservoir stage, volume, and operational directives (`kotivizig.hu`, `vizugy.hu`).

## Super Intelligence (SI) Acknowledgment
Large language model assistants (Gemini and Claude) were used during this hackathon to accelerate Python HDF5 streaming code, statistical metrology verification, HTML/SVG frontend layout refinement, and editorial clarity. All satellite data queries, numerical computations, physical interpretations, and validation checks were executed and verified against live NASA Earthdata and KÖTIVIZIG records by the team.
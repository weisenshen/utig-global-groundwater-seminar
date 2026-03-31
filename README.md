# UTIG Global Groundwater Seminar (Spring 2026)

This repository stores course materials and collaborative work for the **UTIG seminar (Spring 2026)** focused on **global groundwater storage estimates**.

## Purpose
- Organize seminar notes and discussion topics.
- Track readings, references, and data resources.
- Develop and version analysis workflows and results.

## Suggested repository organization
- `notes/` — meeting notes and weekly summaries
- `readings/` — reading lists and paper annotations
- `data/` — data inventories, source links, and metadata
- `analysis/` — scripts, notebooks, and reproducible workflows
- `notebooks/` — shared student notebooks and exercises
- `figures/` — plots and presentation-ready graphics

## Scope
The seminar explores methods, datasets, uncertainty, and interpretation related to estimating groundwater storage at global scales.

## Current Workflow
- [Project summary notebook](./notebooks/project_summary.ipynb)
- [Point-profile extraction script](./analysis/extract_usgs_point_profile.py)
- [Profile plotting script](./analysis/plot_point_profile.py)
- [Batch CONUS workflow script](./analysis/run_conus_porosity_workflow.py)
- [SQLite extraction script](./analysis/extract_location_from_master.py)

## Example Outputs
- [Williston Basin point-profile summary (JSON)](./data/point_profile_lat_47p5000N_lon_102p0000W.json)
- [Williston Basin point-profile layers (CSV)](./data/point_profile_lat_47p5000N_lon_102p0000W_layers.csv)
- [Williston Basin profile figure (PNG)](./figures/point_profile_lat_47p5000N_lon_102p0000W_profile.png)
- [Verified compact-workflow aggregate summary (JSON)](./data/conus_0p25_sqlite_test/conus_0p25_aggregate.json)
- [Extracted Williston Basin record from SQLite master store (JSON)](./data/conus_0p25_sqlite_test/extract_lat_47p5000_lon_-102p0000.json)
- [Extracted Williston Basin intervals from SQLite master store (CSV)](./data/conus_0p25_sqlite_test/extract_lat_47p5000_lon_-102p0000_intervals.csv)

## Notes
- The geologic framework geometry comes from the USGS National Crustal Model files used locally during analysis.
- The current porosity, conductivity, and seismic property assignments are provisional placeholder values for workflow development and benchmarking.
- Large raw data files are not stored in this repository; this repo tracks the workflow, summary products, and lightweight examples.

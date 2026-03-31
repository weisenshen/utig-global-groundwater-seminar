#!/opt/anaconda3/bin/python

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator


DEFAULT_DATA_DIR = Path("USGS-geology model")
DEFAULT_OUTPUT_DIR = Path("outputs") / "conus_0p25"
CONUS_BOUNDS = {
    "lat_min": 22.0,
    "lat_max": 53.0,
    "lon_min": -130.0,
    "lon_max": -63.0,
}
EARTH_RADIUS_M = 6_371_000.0


def decode_name(table, index: int) -> str:
    if index <= 0:
        return ""
    raw = table[:, index - 1].tobytes().decode("utf-8", errors="ignore")
    return raw.replace("\x00", "").strip()


def format_coord(value: float, positive_tag: str, negative_tag: str) -> str:
    tag = positive_tag if value >= 0 else negative_tag
    magnitude = f"{abs(value):.4f}".replace(".", "p")
    return f"{magnitude}{tag}"


def cell_area_m2(latitude_deg: float, step_deg: float) -> float:
    lat1 = math.radians(latitude_deg - step_deg / 2)
    lat2 = math.radians(latitude_deg + step_deg / 2)
    dlon = math.radians(step_deg)
    return (EARTH_RADIUS_M**2) * dlon * abs(math.sin(lat2) - math.sin(lat1))


def provisional_properties(rock_type_name: str) -> dict:
    name = rock_type_name.lower()

    if any(token in name for token in ["alluv", "soil", "sediment", "residuum", "till", "loess"]):
        return {"porosity_fraction": 0.25, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.03, "vp_km_s": 2.0, "vs_km_s": 0.8, "property_class": "unconsolidated"}
    if any(token in name for token in ["sandstone", "conglomerate", "breccia", "arkose"]):
        return {"porosity_fraction": 0.12, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.01, "vp_km_s": 4.0, "vs_km_s": 2.2, "property_class": "clastic_coarse"}
    if any(token in name for token in ["shale", "claystone", "mudstone", "siltstone", "argillite"]):
        return {"porosity_fraction": 0.06, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.02, "vp_km_s": 3.2, "vs_km_s": 1.6, "property_class": "clastic_fine"}
    if any(token in name for token in ["limestone", "dolomite", "chalk", "marl"]):
        return {"porosity_fraction": 0.05, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.005, "vp_km_s": 5.8, "vs_km_s": 3.1, "property_class": "carbonate"}
    if any(token in name for token in ["evaporite", "anhydrite", "gypsum", "halite", "salt"]):
        return {"porosity_fraction": 0.02, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.001, "vp_km_s": 4.5, "vs_km_s": 2.5, "property_class": "evaporite"}
    if any(token in name for token in ["basalt", "gabbro", "diabase", "peridotite", "amphibolite", "granulite"]):
        return {"porosity_fraction": 0.005, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.0005, "vp_km_s": 6.8, "vs_km_s": 3.9, "property_class": "mafic_ultramafic"}
    if any(token in name for token in ["rhyolite", "andesite", "dacite", "tuff", "volcanic"]):
        return {"porosity_fraction": 0.08, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.003, "vp_km_s": 4.5, "vs_km_s": 2.5, "property_class": "volcanic"}
    if any(token in name for token in ["granite", "granodiorite", "gneiss", "schist", "quartzite", "metamorphic"]):
        return {"porosity_fraction": 0.01, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.001, "vp_km_s": 6.0, "vs_km_s": 3.5, "property_class": "felsic_metamorphic"}
    return {"porosity_fraction": 0.03, "fluid_saturation_fraction": 1.0, "conductivity_s_per_m": 0.002, "vp_km_s": 5.0, "vs_km_s": 2.8, "property_class": "fallback"}


class USGSProfileSampler:
    def __init__(self, data_dir: Path):
        self.spatial_ds = Dataset(data_dir / "NCM_SpatialGrid.nc")
        self.grids_ds = Dataset(data_dir / "NCM_GeologicFrameworkGrids.nc")
        self.volume_ds = Dataset(data_dir / "NCM_GeologicFrameworkVolume.nc")

        lat_vec = self.spatial_ds.variables["Latitude vector"][:]
        lon_vec = self.spatial_ds.variables["Longitude vector"][:]
        j_grid = self.spatial_ds.variables["Index j grid"][:]
        k_grid = self.spatial_ds.variables["Index k grid"][:]
        self.j_interp = RegularGridInterpolator((lon_vec, lat_vec), j_grid, bounds_error=False, fill_value=np.nan)
        self.k_interp = RegularGridInterpolator((lon_vec, lat_vec), k_grid, bounds_error=False, fill_value=np.nan)

        self.lat_grid = self.spatial_ds.variables["Latitude"]
        self.lon_grid = self.spatial_ds.variables["Longitude"]
        self.rock_names = self.volume_ds.variables["Rock Type Name"][:]
        self.age_names = self.volume_ds.variables["Age Name"][:]
        self.age_min = self.volume_ds.variables["Age Min"][:]
        self.age_max = self.volume_ds.variables["Age Max"][:]

    def close(self) -> None:
        self.spatial_ds.close()
        self.grids_ds.close()
        self.volume_ds.close()

    def nearest_index(self, latitude: float, longitude: float) -> tuple[int, int] | None:
        point = np.array([[longitude, latitude]])
        j_est = float(self.j_interp(point)[0])
        k_est = float(self.k_interp(point)[0])
        if np.isnan(j_est) or np.isnan(k_est):
            return None

        j_center = int(round(j_est)) - 1
        k_center = int(round(k_est)) - 1
        if j_center < 0 or k_center < 0:
            return None
        if j_center >= self.lat_grid.shape[1] or k_center >= self.lat_grid.shape[0]:
            return None

        k0 = max(0, k_center - 2)
        k1 = min(self.lat_grid.shape[0], k_center + 3)
        j0 = max(0, j_center - 2)
        j1 = min(self.lat_grid.shape[1], j_center + 3)
        if k0 >= k1 or j0 >= j1:
            return None

        lat_window = self.lat_grid[k0:k1, j0:j1]
        lon_window = self.lon_grid[k0:k1, j0:j1]
        distance2 = (lat_window - latitude) ** 2 + (lon_window - longitude) ** 2
        if distance2.size == 0 or np.all(np.isnan(distance2)):
            return None
        local_k, local_j = np.unravel_index(np.nanargmin(distance2), distance2.shape)
        return k0 + int(local_k), j0 + int(local_j)

    def sample_profile(self, latitude: float, longitude: float) -> dict | None:
        idx = self.nearest_index(latitude, longitude)
        if idx is None:
            return None

        k_idx, j_idx = idx
        nearest_lat = float(self.lat_grid[k_idx, j_idx])
        nearest_lon = float(self.lon_grid[k_idx, j_idx])
        distance_deg = float(math.sqrt((nearest_lat - latitude) ** 2 + (nearest_lon - longitude) ** 2))

        grid_fields = [
            "Surface Elevation",
            "Bedrock Elevation",
            "Bottom Cenozoic Elevation",
            "Bottom Phanerozoic Elevation",
            "Upper Mid Crustal Elevation",
            "Lower Mid Crustal Elevation",
            "Moho Elevation",
            "Top Ocean Plate Elevation",
        ]
        elevation_summary = {}
        for field in grid_fields:
            if field in self.grids_ds.variables:
                value = float(self.grids_ds.variables[field][k_idx, j_idx])
                elevation_summary[field] = None if np.isnan(value) else value

        rock_idx = self.volume_ds.variables["Rock Type Profile Index"][k_idx, j_idx, :]
        age_idx = self.volume_ds.variables["Age Profile Index"][k_idx, j_idx, :]
        layer_elev = self.volume_ds.variables["Layer Elevation"][k_idx, j_idx, :]
        layers = []
        for layer_number in range(len(layer_elev)):
            rock_code = int(rock_idx[layer_number]) if layer_number < len(rock_idx) else 0
            age_code = int(age_idx[layer_number]) if layer_number < len(age_idx) else 0
            if rock_code == 0 and age_code == 0:
                continue
            boundary_elevation_m = float(layer_elev[layer_number])
            if not np.isfinite(boundary_elevation_m):
                boundary_elevation_m = None
            layers.append(
                {
                    "layer_number": layer_number + 1,
                    "boundary_elevation_m": boundary_elevation_m,
                    "rock_type_name": decode_name(self.rock_names, rock_code),
                    "age_name": decode_name(self.age_names, age_code),
                    "age_min_ma": float(self.age_min[age_code - 1]) if age_code > 0 else None,
                    "age_max_ma": float(self.age_max[age_code - 1]) if age_code > 0 else None,
                }
            )

        return {
            "requested_point": {"latitude": latitude, "longitude": longitude},
            "nearest_grid_point": {
                "k_index": k_idx,
                "j_index": j_idx,
                "latitude": nearest_lat,
                "longitude": nearest_lon,
                "distance_degrees": distance_deg,
            },
            "elevation_summary_m": elevation_summary,
            "layers": layers,
        }


def build_intervals(profile: dict) -> list[dict]:
    surface = profile["elevation_summary_m"]["Surface Elevation"]
    top = surface
    intervals = []
    for layer in profile["layers"]:
        bottom = layer["boundary_elevation_m"]
        if bottom is None or bottom >= top:
            continue
        intervals.append(
            {
                "top_elevation_m": top,
                "bottom_elevation_m": bottom,
                "thickness_m": top - bottom,
                "rock_type_name": layer["rock_type_name"] or "Unknown",
                "age_name": layer["age_name"] or "Unknown",
                "age_min_ma": layer["age_min_ma"],
                "age_max_ma": layer["age_max_ma"],
            }
        )
        top = bottom
    return intervals


def enrich_intervals(intervals: list[dict]) -> list[dict]:
    enriched = []
    for interval in intervals:
        props = provisional_properties(interval["rock_type_name"])
        enriched.append({**interval, **props})
    return enriched


def compute_top_depth_pore_volume(intervals: list[dict], surface_elevation_m: float, top_depth_m: float, area_m2: float) -> tuple[float, float]:
    lower_limit = surface_elevation_m - top_depth_m
    pore_volume_m3 = 0.0
    fluid_volume_m3 = 0.0
    for interval in intervals:
        overlap_top = min(interval["top_elevation_m"], surface_elevation_m)
        overlap_bottom = max(interval["bottom_elevation_m"], lower_limit)
        overlap_thickness_m = overlap_top - overlap_bottom
        if overlap_thickness_m <= 0:
            continue
        interval_volume_m3 = overlap_thickness_m * area_m2
        pore_volume_m3 += interval_volume_m3 * interval["porosity_fraction"]
        fluid_volume_m3 += interval_volume_m3 * interval["porosity_fraction"] * interval["fluid_saturation_fraction"]
    return pore_volume_m3, fluid_volume_m3


def write_model_csv(model_path: Path, intervals: list[dict]) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "top_elevation_m",
                "bottom_elevation_m",
                "thickness_m",
                "rock_type_name",
                "age_name",
                "age_min_ma",
                "age_max_ma",
                "porosity_fraction",
                "fluid_saturation_fraction",
                "conductivity_s_per_m",
                "vp_km_s",
                "vs_km_s",
                "property_class",
            ],
        )
        writer.writeheader()
        writer.writerows(intervals)


def initialize_database(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS location_summaries (
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            nearest_latitude REAL NOT NULL,
            nearest_longitude REAL NOT NULL,
            distance_degrees REAL NOT NULL,
            surface_elevation_m REAL,
            bedrock_elevation_m REAL,
            bottom_cenozoic_elevation_m REAL,
            bottom_phanerozoic_elevation_m REAL,
            cell_area_km2 REAL NOT NULL,
            interval_count INTEGER NOT NULL,
            top_depth_km REAL NOT NULL,
            pore_volume_km3 REAL NOT NULL,
            fluid_volume_km3 REAL NOT NULL,
            PRIMARY KEY (latitude, longitude)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS location_intervals (
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            interval_index INTEGER NOT NULL,
            top_elevation_m REAL NOT NULL,
            bottom_elevation_m REAL NOT NULL,
            thickness_m REAL NOT NULL,
            rock_type_name TEXT NOT NULL,
            age_name TEXT NOT NULL,
            age_min_ma REAL,
            age_max_ma REAL,
            porosity_fraction REAL NOT NULL,
            fluid_saturation_fraction REAL NOT NULL,
            conductivity_s_per_m REAL NOT NULL,
            vp_km_s REAL NOT NULL,
            vs_km_s REAL NOT NULL,
            property_class TEXT NOT NULL,
            PRIMARY KEY (latitude, longitude, interval_index)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rock_properties (
            rock_type_name TEXT PRIMARY KEY,
            property_class TEXT NOT NULL,
            porosity_fraction REAL NOT NULL,
            fluid_saturation_fraction REAL NOT NULL,
            conductivity_s_per_m REAL NOT NULL,
            vp_km_s REAL NOT NULL,
            vs_km_s REAL NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    conn.execute("DELETE FROM location_summaries")
    conn.execute("DELETE FROM location_intervals")
    conn.execute("DELETE FROM rock_properties")
    conn.execute("DELETE FROM run_metadata")
    return conn


def write_database_records(
    conn: sqlite3.Connection,
    summary_rows: list[dict],
    interval_rows: list[tuple],
    property_rows: list[dict],
    run_metadata: dict,
) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO location_summaries (
            latitude, longitude, nearest_latitude, nearest_longitude, distance_degrees,
            surface_elevation_m, bedrock_elevation_m, bottom_cenozoic_elevation_m,
            bottom_phanerozoic_elevation_m, cell_area_km2, interval_count, top_depth_km,
            pore_volume_km3, fluid_volume_km3
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["latitude"],
                row["longitude"],
                row["nearest_latitude"],
                row["nearest_longitude"],
                row["distance_degrees"],
                row["surface_elevation_m"],
                row["bedrock_elevation_m"],
                row["bottom_cenozoic_elevation_m"],
                row["bottom_phanerozoic_elevation_m"],
                row["cell_area_km2"],
                row["interval_count"],
                row["top_depth_km"],
                row["pore_volume_km3"],
                row["fluid_volume_km3"],
            )
            for row in summary_rows
        ],
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO location_intervals (
            latitude, longitude, interval_index, top_elevation_m, bottom_elevation_m,
            thickness_m, rock_type_name, age_name, age_min_ma, age_max_ma,
            porosity_fraction, fluid_saturation_fraction, conductivity_s_per_m,
            vp_km_s, vs_km_s, property_class
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        interval_rows,
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO rock_properties (
            rock_type_name, property_class, porosity_fraction, fluid_saturation_fraction,
            conductivity_s_per_m, vp_km_s, vs_km_s, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["rock_type_name"],
                row["property_class"],
                row["porosity_fraction"],
                row["fluid_saturation_fraction"],
                row["conductivity_s_per_m"],
                row["vp_km_s"],
                row["vs_km_s"],
                row["source"],
            )
            for row in property_rows
        ],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO run_metadata (key, value) VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in run_metadata.items()],
    )
    conn.commit()


def generate_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float, step: float) -> list[tuple[float, float]]:
    latitudes = np.arange(lat_min, lat_max + 0.5 * step, step)
    longitudes = np.arange(lon_min, lon_max + 0.5 * step, step)
    return [(round(float(lat), 6), round(float(lon), 6)) for lat in latitudes for lon in longitudes]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a coarse-grid CONUS profile extraction and provisional pore-volume workflow."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-deg", type=float, default=0.25)
    parser.add_argument("--lat-min", type=float, default=CONUS_BOUNDS["lat_min"])
    parser.add_argument("--lat-max", type=float, default=CONUS_BOUNDS["lat_max"])
    parser.add_argument("--lon-min", type=float, default=CONUS_BOUNDS["lon_min"])
    parser.add_argument("--lon-max", type=float, default=CONUS_BOUNDS["lon_max"])
    parser.add_argument("--top-depth-km", type=float, default=10.0)
    parser.add_argument("--max-points", type=int, default=None, help="Optional cap for verification runs.")
    parser.add_argument("--write-legacy-csv", action="store_true", help="Also write CSV summary/property files for compatibility.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "conus_0p25_master.sqlite"
    conn = initialize_database(db_path)

    grid_points = generate_grid(args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.step_deg)
    if args.max_points is not None:
        grid_points = grid_points[: args.max_points]

    sampler = USGSProfileSampler(args.data_dir)
    summaries = []
    interval_rows = []
    property_rows = {}
    total_pore_volume_km3 = 0.0
    total_fluid_volume_km3 = 0.0
    processed_points = 0

    try:
        for latitude, longitude in grid_points:
            profile = sampler.sample_profile(latitude, longitude)
            if profile is None:
                continue

            intervals = enrich_intervals(build_intervals(profile))
            surface_elevation_m = profile["elevation_summary_m"]["Surface Elevation"]
            area_m2 = cell_area_m2(latitude, args.step_deg)
            pore_volume_m3, fluid_volume_m3 = compute_top_depth_pore_volume(
                intervals,
                surface_elevation_m,
                args.top_depth_km * 1000.0,
                area_m2,
            )

            for interval in intervals:
                property_rows[interval["rock_type_name"]] = {
                    "rock_type_name": interval["rock_type_name"],
                    "property_class": interval["property_class"],
                    "porosity_fraction": interval["porosity_fraction"],
                    "fluid_saturation_fraction": interval["fluid_saturation_fraction"],
                    "conductivity_s_per_m": interval["conductivity_s_per_m"],
                    "vp_km_s": interval["vp_km_s"],
                    "vs_km_s": interval["vs_km_s"],
                    "source": "provisional_internal_defaults",
                }
            for interval_index, interval in enumerate(intervals, start=1):
                interval_rows.append(
                    (
                        latitude,
                        longitude,
                        interval_index,
                        interval["top_elevation_m"],
                        interval["bottom_elevation_m"],
                        interval["thickness_m"],
                        interval["rock_type_name"],
                        interval["age_name"],
                        interval["age_min_ma"],
                        interval["age_max_ma"],
                        interval["porosity_fraction"],
                        interval["fluid_saturation_fraction"],
                        interval["conductivity_s_per_m"],
                        interval["vp_km_s"],
                        interval["vs_km_s"],
                        interval["property_class"],
                    )
                )

            summaries.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "nearest_latitude": profile["nearest_grid_point"]["latitude"],
                    "nearest_longitude": profile["nearest_grid_point"]["longitude"],
                    "distance_degrees": profile["nearest_grid_point"]["distance_degrees"],
                    "surface_elevation_m": profile["elevation_summary_m"]["Surface Elevation"],
                    "bedrock_elevation_m": profile["elevation_summary_m"]["Bedrock Elevation"],
                    "bottom_cenozoic_elevation_m": profile["elevation_summary_m"]["Bottom Cenozoic Elevation"],
                    "bottom_phanerozoic_elevation_m": profile["elevation_summary_m"]["Bottom Phanerozoic Elevation"],
                    "cell_area_km2": area_m2 / 1e6,
                    "interval_count": len(intervals),
                    "top_depth_km": args.top_depth_km,
                    "pore_volume_km3": pore_volume_m3 / 1e9,
                    "fluid_volume_km3": fluid_volume_m3 / 1e9,
                }
            )
            total_pore_volume_km3 += pore_volume_m3 / 1e9
            total_fluid_volume_km3 += fluid_volume_m3 / 1e9
            processed_points += 1
    finally:
        sampler.close()

    aggregate_json = output_dir / "conus_0p25_aggregate.json"
    property_rows_sorted = sorted(property_rows.values(), key=lambda row: row["rock_type_name"])
    aggregate = {
        "grid": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
            "step_degrees": args.step_deg,
            "requested_point_count": len(grid_points),
            "processed_point_count": processed_points,
        },
        "integration": {
            "top_depth_km_below_surface": args.top_depth_km,
            "total_pore_volume_km3": total_pore_volume_km3,
            "total_fluid_volume_km3": total_fluid_volume_km3,
        },
        "property_model": {
            "type": "provisional_internal_defaults",
            "note": "These porosity, conductivity, and seismic properties are broad lithology-based placeholders and should be replaced with a calibrated property table before scientific use.",
        },
        "outputs": {
            "master_sqlite": str(db_path),
            "aggregate_json": str(aggregate_json),
            "legacy_summary_csv": str(output_dir / "conus_0p25_summary.csv") if args.write_legacy_csv else None,
            "legacy_rock_properties_csv": str(output_dir / "rock_properties_used.csv") if args.write_legacy_csv else None,
        },
    }
    write_database_records(conn, summaries, interval_rows, property_rows_sorted, aggregate)
    conn.close()

    summary_csv = output_dir / "conus_0p25_summary.csv"
    properties_csv = output_dir / "rock_properties_used.csv"
    if args.write_legacy_csv:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()) if summaries else [])
            if summaries:
                writer.writeheader()
                writer.writerows(summaries)
        with properties_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(property_rows_sorted[0].keys()) if property_rows_sorted else [])
            if property_rows_sorted:
                writer.writeheader()
                writer.writerows(property_rows_sorted)

    with aggregate_json.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Processed {processed_points} locations.")
    print(f"Wrote {db_path}")
    print(f"Wrote {aggregate_json}")
    if args.write_legacy_csv:
        print(f"Wrote {summary_csv}")
        print(f"Wrote {properties_csv}")


if __name__ == "__main__":
    main()

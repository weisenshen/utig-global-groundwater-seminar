#!/opt/anaconda3/bin/python

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


DEFAULT_DATA_DIR = Path("USGS-geology model")


def format_coord(value: float, positive_tag: str, negative_tag: str) -> str:
    tag = positive_tag if value >= 0 else negative_tag
    magnitude = f"{abs(value):.4f}".replace(".", "p")
    return f"{magnitude}{tag}"


def build_output_stem(lat: float, lon: float) -> Path:
    lat_part = format_coord(lat, "N", "S")
    lon_part = format_coord(lon, "E", "W")
    return Path("outputs") / f"point_profile_lat_{lat_part}_lon_{lon_part}"


def decode_name(table, index: int) -> str:
    if index <= 0:
        return ""
    raw = table[:, index - 1].tobytes().decode("utf-8", errors="ignore")
    return raw.replace("\x00", "").strip()


def load_point_profile(data_dir: Path, lat0: float, lon0: float) -> dict:
    spatial_path = data_dir / "NCM_SpatialGrid.nc"
    grids_path = data_dir / "NCM_GeologicFrameworkGrids.nc"
    volume_path = data_dir / "NCM_GeologicFrameworkVolume.nc"

    with Dataset(spatial_path) as spatial_ds:
        lat = spatial_ds.variables["Latitude"][:]
        lon = spatial_ds.variables["Longitude"][:]
        distance2 = (lat - lat0) ** 2 + (lon - lon0) ** 2
        x_idx, y_idx = np.unravel_index(np.nanargmin(distance2), distance2.shape)
        nearest_lat = float(lat[x_idx, y_idx])
        nearest_lon = float(lon[x_idx, y_idx])
        distance_deg = float(np.sqrt(distance2[x_idx, y_idx]))

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
    grid_summary = {}
    with Dataset(grids_path) as grids_ds:
        for field in grid_fields:
            if field in grids_ds.variables:
                value = float(grids_ds.variables[field][x_idx, y_idx])
                grid_summary[field] = None if np.isnan(value) else value

    layers = []
    with Dataset(volume_path) as volume_ds:
        rock_idx = volume_ds.variables["Rock Type Profile Index"][x_idx, y_idx, :]
        age_idx = volume_ds.variables["Age Profile Index"][x_idx, y_idx, :]
        layer_elev = volume_ds.variables["Layer Elevation"][x_idx, y_idx, :]
        rock_names = volume_ds.variables["Rock Type Name"][:]
        age_names = volume_ds.variables["Age Name"][:]
        age_min = volume_ds.variables["Age Min"][:]
        age_max = volume_ds.variables["Age Max"][:]

        for layer_number in range(len(layer_elev)):
            rock_code = int(rock_idx[layer_number]) if layer_number < len(rock_idx) else 0
            age_code = int(age_idx[layer_number]) if layer_number < len(age_idx) else 0
            boundary_elevation_m = float(layer_elev[layer_number])
            if not np.isfinite(boundary_elevation_m):
                boundary_elevation_m = None

            rock_name = decode_name(rock_names, rock_code)
            age_name = decode_name(age_names, age_code)
            if rock_code == 0 and age_code == 0:
                continue

            layers.append(
                {
                    "layer_number": layer_number + 1,
                    "boundary_elevation_m": boundary_elevation_m,
                    "rock_type_code": rock_code,
                    "rock_type_name": rock_name,
                    "age_code": age_code,
                    "age_name": age_name,
                    "age_min_ma": float(age_min[age_code - 1]) if age_code > 0 else None,
                    "age_max_ma": float(age_max[age_code - 1]) if age_code > 0 else None,
                }
            )

    return {
        "requested_point": {"latitude": lat0, "longitude": lon0},
        "nearest_grid_point": {
            "x_index": int(x_idx),
            "y_index": int(y_idx),
            "latitude": nearest_lat,
            "longitude": nearest_lon,
            "distance_degrees": distance_deg,
        },
        "elevation_summary_m": grid_summary,
        "layers": layers,
    }


def write_outputs(profile: dict, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    summary_path = output_stem.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    layers_path = output_stem.with_name(f"{output_stem.name}_layers.csv")
    with layers_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_number",
                "boundary_elevation_m",
                "rock_type_code",
                "rock_type_name",
                "age_code",
                "age_name",
                "age_min_ma",
                "age_max_ma",
            ],
        )
        writer.writeheader()
        writer.writerows(profile["layers"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the nearest USGS geologic framework profile for a latitude/longitude."
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude in degrees north.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude in decimal degrees east; western longitudes are negative.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the USGS netCDF files.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output path without extension. Defaults to a filename derived from lat/lon.",
    )
    args = parser.parse_args()

    output_stem = args.output_stem or build_output_stem(args.lat, args.lon)
    profile = load_point_profile(args.data_dir, args.lat, args.lon)
    write_outputs(profile, output_stem)

    print(f"Requested point: ({args.lat:.4f}, {args.lon:.4f})")
    nearest = profile["nearest_grid_point"]
    print(
        "Nearest grid point: "
        f"({nearest['latitude']:.6f}, {nearest['longitude']:.6f}) "
        f"at indices ({nearest['x_index']}, {nearest['y_index']})"
    )
    print(f"Grid offset: {nearest['distance_degrees']:.6f} degrees")
    print("Elevation summary (m):")
    for key, value in profile["elevation_summary_m"].items():
        print(f"  {key}: {value}")
    print(f"Layer count: {len(profile['layers'])}")
    print(f"Wrote {output_stem.with_suffix('.json')}")
    print(f"Wrote {output_stem.with_name(f'{output_stem.name}_layers.csv')}")


if __name__ == "__main__":
    main()

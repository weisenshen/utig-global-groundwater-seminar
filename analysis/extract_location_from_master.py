#!/opt/anaconda3/bin/python

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("outputs") / "conus_0p25" / "conus_0p25_master.sqlite"


def fetch_location(conn: sqlite3.Connection, lat: float, lon: float) -> tuple[dict, list[dict]] | None:
    conn.row_factory = sqlite3.Row
    summary = conn.execute(
        """
        SELECT *
        FROM location_summaries
        WHERE latitude = ? AND longitude = ?
        """,
        (lat, lon),
    ).fetchone()
    if summary is None:
        return None

    intervals = conn.execute(
        """
        SELECT interval_index, top_elevation_m, bottom_elevation_m, thickness_m,
               rock_type_name, age_name, age_min_ma, age_max_ma,
               porosity_fraction, fluid_saturation_fraction, conductivity_s_per_m,
               vp_km_s, vs_km_s, property_class
        FROM location_intervals
        WHERE latitude = ? AND longitude = ?
        ORDER BY interval_index
        """,
        (lat, lon),
    ).fetchall()
    return dict(summary), [dict(row) for row in intervals]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one location from the compact CONUS master SQLite file.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to the master SQLite file.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the stored grid point.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the stored grid point.")
    parser.add_argument("--output-stem", type=Path, default=None, help="Optional output path stem; defaults beside the database.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        result = fetch_location(conn, args.lat, args.lon)
    finally:
        conn.close()

    if result is None:
        raise SystemExit(f"No location found for lat={args.lat} lon={args.lon}")

    summary, intervals = result
    output_stem = args.output_stem or args.db.parent / f"extract_lat_{args.lat:.4f}_lon_{args.lon:.4f}".replace(".", "p")

    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_name(f"{output_stem.name}_intervals.csv")

    payload = {"summary": summary, "intervals": intervals}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(intervals[0].keys()) if intervals else [])
        if intervals:
            writer.writeheader()
            writer.writerows(intervals)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

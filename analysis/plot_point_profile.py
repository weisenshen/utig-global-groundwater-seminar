#!/opt/anaconda3/bin/python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LOCAL_CACHE_DIR = Path(".cache")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG_DIR = Path(".mplconfig")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def format_coord(value: float, positive_tag: str, negative_tag: str) -> str:
    tag = positive_tag if value >= 0 else negative_tag
    magnitude = f"{abs(value):.4f}".replace(".", "p")
    return f"{magnitude}{tag}"


def build_output_stem(lat: float, lon: float) -> Path:
    lat_part = format_coord(lat, "N", "S")
    lon_part = format_coord(lon, "E", "W")
    return Path("outputs") / f"point_profile_lat_{lat_part}_lon_{lon_part}"


def load_profile(profile_json: Path) -> dict:
    with profile_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def color_for_rock(rock_name: str, cache: dict[str, tuple]) -> tuple:
    if rock_name not in cache:
        cmap = plt.get_cmap("tab20")
        cache[rock_name] = cmap(len(cache) % cmap.N)
    return cache[rock_name]


def build_intervals(profile: dict) -> list[dict]:
    surface = profile["elevation_summary_m"]["Surface Elevation"]
    intervals = []
    top = surface
    for layer in profile["layers"]:
        bottom = layer["boundary_elevation_m"]
        if bottom is None or bottom >= top:
            continue
        intervals.append(
            {
                "layer_number": layer["layer_number"],
                "top_m": top,
                "bottom_m": bottom,
                "thickness_m": top - bottom,
                "rock_type_name": layer["rock_type_name"] or "Unknown",
                "age_name": layer["age_name"] or "Unknown",
            }
        )
        top = bottom
    return intervals


def draw_intervals(
    ax,
    intervals: list[dict],
    color_cache: dict[str, tuple],
    min_label_thickness_m: float,
    label_mode: str = "inside",
) -> None:
    outside_labels = []
    y_min, y_max = ax.get_ylim()

    for interval in intervals:
        color = color_for_rock(interval["rock_type_name"], color_cache)
        rect = Rectangle(
            (0, interval["bottom_m"]),
            1,
            interval["thickness_m"],
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(rect)

        if interval["thickness_m"] >= min_label_thickness_m:
            label = (
                f"{interval['rock_type_name']}\n"
                f"{interval['thickness_m']:.0f} m"
            )
            y_mid = (interval["top_m"] + interval["bottom_m"]) / 2

            if label_mode == "outside":
                if interval["top_m"] >= y_min and interval["bottom_m"] <= y_max:
                    outside_labels.append((y_mid, label))
            else:
                ax.text(
                    0.5,
                    y_mid,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    if label_mode == "outside" and outside_labels:
        ymin, ymax = ax.get_ylim()
        margin = 200
        y_start = ymax - margin
        y_end = ymin + margin
        if len(outside_labels) == 1:
            label_positions = [(y_start + y_end) / 2]
        else:
            step = (y_start - y_end) / (len(outside_labels) - 1)
            label_positions = [y_start - i * step for i in range(len(outside_labels))]

        for (y_mid, label), label_y in zip(sorted(outside_labels, key=lambda item: item[0], reverse=True), label_positions):
            ax.annotate(
                label,
                xy=(1.0, y_mid),
                xytext=(1.22, label_y),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=7,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.35",
                    "lw": 0.7,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                annotation_clip=False,
            )


def plot_profile(profile: dict, output_path: Path) -> Path:
    intervals = build_intervals(profile)
    if not intervals:
        raise ValueError("No valid intervals found to plot.")

    color_cache: dict[str, tuple] = {}
    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(11.5, 11),
        gridspec_kw={"width_ratios": [0.72, 0.72]},
        sharex=True,
    )

    elevations = [interval["top_m"] for interval in intervals] + [interval["bottom_m"] for interval in intervals]
    profile_top = max(elevations) + 250
    profile_bottom = min(elevations) - 500
    zoom_bottom = -10000

    for ax in (ax_full, ax_zoom):
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.axhline(0, color="steelblue", linestyle="--", linewidth=1.1)
        ax.grid(axis="y", color="0.85", linewidth=0.7)

    ax_full.set_ylim(profile_bottom, profile_top)
    ax_zoom.set_ylim(zoom_bottom, profile_top)
    draw_intervals(ax_full, intervals, color_cache, min_label_thickness_m=5000)
    draw_intervals(ax_zoom, intervals, color_cache, min_label_thickness_m=0, label_mode="outside")
    ax_full.set_ylabel("Elevation relative to sea level (m)")
    ax_full.set_title("Full profile")
    ax_zoom.set_title("Top 10 km zoom")

    ax_zoom.text(1.02, 0, "Sea level", va="center", ha="left", fontsize=9, color="steelblue")

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.6, label=rock_name)
        for rock_name, color in color_cache.items()
    ]
    fig.legend(
        handles=legend_handles,
        title="Rock type",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    fig.suptitle(
        "Geologic Profile\n"
        f"Requested ({profile['requested_point']['latitude']:.4f}, {profile['requested_point']['longitude']:.4f})\n"
        f"Nearest grid ({profile['nearest_grid_point']['latitude']:.4f}, {profile['nearest_grid_point']['longitude']:.4f})",
        y=0.97,
    )
    fig.subplots_adjust(top=0.9, bottom=0.14, left=0.09, right=0.88, wspace=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a point-based vertical geologic profile from the extracted JSON summary."
    )
    parser.add_argument("--lat", type=float, help="Latitude in degrees north.")
    parser.add_argument("--lon", type=float, help="Longitude in decimal degrees east; western longitudes are negative.")
    parser.add_argument("--profile-json", type=Path, default=None, help="Path to the extracted profile JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    args = parser.parse_args()

    if args.profile_json is None:
        if args.lat is None or args.lon is None:
            raise ValueError("Provide either --profile-json or both --lat and --lon.")
        output_stem = build_output_stem(args.lat, args.lon)
        profile_json = output_stem.with_suffix(".json")
    else:
        profile_json = args.profile_json
        output_stem = profile_json.with_suffix("")

    output_path = args.output or output_stem.with_name(f"{output_stem.name}_profile.png")
    profile = load_profile(profile_json)
    written = plot_profile(profile, output_path)
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()

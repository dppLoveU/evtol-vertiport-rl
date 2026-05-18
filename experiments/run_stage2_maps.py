"""Stage-2 task 6: folium interactive maps.

Renders three maps from the stage-2 build artifacts (zones, candidates,
coverage matrix) into ``results/stage2/maps/``:

  1. zones_map.html      — H3 demand zones shaded by endpoint count.
  2. candidates_map.html — candidate sites colored by source category.
  3. coverage_map.html   — the candidate that covers the most zones,
     with the zones inside its walk radius highlighted.

This is a visualization-only step; it reads existing files and writes
no data artifacts.

Run:
    python -m experiments.run_stage2_maps
    python -m experiments.run_stage2_maps --config configs/spatial.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "spatial.yaml"

# Per-source marker colors for the candidates map.
_SOURCE_COLORS: dict[str, str] = {
    "poi_airport": "red",
    "poi_subway": "blue",
    "poi_mall": "green",
    "poi_hospital": "purple",
    "poi_industrial": "orange",
    "grid": "gray",
}


def _resolve(path_str: str) -> Path:
    """Resolve a yaml path relative to repo root if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _center(zones: gpd.GeoDataFrame) -> list[float]:
    """Map center as the mean of zone centroids ([lat, lon] for folium)."""
    return [
        float(zones["centroid_lat"].mean()),
        float(zones["centroid_lon"].mean()),
    ]


def _legend(entries: list[tuple[str, str]], title: str) -> folium.Element:
    """A small fixed-position HTML legend (label, color) pairs."""
    rows = "".join(
        f'<div><span style="display:inline-block;width:12px;height:12px;'
        f'background:{color};border-radius:50%;margin-right:6px;"></span>'
        f"{label}</div>"
        for label, color in entries
    )
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        'background:white;padding:10px 14px;border:1px solid #999;'
        'border-radius:4px;font-size:13px;font-family:sans-serif;">'
        f"<b>{title}</b>{rows}</div>"
    )
    return folium.Element(html)


def build_zones_map(zones: gpd.GeoDataFrame, out_path: Path) -> None:
    """H3 demand zones shaded by ``n_orders`` (endpoint count)."""
    m = folium.Map(location=_center(zones), zoom_start=10, tiles="cartodbpositron")

    counts = zones["n_orders"].to_numpy(dtype=float)
    # Quantile bins: demand is heavily skewed, so a linear ramp would wash
    # out all but the few densest zones.
    edges = np.unique(np.quantile(counts, np.linspace(0.0, 1.0, 9)))
    colormap = cm.linear.YlOrRd_09.to_step(index=list(edges))
    colormap.caption = "Demand zone endpoint count (n_orders)"

    def style(feature: dict[str, Any]) -> dict[str, Any]:
        return {
            "fillColor": colormap(feature["properties"]["n_orders"]),
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        zones,
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone_id", "h3_index", "n_orders"],
            aliases=["zone_id", "h3", "n_orders"],
        ),
        name="demand zones",
    ).add_to(m)
    colormap.add_to(m)
    m.save(str(out_path))


def build_candidates_map(cands: gpd.GeoDataFrame, out_path: Path) -> None:
    """Candidate vertiport sites colored by ``source`` category."""
    center = [
        float(cands["lat"].mean()),
        float(cands["lon"].mean()),
    ]
    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    counts = cands["source"].value_counts()
    for _, row in cands.iterrows():
        src = str(row["source"])
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color=_SOURCE_COLORS.get(src, "black"),
            fill=True,
            fill_color=_SOURCE_COLORS.get(src, "black"),
            fill_opacity=0.8,
            weight=0,
            tooltip=f"cand_id={row['cand_id']} | {src} | zone {row['zone_id']}",
        ).add_to(m)

    entries = [
        (f"{src} ({int(counts.get(src, 0))})", color)
        for src, color in _SOURCE_COLORS.items()
        if counts.get(src, 0) > 0
    ]
    m.get_root().html.add_child(_legend(entries, "Candidate source"))
    m.save(str(out_path))


def build_coverage_map(
    zones: gpd.GeoDataFrame,
    cands: gpd.GeoDataFrame,
    covers: np.ndarray,
    walk_radius_km: float,
    out_path: Path,
) -> int:
    """Highlight the zones covered by the highest-coverage candidate.

    Returns the chosen candidate's row index (== ``cand_id``).
    """
    coverage_counts = covers.sum(axis=1)
    best = int(np.argmax(coverage_counts))
    covered = set(np.where(covers[best])[0].tolist())
    cand = cands.iloc[best]

    m = folium.Map(location=_center(zones), zoom_start=11, tiles="cartodbpositron")

    def style(feature: dict[str, Any]) -> dict[str, Any]:
        is_covered = feature["properties"]["zone_id"] in covered
        return {
            "fillColor": "#2c7fb8" if is_covered else "#dddddd",
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.6 if is_covered else 0.15,
        }

    folium.GeoJson(
        zones,
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["zone_id", "n_orders"]),
        name="zones",
    ).add_to(m)

    # Walk-radius disk + marker for the chosen candidate.
    folium.Circle(
        location=[float(cand["lat"]), float(cand["lon"])],
        radius=walk_radius_km * 1000.0,
        color="#d7301f",
        weight=2,
        fill=True,
        fill_opacity=0.05,
    ).add_to(m)
    folium.Marker(
        location=[float(cand["lat"]), float(cand["lon"])],
        icon=folium.Icon(color="red", icon="star"),
        tooltip=(
            f"cand_id={cand['cand_id']} | {cand['source']} | "
            f"covers {len(covered)} zones within {walk_radius_km} km"
        ),
    ).add_to(m)

    m.get_root().html.add_child(
        _legend(
            [
                (f"covered zones ({len(covered)})", "#2c7fb8"),
                ("other zones", "#dddddd"),
                (f"candidate {cand['cand_id']} ({cand['source']})", "#d7301f"),
            ],
            f"Coverage: {walk_radius_km} km walk radius",
        )
    )
    m.save(str(out_path))
    return best


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Render the three stage-2 folium maps. Returns a summary dict."""
    zones_path = _resolve(cfg["zones"]["output_path"])
    cands_path = _resolve(cfg["candidates"]["output_path"])
    covers_path = _resolve(cfg["matrices"]["cand_covers_zones_path"])
    meta_path = _resolve(cfg["meta_path"])
    out_dir = _resolve(cfg["maps"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    zones = gpd.read_file(zones_path)
    cands = gpd.read_file(cands_path)
    covers = np.load(covers_path)
    with open(meta_path) as fh:
        walk_radius_km = float(json.load(fh)["walk_radius_km"])

    print(f"[maps] |Z|={len(zones)}  |C|={len(cands)}  covers={covers.shape}")

    zones_map = out_dir / "zones_map.html"
    cands_map = out_dir / "candidates_map.html"
    coverage_map = out_dir / "coverage_map.html"

    build_zones_map(zones, zones_map)
    print(f"  zones_map      -> {zones_map}")

    build_candidates_map(cands, cands_map)
    print(f"  candidates_map -> {cands_map}")

    best = build_coverage_map(zones, cands, covers, walk_radius_km, coverage_map)
    n_cov = int(covers[best].sum())
    print(
        f"  coverage_map   -> {coverage_map} "
        f"(cand_id={best} covers {n_cov} zones within {walk_radius_km} km)"
    )

    return {
        "zones_map": str(zones_map),
        "candidates_map": str(cands_map),
        "coverage_map": str(coverage_map),
        "coverage_cand_id": best,
        "coverage_n_zones": n_cov,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to spatial yaml (default: {DEFAULT_CONFIG.relative_to(REPO)})",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    run(cfg)


if __name__ == "__main__":
    main()
